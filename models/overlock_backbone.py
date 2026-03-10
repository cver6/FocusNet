"""
- 输入: OverLoCK-T双分支 640维(main) + 512维(ctx)
- 输出: 384维
- 支持三种融合模式: concat, weighted_add, main_only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# 解决models文件夹冲突的导入方式
overlock_path = Path(__file__).parent.parent / 'OverLoCK'

try:
    # 添加OverLoCK路径到sys.path
    if str(overlock_path) not in sys.path:
        sys.path.insert(0, str(overlock_path))
    
    # 现在可以直接导入，因为没有名称冲突了
    from model.overlock import overlock_t
    OVERLOCK_AVAILABLE = True
except Exception as e:
    print(f"警告: OverLoCK模块导入失败: {e}")
    print("将使用ConvNeXt作为fallback")
    OVERLOCK_AVAILABLE = False
    overlock_t = None


class FeatureAdapter(nn.Module):
    """特征适配器：将OverLoCK-T的双分支特征映射到384维单一特征"""
    
    def __init__(self, main_dim=640, ctx_dim=512, output_dim=384, fusion_mode='concat'):
        super().__init__()
        self.main_dim = main_dim
        self.ctx_dim = ctx_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'no_compress':
            # 无压缩模式，直接拼接保留所有信息
            # 640+512=1152维，不进行维度压缩
            print(f"使用无压缩模式: {main_dim}+{ctx_dim}={main_dim+ctx_dim}维 → 直接输出")
            self.adapter = None  # 不使用适配器
            
        elif fusion_mode == 'main_only_no_compress':
            # 修正：与原始分类任务一致的融合特征模式 - 使用640维融合特征（主+上下文），不压缩
            self.adapter = None  # 不使用适配器，直接输出融合特征
            
        elif fusion_mode == 'concat':
            # 拼接后映射: 640+512=1152 -> 384
            self.adapter = nn.Conv2d(main_dim + ctx_dim, output_dim, 1, bias=False)
            
        elif fusion_mode == 'weighted_add':
            # 分别映射后加权相加
            self.main_adapter = nn.Conv2d(main_dim, output_dim, 1, bias=False)
            self.ctx_adapter = nn.Conv2d(ctx_dim, output_dim, 1, bias=False)
            # 可学习权重
            self.main_weight = nn.Parameter(torch.tensor(0.6))  # 主分支权重稍高
            self.ctx_weight = nn.Parameter(torch.tensor(0.4))   # 上下文分支权重
            
        elif fusion_mode == 'main_only':
            # 仅使用主分支: 640 -> 384
            self.adapter = nn.Conv2d(main_dim, output_dim, 1, bias=False)
            
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")
    
    def forward(self, main_features, ctx_features=None):
        """
        Args:
            main_features: [B, 640, H, W] 主分支特征
            ctx_features: [B, 512, H, W] 上下文分支特征 (可选，某些模式下可为None)
        Returns:
            features: [B, output_dim, H, W] 融合后的特征
        """
        if self.fusion_mode == 'no_compress':
            # 无压缩模式：直接拼接，保留所有信息
            if ctx_features is None:
                raise ValueError("no_compress模式需要上下文特征")
            features = torch.cat([main_features, ctx_features], dim=1)  # [B, 1152, H, W]
            
        elif self.fusion_mode == 'main_only_no_compress':
            # 注意：这里的main_features实际上已经是融合特征（640维），包含主分支+上下文特征
            features = main_features  # [B, 640, H, W] 
            

            
        elif self.fusion_mode == 'concat':
            # 拼接融合
            if ctx_features is None:
                raise ValueError("concat模式需要上下文特征")
            concat_features = torch.cat([main_features, ctx_features], dim=1)  # [B, 1152, H, W]
            features = self.adapter(concat_features)  # [B, 384, H, W]
            
        elif self.fusion_mode == 'weighted_add':
            # 加权融合
            if ctx_features is None:
                raise ValueError("weighted_add模式需要上下文特征")
            main_adapted = self.main_adapter(main_features)  # [B, 384, H, W]
            ctx_adapted = self.ctx_adapter(ctx_features)     # [B, 384, H, W]
            features = self.main_weight * main_adapted + self.ctx_weight * ctx_adapted
            
        elif self.fusion_mode == 'main_only':
            # 仅主分支
            features = self.adapter(main_features)  # [B, 384, H, W]
            
        return features


class OverLoCKTFeaturesBackbone(nn.Module):
    """
    OverLoCK-T特征提取骨干网络
    """
    
    def __init__(self, output_dim=384, fusion_mode='concat', pretrained_path=None):
        super().__init__()
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        
        if not OVERLOCK_AVAILABLE:
            raise ImportError("OverLoCK模块不可用，请检查依赖安装")
        
        # 创建OverLoCK-T模型（不加载预训练权重）
        self.features = overlock_t(pretrained=False)
        
        # 移除分类相关的头部（如果存在）
        if hasattr(self.features, 'head'):
            delattr(self.features, 'head')
        if hasattr(self.features, 'aux_head'):
            delattr(self.features, 'aux_head')
        
        # 创建特征适配器
        self.adapter = FeatureAdapter(
            main_dim=640,
            ctx_dim=512, 
            output_dim=output_dim,
            fusion_mode=fusion_mode
        )
        
        # 根据融合模式更新输出维度
        if fusion_mode == 'no_compress':
            self.output_dim = 640 + 512  # 1152维
            print(f"OverLoCK-T无压缩模式: 输出维度 = {self.output_dim}")
        elif fusion_mode == 'main_only_no_compress':
            self.output_dim = 640  # 640维，仅主特征
        
        # 加载预训练权重（如果提供）
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)
    
    def forward(self, x):
        """
        前向传播，仅进行特征提取
        
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            features: [B, output_dim, H, W] 提取的特征 (384维压缩、640维主特征 或 1152维无压缩)
        """
        # 根据融合模式选择特征提取方式
        if self.fusion_mode == 'main_only_no_compress':
            main_features, ctx_features = self.features.forward_features(x)  # 使用默认参数
            features = self.adapter(main_features, ctx_features=None)  # 直接使用已融合的main_features
        else:
            # 提取完整特征
            main_features, ctx_features = self.features.forward_features(x)
            features = self.adapter(main_features, ctx_features)
        
        return features
    
    def load_pretrained_weights(self, weight_path):
        """
        加载预训练权重，仅加载特征提取相关权重
        
        Args:
            weight_path: 预训练权重文件路径
        """
        if not os.path.exists(weight_path):
            print(f"警告: 预训练权重文件不存在: {weight_path}")
            return
        
        try:
            state_dict = torch.load(weight_path, map_location='cpu')
            
            # 如果state_dict包含'model'键，提取实际的权重
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not (key.startswith('head.') or key.startswith('aux_head.')):
                    filtered_state_dict[key] = value
            
            # 加载权重，允许部分不匹配
            missing_keys, unexpected_keys = self.features.load_state_dict(
                filtered_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"缺失的权重键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"意外的权重键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")


def create_overlock_features_backbone(output_dim=384, fusion_mode='concat', pretrained_path=None):
    """    
    Args:
        output_dim: 输出特征维度，默认384
        fusion_mode: 特征融合模式，可选 'concat', 'weighted_add', 'main_only'
        pretrained_path: 预训练权重路径，可选
    
    Returns:
        backbone: OverLoCK-T特征提取骨干网络
    """
    if not OVERLOCK_AVAILABLE:
        print("警告: OverLoCK不可用，请使用ConvNeXt作为替代")
        return None
    
    return OverLoCKTFeaturesBackbone(
        output_dim=output_dim,
        fusion_mode=fusion_mode, 
        pretrained_path=pretrained_path
    )


class ConvNeXtFallback(nn.Module):
    
    def __init__(self, output_dim=384):
        super().__init__()
        import torchvision
        
        # 使用ConvNeXt-Tiny作为fallback
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)
        
        # ConvNeXt-Tiny输出768维，需要映射到384维
        self.adapter = nn.Conv2d(768, output_dim, 1, bias=False)
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 768, H/16, W/16]
        features = self.adapter(features)  # [B, 384, H/16, W/16]
        return features


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    if OVERLOCK_AVAILABLE:
        # 测试OverLoCK-T backbone
        print("测试OverLoCK-T特征提取...")
        backbone = create_overlock_features_backbone(
            output_dim=384,
            fusion_mode='concat'
        )
        
        if backbone is not None:
            backbone = backbone.to(device)
            
            # 测试不同分辨率
            test_sizes = [224, 384]
            for size in test_sizes:
                x = torch.randn(1, 3, size, size).to(device)
                with torch.no_grad():
                    features = backbone(x)
                print(f"输入尺寸 {size}x{size}: 输出特征 {features.shape}")
        
        # 测试不同融合模式
        fusion_modes = ['concat', 'weighted_add', 'main_only']
        for mode in fusion_modes:
            print(f"\n测试融合模式: {mode}")
            backbone = create_overlock_features_backbone(fusion_mode=mode)
            if backbone is not None:
                backbone = backbone.to(device)
                x = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    features = backbone(x)
                print(f"融合模式 {mode}: 输出特征 {features.shape}")
    else:
        print("OverLoCK不可用，测试ConvNeXt fallback...")
        fallback = ConvNeXtFallback(output_dim=384).to(device)
        x = torch.randn(1, 3, 384, 384).to(device)
        with torch.no_grad():
            features = fallback(x)
        print(f"ConvNeXt fallback输出: {features.shape}")
