"""
OverLoCK-T 24×24版本的特征提取适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# 解决models文件夹冲突的导入方式
overlock_path = Path(__file__).parent.parent / 'OverLoCK'

# 检查OverLoCK是否可用
OVERLOCK_AVAILABLE = False
try:
    # 添加OverLoCK路径
    if str(overlock_path) not in sys.path:
        sys.path.insert(0, str(overlock_path))
    
    # 导入24×24版本的OverLoCK
    from model.overlock_24x24 import overlock_t_24x24
    OVERLOCK_AVAILABLE = True
except ImportError as e:
    print(f"OverLoCK 24×24模块导入失败: {e}")
    print("   请确保OverLoCK路径正确且模块已安装")


class FeatureAdapter_24x24(nn.Module):
    """24×24版本的特征适配器：将OverLoCK-T的双分支特征映射到1152维单一特征"""
    
    def __init__(self, main_dim=640, ctx_dim=512, output_dim=1152, fusion_mode='no_compress'):
        super().__init__()
        self.main_dim = main_dim
        self.ctx_dim = ctx_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'no_compress':
            # 无压缩模式：直接拼接保留所有信息
            self.adapter = None  # 不使用适配器
            
        elif fusion_mode == 'concat':
            # 拼接后映射: 640+512=1152 -> output_dim
            self.adapter = nn.Conv2d(main_dim + ctx_dim, output_dim, 1, bias=False)
            
        elif fusion_mode == 'weighted_add':
            # 分别映射后加权相加
            self.main_adapter = nn.Conv2d(main_dim, output_dim, 1, bias=False)
            self.ctx_adapter = nn.Conv2d(ctx_dim, output_dim, 1, bias=False)
            # 可学习权重
            self.main_weight = nn.Parameter(torch.tensor(0.6))  # 主分支权重稍高
            self.ctx_weight = nn.Parameter(torch.tensor(0.4))   # 上下文分支权重
            
        elif fusion_mode == 'main_only':
            # 仅使用主分支: 640 -> output_dim
            self.adapter = nn.Conv2d(main_dim, output_dim, 1, bias=False)
            
        elif fusion_mode == 'main_only_no_compress':
            self.adapter = None  # 不使用适配器，直接输出融合特征
            
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")
    
    def forward(self, main_features, ctx_features):
        """
        Args:
            main_features: [B, 640, 24, 24] 主分支特征
            ctx_features: [B, 512, 24, 24] 上下文分支特征
        Returns:
            features: [B, output_dim, 24, 24] 融合后的特征
        """
        if self.fusion_mode == 'no_compress':
            # 无压缩模式：直接拼接，保留所有信息
            features = torch.cat([main_features, ctx_features], dim=1)  # [B, 1152, 24, 24]
            
        elif self.fusion_mode == 'concat':
            # 拼接融合
            concat_features = torch.cat([main_features, ctx_features], dim=1)  # [B, 1152, 24, 24]
            features = self.adapter(concat_features)  # [B, output_dim, 24, 24]
            
        elif self.fusion_mode == 'weighted_add':
            # 加权融合
            main_adapted = self.main_adapter(main_features)  # [B, output_dim, 24, 24]
            ctx_adapted = self.ctx_adapter(ctx_features)     # [B, output_dim, 24, 24]
            features = self.main_weight * main_adapted + self.ctx_weight * ctx_adapted
            
        elif self.fusion_mode == 'main_only':
            # 仅主分支
            features = self.adapter(main_features)  # [B, output_dim, 24, 24]
            
        elif self.fusion_mode == 'main_only_no_compress':
            # 注意：这里的main_features实际上已经是融合特征（640维），包含主分支+上下文特征
            features = main_features  # [B, 640, 24, 24] 直接返回融合特征
            
        return features


class OverLoCKTFeaturesBackbone_24x24(nn.Module):
    """
    OverLoCK-T 24×24版本的特征提取骨干网络
    输出24×24×1152的高分辨率特征
    """
    
    def __init__(self, output_dim=1152, fusion_mode='no_compress', pretrained_path=None, img_size=224):
        super().__init__()
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        self.img_size = img_size
        
        if not OVERLOCK_AVAILABLE:
            raise ImportError("OverLoCK 24×24模块不可用，请检查依赖安装")
        
        # 创建OverLoCK-T 24×24模型
        self.features = overlock_t_24x24(pretrained=False)
        
        # 创建特征适配器
        self.adapter = FeatureAdapter_24x24(
            main_dim=640,
            ctx_dim=512, 
            output_dim=output_dim,
            fusion_mode=fusion_mode
        )
        
        # 动态计算实际输出尺寸（16倍下采样）
        spatial_size = img_size // 16
        spatial = spatial_size * spatial_size
        
        # 根据融合模式更新输出维度
        if fusion_mode == 'no_compress':
            self.output_dim = 640 + 512  # 1152维
        elif fusion_mode == 'main_only_no_compress':
            self.output_dim = 640  # 640维融合特征
        
        # 加载预训练权重（如果提供）
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)
    
    def forward(self, x):
        """
        前向传播，输出24×24特征
        
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            features: [B, output_dim, 24, 24] 提取的特征 (640维或1152维)
        """
        # 根据融合模式选择特征提取方式
        if self.fusion_mode == 'main_only_no_compress':
            # 与原始分类任务完全一致：直接使用forward_features的默认输出（融合特征）
            main_features, ctx_features = self.features.forward_features(x)  # 使用默认参数
            features = self.adapter(main_features, ctx_features=None)  # 直接使用已融合的main_features
        else:
            # 提取完整特征
            main_features, ctx_features = self.features.forward_features(x)
            features = self.adapter(main_features, ctx_features)
        
        return features
    
    def load_pretrained_weights(self, weight_path):
        """
        加载预训练权重（兼容性加载）
        
        修改了patch_embed4，特殊处理权重加载
        """
        if not os.path.exists(weight_path):
            print(f"预训练权重文件不存在: {weight_path}")
            return
        
        print(f"加载预训练权重: {weight_path}")
        
        try:
            # 加载原始权重
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 获取当前模型的状态字典
            model_state_dict = self.features.state_dict()
            
            # 兼容性加载：跳过不匹配的权重
            loaded_keys = []
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        model_state_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch)")
                else:
                    skipped_keys.append(f"{key} (not found)")
            
            # 加载兼容的权重
            self.features.load_state_dict(model_state_dict, strict=False)
            
            print(f"   成功加载: {len(loaded_keys)} 个参数")
            print(f"   跳过参数: {len(skipped_keys)} 个")
            
            if skipped_keys and len(skipped_keys) < 10:  # 只显示前10个跳过的键
                print(f"   跳过的键: {skipped_keys[:10]}")
                
        except Exception as e:
            print(f"预训练权重加载失败: {e}")
            print("   将使用随机初始化权重")


def create_overlock_features_backbone_24x24(output_dim=1152, fusion_mode='no_compress', pretrained_path=None, img_size=224):
    
    if not OVERLOCK_AVAILABLE:
        raise ImportError(
            "OverLoCK 24×24模块不可用。请确保:\n"
            "1. OverLoCK目录存在\n"
            "2. overlock_24x24.py文件已创建\n"
            "3. 相关依赖已安装"
        )
    
    return OverLoCKTFeaturesBackbone_24x24(
        output_dim=output_dim,
        fusion_mode=fusion_mode,
        pretrained_path=pretrained_path,
        img_size=img_size
    )
