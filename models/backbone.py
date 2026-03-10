import torch
import torchvision
from torch import nn
from .overlock_backbone import create_overlock_features_backbone, OVERLOCK_AVAILABLE
from .overlock_backbone_24x24 import create_overlock_features_backbone_24x24, OVERLOCK_AVAILABLE as OVERLOCK_24X24_AVAILABLE


class ConvNextTiny(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)
        
        # ConvNeXt-Tiny实际输出384维，添加适配层（如果需要不同输出维度）
        self.adapter = nn.Conv2d(384, output_dim, 1, bias=False) if output_dim != 384 else nn.Identity()

    def forward(self, x):
        features = self.backbone(x)  # (B, 384, H/16, W/16)
        return self.adapter(features)  # (B, output_dim, H/16, W/16)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 512, H/32, W/32)


def get_backbone(backbone='ConvNeXt', output_dim=384, fusion_mode='concat', pretrained_path=None, **kwargs):
    """
    获取骨干网络
    
    Args:
        backbone: 骨干网络类型 ('convnext', 'resnet18', 'overlock_t')
        output_dim: 输出特征维度，默认384
        fusion_mode: OverLoCK-T融合模式 ('concat', 'weighted_add', 'main_only')
        pretrained_path: OverLoCK-T预训练权重路径
        **kwargs: 其他参数
    
    Returns:
        backbone: 骨干网络模型
    """
    backbone = backbone.lower()
    
    if backbone == 'convnext':
        return ConvNextTiny(output_dim=output_dim)
    
    elif backbone == 'overlock_t':
        if not OVERLOCK_AVAILABLE:
            print("警告: OverLoCK不可用，回退到ConvNeXt")
            return ConvNextTiny(output_dim=output_dim)
        
        # 设置默认预训练权重路径
        if pretrained_path is None:
            pretrained_path = '/media/hk/soft/zx/GRSL-paper/fast/LRFR-overlock/OverLoCK/weight/overlock_t_in1k_224.pth'
        
        return create_overlock_features_backbone(
            output_dim=output_dim,
            fusion_mode=fusion_mode,
            pretrained_path=pretrained_path
        )
    
    elif backbone == 'overlock_t_24x24':
        if not OVERLOCK_24X24_AVAILABLE:
            print("警告: OverLoCK 24×24不可用，回退到ConvNeXt")
            return ConvNextTiny(output_dim=output_dim)
        
        # 设置默认预训练权重路径
        if pretrained_path is None:
            pretrained_path = '/media/hk/soft/zx/GRSL-paper/fast/LRFR-overlock/OverLoCK/weight/overlock_t_in1k_224.pth'
        
        return create_overlock_features_backbone_24x24(
            output_dim=output_dim,
            fusion_mode=fusion_mode,
            pretrained_path=pretrained_path,
            img_size=kwargs.get('img_size', 224)
        )
    
    elif backbone == 'resnet18':
        return ResNet18()
    
    else:
        print(f"警告: 未知的backbone类型 '{backbone}'，使用默认ConvNeXt")
        return ConvNextTiny(output_dim=output_dim)
