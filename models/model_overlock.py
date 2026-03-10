import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone
from .attention_overlock import get_attention_overlock
from .aggregation_overlock import get_aggregation_overlock


class LRFROverLoCK(nn.Module):
    
    def __init__(self, config):
        super(LRFROverLoCK, self).__init__()
        self.config = config
        self.backbone_type = config.backbone.lower()
        
        # Use standard backbone network
        self.backbone = get_backbone(
            backbone=config.backbone,
            output_dim=config.num_channels,
            fusion_mode=getattr(config, 'fusion_mode', 'concat'),
            pretrained_path=getattr(config, 'pretrained_path', None),
            img_size=config.img_size
        )
        
        # Dynamically compute the spatial parameter
        self.spatial = self._calculate_spatial(config)
        
        # Use the OverLoCK-optimized attention module
        self.attention = get_attention_overlock(
            attention=config.attention, 
            channel=config.num_channels,
            spatial=self.spatial
        )
        
        # Use the OverLoCK-optimized aggregation module
        self.aggregation = get_aggregation_overlock(
            aggregation=config.aggregation, 
            num_channels=config.num_channels,
            num_clusters=config.num_clusters, 
            cluster_dim=config.cluster_dim
        )
        
        # Detect the actual module types in use
        if isinstance(self.attention, nn.Identity):
            attention_type = "禁用"
        elif self.attention.__class__.__name__ in ('GSRA', 'SEAttention', 'CoordAttention', 'ChannelAttention', 'SpatialAttention'):
            # Original attention module
            attention_type = "原始"
        else:
            # Optimized variants such as GSRAOverLoCK
            attention_type = "OverLoCK优化"
        
        if isinstance(self.aggregation, nn.Identity):
            aggregation_type = "禁用"
        elif self.aggregation.__class__.__name__ in ('SALAD', 'MixVPR', 'GeM', 'ConvAP'):
            # Original aggregation module
            aggregation_type = "原始"
        else:
            # OverLoCK-optimized variant
            aggregation_type = "OverLoCK优化"
        
    def _calculate_spatial(self, config):
        import torch
        
        backbone_type = config.backbone.lower()
        
        # Compute directly based on known backbone output patterns
        if backbone_type == 'overlock_t':
            # Original OverLoCK-T: 384x384 -> 12x12 (32x downsampling)
            spatial_size = config.img_size // 32
            spatial = spatial_size * spatial_size
            
        elif backbone_type == 'overlock_t_24x24':
            # OverLoCK-T 24x24 version: 384x384 -> 24x24 (16x downsampling)
            spatial_size = config.img_size // 16
            spatial = spatial_size * spatial_size
            
        elif backbone_type == 'convnext':
            # ConvNeXt: 384x384 -> 24x24 (16x downsampling)
            spatial_size = config.img_size // 16
            spatial = spatial_size * spatial_size
            
        elif backbone_type == 'resnet18':
            # ResNet18: 32x downsampling
            spatial_size = config.img_size // 32
            spatial = spatial_size * spatial_size
            
        else:
            # For unknown backbones, attempt runtime inference
            device = torch.device(config.device if hasattr(config, 'device') and config.device else 'cuda' if torch.cuda.is_available() else 'cpu')
            test_input = torch.randn(1, 3, config.img_size, config.img_size).to(device)
            
            with torch.no_grad():
                test_output = self.backbone(test_input)
                _, _, h, w = test_output.shape
                spatial = h * w
        
        return spatial

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        
        # Attention mechanism
        x = self.attention(x)
        
        # Feature aggregation
        x = self.aggregation(x)
        
        # L2 normalization
        return F.normalize(x.flatten(1), p=2, dim=1)


class GeoModelOverLoCK(nn.Module):
    
    def __init__(self, config):
        super(GeoModelOverLoCK, self).__init__()
        self.model = LRFROverLoCK(config=config)

    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        else:
            image_features = self.model(img1)
            return image_features


def create_overlock_optimized_model(config):
    
    # Check whether the backbone is OverLoCK
    if config.backbone.lower().startswith('overlock'):
        print("检测到OverLoCK backbone，使用优化版本")
        return GeoModelOverLoCK(config)
    else:
        print("非OverLoCK backbone，使用标准版本")
        from .model import GeoModel
        return GeoModel(config)


class LRFRAdaptive(nn.Module):
    
    def __init__(self, config):
        super(LRFRAdaptive, self).__init__()
        
        # Select model based on backbone type
        if config.backbone.lower().startswith('overlock'):
            self.model = LRFROverLoCK(config)
            self.mode = "overlock_optimized"
        else:
            from .model import LRFR
            self.model = LRFR(config)
            self.mode = "standard"
            
        print(f"LRFRAdaptive选择模式: {self.mode}")
    
    def forward(self, x):
        return self.model(x)


class GeoModelAdaptive(nn.Module):
    
    def __init__(self, config):
        super(GeoModelAdaptive, self).__init__()
        self.model = LRFRAdaptive(config=config)

    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        else:
            image_features = self.model(img1)
            return image_features


if __name__ == "__main__":
    import torch
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        backbone: str = 'overlock_t_24x24'
        attention: str = 'GSRA'
        aggregation: str = 'SALAD'
        fusion_mode: str = 'main_only_no_compress'
        
        num_channels: int = 640
        img_size: int = 384
        num_clusters: int = 128
        cluster_dim: int = 128
        
        pretrained_path: str = None
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        config = TestConfig()
        config.device = device
        model = GeoModelOverLoCK(config).to(device)
        
        test_input = torch.randn(2, 3, 384, 384).to(device)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"\n模型测试成功")
            print(f"   输入: {test_input.shape}")
            print(f"   输出: {output.shape}")
            print(f"   参数数: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("CUDA不可用，跳过测试")
