"""
视觉编码器模块
支持 CLIP ViT-B/32, CLIP RN50, ResNet-50 等预训练模型
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CLIPViTEncoder(nn.Module):
    """基于 CLIP ViT-B/32 的视觉编码器，输出多尺度特征"""

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        except ImportError:
            raise ImportError("请安装 CLIP: pip install git+https://github.com/openai/CLIP.git")

        if not pretrained:
            # 如果需要随机初始化，可以在此处理
            pass

        self.embed_dim = 512  # ViT-B/32 的视觉特征维度

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
        Returns:
            features: 全局视觉特征 [B, 512]
        """
        features = self.clip_model.encode_image(x)
        return features.float()


class CLIPResNetEncoder(nn.Module):
    """基于 CLIP ResNet-50 的视觉编码器"""

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            import clip
            self.clip_model, _ = clip.load("RN50", device="cpu")
        except ImportError:
            raise ImportError("请安装 CLIP: pip install git+https://github.com/openai/CLIP.git")

        self.embed_dim = 1024

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return features.float()


class ResNet50Encoder(nn.Module):
    """基于 torchvision ResNet-50 的视觉编码器"""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # 去掉最后的 FC 层，保留到 avgpool 之前的特征
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.embed_dim = 2048
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
        Returns:
            features: 全局视觉特征 [B, 2048]
        """
        feat_map = self.features(x)  # [B, 2048, 7, 7]
        features = self.pool(feat_map).flatten(1)  # [B, 2048]
        return features


class VisualEncoder(nn.Module):
    """
    统一的视觉编码器接口
    支持多种预训练视觉模型
    """

    SUPPORTED_MODELS = {
        "clip_vit_b32": CLIPViTEncoder,
        "clip_rn50": CLIPResNetEncoder,
        "resnet50": ResNet50Encoder,
    }

    def __init__(self, model_name="clip_vit_b32", pretrained=True, freeze=False):
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}，"
                f"可选: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.encoder = self.SUPPORTED_MODELS[model_name](pretrained=pretrained)
        self.embed_dim = self.encoder.embed_dim
        self.model_name = model_name

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)
