"""
文本编码器模块
支持两种文本编码：
1. 语义编码器：编码图像内容描述（原有结构）
2. 情感编码器：编码情感描述（新增，提示词升级后）

训练策略：将情感单独作为特征进行训练
"""
import torch
import torch.nn as nn


class CLIPEmbeddingTextEncoder(nn.Module):
    """使用 CLIP 的文本编码器"""

    def __init__(self):
        super().__init__()
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        except ImportError:
            raise ImportError("请安装 CLIP: pip install git+https://github.com/openai/CLIP.git")

        import clip as clip_module
        self.tokenize = clip_module.tokenize
        self.embed_dim = 512

    def forward(self, text_features):
        """
        Args:
            text_features: 已经编码好的文本特征 [B, 512]
        Returns:
            text_features: 归一化后的文本特征 [B, 512]
        """
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def encode_text(self, text_list):
        """
        从原始文本字符串编码
        Args:
            text_list: 文本列表（可以是内容描述或情感描述）
        Returns:
            text_features: 文本特征 [B, 512]
        """
        tokens = self.tokenize(text_list).to(next(self.parameters()).device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
        return features.float()


class SimpleTextEncoder(nn.Module):
    """
    简单的文本编码器（不依赖 CLIP）
    用于在没有 CLIP 环境时的替代方案
    """

    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, max_length=77):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )
        self.projection = nn.Linear(hidden_dim, 512)
        self.embed_dim = 512
        self.max_length = max_length

    def forward(self, text_features):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()


class SemanticEncoder(nn.Module):
    """
    语义编码器：编码图像内容描述
    对应原有结构中的"文字图"
    """

    def __init__(self, encoder_type="clip"):
        super().__init__()
        if encoder_type == "clip":
            self.encoder = CLIPEmbeddingTextEncoder()
        elif encoder_type == "simple":
            self.encoder = SimpleTextEncoder()
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        self.embed_dim = self.encoder.embed_dim

    def forward(self, text_features):
        """
        编码内容描述
        Args:
            text_features: 内容描述特征 [B, text_dim]
        Returns:
            semantic_features: 语义特征 [B, embed_dim]
        """
        return self.encoder(text_features)

    def encode_text(self, text_list):
        """从原始文本编码"""
        if hasattr(self.encoder, 'encode_text'):
            return self.encoder.encode_text(text_list)
        raise NotImplementedError("当前编码器不支持从原始文本编码")


class EmotionEncoder(nn.Module):
    """
    情感编码器：编码情感描述（独立特征）
    对应提示词升级后新增的情感描述部分
    
    训练策略：将情感单独作为特征进行训练
    """

    def __init__(self, encoder_type="clip"):
        super().__init__()
        if encoder_type == "clip":
            self.encoder = CLIPEmbeddingTextEncoder()
        elif encoder_type == "simple":
            self.encoder = SimpleTextEncoder()
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        self.embed_dim = self.encoder.embed_dim

    def forward(self, emotion_features):
        """
        编码情感描述（独立特征）
        Args:
            emotion_features: 情感描述特征 [B, text_dim]
        Returns:
            emotion_features: 情感特征 [B, embed_dim]
        """
        return self.encoder(emotion_features)

    def encode_text(self, text_list):
        """从原始情感描述文本编码"""
        if hasattr(self.encoder, 'encode_text'):
            return self.encoder.encode_text(text_list)
        raise NotImplementedError("当前编码器不支持从原始文本编码")


class TextEncoder(nn.Module):
    """
    统一的文本编码器接口
    包含语义编码器和情感编码器两个独立编码器
    """

    def __init__(self, encoder_type="clip"):
        super().__init__()
        # 语义编码器：编码内容描述
        self.semantic_encoder = SemanticEncoder(encoder_type)
        # 情感编码器：编码情感描述（独立特征）
        self.emotion_encoder = EmotionEncoder(encoder_type)
        
        self.embed_dim = self.semantic_encoder.embed_dim

    def forward(self, semantic_features, emotion_features):
        """
        分别编码语义特征和情感特征
        Args:
            semantic_features: 内容描述特征 [B, text_dim]
            emotion_features: 情感描述特征 [B, text_dim]
        Returns:
            semantic_encoded: 编码后的语义特征 [B, embed_dim]
            emotion_encoded: 编码后的情感特征 [B, embed_dim]
        """
        semantic_encoded = self.semantic_encoder(semantic_features)
        emotion_encoded = self.emotion_encoder(emotion_features)
        return semantic_encoded, emotion_encoded

    def encode_semantic(self, text_list):
        """编码内容描述"""
        return self.semantic_encoder.encode_text(text_list)

    def encode_emotion(self, text_list):
        """编码情感描述"""
        return self.emotion_encoder.encode_text(text_list)
