import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import models

class AttentionFusionModel(nn.Module):
    """注意力融合模型 - 使用注意力机制融合多模态特征，不依赖NLTK"""
    
    def __init__(self, config):
        super(AttentionFusionModel, self).__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        self.text_proj = nn.Linear(config.text_feature_dim, config.fusion_hidden_dim)
        
        # 图像编码器
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        self.image_proj = nn.Linear(config.image_feature_dim, config.fusion_hidden_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_dim,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_dim, config.num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        batch_size = input_ids.size(0)
        
        # 提取文本特征
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS]
        text_features = self.text_proj(text_features).unsqueeze(1)  # [B, 1, D]
        
        # 提取图像特征
        image_features = self.image_encoder(image)
        image_features = image_features.view(batch_size, -1)
        image_features = self.image_proj(image_features).unsqueeze(1)  # [B, 1, D]
        
        # 拼接特征用于注意力
        multimodal_features = torch.cat([text_features, image_features], dim=1)  # [B, 2, D]
        
        # 自注意力融合
        attended_features, _ = self.cross_attention(
            multimodal_features, multimodal_features, multimodal_features
        )
        
        # 池化
        pooled_features = attended_features.mean(dim=1)  # [B, D]
        max_features = attended_features.max(dim=1)[0]   # [B, D]
        combined_features = torch.cat([pooled_features, max_features], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits, {
            'text_features': text_features.squeeze(1),
            'image_features': image_features.squeeze(1),
            'attention_weights': attended_features
        }