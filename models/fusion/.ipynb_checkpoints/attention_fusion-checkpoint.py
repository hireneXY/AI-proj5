# models/fusion/attention_fusion.py
import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionFusionModel(nn.Module):
    """注意力机制融合模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            *list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).children())[:-1]
        )
        image_dim = 2048
        
        # 特征投影
        self.text_proj = nn.Linear(text_dim, config.hidden_dim)
        self.image_proj = nn.Linear(image_dim, config.hidden_dim)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        batch_size = input_ids.size(0)
        
        # 文本特征
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, H]
        
        # 图像特征
        image_features = self.image_encoder(image)
        image_features = torch.flatten(image_features, 1)
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, H]
        
        # 拼接特征用于注意力
        features = torch.cat([text_proj, image_proj], dim=1)  # [B, 2, H]
        
        # 自注意力
        attended, _ = self.attention(features, features, features)
        
        # 全局平均池化
        attended_pooled = attended.mean(dim=1)  # [B, H]
        
        # 与原始文本特征拼接
        combined = torch.cat([text_features, attended_pooled], dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits