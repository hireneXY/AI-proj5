# models/fusion/early_fusion.py
import torch
import torch.nn as nn
from transformers import AutoModel

class EarlyFusionModel(nn.Module):
    """早期特征拼接融合模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            *list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).children())[:-1]
        )
        image_dim = 2048  # ResNet50特征维度
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        # 文本特征
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 图像特征
        image_features = self.image_encoder(image)
        image_features = torch.flatten(image_features, 1)
        
        # 特征拼接
        combined = torch.cat([text_features, image_features], dim=1)
        
        # 分类
        logits = self.fusion(combined)
        
        return logits