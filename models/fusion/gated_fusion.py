import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import models

class GatedFusionModel(nn.Module):
    """门控融合模型 - 使用门控机制控制信息流"""
    
    def __init__(self, config):
        super(GatedFusionModel, self).__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        # use actual encoder hidden size to avoid config/model mismatch (e.g. DeBERTa-large)
        text_hidden_size = getattr(self.text_encoder.config, "hidden_size", config.text_feature_dim)
        self.text_proj = nn.Linear(text_hidden_size, config.fusion_hidden_dim)
        
        # 图像编码器
        backbone = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.image_proj = nn.Linear(2048, config.fusion_hidden_dim)
        
        # 门控机制
        self.text_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.Sigmoid()
        )
        
        self.image_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_dim, config.num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        # 文本特征
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_features)
        
        # 图像特征
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)
        
        # 门控
        text_gate = self.text_gate(text_features)
        image_gate = self.image_gate(image_features)
        
        # 门控后的特征
        gated_text = text_features * text_gate
        gated_image = image_features * image_gate
        
        # 融合
        combined = torch.cat([gated_text, gated_image], dim=1)
        logits = self.fusion_layer(combined)
        
        return logits, {
            'text_features': text_features,
            'image_features': image_features,
            'text_gate': text_gate,
            'image_gate': image_gate,
            'gated_text': gated_text,
            'gated_image': gated_image
        }