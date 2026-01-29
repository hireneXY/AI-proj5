# models/fusion/late_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class LateFusionModel(nn.Module):
    """晚期决策融合模型"""
    
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
        
        # 文本分类器
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # 图像分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(image_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(self, input_ids, attention_mask, image):
        # 文本分支
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_logits = self.text_classifier(text_features)
        
        # 图像分支
        image_features = self.image_encoder(image)
        image_features = torch.flatten(image_features, 1)
        image_logits = self.image_classifier(image_features)
        
        # 权重归一化
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权融合
        combined_logits = weights[0] * text_logits + weights[1] * image_logits
        
        return combined_logits