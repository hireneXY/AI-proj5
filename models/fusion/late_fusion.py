# late_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import models

# 导入临时配置覆盖
try:
    from temp_config import OVERRIDE_CONFIG
    USE_OVERRIDE = True
except:
    USE_OVERRIDE = False
    OVERRIDE_CONFIG = {}

def get_config_value(config, key, default):
    """安全获取配置值，优先使用覆盖配置"""
    if USE_OVERRIDE and key in OVERRIDE_CONFIG:
        return OVERRIDE_CONFIG[key]
    return getattr(config, key, default)

class LateFusionModel(nn.Module):
    """晚期融合模型 - 决策级融合"""
    
    def __init__(self, config):
        super(LateFusionModel, self).__init__()
        self.config = config
        
        # 安全获取配置参数
        text_model_name = getattr(config, 'text_model_name', 'bert-base-uncased')
        image_model_name = getattr(config, 'image_model_name', 'resnet50')
        num_classes = getattr(config, 'num_classes', 3)
        dropout_rate = getattr(config, 'dropout_rate', 0.3)
        
        # 文本编码器（BERT模型）
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # 文本分类器 - 使用实际 transformer 输出维度（fallback 到 config/text_feature_dim）
        text_feat_dim = getattr(self.text_encoder.config, "hidden_size", getattr(config, "text_feature_dim", 768))
        text_hidden = get_config_value(config, "TEXT_HIDDEN", 256)
        self.text_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(text_feat_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, num_classes)
        )
        
        # 图像编码器（ResNet）
        if image_model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            image_feature_dim = 2048
        else:  # resnet34
            backbone = models.resnet34(pretrained=True)
            image_feature_dim = 512
        
        # 移除最后的全连接层，保留特征提取器
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # 自适应池化层（将特征图转换为向量）
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 图像分类器
        image_hidden = get_config_value(config, "IMAGE_HIDDEN", 256)
        self.image_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(image_feature_dim, image_hidden),
            nn.ReLU(),
            nn.Linear(image_hidden, num_classes)
        )
        
        # 融合权重（可学习的）
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        self.image_weight = nn.Parameter(torch.tensor(0.5))
        self.ensemble_method = "weighted"
        
        # 如果是stacking方法，添加meta分类器
        if hasattr(config, 'ensemble_method') and config.ensemble_method == "stacking":
            self.meta_classifier = nn.Sequential(
                nn.Linear(2 * num_classes, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, input_ids, attention_mask, images):
        # 1. 提取文本特征和logits
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_logits = self.text_classifier(text_features)
        
        # 2. 提取图像特征和logits
        image_features = self.image_encoder(images)
        image_features = self.image_pool(image_features)  # [batch, 2048, 1, 1] → [batch, 2048]
        image_logits = self.image_classifier(image_features)
        
        # 3. 集成策略
        ensemble_method = getattr(self, 'ensemble_method', 'weighted')
        
        if ensemble_method == "weighted":
            # 使用可学习的权重
            text_weight = torch.sigmoid(self.text_weight)
            image_weight = torch.sigmoid(self.image_weight)
            total_weight = text_weight + image_weight
            final_logits = (text_weight * text_logits + image_weight * image_logits) / total_weight
            
        elif ensemble_method == "voting":
            # 简单平均
            final_logits = (text_logits + image_logits) / 2
            
        else:  # stacking
            if hasattr(self, 'meta_classifier'):
                combined = torch.cat([text_logits, image_logits], dim=1)
                final_logits = self.meta_classifier(combined)
            else:
                # 回退到加权平均
                text_weight = torch.sigmoid(self.text_weight)
                image_weight = torch.sigmoid(self.image_weight)
                total_weight = text_weight + image_weight
                final_logits = (text_weight * text_logits + image_weight * image_logits) / total_weight
        
        # 4. 返回结果
        return final_logits, {
            "text_logits": text_logits,
            "image_logits": image_logits,
            "text_weight": self.text_weight,
            "image_weight": self.image_weight
        }