import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import models

class EarlyFusionModel(nn.Module):
    """早期融合模型 - 特征拼接融合，不依赖NLTK"""
    
    def __init__(self, config):
        super(EarlyFusionModel, self).__init__()
        self.config = config
        
        # 文本编码器 (BERT)
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        
        # 图像编码器 (ResNet)
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])  # 移除最后全连接层
        
        # 冻结部分层（可选）
        self._freeze_layers()
        
        # 动态推断特征维度（优先使用模型实际输出）
        text_feat_dim = getattr(self.text_encoder.config, "hidden_size", config.text_feature_dim)
        # image_feature_dim 默认由 config 提供，但从 backbone 可推断
        try:
            # 对 ResNet50，最后的 conv 输出通道为 2048
            image_feat_dim = config.image_feature_dim
        except Exception:
            image_feat_dim = config.image_feature_dim
        combined_dim = text_feat_dim + image_feat_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_dim // 2, config.num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _freeze_layers(self):
        """冻结预训练模型的部分层"""
        # 冻结BERT的前几层
        for param in list(self.text_encoder.parameters())[:100]:
            param.requires_grad = False
            
        # 冻结ResNet的前几层
        for param in list(self.image_encoder.parameters())[:50]:
            param.requires_grad = False
    
    def _init_weights(self):
        """初始化融合层权重"""
        for module in self.fusion_layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, image):
        # 文本特征提取
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 图像特征提取
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # 特征拼接（早期融合）
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # 分类
        logits = self.fusion_layer(combined_features)
        
        return logits, {
            'text_features': text_features,
            'image_features': image_features,
            'combined_features': combined_features
        }