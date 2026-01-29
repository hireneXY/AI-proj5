import torch
import torch.nn as nn
from torchvision import models

class ImageOnlyModel(nn.Module):
    """纯图像模型"""
    
    def __init__(self, config):
        super(ImageOnlyModel, self).__init__()
        self.config = config
        
        # ResNet模型
        if config.image_model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif config.image_model_name == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif config.image_model_name == 'efficientnet':
            backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {config.image_model_name}")
        
        # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, config.num_classes)
        )
        
        # 冻结部分层（可选）
        if not config.image_finetune:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        """前向传播 - 统一接口，但只使用image参数"""
        # 确保有图像输入
        if image is None:
            raise ValueError("ImageOnlyModel需要image参数")
        
        # 特征提取
        features = self.feature_extractor(image)
        features = features.view(features.size(0), -1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, {}  # 返回空字典保持接口一致