import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextOnlyModel(nn.Module):
    """纯文本模型 - 不使用NLTK"""
    
    def __init__(self, config):
        super(TextOnlyModel, self).__init__()
        self.config = config
        
        # BERT模型
        self.bert = AutoModel.from_pretrained(config.text_model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.text_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
        
        # 冻结BERT层（可选）
        if not config.text_finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, images=None):
        """前向传播 - 添加images参数以保持接口一致"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits, {}  # 返回空字典保持接口一致