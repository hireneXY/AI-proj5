# models/pretrained_mm/misa.py
"""
MISA模型 (Modality-Invariant and -Specific Representations)
TODO: 实现MISA架构
"""

import torch
import torch.nn as nn

class MISAModel(nn.Module):
    """MISA模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        print("⚠️  MISAModel: 尚未实现")
        
        # 占位模型
        self.dummy = nn.Linear(10, config.num_classes)
    
    def forward(self, input_ids, attention_mask, image):
        print("警告: 使用MISA占位模型")
        batch_size = input_ids.shape[0]
        return self.dummy(torch.randn(batch_size, 10))