import torch
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class BaseConfig:
    """基础配置类 - 不依赖NLTK"""
    
    # 数据路径（已设置为项目相对路径）
    data_dir: str = "data"
    dataset_subdir: str = "dataset"
    train_file: str = "data/train.txt"
    test_file: str = "data/test_without_label.txt"
    
    # 模型参数
    # 使用 HuggingFace 模型名或本地 Linux 路径；之前仓库中默认值为 Windows 路径，已修改为通用模型名
    text_model_name: str = "bert-base-uncased"
    image_model_name: str = "resnet50"
    num_classes: int = 3

    text_finetune: bool = True      # 是否微调文本模型
    image_finetune: bool = True     # 是否微调图像模型
    text_hidden: int = 256          # 文本隐藏层维度
    image_hidden: int = 256         # 图像隐藏层维度
    
    # 标签映射（不使用NLTK）
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    rev_label_map = {0: "positive", 1: "neutral", 2: "negative"}
    
    # 文本处理参数
    max_seq_length: int = 128
    text_feature_dim: int = 768  # BERT-base输出维度
    
    # 图像处理参数
    image_size: int = 224
    image_feature_dim: int = 2048  # ResNet50输出维度
    
    # 训练参数
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.3
    
    # 融合参数
    fusion_hidden_dim: int = 512
    fusion_type: str = "early"  # early, late, attention, gated
    
    # 实验设置
    val_ratio: float = 0.2
    seed: int = 42
    early_stop_patience: int = 5
    
    # 训练优化 - mixed precision 与 gradient accumulation
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    
    # 数据增强（默认关闭）
    # 图像增强开关与参数
    image_augmentation_enable: bool = True
    image_aug_random_resized_crop_scale: tuple = (0.8, 1.0)
    image_aug_hflip_prob: float = 0.5
    image_aug_color_jitter: tuple = (0.2, 0.2, 0.2, 0.02)
    image_aug_random_erasing_prob: float = 0.0
    # 是否在验证/测试阶段也使用训练时的增强（默认关闭，若开启会使验证不可比）
    image_augmentation_eval_enable: bool = True

    # 文本增强开关与参数
    text_augmentation_enable: bool = False
    text_aug_synonym_replace_prob: float = 0.0
    text_aug_random_deletion_prob: float = 0.0

    # MixUp / CutMix（只对图像实现）
    mixup_enable: bool = False
    mixup_alpha: float = 0.4
    cutmix_enable: bool = False
    cutmix_alpha: float = 1.0
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 路径设置
    model_save_dir: str = "saved_models"
    results_dir: str = "results"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def get_label_id(self, label_str: str) -> int:
        """将标签字符串转换为ID（不使用NLTK）"""
        return self.label_map.get(label_str.lower(), 1)  # 默认neutral
    
    def get_label_str(self, label_id: int) -> str:
        """将标签ID转换为字符串（不使用NLTK）"""
        return self.rev_label_map.get(label_id, "neutral")