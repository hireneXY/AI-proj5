from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class EarlyFusionConfig(BaseConfig):
    """早期融合实验配置"""
    fusion_type: str = "early"
    fusion_hidden_dim: int = 1024
    learning_rate: float = 2e-5
    text_finetune: bool = True
    image_finetune: bool = True

@dataclass
class AttentionFusionConfig(BaseConfig):
    """注意力融合实验配置"""
    fusion_type: str = "attention"
    attention_heads: int = 8
    attention_dropout: float = 0.1
    fusion_hidden_dim: int = 768
    learning_rate: float = 3e-5

@dataclass
class LateFusionConfig(BaseConfig):
    """晚期融合实验配置"""
    fusion_type: str = "late"
    text_hidden_dim: int = 256
    image_hidden_dim: int = 256
    learning_rate: float = 1e-4
    ensemble_method: str = "weighted"  # weighted, voting, stacking

class ExperimentConfigs:
    """实验配置集合"""
    
    @staticmethod
    def get_config(experiment_name: str):
        """获取指定实验配置"""
        configs = {
            "early_fusion": EarlyFusionConfig(),
            "attention_fusion": AttentionFusionConfig(),
            "late_fusion": LateFusionConfig(),
            "baseline_text": BaseConfig(fusion_type="text_only"),
            "baseline_image": BaseConfig(fusion_type="image_only"),
        }
        return configs.get(experiment_name, BaseConfig())