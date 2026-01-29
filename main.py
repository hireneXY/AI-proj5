#!/usr/bin/env python3
"""
多模态情感分类主入口 - 修改版（支持命令行参数）
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datetime import datetime
from configs.base_config import BaseConfig
print("✅ 使用基础配置 BaseConfig")

from data.dataset import MultimodalDataset
from models.fusion.early_fusion import EarlyFusionModel
from models.fusion.attention_fusion import AttentionFusionModel
from models.fusion.late_fusion import LateFusionModel
from models.fusion.gated_fusion import GatedFusionModel
from models.unimodal.text_model import TextOnlyModel
from models.unimodal.image_model import ImageOnlyModel
from core.trainer import MultimodalTrainer
from utils.logger import setup_logger
from utils.seed import set_seed

# ============= 添加命令行参数解析 =============
def parse_args():
    parser = argparse.ArgumentParser(description='多模态情感分类实验')
    parser.add_argument('--model', type=str, default='attention',
                       choices=['text', 'image', 'early', 'late', 'attention', 'gated', 'misa'],
                       help='模型类型: text(纯文本), image(纯图像), early(早期融合), late(晚期融合), attention(注意力融合), gated(门控融合)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps to simulate larger batch')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision (AMP) training')
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/multimodel_experiment/data', help='数据目录')
    parser.add_argument('--config', type=str, default=None, help='可选的 yaml 配置文件，用于覆盖部分配置（例如 loss_weights）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（覆盖 config.seed）')
    return parser.parse_args()
# ===========================================

def load_data(config):
    """Robust loader for train/test files (handles encodings and header)."""
    print(f"📂 加载训练数据从: {config.train_file}")
    print(f"📂 加载测试数据从: {config.test_file}")

    def _read_train(path):
        records = []
        for enc in ("utf-8", "utf-8-sig", "latin-1", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        # 忽略 header
                        if line.lower().startswith("guid,"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 2:
                            guid = parts[0].strip()
                            label = parts[1].strip().lower()
                            if label in ["positive", "neutral", "negative"]:
                                records.append((guid, label))
                            else:
                                print(f"⚠️  训练数据第{line_num}行标签无效: {label}")
                return records
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"⚠️  读取训练文件时发生错误（enc={enc}）: {e}")
                continue
        return records

    def _read_test(path):
        guids = []
        for enc in ("utf-8", "utf-8-sig", "latin-1", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        if line.lower().startswith("guid,"):
                            continue
                        parts = line.split(",")
                        if len(parts) > 0:
                            guid = parts[0].strip()
                            if guid:
                                guids.append(guid)
                            else:
                                print(f"⚠️  测试数据第{line_num}行guid为空")
                return guids
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"⚠️  读取测试文件时发生错误（enc={enc}）: {e}")
                continue
        return guids

    try:
        train_data = _read_train(config.train_file)
        print(f"✅ 成功加载 {len(train_data)} 个训练样本")
    except FileNotFoundError:
        print(f"❌ 找不到训练文件: {config.train_file}")
        train_data = []
    except Exception as e:
        print(f"❌ 加载训练数据失败: {e}")
        train_data = []

    try:
        test_data = _read_test(config.test_file)
        print(f"✅ 成功加载 {len(test_data)} 个测试样本")
    except FileNotFoundError:
        print(f"❌ 找不到测试文件: {config.test_file}")
        test_data = []
    except Exception as e:
        print(f"❌ 加载测试数据失败: {e}")
        test_data = []

    # show examples
    if test_data:
        print(f"  测试guid示例: {test_data[:5]}")

    return train_data, test_data

def create_model(config, model_type='attention'):
    """创建模型 - 修改为支持多种类型"""
    if model_type == 'text':
        from models.unimodal.text_model import TextOnlyModel
        return TextOnlyModel(config)
    elif model_type == 'misa':
        # MISA 实现位于 models/pretrained_mm/misa_impl.py (clean implementation)
        from models.pretrained_mm.misa_impl import MISAModel
        return MISAModel(config)
    elif model_type == 'image':
        from models.unimodal.image_model import ImageOnlyModel
        return ImageOnlyModel(config)
    elif model_type == 'early':
        return EarlyFusionModel(config)
    elif model_type == 'late':
        return LateFusionModel(config)
    elif model_type == 'attention':
        return AttentionFusionModel(config)
    elif model_type == 'gated':
        return GatedFusionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """主函数 - 修改为接受参数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    config = BaseConfig()
    config.fusion_type = args.model  # 使用命令行指定的模型类型
    
    # 如果提供了 YAML 配置文件，尝试加载并合并关键字段（training.loss_weights, augmentation.text.text_cleaning_enable）
    if getattr(args, "config", None):
        try:
            import yaml
            user_cfg = yaml.safe_load(open(args.config))
            # training.loss_weights -> config.loss_weights (dict)
            if user_cfg and isinstance(user_cfg, dict):
                training = user_cfg.get("training") or {}
                loss_weights = training.get("loss_weights")
                if loss_weights:
                    setattr(config, "loss_weights", loss_weights)
                # augmentation text cleaning
                aug = user_cfg.get("augmentation") or {}
                text_aug = aug.get("text") or {}
                text_clean_flag = text_aug.get("text_cleaning_enable", user_cfg.get("text_cleaning_enable", None))
                if text_clean_flag is not None:
                    setattr(config, "text_cleaning_enable", bool(text_clean_flag))
        except Exception as e:
            print(f"⚠️ 无法加载/解析 config yaml ({args.config}): {e}")
    
    # 🔍 添加：打印所有关键路径
    print("=" * 80)
    print("🔍 路径调试信息")
    print("=" * 80)
    print(f"config.data_dir: {config.data_dir}")
    print(f"config.train_file: {config.train_file}")
    print(f"config.test_file: {config.test_file}")

    # 覆盖配置参数
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.data_dir = args.data_dir
    # seed override
    config.seed = args.seed
    # 更新 accumulation 与 AMP 设置
    config.gradient_accumulation_steps = max(1, getattr(args, "accumulation_steps", 1))
    config.use_mixed_precision = getattr(args, "use_amp", False)
    
    # [其余原有代码不变...]
    # 设置日志
    logger = setup_logger(config)
    logger.info(f"Starting multimodal sentiment classification experiment")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {config.device}")
    
    # 加载数据
    train_data, test_data = load_data(config)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(
        train_data, test_size=config.val_ratio, random_state=config.seed, 
        stratify=[d[1] for d in train_data]
    )
    
    # 创建数据集
    train_dataset = MultimodalDataset(train_split, config, is_train=True)
    # 验证集需要标签以计算指标，因此is_train应为True
    val_dataset = MultimodalDataset(val_split, config, is_train=True)
    
    # 创建数据加载器
    # If config provides class_weights, use WeightedRandomSampler to address imbalance
    try:
        cfg_class_weights = None
        # support structure from YAML: training.class_weights or top-level class_weights
        import yaml as _yaml
        cfg_class_weights = getattr(config, "class_weights", None)
        if cfg_class_weights is None and hasattr(config, "training"):
            try:
                cfg_training = getattr(config, "training")
                if isinstance(cfg_training, dict):
                    cfg_class_weights = cfg_training.get("class_weights", None)
            except Exception:
                cfg_class_weights = None
    except Exception:
        cfg_class_weights = None

    if cfg_class_weights and isinstance(cfg_class_weights, dict):
        # build per-sample weights according to config mapping
        from torch.utils.data import WeightedRandomSampler
        sample_weights = []
        for guid, label in train_split:
            lid = config.get_label_id(label)
            # class name lookup
            # try to map by rev_label_map
            class_name = config.get_label_str(lid)
            w = float(cfg_class_weights.get(class_name, 1.0))
            sample_weights.append(w)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
        )
        
    # 创建模型
    model = create_model(config, args.model)  # 使用args.model
    # 如果本地 tokenizer 与模型词表不一致，确保模型 embedding 大小与 tokenizer 匹配
    try:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.text_model_name, local_files_only=True)
        except Exception:
            tokenizer = None

        if tokenizer is not None:
            # 如果模型是 HF 模型包装（例如在 TextOnlyModel 中有 .bert 属性），调整其 embedding 大小
            vocab_size = len(tokenizer)
            adjusted = False
            if hasattr(model, "bert") and hasattr(model.bert, "resize_token_embeddings"):
                try:
                    model.bert.resize_token_embeddings(vocab_size)
                    adjusted = True
                except Exception:
                    adjusted = False
            elif hasattr(model, "resize_token_embeddings"):
                try:
                    model.resize_token_embeddings(vocab_size)
                    adjusted = True
                except Exception:
                    adjusted = False

            if adjusted:
                print(f"🔧 调整模型 embedding 大小以匹配 tokenizer（vocab_size={vocab_size}）")
            # 如果模型包装中包含 classifier 且 classifier 输入维度与 transformer hidden_size 不匹配，重建 classifier
            try:
                hidden_size = None
                if hasattr(model, "bert") and hasattr(model.bert, "config"):
                    hidden_size = getattr(model.bert.config, "hidden_size", None)
                if hidden_size is not None and hasattr(model, "classifier"):
                    # 尝试匹配现有 classifier 的输出类别数
                    from torch import nn as _nn
                    out_classes = None
                    # 假设最后一层是 Linear(..., num_classes)
                    try:
                        last_linear = None
                        for module in reversed(list(model.classifier)):
                            if isinstance(module, _nn.Linear):
                                last_linear = module
                                break
                        if last_linear is not None:
                            out_classes = last_linear.out_features
                    except Exception:
                        out_classes = getattr(config, "num_classes", None)

                    if out_classes is None:
                        out_classes = getattr(config, "num_classes", 3)

                    # rebuild classifier to accept hidden_size
                    try:
                        new_clf = _nn.Sequential(
                            _nn.Dropout(config.dropout_rate),
                            _nn.Linear(hidden_size, 256),
                            _nn.ReLU(),
                            _nn.Dropout(config.dropout_rate),
                            _nn.Linear(256, out_classes)
                        )
                        model.classifier = new_clf
                        print(f"🔧 重建模型 classifier，以输入维度 {hidden_size} 匹配 Transformer 输出")
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        # 忽略任何调整错误，后续训练可能仍然可运行
        pass
    logger.info(f"Created {args.model} model")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # [继续原有代码...]
    # 创建训练器
    trainer = MultimodalTrainer(model, config)
    
    # 训练循环
    logger.info("Starting training...")
    for epoch in range(config.num_epochs):
        # 训练
        train_result = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch+1} Train - Loss: {train_result['loss']:.4f}, "
                   f"Acc: {train_result['accuracy']:.4f}, F1: {train_result['f1']:.4f}")
        
        # 验证
        val_result = trainer.validate(val_loader)
        logger.info(f"Epoch {epoch+1} Val - Loss: {val_result['loss']:.4f}, "
                   f"Acc: {val_result['accuracy']:.4f}, F1: {val_result['f1']:.4f}")
        
        # 学习率调整
        trainer.scheduler.step(val_result['accuracy'])
        
        # 早停检查
        if epoch > config.early_stop_patience:
            recent_accs = trainer.val_accs[-config.early_stop_patience:]
            if max(recent_accs) <= trainer.best_val_acc:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # [继续原有代码...]
    logger.info(f"Training completed. Best validation accuracy: {trainer.best_val_acc:.4f}")
    
    # 生成测试集预测（原有函数）
    generate_predictions(model, test_data, config, logger)

def generate_predictions(model, test_data, config, logger):
    """生成测试集预测 - test_data只包含guid列表"""
    logger.info("生成测试集预测...")
    
    from tqdm import tqdm
    
    # 测试数据只包含guid，没有标签
    print(f"🔍 测试数据: {len(test_data)} 个guid")
    
    # 创建测试数据集（注意：测试集没有标签）
    test_dataset = MultimodalDataset(test_data, config, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    
    model.eval()
    predictions = []
    guids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            images = batch['image'].to(config.device)
            
            # 根据模型类型调用
            if config.fusion_type in ['text', 'text_only']:
                logits, _ = model(input_ids, attention_mask)
            else:
                logits, _ = model(input_ids, attention_mask, images)
            
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            guids.extend(batch['guid'])
    
    # 验证预测数量与guid数量一致
    print(f"📊 预测统计:")
    print(f"  GUID数量: {len(guids)}")
    print(f"  预测数量: {len(predictions)}")
    print(f"  标签分布: {np.bincount(predictions, minlength=3)}")
    
    # 保存预测结果
    output_file = os.path.join(config.results_dir, f'test_predictions_{config.fusion_type}.csv')
    
    with open(output_file, 'w') as f:
        f.write("guid,label\n")
        for guid, pred in zip(guids, predictions):
            label_str = config.get_label_str(pred)
            f.write(f"{guid},{label_str}\n")
    
    logger.info(f"预测保存到: {output_file}")
    
    # 显示前几个预测结果
    print(f"📝 预测结果示例:")
    for i in range(min(5, len(guids))):
        print(f"  {guids[i]} → {config.get_label_str(predictions[i])}")
    
    return output_file

# [保留原有的generate_predictions函数]

if __name__ == "__main__":
    main()