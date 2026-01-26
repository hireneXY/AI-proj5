#!/usr/bin/env python3
"""
多模态情感分类主入口 - 修改版（支持命令行参数）
"""

import sys
import os
import argparse  # 添加这行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datetime import datetime

try:
    from configs.absolute_config import AbsoluteConfig as BaseConfig
    print("✅ 使用绝对路径配置")
except ImportError:
    from configs.base_config import BaseConfig
    print("⚠️ 使用默认配置（可能需要修改路径）")

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
                       choices=['text', 'image', 'early', 'late', 'attention', 'gated'],
                       help='模型类型: text(纯文本), image(纯图像), early(早期融合), late(晚期融合), attention(注意力融合), gated(门控融合)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/multimodel_experiment/data', help='数据目录')
    return parser.parse_args()
# ===========================================

def load_data(config):
    """加载数据 - 区分训练和测试数据格式"""
    print(f"📂 加载训练数据从: {config.train_file}")
    print(f"📂 加载测试数据从: {config.test_file}")
    
    # ===== 1. 加载训练数据（guid,label格式） =====
    train_data = []
    try:
        with open(config.train_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    guid = parts[0].strip()
                    label = parts[1].strip().lower()
                    
                    # 验证标签有效性
                    if label in ['positive', 'neutral', 'negative']:
                        train_data.append((guid, label))
                    else:
                        print(f"⚠️  训练数据第{line_num}行标签无效: {label}")
        
        print(f"✅ 成功加载 {len(train_data)} 个训练样本")
    except Exception as e:
        print(f"❌ 加载训练数据失败: {e}")
        return [], []
    
    # ===== 2. 加载测试数据（guid,null格式，只取guid） =====
    test_data = []
    try:
        with open(config.test_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 处理 guid,null 格式
                parts = line.split(',')
                if len(parts) > 0:
                    guid = parts[0].strip()
                    
                    # 只取guid，忽略后面的null标签
                    if guid:  # guid非空
                        test_data.append(guid)
                    else:
                        print(f"⚠️  测试数据第{line_num}行guid为空")
        
        print(f"✅ 成功加载 {len(test_data)} 个测试样本")
        print(f"  测试文件原始行数: {line_num}")
        print(f"  有效guid数量: {len(test_data)}")
        
        # 显示前几个guid
        if test_data:
            print(f"  测试guid示例: {test_data[:5]}")
        
    except Exception as e:
        print(f"❌ 加载测试数据失败: {e}")
    
    return train_data, test_data

def create_model(config, model_type='attention'):
    """创建模型 - 修改为支持多种类型"""
    if model_type == 'text':
        from models.unimodal.text_model import TextOnlyModel
        return TextOnlyModel(config)
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
    set_seed(42)
    
    # 创建配置
    config = BaseConfig()
    config.fusion_type = args.model  # 使用命令行指定的模型类型
    
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
    val_dataset = MultimodalDataset(val_split, config, is_train=False)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    
    # 创建模型
    model = create_model(config, args.model)  # 使用args.model
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