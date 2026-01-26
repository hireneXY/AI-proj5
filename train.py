#!/usr/bin/env python3
"""
多模态情感分析训练脚本
"""

import os
import sys
import torch
import argparse
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import Config
from utils.experiment_manager import ExperimentManager
from data.processor import DataProcessor
from models import create_model
from training.trainer import MultimodalTrainer
from utils.logger import TrainingLogger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态情感分析训练')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称 (覆盖配置文件中的设置)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--list_experiments', action='store_true',
                       help='列出所有实验')
    
    return parser.parse_args()

def list_experiments():
    """列出所有实验"""
    # 临时创建实验管理器来列出实验
    temp_config = {'data': {'output_dir': './experiments'}}
    exp_manager = ExperimentManager(temp_config)

    experiments = exp_manager.list_experiments()

    if not experiments:
        print("没有找到任何实验")
        return

    print(f"找到 {len(experiments)} 个实验:")
    print("-" * 80)
    print(f"{'实验ID':<20} {'状态':<15} {'最佳F1':<15} {'完成时间':<30}")
    print("-" * 80)

    for exp in experiments:
        info = exp['info']
        best_f1 = info.get('best_val_f1_macro', 'N/A')
        completed_time = info.get('completed_time', 'N/A')
        status = info.get('status', 'unknown')
        print(f"{exp['id']:<20} {status:<15} {best_f1:<15} {completed_time:<30}")

def main():
    """主函数"""
    args = parse_args()
    
    # 如果只是列出实验
    if args.list_experiments:
        list_experiments()
        return

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = Config.from_yaml(args.config)
    
    # 覆盖实验名称
    if args.experiment_name:
        config.experiment.name = args.experiment_name

    # 覆盖配置
    if args.output_dir:
        config.data.output_dir = args.output_dir
    
    if args.device:
        config.device.use_cuda = (args.device.lower() == 'cuda')
    
    # 创建实验管理器
    exp_manager = ExperimentManager(config)
    print(f"实验ID: {exp_manager.experiment_id}")
    print(f"实验目录: {exp_manager.experiment_dir}")

    # 保存实验信息
    exp_manager.save_experiment_info()

    # 设置设备
    device = torch.device('cuda' if (
        torch.cuda.is_available() and config.device.use_cuda
    ) else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取实验目录
    model_dir = exp_manager.model_dir
    log_dir = exp_manager.log_dir
    
    # 设置日志
    logger = TrainingLogger(log_dir, use_tensorboard=config.logging.use_tensorboard)
    logger.log_config(config.__dict__)
    
    # 数据准备
    print("准备数据...")
    data_processor = DataProcessor(config)
    
    # 数据分析
    analysis = data_processor.analyze_data()
    print(f"训练样本数: {analysis['train_samples']}")
    print(f"测试样本数: {analysis['test_samples']}")
    print(f"标签分布: {analysis['label_distribution']}")
    
    # 检查缺失文件
    missing = analysis['missing_data']
    if missing['train'] or missing['test']:
        print(f"警告: 发现缺失文件 - 训练集: {len(missing['train'])}, "
              f"测试集: {len(missing['test'])}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_processor.create_dataloaders()
    
    # 创建模型
    print(f"创建模型: {config.model.name}")
    model = create_model(config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 创建训练器
    trainer = MultimodalTrainer(model, config, device, exp_manager)
    
    # 恢复训练（如果指定）
    if args.resume and os.path.exists(args.resume):
        print(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 训练模型
    print("开始训练...")
    history = trainer.train(train_loader, val_loader, model_dir)
    
    # 保存最终模型
    trainer.save_checkpoint(model_dir, is_best=False)
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    # 这里可以添加测试集评估代码
    
    # 关闭日志
    logger.close()
    
    print("\n训练完成!")
    print(f"实验ID: {exp_manager.experiment_id}")
    print(f"最佳模型: {exp_manager.get_best_model_path()}")
    print(f"实验目录: {exp_manager.experiment_dir}")
    print(f"日志目录: {log_dir}")

if __name__ == "__main__":
    main()