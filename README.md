# AI-proj5

## 文件功能说明

### configs/
- `base_config.py`: 定义基础配置类，包含所有模型和训练的默认参数，如数据路径、模型名称、训练参数等
- `absolute_config.py`: 继承自BaseConfig，使用绝对路径配置数据文件位置
- `experiment_configs.py`: 定义特定实验的配置类，如EarlyFusionConfig、AttentionFusionConfig等，每种配置针对特定模型类型进行了优化

### core/
- `trainer.py`: 实现多模态训练器的核心功能，包括训练循环、验证、损失计算、模型保存、混合精度训练等
- `fusion_trainer.py`: 继承自MultimodalTrainer，专门针对融合模型的特殊需求，如多模态优化器设置和模态dropout训练
- `metrics.py`: 提供多种评估指标的计算方法，包括准确率、精确率、召回率、F1分数以及混淆矩阵的可视化

### models/fusion/
- `early_fusion.py`: 实现早期融合策略，将文本和图像特征在模型的早期阶段进行拼接融合
- `late_fusion.py`: 实现晚期融合策略，分别处理文本和图像特征后再进行融合
- `gated_fusion.py`: 实现门控融合策略，使用门控机制控制不同模态信息的重要性
- `attention_fusion.py`: 实现注意力融合策略，使用注意力机制动态加权多模态特征

### models/pretrained_mm/
- `misa_impl.py`: MISA（Multimodal Sentiment Analysis）模型的核心实现，通过学习共享和私有表示进行多模态情感分析
- `blip_finetune.py`: 基于BLIP模型的微调实现
- `clip_fusion.py`: 基于CLIP的多模态融合实现

### models/unimodal/
- `image_model.py`: 纯图像模型，仅使用图像输入进行分类
- `text_model.py`: 纯文本模型，仅使用文本输入进行分类

### experiments/
- `main_experiment.py`: 主实验执行文件，整合各种实验设置
- `ablation_study.py`: 进行消融实验，验证不同组件对整体性能的贡献
- `hyperparam_search.py`: 自动搜索最优超参数组合

### scripts/
- `run_experiments.py`: 用于运行MISA相关实验的脚本，支持网格搜索重构、相似性和差异性损失权重
- `run_baseline.sh`: 运行基线实验的shell脚本
- `check_overlap.py`: 检查数据集中可能存在的重复或泄露问题

### utils/
- `logger.py`: 提供结构化日志记录功能，便于跟踪训练过程和调试
- `seed.py`: 设置随机种子，确保实验结果的可重现性
- `visualization.py`: 提供训练曲线、特征可视化等功能

## 实验运行命令

### 1. 基础实验运行
```bash
# 运行主要实验，默认使用注意力融合模型
python main.py

# 指定模型类型运行
python main.py --model early    # 早期融合
python main.py --model late     # 晚期融合
python main.py --model attention # 注意力融合
python main.py --model gated    # 门控融合
python main.py --model misa     # MISA模型
python main.py --model text     # 纯文本模型
python main.py --model image    # 纯图像模型


## 实验运行命令
# 替换参数就可以运行不同融合方法、配置等等的实验

python3 main.py --model early --epochs 20 --batch_size 4 --accumulation_steps 8 --lr 1e-4 --data_dir data --config --seed 42

python3 main.py --model late --epochs 20 --batch_size 4 --accumulation_steps 8 --lr 1e-4 --data_dir data --config --seed 42

python3 main.py --model gated --epochs 20 --batch_size 4 --accumulation_steps 8 --lr 1e-4 --data_dir data --config --seed 42

python3 main.py --model attention --epochs 20 --batch_size 4 --accumulation_steps 8 --lr 1e-4 --data_dir data --config --seed 42

python3 main.py --model misa --epochs 20 --batch_size 4 --accumulation_steps 8 --lr 1e-4 --data_dir data --config --seed 42

## 项目特点
- 多种融合策略: 支持早期融合、晚期融合、注意力融合、门控融合等多种多模态融合策略
- MISA模型实现: 包含完整的MISA模型实现，支持共享/私有表示学习及重构、相似性和差异性损失
- 灵活的配置系统: 通过配置类管理所有模型和训练参数，支持命令行参数和YAML配置文件
- 全面的实验框架: 提供消融实验、超参数搜索、对比实验等完整的实验管理功能
- 混合精度训练: 支持AMP（Automatic Mixed Precision）以提高训练效率并减少显存占用
- 梯度累积: 支持梯度累积以模拟更大的有效批次大小
- 可重现性: 通过随机种子设置确保实验结果可重现
- 丰富的评估指标: 提供准确率、F1分数、混淆矩阵等多种评估指标