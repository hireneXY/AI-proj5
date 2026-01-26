#!/bin/bash

# 多模态情感分类实验 - 基线实验运行脚本

# 设置环境
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 实验阶段1: 消融实验
echo "阶段1: 运行消融实验"
python experiments/ablation_study.py \
    --data_dir data \
    --train_file train.txt \
    --test_file test_without_label.txt \
    --experiment_name ablation_study \
    --batch_size 32 \
    --num_epochs 15 \
    --seed 42

# 实验阶段2: 主实验
echo "阶段2: 运行主实验"
python experiments/main_experiment.py \
    --data_dir data \
    --train_file train.txt \
    --test_file test_without_label.txt \
    --experiment_name main_experiment \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --seed 42

# 实验阶段3: 超参数搜索（可选）
echo "阶段3: 运行超参数搜索"
python experiments/hyperparam_search.py \
    --data_dir data \
    --train_file train.txt \
    --model_type early_fusion \
    --n_trials 20 \
    --seed 42

echo "所有实验完成!"