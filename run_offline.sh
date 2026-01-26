#!/bin/bash

echo "🚀 强制离线实验模式"

cd /mnt/workspace/multimodel_experiment

# 设置环境变量（再次确保）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE=/mnt/workspace/models
export HF_HOME=/mnt/workspace/models

# 打印环境
echo "环境变量:"
env | grep -E "TRANSFORMERS|HF_" | sort

# 创建force_offline.py
cat > force_offline.py << 'FORCE_EOF'
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/workspace/models'
FORCE_EOF

echo "步骤1: 运行消融实验"
python -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from experiments.ablation_study import run_ablation_study
run_ablation_study()
"

echo "✅ 实验完成"
