#!/bin/bash
# 增强正则化训练

# 文本模型（增加Dropout和权重衰减）
python main.py --model text --dropout 0.5 --weight_decay 1e-4 --text_hidden 128

# Late Fusion（更强的正则化）
python main.py --model late --dropout 0.5 --weight_decay 1e-4 \
  --text_hidden 128 --image_hidden 128 --lr 5e-5

# 添加标签平滑
python main.py --model late --label_smoothing 0.1
