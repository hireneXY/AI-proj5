#!/bin/bash

echo "🔧 修复所有模型文件的BERT加载..."

cd /mnt/workspace/multimodel_experiment

# 1. 修复early_fusion.py
if [ -f "models/fusion/early_fusion.py" ]; then
    echo "修复 early_fusion.py..."
    sed -i "s/AutoModel.from_pretrained(\"bert-base-uncased\")/AutoModel.from_pretrained(\"\/mnt\/workspace\/models\/bert-base-uncased\", local_files_only=True)/g" models/fusion/early_fusion.py
fi

# 2. 修复late_fusion.py
if [ -f "models/fusion/late_fusion.py" ]; then
    echo "修复 late_fusion.py..."
    sed -i "s/AutoModel.from_pretrained(\"bert-base-uncased\")/AutoModel.from_pretrained(\"\/mnt\/workspace\/models\/bert-base-uncased\", local_files_only=True)/g" models/fusion/late_fusion.py
fi

# 3. 修复attention_fusion.py
if [ -f "models/fusion/attention_fusion.py" ]; then
    echo "修复 attention_fusion.py..."
    sed -i "s/AutoModel.from_pretrained(\"bert-base-uncased\")/AutoModel.from_pretrained(\"\/mnt\/workspace\/models\/bert-base-uncased\", local_files_only=True)/g" models/fusion/attention_fusion.py
fi

# 4. 修复gated_fusion.py
if [ -f "models/fusion/gated_fusion.py" ]; then
    echo "修复 gated_fusion.py..."
    sed -i "s/AutoModel.from_pretrained(\"bert-base-uncased\")/AutoModel.from_pretrained(\"\/mnt\/workspace\/models\/bert-base-uncased\", local_files_only=True)/g" models/fusion/gated_fusion.py
fi

# 5. 修复text_model.py
if [ -f "models/unimodal/text_model.py" ]; then
    echo "修复 text_model.py..."
    sed -i "s/AutoModel.from_pretrained(\"bert-base-uncased\")/AutoModel.from_pretrained(\"\/mnt\/workspace\/models\/bert-base-uncased\", local_files_only=True)/g" models/unimodal/text_model.py
fi

echo "✅ 修复完成"
