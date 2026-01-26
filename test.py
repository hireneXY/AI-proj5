# analyze_neutral_samples_correct.py
import os
import pandas as pd
from collections import Counter

print("分析neutral样本特征（使用正确路径）")
print("=" * 60)

# 数据路径
data_dir = "/mnt/workspace/multimodel_experiment/data/dataset/"
train_file = "/mnt/workspace/multimodel_experiment/data/train.txt"

print(f"数据目录: {data_dir}")
print(f"训练文件: {train_file}")

# 检查路径
if not os.path.exists(data_dir):
    print(f"❌ 数据目录不存在: {data_dir}")
    exit(1)

if not os.path.exists(train_file):
    print(f"❌ 训练文件不存在: {train_file}")
    exit(1)

# 读取训练数据
train_df = pd.read_csv(train_file, header=None, names=['guid', 'label'])

# 分离各类样本
neutral_samples = train_df[train_df['label'] == 'neutral']
positive_samples = train_df[train_df['label'] == 'positive']
negative_samples = train_df[train_df['label'] == 'negative']

print(f"\n样本统计:")
print(f"  neutral:  {len(neutral_samples)} 个 ({len(neutral_samples)/len(train_df)*100:.1f}%)")
print(f"  positive: {len(positive_samples)} 个 ({len(positive_samples)/len(train_df)*100:.1f}%)")
print(f"  negative: {len(negative_samples)} 个 ({len(negative_samples)/len(train_df)*100:.1f}%)")

# 分析neutral样本的文本特征
print(f"\n分析neutral文本特征（前10个）:")
text_lengths = []
neutral_texts = []

for guid in neutral_samples['guid'].head(10):
    txt_file = os.path.join(data_dir, f"{guid}.txt")
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        text_lengths.append(len(text))
        neutral_texts.append(text)
        print(f"  GUID {guid}: {text[:80]}... (长度: {len(text)})")
    except Exception as e:
        print(f"  GUID {guid}: 无法读取 - {e}")

if text_lengths:
    print(f"\nneutral文本统计:")
    print(f"  平均长度: {sum(text_lengths)/len(text_lengths):.1f} 字符")
    print(f"  最短: {min(text_lengths)} 字符")
    print(f"  最长: {max(text_lengths)} 字符")

# 对比positive文本特征
print(f"\n对比positive文本特征（前10个）:")
positive_texts = []
for guid in positive_samples['guid'].head(10):
    txt_file = os.path.join(data_dir, f"{guid}.txt")
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        positive_texts.append(text)
        print(f"  GUID {guid}: {text[:80]}... (长度: {len(text)})")
    except Exception as e:
        print(f"  GUID {guid}: 无法读取 - {e}")

# 简单词汇分析
print(f"\n简单词汇分析:")

def get_common_words(texts, top_n=10):
    """获取常见词汇"""
    all_words = []
    for text in texts:
        # 简单分词（按空格分割）
        words = text.lower().split()
        # 过滤短词和常见停用词
        stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered = [w for w in words if len(w) > 2 and w not in stop_words]
        all_words.extend(filtered)
    
    return Counter(all_words)

if neutral_texts and positive_texts:
    neutral_counter = get_common_words(neutral_texts, top_n=15)
    positive_counter = get_common_words(positive_texts, top_n=15)
    
    print(f"\nneutral常见词（前15）:")
    for word, count in neutral_counter.most_common(15):
        print(f"  '{word}': {count}次")
    
    print(f"\npositive常见词（前15）:")
    for word, count in positive_counter.most_common(15):
        print(f"  '{word}': {count}次")
    
    # 找出neutral特有的词
    neutral_words = set([w for w, _ in neutral_counter.most_common(20)])
    positive_words = set([w for w, _ in positive_counter.most_common(20)])
    neutral_specific = neutral_words - positive_words
    
    if neutral_specific:
        print(f"\nneutral特有词汇: {', '.join(neutral_specific)}")
    else:
        print(f"\n⚠️ 没有明显的neutral特有词汇")

# 检查图像文件
print(f"\n检查图像文件:")
for guid in neutral_samples['guid'].head(5):
    jpg_file = os.path.join(data_dir, f"{guid}.jpg")
    if os.path.exists(jpg_file):
        print(f"  GUID {guid}: 有图像文件 ({os.path.getsize(jpg_file)} bytes)")
    else:
        print(f"  GUID {guid}: 无图像文件")

print("\n" + "=" * 60)
print("分析完成！")