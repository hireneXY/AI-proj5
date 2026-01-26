import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(true_labels, predictions, average='weighted'):
    """计算多种评估指标"""
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, average=average),
        'precision': precision_score(true_labels, predictions, average=average),
        'recall': recall_score(true_labels, predictions, average=average),
    }
    
    # 每个类别的指标
    if len(np.unique(true_labels)) == 3:  # 三分类
        for i, label in enumerate(['positive', 'neutral', 'negative']):
            metrics[f'{label}_precision'] = precision_score(
                true_labels, predictions, average=None)[i]
            metrics[f'{label}_recall'] = recall_score(
                true_labels, predictions, average=None)[i]
            metrics[f'{label}_f1'] = f1_score(
                true_labels, predictions, average=None)[i]
    
    return metrics

def plot_confusion_matrix(true_labels, predictions, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return cm

def compute_modality_importance(model, val_loader, device):
    """计算模态重要性"""
    model.eval()
    
    text_only_acc = 0
    image_only_acc = 0
    multimodal_acc = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 多模态预测
            logits, _ = model(input_ids, attention_mask, images)
            multimodal_preds = torch.argmax(logits, dim=1)
            multimodal_acc += (multimodal_preds == labels).sum().item()
    
    multimodal_acc = multimodal_acc / len(val_loader.dataset)
    
    # 这里可以扩展计算单模态性能
    # 需要修改模型以支持单模态推理
    
    return {
        'multimodal_accuracy': multimodal_acc,
        'text_importance': 0.0,  # 需要具体实现
        'image_importance': 0.0   # 需要具体实现
    }