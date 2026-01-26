import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_distributions(model, dataloader, device, save_path=None):
    """绘制特征分布"""
    model.eval()
    text_features = []
    image_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            _, features = model(input_ids, attention_mask, images)
            
            if 'text_features' in features:
                text_features.append(features['text_features'].cpu())
            if 'image_features' in features:
                image_features.append(features['image_features'].cpu())
    
    if text_features:
        text_features = torch.cat(text_features, dim=0)
        image_features = torch.cat(image_features, dim=0)
        
        # 绘制特征分布
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # 文本特征分布
        axes[0].hist(text_features.mean(dim=1).numpy(), bins=50, alpha=0.7)
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Text Features Distribution')
        
        # 图像特征分布
        axes[1].hist(image_features.mean(dim=1).numpy(), bins=50, alpha=0.7, color='orange')
        axes[1].set_xlabel('Feature Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Image Features Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def visualize_attention(model, dataloader, device, num_examples=3):
    """可视化注意力权重"""
    model.eval()
    
    examples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_examples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            texts = batch['raw_text']
            
            _, features = model(input_ids, attention_mask, images)
            
            if 'attention_weights' in features:
                examples.append({
                    'text': texts[0],
                    'attention': features['attention_weights'][0].cpu().numpy(),
                    'guid': batch['guid'][0]
                })
    
    # 绘制注意力可视化
    for i, example in enumerate(examples):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 文本注意力（简化版）
        if 'attention' in example:
            attention_matrix = example['attention']
            axes[0].imshow(attention_matrix, cmap='viridis', aspect='auto')
            axes[0].set_title(f'Attention Weights - {example["guid"]}')
            axes[0].set_xlabel('Token Position')
            axes[0].set_ylabel('Head')
        
        # 文本内容
        axes[1].text(0.1, 0.5, example['text'], fontsize=10, 
                    verticalalignment='center', wrap=True)
        axes[1].axis('off')
        axes[1].set_title('Input Text')
        
        plt.tight_layout()
        plt.savefig(f'results/attention_visualization_{i}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()