import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

class MultimodalTrainer:
    """多模态训练器 - 支持不同模型类型"""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device(config.device)
        self.model.to(self.device)
        
        # 检测模型类型
        self.model_type = self._detect_model_type(model, config)
        print(f"检测到模型类型: {self.model_type}")
        
        # 损失函数和优化器
        class_weights = torch.tensor([0.8, 3.0, 1.5]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=3, factor=0.5, verbose=True
            )
        except TypeError:
            # 如果verbose参数不被支持，去掉它
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=3, factor=0.5
            )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def _detect_model_type(self, model, config):
        """检测模型类型"""
        model_class_name = model.__class__.__name__
        
        # 从类名判断
        if 'TextOnly' in model_class_name:
            return 'text_only'
        elif 'ImageOnly' in model_class_name:
            return 'image_only'
        elif 'EarlyFusion' in model_class_name:
            return 'early_fusion'
        elif 'LateFusion' in model_class_name:
            return 'late_fusion'
        elif 'AttentionFusion' in model_class_name:
            return 'attention_fusion'
        elif 'GatedFusion' in model_class_name:
            return 'gated_fusion'
        else:
            # 从配置判断
            return getattr(config, 'fusion_type', 'unknown')
    
    def _model_forward(self, input_ids, attention_mask, images):
        """统一的模型前向传播适配器"""
        if self.model_type in ['text', 'text_only']:
            # 纯文本模型：只传文本参数
            # 检查模型是否需要images参数（有些已修改为接受但忽略）
            try:
                return self.model(input_ids, attention_mask, images)
            except TypeError:
                # 如果模型不接受images参数
                return self.model(input_ids, attention_mask)
                
        elif self.model_type in ['image', 'image_only']:
            # 纯图像模型：只传图像参数
            # 检查模型是否需要文本参数（有些已修改为接受但忽略）
            try:
                return self.model(input_ids, attention_mask, images)
            except TypeError:
                # 如果模型不接受文本参数
                return self.model(images)
                
        else:
            # 多模态模型：传递所有参数
            return self.model(input_ids, attention_mask, images)
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播 - 使用适配器
            self.optimizer.zero_grad()
            logits, _ = self._model_forward(input_ids, attention_mask, images)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'type': self.model_type[:10]  # 显示模型类型前10个字符
            })
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 使用macro F1作为主要指标
        f1 = macro_f1
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 使用适配器
                logits, _ = self._model_forward(input_ids, attention_mask, images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        # 保存最佳模型
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_model_state = self.model.state_dict().copy()
            
            # 保存模型
            model_save_path = os.path.join(
                self.config.model_save_dir, 
                f'best_model_{self.model_type}.pth'
            )
            
            torch.save({
                'epoch': len(self.train_losses),
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_accuracy': accuracy,
                'model_type': self.model_type,
                'config': self.config
            }, model_save_path)
            
            print(f"✅ 保存最佳模型到: {model_save_path}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'report': classification_report(all_labels, all_preds, 
                                          target_names=['positive', 'neutral', 'negative'])
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'model_type': self.model_type,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.model_save_dir,
            f'checkpoint_{self.model_type}_epoch{epoch}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.config.model_save_dir,
                f'best_model_{self.model_type}.pth'
            )
            torch.save(checkpoint, best_path)
        
        return checkpoint_path