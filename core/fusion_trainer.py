import torch
import torch.nn as nn
from .trainer import MultimodalTrainer
from tqdm import tqdm

class FusionTrainer(MultimodalTrainer):
    """融合模型专用训练器"""
    
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        
        # 为不同模态设置不同的学习率
        if hasattr(model, 'text_encoder') and hasattr(model, 'image_encoder'):
            self.optimizer = self._create_multimodal_optimizer(model, config)
    
    def _create_multimodal_optimizer(self, model, config):
        """为多模态模型创建优化器（不同模态不同学习率）"""
        # 文本参数
        text_params = []
        if hasattr(model, 'text_encoder'):
            text_params = [
                {'params': model.text_encoder.parameters(), 'lr': config.learning_rate * 0.1}
            ]
        
        # 图像参数
        image_params = []
        if hasattr(model, 'image_encoder'):
            image_params = [
                {'params': model.image_encoder.parameters(), 'lr': config.learning_rate * 0.1}
            ]
        
        # 融合层参数
        fusion_params = []
        for name, param in model.named_parameters():
            if 'text_encoder' not in name and 'image_encoder' not in name:
                fusion_params.append(param)
        
        all_params = []
        if text_params:
            all_params.extend(text_params)
        if image_params:
            all_params.extend(image_params)
        if fusion_params:
            all_params.append({'params': fusion_params, 'lr': config.learning_rate})
        
        return torch.optim.AdamW(all_params, weight_decay=config.weight_decay)
    
    def train_with_modality_dropout(self, train_loader, epoch, dropout_rate=0.1):
        """使用模态dropout训练（随机丢弃某个模态）"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 随机模态dropout
            use_text = torch.rand(1).item() > dropout_rate
            use_image = torch.rand(1).item() > dropout_rate
            
            if not use_text:
                # 丢弃文本模态
                input_ids = torch.zeros_like(input_ids)
                attention_mask = torch.zeros_like(attention_mask)
            
            if not use_image:
                # 丢弃图像模态
                images = torch.zeros_like(images)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask, images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': loss.item(),
                'text': use_text,
                'image': use_image
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, all_preds, all_labels