import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import json
import csv
from datetime import datetime
import random

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
        # build class weights from config if provided, otherwise fallback to reasonable defaults
        cfg_class_weights = None
        try:
            # support config.training.class_weights or config.class_weights
            tw = getattr(self.config, "loss_weights", None)
            cfg_class_weights = getattr(self.config, "class_weights", None) or getattr(self.config, "training", {}).get("class_weights", None) if isinstance(getattr(self.config, "training", None), dict) else getattr(self.config, "class_weights", None)
        except Exception:
            cfg_class_weights = None

        if cfg_class_weights and isinstance(cfg_class_weights, dict):
            # order by label_map in config (assume mapping exists)
            try:
                label_map = getattr(self.config, "label_map", {"positive": 0, "neutral": 1, "negative": 2})
                # build list in label id order (0,1,2)
                ordered_names = sorted(label_map, key=lambda x: label_map[x])
                weights_list = [float(cfg_class_weights.get(k, 1.0)) for k in ordered_names]
                class_weights = torch.tensor(weights_list).to(self.device)
            except Exception:
                class_weights = torch.tensor([0.8, 3.0, 1.5]).to(self.device)
        else:
            class_weights = torch.tensor([0.8, 3.0, 1.5]).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # AMP scaler（可选）
        self.use_amp = getattr(config, "use_mixed_precision", False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # gradient accumulation
        self.accumulation_steps = max(1, getattr(config, "gradient_accumulation_steps", 1))
        
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
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer_step_count = 0

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # mixup / cutmix (optional)
            use_mixup = getattr(self.config, "mixup_enable", False)
            use_cutmix = getattr(self.config, "cutmix_enable", False)
            lam = None
            y_a = None
            y_b = None
            if (use_mixup or use_cutmix) and images is not None:
                batch_size = images.size(0)
                if use_mixup:
                    alpha = getattr(self.config, "mixup_alpha", 0.4)
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    index = torch.randperm(batch_size).to(self.device)
                    images = lam * images + (1 - lam) * images[index, :]
                    y_a, y_b = labels, labels[index]
                else:
                    alpha = getattr(self.config, "cutmix_alpha", 1.0)
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    index = torch.randperm(batch_size).to(self.device)
                    bbx1 = random.randint(0, images.size(2) - 1)
                    bby1 = random.randint(0, images.size(3) - 1)
                    bbx2 = int(bbx1 + images.size(2) * np.sqrt(1 - lam))
                    bby2 = int(bby1 + images.size(3) * np.sqrt(1 - lam))
                    bbx2 = min(bbx2, images.size(2) - 1)
                    bby2 = min(bby2, images.size(3) - 1)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
                    area = (bbx2 - bbx1) * (bby2 - bby1)
                    lam = 1 - area / (images.size(2) * images.size(3))
                    y_a, y_b = labels, labels[index]

            # forward (with AMP if enabled)
            try:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits, aux = self._model_forward(input_ids, attention_mask, images)
                    if lam is not None and y_a is not None and y_b is not None:
                        cls_loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                    else:
                        cls_loss = self.criterion(logits, labels)

                    combined_loss = cls_loss
                    if hasattr(self.model, "compute_misa_losses") and aux is not None:
                        misa_losses = self.model.compute_misa_losses(aux)
                        lw = getattr(self.config, "loss_weights", None) or {}
                        rec_w = lw.get("reconstruction", 0.1)
                        sim_w = lw.get("similarity", 0.1)
                        diff_w = lw.get("difference", 0.05)
                        combined_loss = combined_loss + rec_w * misa_losses.get("reconstruction", 0.0) \
                            + sim_w * misa_losses.get("similarity", 0.0) \
                            + diff_w * misa_losses.get("difference", 0.0)

                    loss = combined_loss / self.accumulation_steps
            except RuntimeError as e:
                msg = str(e)
                if 'cannot be converted to type c10::Half' in msg or 'value cannot be converted to type c10::Half' in msg:
                    print("⚠️ AMP caused overflow/conversion error; disabling AMP and retrying without autocast.")
                    self.use_amp = False
                    self.scaler = torch.cuda.amp.GradScaler(enabled=False)
                    logits, aux = self._model_forward(input_ids, attention_mask, images)
                    cls_loss = self.criterion(logits, labels)
                    combined_loss = cls_loss
                    if hasattr(self.model, "compute_misa_losses") and aux is not None:
                        misa_losses = self.model.compute_misa_losses(aux)
                        lw = getattr(self.config, "loss_weights", None) or {}
                        rec_w = lw.get("reconstruction", 0.1)
                        sim_w = lw.get("similarity", 0.1)
                        diff_w = lw.get("difference", 0.05)
                        combined_loss = combined_loss + rec_w * misa_losses.get("reconstruction", 0.0) \
                            + sim_w * misa_losses.get("similarity", 0.0) \
                            + diff_w * misa_losses.get("difference", 0.0)
                    loss = combined_loss / self.accumulation_steps
                else:
                    raise

            # backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer step (when accumulation reached)
            if (step % self.accumulation_steps) == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                optimizer_step_count += 1

            # statistics
            total_loss += (loss.item() * self.accumulation_steps)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': loss.item(),
                'type': self.model_type[:10]
            })

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0

        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': macro_f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                try:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits, _ = self._model_forward(input_ids, attention_mask, images)
                        loss = self.criterion(logits, labels)
                except RuntimeError as e:
                    msg = str(e)
                    if 'cannot be converted to type c10::Half' in msg or 'value cannot be converted to type c10::Half' in msg:
                        print("⚠️ AMP caused overflow/conversion error in validation; disabling AMP for remaining validation.")
                        self.use_amp = False
                        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
                        logits, _ = self._model_forward(input_ids, attention_mask, images)
                        loss = self.criterion(logits, labels)
                    else:
                        raise

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_labels) > 0 else 0.0

        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)

        # 保存最佳模型
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_model_state = self.model.state_dict().copy()

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

        # 汇总并保存指标
        try:
            labels_order = [0, 1, 2]
            target_names = ['positive', 'neutral', 'negative']
            cm = confusion_matrix(all_labels, all_preds, labels=labels_order).tolist()
            precision, recall, fscore, support = precision_recall_fscore_support(
                all_labels, all_preds, labels=labels_order, zero_division=0
            )

            results_dir = getattr(self.config, "results_dir", "results")
            os.makedirs(results_dir, exist_ok=True)
            epoch_idx = len(self.train_losses)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            base_name = f"{self.model_type}_val_metrics_epoch{epoch_idx}_{timestamp}"
            json_path = os.path.join(results_dir, base_name + ".json")
            csv_path = os.path.join(results_dir, base_name + ".csv")

            metrics = {
                "model_type": self.model_type,
                "epoch": epoch_idx,
                "timestamp": timestamp,
                "loss": avg_loss,
                "accuracy": accuracy,
                "f1_weighted": float(f1),
                "confusion_matrix": cm,
                "per_class": []
            }
            for i, name in enumerate(target_names):
                metrics["per_class"].append({
                    "class_id": int(labels_order[i]),
                    "class_name": name,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(fscore[i]),
                    "support": int(support[i])
                })

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(metrics, jf, ensure_ascii=False, indent=2)

            with open(csv_path, "w", encoding="utf-8", newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
                for row in metrics["per_class"]:
                    writer.writerow([row["class_id"], row["class_name"], row["precision"], row["recall"], row["f1"], row["support"]])

            print(f"🔍 验证指标已保存: {json_path}, {csv_path}")
        except Exception as e:
            print("⚠️ 保存验证指标时出错:", e)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'report': classification_report(all_labels, all_preds,
                                           target_names=['positive', 'neutral', 'negative']),
            'metrics_json': json_path if 'json_path' in locals() else None,
            'metrics_csv': csv_path if 'csv_path' in locals() else None
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