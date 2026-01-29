import torch
import pandas as pd
from tqdm import tqdm
import os

from configs.experiment_configs import ExperimentConfigs
from data.data_loader import create_data_loaders
from models import *
from core.fusion_trainer import FusionTrainer

class MainExperiment:
    """主实验类"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.config = ExperimentConfigs.get_config(experiment_name)
        
        # 创建必要的目录
        os.makedirs(f"results/{experiment_name}", exist_ok=True)
        os.makedirs(f"logs/{experiment_name}", exist_ok=True)
        
    def run(self):
        """运行实验"""
        print(f"Starting experiment: {self.experiment_name}")
        
        # 1. 加载数据
        train_data, test_data = self._load_data()
        
        # 2. 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            self.config, train_data, test_data
        )
        
        # 3. 创建模型
        model = self._create_model()
        
        # 4. 创建训练器
        trainer = FusionTrainer(model, self.config)
        
        # 5. 训练
        history = self._train_model(trainer, train_loader, val_loader)
        
        # 6. 评估
        results = self._evaluate_model(trainer, val_loader, test_loader)
        
        # 7. 保存结果
        self._save_results(history, results)
        
        return results
    
    def _load_data(self):
        """加载数据"""
        train_data = []
        with open(os.path.join(self.config.data_dir, self.config.train_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    guid = parts[0].strip()
                    label = parts[1].strip()
                    train_data.append((guid, label))
        
        test_data = []
        with open(os.path.join(self.config.data_dir, self.config.test_file), 'r') as f:
            for line in f:
                guid = line.strip()
                test_data.append(guid)
        
        return train_data, test_data
    
    def _create_model(self):
        """创建模型"""
        if self.config.fusion_type == "early":
            return EarlyFusionModel(self.config)
        elif self.config.fusion_type == "late":
            return LateFusionModel(self.config)
        elif self.config.fusion_type == "attention":
            return AttentionFusionModel(self.config)
        elif self.config.fusion_type == "gated":
            return GatedFusionModel(self.config)
        elif self.config.fusion_type == "text_only":
            return TextOnlyModel(self.config)
        elif self.config.fusion_type == "image_only":
            return ImageOnlyModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.fusion_type}")
    
    def _train_model(self, trainer, train_loader, val_loader):
        """训练模型"""
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
        
        for epoch in range(self.config.num_epochs):
            # 训练
            if hasattr(trainer, 'train_with_modality_dropout'):
                loss, preds, labels = trainer.train_with_modality_dropout(
                    train_loader, epoch, dropout_rate=0.1
                )
            else:
                train_result = trainer.train_epoch(train_loader, epoch)
                loss = train_result['loss']
                preds = train_result['predictions']
                labels = train_result['labels']
            
            # 验证
            val_result = trainer.validate(val_loader)
            
            # 记录历史
            history['train_loss'].append(loss)
            history['val_loss'].append(val_result['loss'])
            history['train_acc'].append(accuracy_score(labels, preds))
            history['val_acc'].append(val_result['accuracy'])
            history['learning_rates'].append(
                trainer.optimizer.param_groups[0]['lr']
            )
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {loss:.4f}, Acc: {history['train_acc'][-1]:.4f} | "
                  f"Val Loss: {val_result['loss']:.4f}, Acc: {val_result['accuracy']:.4f}")
            
            # 早停检查
            if epoch >= self.config.early_stop_patience:
                recent_accs = history['val_acc'][-self.config.early_stop_patience:]
                if max(recent_accs) <= max(history['val_acc']):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def _evaluate_model(self, trainer, val_loader, test_loader):
        """评估模型"""
        results = {}
        
        # 在验证集上的最终评估
        val_result = trainer.validate(val_loader)
        results['validation'] = val_result
        
        # 在测试集上生成预测
        predictions = self._predict_test_set(trainer.model, test_loader)
        results['test_predictions'] = predictions
        
        return results
    
    def _predict_test_set(self, model, test_loader):
        """在测试集上生成预测"""
        model.eval()
        all_predictions = []
        all_guids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Prediction"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                images = batch['image'].to(self.config.device)
                
                logits, _ = model(input_ids, attention_mask, images)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_guids.extend(batch['guid'])
        
        return list(zip(all_guids, all_predictions))
    
    def _save_results(self, history, results):
        """保存结果"""
        # 保存训练历史
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"results/{self.experiment_name}/training_history.csv", index=False)
        
        # 保存验证结果
        with open(f"results/{self.experiment_name}/validation_report.txt", 'w') as f:
            f.write(results['validation']['report'])
        
        # 保存测试预测
        test_df = pd.DataFrame(results['test_predictions'], columns=['guid', 'label_id'])
        test_df['label'] = test_df['label_id'].map(self.config.rev_label_map)
        test_df[['guid', 'label']].to_csv(
            f"results/{self.experiment_name}/test_predictions.csv", index=False
        )
        
        print(f"Results saved to results/{self.experiment_name}/")