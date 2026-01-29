import itertools
import numpy as np
from sklearn.model_selection import ParameterGrid
from .main_experiment import MainExperiment

class HyperparameterSearch:
    """超参数搜索"""
    
    def __init__(self, experiment_type="early_fusion"):
        self.experiment_type = experiment_type
        self.results = []
        
        # 定义搜索空间
        self.param_grid = {
            'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.1, 0.3, 0.5],
            'fusion_hidden_dim': [256, 512, 1024],
            'weight_decay': [0.0, 1e-5, 1e-4]
        }
    
    def run_grid_search(self, n_trials=10):
        """运行网格搜索"""
        # 随机采样参数组合
        param_list = list(ParameterGrid(self.param_grid))
        
        if len(param_list) > n_trials:
            indices = np.random.choice(len(param_list), n_trials, replace=False)
            param_list = [param_list[i] for i in indices]
        
        print(f"Running {len(param_list)} hyperparameter trials...")
        
        for i, params in enumerate(param_list):
            print(f"\n{'='*60}")
            print(f"Trial {i+1}/{len(param_list)}")
            print(f"Parameters: {params}")
            print(f"{'='*60}")
            
            try:
                # 创建实验配置
                experiment = MainExperiment(self.experiment_type)
                
                # 更新配置参数
                for key, value in params.items():
                    setattr(experiment.config, key, value)
                
                # 运行实验
                results = experiment.run()
                
                # 记录结果
                self.results.append({
                    'params': params,
                    'val_accuracy': results['validation']['accuracy'],
                    'val_f1': results['validation']['f1'],
                    'trial': i
                })
                
                print(f"Results - Accuracy: {results['validation']['accuracy']:.4f}, "
                      f"F1: {results['validation']['f1']:.4f}")
                
            except Exception as e:
                print(f"Error in trial {i+1}: {e}")
                continue
        
        self._analyze_results()
    
    def _analyze_results(self):
        """分析超参数搜索结果"""
        if not self.results:
            print("No results to analyze")
            return
        
        import pandas as pd
        
        # 转换为DataFrame
        results_df = pd.DataFrame(self.results)
        
        # 展开参数字典
        params_df = pd.json_normalize(results_df['params'])
        results_df = pd.concat([results_df.drop('params', axis=1), params_df], axis=1)
        
        # 按准确率排序
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        print("\nHyperparameter Search Results:")
        print(results_df[['trial', 'val_accuracy', 'val_f1', 
                         'learning_rate', 'batch_size', 'dropout_rate']].to_string(index=False))
        
        # 保存结果
        results_df.to_csv(f"results/{self.experiment_type}/hyperparam_search.csv", index=False)
        
        # 找到最佳参数
        best_result = results_df.iloc[0]
        print(f"\nBest Parameters:")
        print(f"Accuracy: {best_result['val_accuracy']:.4f}")
        print(f"F1 Score: {best_result['val_f1']:.4f}")
        print(f"Learning Rate: {best_result['learning_rate']}")
        print(f"Batch Size: {best_result['batch_size']}")
        print(f"Dropout Rate: {best_result['dropout_rate']}")
        
        # 可视化
        self._visualize_results(results_df)
    
    def _visualize_results(self, results_df):
        """可视化超参数搜索结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 学习率 vs 准确率
        axes[0, 0].scatter(np.log10(results_df['learning_rate']), 
                          results_df['val_accuracy'], alpha=0.6)
        axes[0, 0].set_xlabel('log10(Learning Rate)')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].set_title('Learning Rate Impact')
        
        # Dropout vs 准确率
        axes[0, 1].scatter(results_df['dropout_rate'], 
                          results_df['val_accuracy'], alpha=0.6)
        axes[0, 1].set_xlabel('Dropout Rate')
        axes[0, 1].set_ylabel('Validation Accuracy')
        axes[0, 1].set_title('Dropout Impact')
        
        # 批量大小 vs 准确率
        axes[1, 0].scatter(results_df['batch_size'], 
                          results_df['val_accuracy'], alpha=0.6)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Validation Accuracy')
        axes[1, 0].set_title('Batch Size Impact')
        
        # 隐藏层维度 vs 准确率
        if 'fusion_hidden_dim' in results_df.columns:
            axes[1, 1].scatter(results_df['fusion_hidden_dim'], 
                              results_df['val_accuracy'], alpha=0.6)
            axes[1, 1].set_xlabel('Hidden Dimension')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].set_title('Hidden Dimension Impact')
        
        plt.tight_layout()
        plt.savefig(f"results/{self.experiment_type}/hyperparam_visualization.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()