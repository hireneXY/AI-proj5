import pandas as pd
import matplotlib.pyplot as plt
from .main_experiment import MainExperiment

class AblationStudy:
    """消融实验"""
    
    def __init__(self):
        self.experiments = [
            "early_fusion",
            "attention_fusion",
            "late_fusion",
            "baseline_text",
            "baseline_image"
        ]
        
        self.results = {}
    
    def run_all(self):
        """运行所有消融实验"""
        for exp_name in self.experiments:
            print(f"\n{'='*50}")
            print(f"Running: {exp_name}")
            print(f"{'='*50}")
            
            experiment = MainExperiment(exp_name)
            results = experiment.run()
            
            self.results[exp_name] = {
                'val_accuracy': results['validation']['accuracy'],
                'val_f1': results['validation']['f1'],
                'config': experiment.config
            }
        
        self._analyze_results()
        self._visualize_results()
    
    def _analyze_results(self):
        """分析结果"""
        analysis = []
        
        for exp_name, result in self.results.items():
            analysis.append({
                'Experiment': exp_name,
                'Val Accuracy': result['val_accuracy'],
                'Val F1': result['val_f1'],
                'Fusion Type': result['config'].fusion_type,
                'Model Size': 'Multimodal' if 'fusion' in exp_name else 'Unimodal'
            })
        
        self.analysis_df = pd.DataFrame(analysis)
        print("\nAblation Study Results:")
        print(self.analysis_df.to_string(index=False))
        
        # 保存分析结果
        self.analysis_df.to_csv("results/ablation_study/summary.csv", index=False)
    
    def _visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 准确率对比
        self.analysis_df.plot.bar(x='Experiment', y='Val Accuracy', 
                                 ax=axes[0], color='skyblue')
        axes[0].set_title('Validation Accuracy by Experiment')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # F1分数对比
        self.analysis_df.plot.bar(x='Experiment', y='Val F1', 
                                 ax=axes[1], color='lightgreen')
        axes[1].set_title('Validation F1 Score by Experiment')
        axes[1].set_ylabel('F1 Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/ablation_study/comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 模态重要性分析
        self._plot_modality_importance()
    
    def _plot_modality_importance(self):
        """绘制模态重要性"""
        multimodal_results = self.analysis_df[
            self.analysis_df['Model Size'] == 'Multimodal'
        ]
        
        if len(multimodal_results) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(multimodal_results['Experiment'], 
                   multimodal_results['Val Accuracy'])
            plt.title('Multimodal Models Performance Comparison')
            plt.ylabel('Validation Accuracy')
            plt.axhline(y=self.results['baseline_text']['val_accuracy'], 
                       color='r', linestyle='--', label='Text Baseline')
            plt.axhline(y=self.results['baseline_image']['val_accuracy'], 
                       color='g', linestyle='--', label='Image Baseline')
            plt.legend()
            plt.tight_layout()
            plt.savefig('results/ablation_study/multimodal_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()