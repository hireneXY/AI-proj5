import logging
import sys
from datetime import datetime
import os

def setup_logger(config, name=None):
    """设置日志记录器"""
    if name is None:
        name = config.fusion_type if hasattr(config, 'fusion_type') else 'experiment'
    
    # 创建日志目录
    log_dir = os.path.join(config.log_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, config):
        self.logger = setup_logger(config)
        self.config = config
        
    def log_config(self):
        """记录配置信息"""
        self.logger.info("=" * 60)
        self.logger.info("Experiment Configuration")
        self.logger.info("=" * 60)
        
        for key, value in vars(self.config).items():
            if not key.startswith('_'):
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 60)
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """记录epoch信息"""
        self.logger.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    def log_best_result(self, best_accuracy, best_epoch):
        """记录最佳结果"""
        self.logger.info("=" * 60)
        self.logger.info(f"Best Validation Accuracy: {best_accuracy:.4f}")
        self.logger.info(f"Achieved at Epoch: {best_epoch}")
        self.logger.info("=" * 60)