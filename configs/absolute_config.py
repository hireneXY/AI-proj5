# configs/absolute_config.py
from .base_config import BaseConfig
import os

class AbsoluteConfig(BaseConfig):
    """使用绝对路径的配置"""
    
    # 基础数据目录
    data_dir: str = "/mnt/workspace/multimodel_experiment/data/dataset"
    
    # dataset目录（包含guid.txt和guid.jpg）
    base_data_dir: str = "/mnt/workspace/multimodel_experiment/data"
    
    # 具体文件路径
    train_file: str = "/mnt/workspace/multimodel_experiment/data/train.txt"
    test_file: str = "/mnt/workspace/multimodel_experiment/data/test_without_label.txt"
    
    
    def __post_init__(self):
        """验证所有路径"""
        super().__post_init__()
        
        print("=" * 60)
        print("数据路径配置:")
        print(f"  data_dir: {self.data_dir}")
        print(f"  train_file: {self.train_file}")
        print(f"  test_file: {self.test_file}")
        
        # 验证路径
        for name, path in [
            ("data_dir", self.data_dir),
            ("train_file", self.train_file),
            ("test_file", self.test_file)
        ]:
            if os.path.exists(path):
                print(f"  ✅ {name}: 存在")
            else:
                print(f"  ❌ {name}: 不存在!")
        
        print("=" * 60)