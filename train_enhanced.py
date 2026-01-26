# config_patch.py
"""
独立的配置补丁系统
可以在不修改原代码的情况下覆盖配置值
"""
import sys
import os

class ConfigPatch:
    """配置补丁类"""
    
    def __init__(self):
        self.overrides = {}
        self.original_modules = {}
        self.applied = False
        
    def set(self, **kwargs):
        """设置要覆盖的配置值"""
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                # 数值类型直接存储
                self.overrides[key] = value
            elif isinstance(value, str):
                # 字符串类型添加引号
                self.overrides[key] = f"'{value}'"
            else:
                self.overrides[key] = value
        return self
    
    def apply_to_model(self, model_type='late'):
        """应用配置补丁到特定模型"""
        
        print(f"应用配置补丁到 {model_type} 模型...")
        print("-" * 40)
        
        # 根据模型类型确定要修改的文件
        if model_type == 'late':
            target_file = 'models/fusion/late_fusion.py'
        elif model_type == 'early':
            target_file = 'models/fusion/early_fusion.py'
        elif model_type == 'attention':
            target_file = 'models/fusion/attention_fusion.py'
        elif model_type == 'text':
            target_file = 'models/text_model.py'  # 可能需要调整
        elif model_type == 'image':
            target_file = 'models/image_model.py'  # 可能需要调整
        else:
            print(f"⚠️ 未知模型类型: {model_type}")
            return False
        
        if not os.path.exists(target_file):
            print(f"❌ 目标文件不存在: {target_file}")
            return False
        
        # 备份原文件
        backup_file = f"{target_file}.backup"
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(target_file, backup_file)
            print(f"✅ 已备份: {backup_file}")
        
        # 读取文件
        with open(target_file, 'r') as f:
            content = f.read()
        
        # 替换配置获取调用
        for config_key, new_value in self.overrides.items():
            # 查找类似 getattr(config, 'DROPOUT', 0.3) 的模式
            import re
            pattern = rf"getattr\(config,\s*['\"]{config_key}['\"],\s*([^)]+)\)"
            
            matches = list(re.finditer(pattern, content))
            if matches:
                for match in matches:
                    old_value = match.group(1)
                    replacement = f"getattr(config, '{config_key}', {new_value})"
                    content = content.replace(match.group(0), replacement)
                    print(f"✓ {config_key}: {old_value} → {new_value}")
            else:
                # 尝试其他模式
                pattern2 = rf"config\.{config_key}"
                if re.search(pattern2, content):
                    # 替换为安全的getattr调用
                    content = re.sub(pattern2, f"getattr(config, '{config_key}', {new_value})", content)
                    print(f"✓ {config_key}: 直接访问 → 安全访问(默认:{new_value})")
        
        # 写回文件
        with open(target_file, 'w') as f:
            f.write(content)
        
        self.applied = True
        print(f"✅ 配置补丁已应用到 {target_file}")
        return True
    
    def apply_to_config_class(self):
        """直接修改配置类"""
        try:
            import main
            
            # 备份原始配置类
            if not hasattr(self, '_original_config'):
                self._original_config = main.AbsoluteConfig
            
            # 创建新的配置类
            class PatchedConfig(self._original_config):
                def __getattr__(self, name):
                    # 检查覆盖值
                    if name in self._overrides:
                        return self._overrides[name]
                    # 回退到原始
                    return super().__getattr__(name)
                
                def _set_overrides(self, overrides):
                    self._overrides = overrides
            
            # 替换配置类
            main.AbsoluteConfig = PatchedConfig
            
            # 创建实例并设置覆盖
            config_instance = PatchedConfig()
            config_instance._set_overrides(self.overrides)
            
            print("✅ 配置类补丁已应用")
            return True
            
        except Exception as e:
            print(f"❌ 配置类补丁失败: {e}")
            return False
    
    def restore(self):
        """恢复原始文件"""
        if not self.applied:
            print("⚠️ 没有应用过补丁")
            return
        
        # 恢复所有备份文件
        import glob
        for backup in glob.glob("models/**/*.py.backup", recursive=True):
            original = backup.replace('.backup', '')
            if os.path.exists(backup):
                import shutil
                shutil.copy2(backup, original)
                print(f"✅ 已恢复: {original}")
        
        # 删除备份文件
        for backup in glob.glob("models/**/*.py.backup", recursive=True):
            os.remove(backup)
        
        # 恢复配置类
        if hasattr(self, '_original_config'):
            import main
            main.AbsoluteConfig = self._original_config
            print("✅ 配置类已恢复")
        
        self.applied = False
        print("✅ 所有补丁已恢复")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='应用配置补丁')
    parser.add_argument('--model', default='late', 
                       choices=['late', 'early', 'attention', 'text', 'image'],
                       help='目标模型类型')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--text_hidden', type=int, default=128, help='文本隐藏层维度')
    parser.add_argument('--image_hidden', type=int, default=128, help='图像隐藏层维度')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--restore', action='store_true', help='恢复原始文件')
    
    args = parser.parse_args()
    
    patch = ConfigPatch()
    
    if args.restore:
        patch.restore()
    else:
        # 设置覆盖值
        patch.set(
            DROPOUT=args.dropout,
            TEXT_HIDDEN=args.text_hidden,
            IMAGE_HIDDEN=args.image_hidden,
            WEIGHT_DECAY=args.weight_decay
        )
        
        # 应用到模型文件
        success = patch.apply_to_model(args.model)
        
        if success:
            print("\n" + "=" * 50)
            print(f"✅ 补丁应用成功！")
            print(f"现在运行: python main.py --model {args.model}")
            print("使用的配置:")
            for key, value in patch.overrides.items():
                print(f"  {key}: {value}")
        else:
            print("\n❌ 补丁应用失败")