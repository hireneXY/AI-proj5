# direct_fix_dataset.py
import os

print("直接修复数据集类的编码处理...")
print("=" * 60)

# 查找数据集文件
dataset_files = []
for root, dirs, files in os.walk("/mnt/workspace/multimodel_experiment"):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
            if 'class MultimodalDataset' in content and '_load_text' in content:
                dataset_files.append(filepath)

if not dataset_files:
    print("❌ 找不到数据集文件")
    exit(1)

dataset_file = dataset_files[0]
print(f"修复文件: {dataset_file}")

# 备份
os.system(f"cp {dataset_file} {dataset_file}.backup_final")

# 读取内容
with open(dataset_file, 'r') as f:
    content = f.read()

# 直接替换整个_load_text方法（最简单可靠）
new_load_text = '''

def _load_text(self, guid):
    """加载文本文件 - 超级健壮版本"""
    # 尝试多个路径
    possible_paths = [
        os.path.join(self.data_dir, "dataset", f"{guid}.txt"),
        os.path.join(self.data_dir, f"{guid}.txt"),
        os.path.join("/mnt/workspace/multimodel_experiment/data/dataset", f"{guid}.txt"),
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if not filepath:
        return "[FILE_NOT_FOUND]"
    
    # 检查文件大小
    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return "[EMPTY_FILE]"
        if file_size > 100000:  # 100KB，可能不是文本文件
            return "[TOO_LARGE]"
    except:
        pass
    
    # 方法1: 首先尝试二进制读取并智能解码
    try:
        with open(filepath, 'rb') as f:
            binary_data = f.read()
        
        # 检查是否包含大量0x00（可能是二进制文件）
        null_count = binary_data.count(b'\x00')
        if null_count > len(binary_data) * 0.1:  # 超过10%是0x00
            return "[BINARY_FILE]"
        
        # 尝试解码 - 按优先级
        encodings = [
            'utf-8-sig',    # 带BOM的UTF-8
            'utf-8',        # 标准UTF-8
            'latin-1',      # 处理0xa1等
            'cp1252',       # Windows西欧
            'iso-8859-1',   # ISO西欧
            'gbk',          # 中文
        ]
        
        for encoding in encodings:
            try:
                text = binary_data.decode(encoding, errors='strict')
                if text.strip():  # 非空文本
                    # 进一步清理
                    text = text.strip()
                    text = ' '.join(text.split())  # 合并空白
                    if 5 <= len(text) <= 1000:  # 合理长度
                        return text
            except UnicodeDecodeError:
                continue
        
        # 如果严格解码都失败，尝试忽略错误
        for encoding in ['utf-8', 'latin-1']:
            try:
                text = binary_data.decode(encoding, errors='ignore')
                text = text.strip()
                if text and len(text) >= 5:
                    return text[:1000]  # 截断
            except:
                continue
        
        return "[DECODE_FAILED]"
        
    except Exception as e:
        return f"[ERROR: {str(e)[:50]}]"
