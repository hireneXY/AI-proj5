# 1. 查看问题文件的基本信息
echo "=== 诊断问题文件 ==="
for guid in 862 550 903 451; do
    echo -e "\n--- 文件: $guid.txt ---"
    filepath="/mnt/workspace/multimodel_experiment/data/dataset/$guid.txt"
    
    # 检查文件是否存在
    if [ -f "$filepath" ]; then
        echo "大小: $(wc -c < "$filepath") bytes"
        
        # 查看文件类型
        echo -n "file命令: "
        file "$filepath"
        
        # 查看前100字节的hex
        echo "前100字节(hex):"
        hexdump -C "$filepath" | head -5
        
        # 尝试用od查看
        echo "ASCII表示:"
        od -c "$filepath" | head -5
        
        # 尝试读取
        echo "尝试读取:"
        echo -n "1. cat直接输出: "
        cat "$filepath" | head -c 100 | cat -A
        echo ""
        
        echo -n "2. iconv latin1转utf8: "
        iconv -f latin1 -t utf8 "$filepath" 2>/dev/null | head -c 100 || echo "失败"
        
    else
        echo "文件不存在!"
    fi
done

# 2. 特别检查862.txt（最常见的错误文件）
echo -e "\n=== 详细检查 862.txt ==="
filepath="/mnt/workspace/multimodel_experiment/data/dataset/862.txt"

if [ -f "$filepath" ]; then
    echo "完整hexdump:"
    hexdump -C "$filepath"
    
    echo -e "\n尝试不同方式读取:"
    
    # 查看是否包含0xa1字符
    echo "查找0xa1字节位置:"
    hexdump -C "$filepath" | grep -n "a1"
    
    # 尝试二进制查看
    echo -e "\n二进制分析:"
    python3 -c "
import sys
path = '$filepath'
with open(path, 'rb') as f:
    data = f.read()
print(f'文件大小: {len(data)} bytes')
print(f'前20字节: {data[:20].hex()}')
print(f'包含0xa1的数量: {sum(1 for b in data if b == 0xa1)}')
print(f'所有字节值: {list(data[:50])}')
print('\\n尝试解码:')
# 尝试不同编码
encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8', 'utf-8-sig', 'gbk', 'mac_roman']
for enc in encodings:
    try:
        text = data.decode(enc)
        print(f'{enc:15}: 成功 - 长度: {len(text)}, 内容前50: {repr(text[:50])}')
    except Exception as e:
        print(f'{enc:15}: 失败 - {str(e)[:50]}')
"
else
    echo "862.txt不存在!"
fi
