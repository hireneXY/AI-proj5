import os

print("测试修复后的文本读取...")
print("=" * 60)

test_guids = ['26', '862', '550', '903', '204', '713']
for guid in test_guids:
    filepath = f"/mnt/workspace/multimodel_experiment/data/dataset/{guid}.txt"
    print(f"\nGUID {guid}:")
    
    if not os.path.exists(filepath):
        print("  文件不存在")
        continue
    
    # 直接测试
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        print(f"  文件大小: {len(data)} bytes")
        
        # 检查0x00
        nulls = data.count(b'\x00')
        if nulls > 0:
            print(f"  包含 {nulls} 个0x00字节 ({nulls/len(data)*100:.1f}%)")
        
        # 尝试解码
        for enc in ['utf-8', 'latin-1', 'utf-8-sig']:
            try:
                text = data.decode(enc, errors='strict')
                print(f"  {enc}: 严格解码成功, 长度: {len(text)}")
                break
            except:
                try:
                    text = data.decode(enc, errors='ignore')
                    if text.strip():
                        print(f"  {enc}: 忽略错误解码, 长度: {len(text)}")
                        print(f"    内容前50: {repr(text[:50])}")
                        break
                except:
                    pass
    except Exception as e:
        print(f"  读取失败: {e}")
