def print_top_brands(file_path, top_n=30):
    """
    打印前N个品牌及其数量的美观输出
    """
    brands = {}
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                brand, count = line.rsplit(':', 1)
                brands[brand] = int(count)
    
    # 排序
    sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 打印表格
    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║ {'Top ' + str(top_n) + ' Brands':^66s} ║")
    print("╠" + "═" * 4 + "╦" + "═" * 38 + "╦" + "═" * 24 + "╣")
    print(f"║ {'No':^2s} ║ {'Brand Name':^36s} ║ {'Count':^22s} ║")
    print("╠" + "═" * 4 + "╬" + "═" * 38 + "╬" + "═" * 24 + "╣")
    
    for i, (brand, count) in enumerate(sorted_brands, 1):
        # 创建可视化条形图
        bar_length = int(count / sorted_brands[0][1] * 20)
        bar = "█" * bar_length
        
        print(f"║ {i:2d} ║ {brand:36s} ║ {count:6d} {bar:20s} ║")
    
    print("╠" + "═" * 4 + "╩" + "═" * 38 + "╩" + "═" * 24 + "╣")
    
    # 统计信息
    total = sum(count for _, count in sorted_brands)
    avg = total / len(sorted_brands)
    
    print(f"║ {'Statistics':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║ Total Images: {total:12,d} {'':39s} ║")
    print(f"║ Average:      {avg:12,.1f} {'':39s} ║")
    print(f"║ Max:          {sorted_brands[0][1]:12,d}  ({sorted_brands[0][0]}) {'':14s} ║")
    print(f"║ Min:          {sorted_brands[-1][1]:12,d}  ({sorted_brands[-1][0]}) {'':14s} ║")
    print("╚" + "═" * 68 + "╝")
    
    # 返回品牌列表和数据
    brand_list = [brand for brand, _ in sorted_brands]
    
    print(f"\n# Python List (Top {top_n})")
    print(f"top_{top_n}_brands = {brand_list}")
    
    return brand_list, sorted_brands


# 使用
file_path = "/home1/lu-wei/repo/EMMA/classifier/YOLO/LogoDet-3K/VOC3000number/logo_num.txt"
brands, data = print_top_brands(file_path, top_n=30)