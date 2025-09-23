# def read_clr_file(clr_path):
#     color_map = {}
#     with open(clr_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue  # 跳过空行和注释
#             parts = line.split()
#             if len(parts) >= 4:
#                 value = int(float(parts[0]))  # 处理可能的浮点值
#                 r, g, b = map(int, parts[1:4])
#                 color_map[value] = (r, g, b)
#     return color_map
#
# # 示例用法
# clr_path = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\Maize_NH3EF_domin_int.clr"
# color_map = read_clr_file(clr_path)
# print("颜色映射表:", color_map)

def read_clr_file(clr_path):
    color_map = {}
    with open(clr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 跳过空行和注释
            parts = line.split()
            if len(parts) >= 4:
                value = int(float(parts[0]))  # 处理可能的浮点值
                r, g, b = map(int, parts[1:4])
                hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b).upper()  # 转换为HEX（大写）
                color_map[value] = {
                    'RGB': (r, g, b),
                    'HEX': hex_color
                }
    return color_map

def save_color_map(color_map, output_path):
    with open(output_path, 'w') as f:
        f.write("颜色映射表 (RGB + HEX):\n")
        f.write("=" * 40 + "\n")
        for value, colors in sorted(color_map.items()):
            line = "值 {:>4}: RGB=({:>3}, {:>3}, {:>3}) | HEX={}\n".format(
                value,
                colors['RGB'][0], colors['RGB'][1], colors['RGB'][2],
                colors['HEX']
            )
            f.write(line)

# 示例用法
clr_path = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\Maize_NH3EF_domin_int.clr"
output_path = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\domin_legend.txt"  # 结果保存路径

color_map = read_clr_file(clr_path)

# 打印到控制台
print("颜色映射表 (RGB + HEX):")
print("=" * 40)
for value, colors in sorted(color_map.items()):
    print("值 {:>4}: RGB=({:>3}, {:>3}, {:>3}) | HEX={}".format(
        value,
        colors['RGB'][0], colors['RGB'][1], colors['RGB'][2],
        colors['HEX']
    ))

# 保存到文件
save_color_map(color_map, output_path)
print(f"\n颜色表已保存到: {output_path}")