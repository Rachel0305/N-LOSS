import os
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Circle
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置工作目录
input_folder = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers"  # 替换为您的文件夹路径
os.chdir(input_folder)

# 颜色映射表
color_dict = {
    # 1: "#1F77B4", 2: "#1F77B4", 3: "#8C510A", 4: "#8C510A", 5: "#7F7F7F",
    # 6: "#8C510A", 7: "#8C510A", 8: "#8C510A", 9: "#11BBBB", 10: "#11BBBB",
    # 11: "#FF7F0E", 12: "#FF7F0E", 13: "#1F77B4", 14: "#1F77B4", 15: "#1F77B4",
    # 16: "#4D9221", 17: "#4D9221", 18: "#4D9221", 19: "#D62728", 20: "#D62728",
    # 21: "#D62728", 22: "#D62728", 23: "#D62728", 24: "#D62728", 25: "#1F77B4",
    # 26: "#1F77B4", 27: "#1F77B4", 28: "#1F77B4", 29: "#1F77B4", 30: "#1F77B4",
    # 31: "#D62728", 32: "#D62728", 33: "#D62728", 34: "#D62728", 35: "#D62728",
    # 36: "#D62728", 37: '#D62728', 38: '#D62728', 39: '#D62728', 40: '#D62728',
    # 41: '#D62728', 42: '#D62728'
    1: "#FF0000", 2: "#00FF00", 3: "#0000FF", 4: "#FFFF00", 5: "#00FFFF",
    6: "#FF00FF", 7: "#800000", 8: "#008000", 9: "#000080", 10: "#808000",
    11: "#008080", 12: "#800080", 13: "#C0C0C0", 14: "#808080", 15: "#993366",
    16: "#339966", 17: "#999933", 18: "#336699", 19: "#A0522D", 20: "#D8BFD8",
    21: "#6A5ACD", 22: "#FFA500", 23: "#FFC0CB", 24: "#4B0082", 25: "#ADFF2F",
    26: "#D2691E", 27: "#DC143C", 28: "#00BFFF", 29: "#32CD32", 30: "#8A2BE2",
    31: "#FFD700", 32: "#F08080", 33: "#00FA9A", 34: "#BA55D3", 35: "#F4A460",
    36: "#9370DB", 37: "#3CB371", 38: "#7B68EE", 39: "#FF6347", 40: "#00CED1",
    41: "#C71585", 42: "#FF8C00"
}


def create_selective_label_pie_chart(tif_file, color_counts, output_folder):
    """创建饼图，仅标注比例大于5%的分类"""
    # 准备饼图数据
    sizes = []
    colors = []

    # 过滤掉计数为0的项并按计数降序排序
    filtered_data = {k: v for k, v in color_counts.items() if v > 0}
    sorted_data = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)

    # 准备饼图参数
    for color, count in sorted_data:
        sizes.append(count)
        colors.append(color)

    # 计算总数量用于计算百分比
    total = sum(sizes)

    # 创建饼图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制饼图（从90度开始，逆时针方向）
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,  # 顺时针方向
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )

    # 添加中心白色圆形使饼图变为环形图效果
    centre_circle = Circle((0, 0), 0.6, fc='white', ec='white')  # ec设置边缘颜色
    ax.add_patch(centre_circle)

    # 仅添加大于5%的标签
    for i, p in enumerate(wedges):
        percentage = 100. * sizes[i] / total
        if percentage > 5:  # 只标注大于5%的分类
            # 计算角度（中点角度）
            ang = (p.theta2 + p.theta1) / 2.
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            # 确定标签位置（在饼图外侧）
            distance = 1.15  # 调整标签距离
            label_x = distance * x
            label_y = distance * y

            # 添加百分比标签
            ax.text(label_x, label_y, f"{percentage:.0f}%",
                    ha='center', va='center',
                    fontsize=36,
                    fontfamily='Times New Roman')

    # 添加中心白色圆形使饼图变为环形图效果
    centre_circle = Circle((0, 0), 0.6, fc='white', ec='white')  # ec设置边缘颜色
    ax.add_patch(centre_circle)

    # 添加标题（使用文件名）
    title = tif_file.replace("_domin.tif", "").replace("_", " ")
    ax.set_title(title, fontsize=16, pad=20, weight='bold')

    # 设置等比例确保饼图是圆形
    ax.axis('equal')

    # 调整布局
    plt.tight_layout()

    # 保存PNG
    output_png = os.path.join(output_folder, f"{tif_file.replace('.tif', '')}_pie.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    return output_png


# 创建输出文件夹
output_folder = "output_pie_charts"
os.makedirs(output_folder, exist_ok=True)

# 初始化结果列表
results = []

# 获取所有符合条件的TIFF文件
tif_files = [f for f in os.listdir(input_folder) if f.endswith("_domin.tif")]

# 处理每个TIFF文件（带进度条显示）
for tif_file in tqdm(tif_files, desc="Processing TIFF files"):
    try:
        with rasterio.open(tif_file) as src:
            # 读取栅格数据
            data = src.read(1)

            # 统计每种颜色的总像元数
            color_counts = {}
            unique, counts = np.unique(data, return_counts=True)
            for pixel_value, count in zip(unique, counts):
                if pixel_value in color_dict:
                    color = color_dict[pixel_value]
                    color_counts[color] = color_counts.get(color, 0) + count

            # 为每个像元值创建记录
            for pixel_value, count in zip(unique, counts):
                if pixel_value in color_dict:
                    results.append({
                        "File": tif_file,
                        "Pixel Value": int(pixel_value),
                        "Color": color_dict[pixel_value],
                        "Count": int(count)
                    })

            # 创建并保存饼图
            pie_path = create_selective_label_pie_chart(tif_file, color_counts, output_folder)
            print(f"Saved pie chart to: {pie_path}")

    except Exception as e:
        print(f"Error processing {tif_file}: {str(e)}")

# 创建DataFrame并保存到Excel
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=["File", "Pixel Value"])

    output_excel = "domin_drivers_pixel_statistics.xlsx"
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Pixel Statistics", index=False)

    print(f"\nProcessing completed! Results saved to {output_excel}")
    print(f"Total files processed: {len(tif_files)}")
    print(f"Total records: {len(df)}")
    print(f"Pie charts saved to folder: {output_folder}")
else:
    print("\nNo valid data found to process.")