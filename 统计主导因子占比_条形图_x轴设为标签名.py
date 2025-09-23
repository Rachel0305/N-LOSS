import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置工作目录
input_folder = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers"
os.chdir(input_folder)

# 颜色映射表
color_dict = {
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

# 编号对应的标签
labels_dict = {
    1: "MAP", 2: "MAT", 3: "Sro_Span_max", 4: "Sro_Span_min", 5: "Sro_Span_mean",
    6: "Ddp_Span_max", 7: "Ddp_Span_min", 8: "Ddp_Span_mean", 9: "Spei_Span_max",
    10: "Spei_Span_min", 11: "Spei_Span_mean", 12: "Till", 13: "Irrigated",
    14: "Sand", 15: "Clay", 16: "Slope", 17: "pH", 18: "Bulk density",
    19: "SOC", 20: "C/N", 21: "N input rate", 22: "LAI_Span_max",
    23: "LAI_Span_min", 24: "LAI_Span_mean", 25: "Rx1_Span_max",
    26: "Rx1_Span_min", 27: "Rx1_Span_mean", 28: "Rx5_Span_max",
    29: "Rx5_Span_min", 30: "Rx5_Span_mean", 31: "Hddw_Span_max",
    32: "Hddw_Span_min", 33: "Hddw_Span_mean", 34: "Cddw_Span_max",
    35: "Cddw_Span_min", 36: "Cddw_Span_mean", 37: "Hddm_Span_max",
    38: "Hddm_Span_min", 39: "Hddm_Span_mean", 40: "Cddm_Span_max",
    41: "Cddm_Span_min", 42: "Cddm_Span_mean"
}


def create_bar_chart(tif_file, color_counts, output_folder):
    """创建条形图，使用对应映射表中的颜色和标签"""
    # 准备条形图数据
    pixel_values = []
    values = []
    colors = []
    labels = []

    # 过滤掉计数为0的项并按计数降序排序
    filtered_data = {k: v for k, v in color_counts.items() if v > 0}
    sorted_data = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)

    if not sorted_data:
        print(f"Warning: No valid data to plot for {tif_file}")
        return None

    # 准备条形图参数
    for pixel_value, count in sorted_data:
        pixel_values.append(pixel_value)
        values.append(count)
        colors.append(color_dict[pixel_value])
        labels.append(labels_dict.get(pixel_value, str(pixel_value)))

    # 计算总数量用于计算百分比
    total = sum(values)

    # 创建条形图
    fig, ax = plt.subplots(figsize=(14, 8))  # 增加宽度以适应长标签

    # 绘制条形图
    bars = ax.bar(labels, values, color=colors)

    # 添加数值标签（可选）
    for bar in bars:
        height = bar.get_height()
        percentage = 100 * height / total
        if percentage > 1:  # 只标注大于1%的分类
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{percentage:.0f}%',
                    ha='center', va='bottom',
                    fontsize=10,  # 减小字体以适应更多标签
                    fontfamily='Times New Roman')

    # 设置标题和标签
    title = tif_file.replace("_domin_int.tif", "").replace("_", " ")
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.set_xlabel('Driver Variables', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    # 旋转x轴标签并调整间距
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.subplots_adjust(bottom=0.3)  # 增加底部边距

    # 调整布局
    plt.tight_layout()

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存PNG
    output_png = os.path.join(output_folder, f"{tif_file.replace('.tif', '')}_bar.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    print(f"Successfully saved bar chart to: {output_png}")
    return output_png


# 创建输出文件夹
output_folder = "output_bar_charts"
os.makedirs(output_folder, exist_ok=True)

# 初始化结果列表
results = []

# 获取所有符合条件的TIFF文件
tif_files = [f for f in os.listdir(input_folder) if f.endswith("_domin_int.tif")]
print(f"Found {len(tif_files)} TIFF files to process")

# 处理每个TIFF文件
for tif_file in tqdm(tif_files, desc="Processing TIFF files"):
    try:
        with rasterio.open(tif_file) as src:
            # 读取栅格数据
            data = src.read(1)

            # 统计每种像素值的总像元数
            color_counts = {}
            unique, counts = np.unique(data, return_counts=True)

            for pixel_value, count in zip(unique, counts):
                if pixel_value in color_dict:
                    color_counts[pixel_value] = count

            # 为每个像元值创建记录
            for pixel_value, count in zip(unique, counts):
                if pixel_value in color_dict:
                    results.append({
                        "File": tif_file,
                        "Pixel Value": int(pixel_value),
                        "Label": labels_dict.get(pixel_value, "Unknown"),
                        "Color": color_dict[pixel_value],
                        "Count": int(count)
                    })

            # 创建并保存条形图
            if color_counts:
                bar_path = create_bar_chart(tif_file, color_counts, output_folder)
            else:
                print(f"No valid pixel values found in {tif_file}")

    except Exception as e:
        print(f"Error processing {tif_file}: {str(e)}")

# 创建DataFrame并保存到Excel
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=["File", "Pixel Value"])

    output_excel = "domin_drivers_pixel_statistics0617.xlsx"
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Pixel Statistics", index=False)

    print(f"\nProcessing completed! Results saved to {output_excel}")
    print(f"Total files processed: {len(tif_files)}")
    print(f"Total records: {len(df)}")
    print(f"Bar charts saved to folder: {os.path.abspath(output_folder)}")

    # 打印生成的图表数量
    generated_charts = len([f for f in os.listdir(output_folder) if f.endswith('_bar.png')])
    print(f"Number of bar charts generated: {generated_charts}")
else:
    print("\nNo valid data found to process.")