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


def process_tif_files(tif_files, max_files=10):
    """处理TIFF文件并准备堆叠条形图数据"""
    file_data = []

    # 只处理前max_files个文件
    for tif_file in tqdm(tif_files[:max_files], desc="Processing TIFF files"):
        try:
            with rasterio.open(tif_file) as src:
                data = src.read(1)

                # 统计每种像素值的总像元数
                color_counts = {}
                unique, counts = np.unique(data, return_counts=True)

                for pixel_value, count in zip(unique, counts):
                    if pixel_value in color_dict:
                        color_counts[pixel_value] = count

                # 计算百分比并按百分比从大到小排序
                total = sum(color_counts.values())
                percentages = {k: (v / total) * 100 for k, v in color_counts.items()}
                sorted_percentages = sorted(percentages.items(), key=lambda x: -x[1])  # 按百分比降序排序

                # 准备数据
                file_name = tif_file.replace("_domin_int.tif", "").replace("_", " ")
                file_data.append({
                    "file_name": file_name,
                    "percentages": dict(sorted_percentages),
                    "sorted_pairs": sorted_percentages  # 保留排序后的(像素值,百分比)对
                })

        except Exception as e:
            print(f"Error processing {tif_file}: {str(e)}")

    return file_data


def create_stacked_bar_chart(file_data, output_folder):
    """创建横向堆叠条形图，每个影像的条带按从长到短排列"""
    # 准备数据
    file_names = [data["file_name"] for data in file_data]

    # 获取所有出现的像素值（按出现频率排序）
    all_pixel_values = set()
    for data in file_data:
        all_pixel_values.update(data["percentages"].keys())

    # 创建颜色和标签映射
    colors = [color_dict[pv] for pv in all_pixel_values]
    labels = [labels_dict.get(pv, str(pv)) for pv in all_pixel_values]

    # 准备堆叠数据
    fig, ax = plt.subplots(figsize=(20, 8))

    # 为每个文件创建堆叠条
    for i, file_name in enumerate(file_names):
        bottom = 0
        # 获取当前文件的排序后数据
        sorted_pairs = file_data[i]["sorted_pairs"]

        # 绘制每个像素值的条带
        for pv, percent in sorted_pairs:
            ax.barh(file_name, percent, left=bottom, color=color_dict[pv],
                    label=labels_dict.get(pv, str(pv)))
            bottom += percent

    # 设置图表属性
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Image Files', fontsize=12)
    ax.set_title('Dominant Drivers Distribution (Stacked by Percentage)', fontsize=14, pad=20, weight='bold')

    # 添加图例（只显示实际出现的变量）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., title="Variables")

    # 调整布局
    plt.tight_layout()

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存PNG
    output_png = os.path.join(output_folder, "stacked_bar_chart_sorted.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    print(f"Successfully saved sorted stacked bar chart to: {output_png}")
    return output_png


# 主程序
def main():
    # 创建输出文件夹
    output_folder = "output_stacked_charts"
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有符合条件的TIFF文件
    tif_files = [f for f in os.listdir(input_folder) if f.endswith("_domin_int.tif")]
    print(f"Found {len(tif_files)} TIFF files")

    # 处理文件并准备数据
    file_data = process_tif_files(tif_files, max_files=10)

    if file_data:
        # 创建堆叠条形图
        chart_path = create_stacked_bar_chart(file_data, output_folder)
        print(f"\nProcessing completed! Chart saved to: {chart_path}")
    else:
        print("\nNo valid data found to process.")


if __name__ == "__main__":
    main()