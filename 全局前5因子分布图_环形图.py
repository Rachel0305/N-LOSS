import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib import font_manager
plt.rcParams['font.family'] = 'Calibri'
# 1. 文件路径配置
base_dir = r"D:\CNPK\N_CODE_PY\N_ML_V2\Maize_domin_factors0907"
world_shp = r"D:\CNPK\Nloss_ML\国家边界矢量\World_countries.shp"

# 2. 加载参考栅格
# reference_raster = os.path.join(base_dir, "Wheat_N2OEF_domin0907", "Wheat_N2OEF_top1_factor.tif")
reference_raster = os.path.join(base_dir, "Maize_N2OEF_domin0907", "Maize_N2OEF_top1_factor.tif")


with rasterio.open(reference_raster) as src:
    mask_data = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    profile = src.profile

# 3. 定义变量颜色映射 (与原代码一致)
variable_info = {
    "MAP": {"order": 1, "color": "#FF0000", "filename": "Column_1_MAP.tif"},
    "MAT": {"order": 2, "color": "#00FF00", "filename": "Column_2_MAT.tif"},
    "Sand": {"order": 3, "color": "#0000FF", "filename": "Column_3_Sand.tif"},
    "Clay": {"order": 4, "color": "#FFFF00", "filename": "Column_4_Clay.tif"},
    "Slope": {"order": 5, "color": "#FF00FF", "filename": "Column_5_Slope.tif"},
    "pH": {"order": 6, "color": "#00FFFF", "filename": "Column_6_pH.tif"},
    "Bulk density": {"order": 7, "color": "#FF8000", "filename": "Column_7_Bulk density.tif"},
    "SOC": {"order": 8, "color": "#8000FF", "filename": "Column_8_SOC.tif"},
    "CN ratio": {"order": 9, "color": "#0080FF", "filename": "Column_9_CN ratio.tif"},
    "N input rate": {"order": 10, "color": "#FF0080", "filename": "Column_10_N input rate.tif"},
    "Tillage": {"order": 11, "color": "#80FF00", "filename": "Column_11_Tillage.tif", "categorical": True},
    "Irrigation": {"order": 12, "color": "#00FF80", "filename": "Column_12_Irrigation.tif", "categorical": True},
    "Sro span max": {"order": 13, "color": "#800000", "filename": "Column_13_Sro span max.tif"},
    "Sro span min": {"order": 14, "color": "#008000", "filename": "Column_14_Sro span min.tif"},
    "Sro key mean": {"order": 15, "color": "#000080", "filename": "Column_15_Sro key mean.tif"},
    "LAI span max": {"order": 16, "color": "#808000", "filename": "Column_16_LAI span max.tif"},
    "LAI span min": {"order": 17, "color": "#800080", "filename": "Column_17_LAI span min.tif"},
    "LAI key mean": {"order": 18, "color": "#008080", "filename": "Column_18_LAI key mean.tif"},
    "Rx1 span max": {"order": 19, "color": "#FF4000", "filename": "Column_19_Rx1 span max.tif"},
    "Rx1 span min": {"order": 20, "color": "#0040FF", "filename": "Column_20_Rx1 span min.tif"},
    "Rx1 key mean": {"order": 21, "color": "#FF0040", "filename": "Column_21_Rx1 key mean.tif"},
    "Rx5 span max": {"order": 22, "color": "#40FF00", "filename": "Column_22_Rx5 span max.tif"},
    "Rx5 span min": {"order": 23, "color": "#0040FF", "filename": "Column_23_Rx5 span min.tif"},
    "Rx5 key mean": {"order": 24, "color": "#FF00BF", "filename": "Column_24_Rx5 key mean.tif"},
    "Ddp span max": {"order": 25, "color": "#BF00FF", "filename": "Column_25_Ddp span max.tif"},
    "Ddp span min": {"order": 26, "color": "#00BFFF", "filename": "Column_26_Ddp span min.tif"},
    "Ddp key mean": {"order": 27, "color": "#FFBF00", "filename": "Column_27_Ddp key mean.tif"},
    "Spei span max": {"order": 28, "color": "#00FFBF", "filename": "Column_28_Spei span max.tif"},
    "Spei span min": {"order": 29, "color": "#BF00BF", "filename": "Column_29_Spei span min.tif"},
    "Spei key mean": {"order": 30, "color": "#00BFBF", "filename": "Column_30_Spei key mean.tif"},
    "SMs span max": {"order": 31, "color": "#BFBF00", "filename": "Column_31_SMs span max.tif"},
    "SMs span min": {"order": 32, "color": "#4000FF", "filename": "Column_32_SMs span min.tif"},
    "SMs key mean": {"order": 33, "color": "#FF0040", "filename": "Column_33_SMs key mean.tif"},
    "SMrz span max": {"order": 34, "color": "#0040FF", "filename": "Column_34_SMrz span max.tif"},
    "SMrz span min": {"order": 35, "color": "#FF00BF", "filename": "Column_35_SMrz span min.tif"},
    "SMrz key mean": {"order": 36, "color": "#BF00FF", "filename": "Column_36_SMrz key mean.tif"},
    "Hddw span max": {"order": 37, "color": "#00BFFF", "filename": "Column_37_Hddw span max.tif"},
    "Hddw span min": {"order": 38, "color": "#FFBF00", "filename": "Column_38_Hddw span min.tif"},
    "Hddw key mean": {"order": 39, "color": "#00FFBF", "filename": "Column_39_Hddw key mean.tif"},
    "Cddw span max": {"order": 40, "color": "#BF00BF", "filename": "Column_40_Cddw span max.tif"},
    "Cddw span min": {"order": 41, "color": "#00BFBF", "filename": "Column_41_Cddw span min.tif"},
    "Cddw key mean": {"order": 42, "color": "#BFBF00", "filename": "Column_42_Cddw key mean.tif"},
    "Hddm span max": {"order": 43, "color": "#FF1493", "filename": "Column_37_Hddm span max.tif"},
    "Hddm span min": {"order": 44, "color": "#8B0000", "filename": "Column_38_Hddm span min.tif"},
    "Hddm key mean": {"order": 45, "color": "#006400", "filename": "Column_39_Hddm key mean.tif"},
    "Cddm span max": {"order": 46, "color": "#00008B", "filename": "Column_40_Cddm span max.tif"},
    "Cddm span min": {"order": 47, "color": "#8B008B", "filename": "Column_41_Cddm span min.tif"},
    "Cddm key mean": {"order": 48, "color": "#8B4513", "filename": "Column_42_Cddm key mean.tif"},
    "Fertilizer type": {"order": 49, "color": "#008B8B", "filename": "Column_43_Fertilizer type.tif",
                        "categorical": True}
}
# 统一的前5因子颜色
UNIFIED_COLORS = [
    '#FF0000',  # 排名1 (红)
    '#FF8000',  # 排名2 (橙)
    '#FFFF00',  # 排名3 (黄)
    '#8BD100',  # 排名4 (浅绿)
    '#38A800',  # 排名5 (深绿)
    'white'  # 其他区域
]


def create_top5_donut_chart(top5_factors, factor_counts, total_pixels, colors, output_path, prefix):
    """创建前五因子数量比例环形图，将变量名称和百分比标注在色块外侧"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 准备数据
    data = []
    labels = []
    colors_list = []
    percentages = []

    for i, factor in enumerate(top5_factors):
        count = factor_counts.get(factor, 0)
        percentage = count / total_pixels * 100
        data.append(count)
        labels.append(factor)
        percentages.append(f"{percentage:.1f}%")
        colors_list.append(colors[i])

    # 计算其他因子的数量
    other_count = total_pixels - sum(data)
    if other_count > 0:
        data.append(other_count)
        labels.append("Others")
        percentages.append(f"{other_count / total_pixels * 100:.1f}%")
        colors_list.append('#CCCCCC')

    # 绘制环形图
    wedges, texts = ax.pie(data, colors=colors_list, startangle=90, wedgeprops=dict(width=0.4))

    # 在色块外侧标注变量名称和百分比
    for i, wedge in enumerate(wedges):
        # 计算标注位置（在环形外侧）
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.25 * np.cos(np.radians(angle))
        y = 1.25 * np.sin(np.radians(angle))

        # 根据角度调整文本对齐方式
        if -90 <= angle <= 90:
            ha = 'left'
        else:
            ha = 'right'

        # 添加变量名称标注（较大字体）
        # ax.text(x, y + 0.02, labels[i], ha=ha, va='center', fontsize=11,
        #         color='black')

        # 添加百分比标注（较小字体）
        ax.text(x + 0.2, y, percentages[i], ha=ha, va='center', fontsize=36,
                color='black')

    # 添加中心标题
    # center_circle = plt.Circle((0, 0), 0.3, color='white')
    # ax.add_artist(center_circle)

    # 根据目标变量类型设置中心标题
    # target_name = prefix.split('_')[-1]
    # if target_name == "N2OEF":
    #     title_text = "N₂O EF\nDominant\nFactors"
    # elif target_name == "NH3EF":
    #     title_text = "NH₃ EF\nDominant\nFactors"
    # elif target_name == "NOEF":
    #     title_text = "NO EF\nDominant\nFactors"
    # elif target_name == "LF":
    #     title_text = "Leaching\nDominant\nFactors"
    # elif target_name == "RF":
    #     title_text = "Runoff\nDominant\nFactors"
    # elif target_name == "NUE":
    #     title_text = "NUE\nDominant\nFactors"
    # else:
    #     title_text = f"{target_name}\nDominant\nFactors"
    #
    # ax.text(0, 0, title_text, ha='center', va='center',
    #         fontsize=12, fontweight='bold', linespacing=1.5)

    # 设置整体标题
    # crop_name = prefix.split('_'plt.rcParams['font.family'] = 'Calibri')[0]
    # ax.set_title(f"{crop_name} - Top5 Dominant Factors Distribution",
    #              fontsize=16, pad=20)

    # 移除坐标轴
    ax.axis('equal')

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存环形图: {output_path}")


def create_top5_donut_for_target(top5_df, top1_raster, valid_pixels, total_pixels, output_dir, prefix):
    """为当前目标创建前五因子环形图"""
    # 获取前五因子
    top5_factors = top5_df['Factor'].head(5).tolist()

    # 计算每个前五因子的像素数量
    factor_counts = {}
    for factor in top5_factors:
        var_index = list(variable_info.keys()).index(factor) + 1
        count = np.sum(top1_raster[valid_pixels] == var_index)
        factor_counts[factor] = count

    # 创建输出目录
    donut_dir = os.path.join(output_dir, "top5_donut_charts")
    os.makedirs(donut_dir, exist_ok=True)

    # 使用统一的颜色方案
    colors = UNIFIED_COLORS[:5]

    # 创建环形图
    output_path = os.path.join(donut_dir, f"{prefix}_top5_donut.png")
    create_top5_donut_chart(top5_factors, factor_counts, total_pixels, colors, output_path, prefix)


def process_crop_target(crop, target_var):
    print(f"\n正在处理: {crop}_{target_var}")

    # 输入文件路径
    input_dir = os.path.join(base_dir, f"{crop}_{target_var}_domin0907")
    top5_excel = os.path.join(input_dir, f"{crop}_{target_var}_top5_factors.xlsx")
    top1_tif = os.path.join(input_dir, f"{crop}_{target_var}_top1_factor.tif")

    if not os.path.exists(top5_excel):
        print(f"警告: 未找到 {top5_excel}，跳过")
        return

    # 读取数据
    top5_df = pd.read_excel(top5_excel)
    with rasterio.open(top1_tif) as src:
        top1_raster = src.read(1)

    # 获取有效像素
    valid_pixels = np.where((top1_raster != -9999) & (mask_data > 0))
    total_pixels = len(valid_pixels[0])

    # 生成前五因子环形图
    create_top5_donut_for_target(top5_df, top1_raster, valid_pixels, total_pixels, input_dir, f"{crop}_{target_var}")


# 批量处理所有作物和变量
crop_targets = [
    # ("Wheat", "N2OEF"),
    # ("Wheat", "NH3EF"),
    # ("Wheat", "NOEF"),
    # ("Wheat", "LF"),
    # ("Wheat", "RF"),
    # ("Wheat", "NUE"),
    ("Maize", "N2OEF"),
    ("Maize", "NH3EF"),
    ("Maize", "NOEF"),
    ("Maize", "LF"),
    ("Maize", "RF"),
    ("Maize", "NUE"),
]

for crop, target in crop_targets:
    process_crop_target(crop, target)

print("所有处理完成！")