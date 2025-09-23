import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
from sympy import false

# 1. 文件路径配置 (请根据实际情况修改)
# base_dir = r"D:\CNPK\N_CODE_PY\N_ML_V2\Wheat_domin_factors0907"  # 原结果目录
base_dir = r"D:\CNPK\N_CODE_PY\N_ML_V2\Maize_domin_factors0907"  # 原结果目录

world_shp = r"D:\CNPK\Nloss_ML\国家边界矢量\World_countries.shp"

# 2. 加载参考栅格（用于获取空间信息）
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
# 统一的前5因子颜色 (红→橙→黄→浅绿→深绿)
UNIFIED_COLORS = [
    '#FF0000',  # 排名1 (红)
    '#FF8000',  # 排名2 (橙)
    '#FFFF00',  # 排名3 (黄)
    '#8BD100',  # 排名4 (浅绿)
    '#38A800',  # 排名5 (深绿)
    'white'     # 其他区域
]


# 4. 主处理函数
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

    # 生成全局主导因子图
    create_global_dominant_map(top5_df, top1_raster, valid_pixels, input_dir, f"{crop}_{target_var}")

    # 生成全局重要性图 (需SHAP值文件，若无可跳过)
    # shap_file = os.path.join(input_dir, f"{crop}_{target_var}_shap_values.npy")
    # if os.path.exists(shap_file):
    #     shap_importance = np.load(shap_file)
    #     create_global_importance_map(top5_df, shap_importance, valid_pixels, input_dir, f"{crop}_{target_var}")
    # else:
    #     print("警告: 未找到SHAP值文件，跳过重要性权重图")


# 5. 可视化函数 (与之前提供的相同)
def create_global_dominant_map(top5_df, top1_raster, valid_pixels, output_dir, prefix):
    """生成全局前5因子分布图（显示变量名称版）"""
    global_top5 = top5_df['Factor'].head(5).tolist()

    # 创建分类栅格 (-1:非前5, 1-5:全局前5的排名)
    dominant_map = np.full_like(mask_data, fill_value=-1, dtype=np.int16)

    for rank, var in enumerate(global_top5, start=1):
        var_index = list(variable_info.keys()).index(var) + 1
        dominant_map[valid_pixels] = np.where(
            top1_raster[valid_pixels] == var_index,
            rank,
            dominant_map[valid_pixels]
        )

    # 保存TIFF
    output_tif = os.path.join(output_dir, f"{prefix}_global_top5_map.tif")
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(dominant_map, 1)

    # 统一颜色方案（仅前5有颜色）
    UNIFIED_COLORS = [
        '#FF0000',  # Rank 1 (红)
        '#FF8000',  # Rank 2 (橙)
        '#FFFF00',  # Rank 3 (黄)
        '#8BD100',  # Rank 4 (浅绿)
        '#38A800',  # Rank 5 (深绿)
    ]

    # 创建颜色映射（前5用指定色，其他用白）
    cmap = mcolors.ListedColormap(['white'] + UNIFIED_COLORS)
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        ncolors=len(UNIFIED_COLORS) + 1
    )

    # 数据分类：-1→0(其他), 1-5→1-5
    plot_data = np.where(dominant_map == -1, 0, dominant_map)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        plot_data,
        cmap=cmap,
        norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )

    # 图例（显示变量名称）
    legend_labels = global_top5 + ['Others']
    legend_colors = UNIFIED_COLORS + ['white']
    patches = [
        mpatches.Patch(color=legend_colors[i], label=legend_labels[i])
        for i in range(len(legend_labels))
    ]

    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        # title="Dominant Factors",
        # framealpha=false
        frameon=False
    )

    # 添加地理边界
    world = gpd.read_file(world_shp).to_crs(crs)
    world.boundary.plot(ax=ax, linewidth=0.4, edgecolor='gray', alpha=0.6)

    # ax.set_title(f"Global Top5 Dominant Factors: {prefix.replace('_', ' ')}", pad=20)
    ax.axis('off')

    # 保存PNG
    output_png = os.path.join(output_dir, f"{prefix}_global_top5_map.png")
    plt.savefig(output_png, dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"已保存可视化图: {output_png}")


# 6. 批量处理所有作物和变量
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