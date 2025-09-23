import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import ListedColormap

# 设置路径和参数
tif_folder = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers'
continent_shapefile = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\大洲边界矢量数据\大洲矢量修正版.shp'
output_folder = './results_stacked'
os.makedirs(output_folder, exist_ok=True)

# 驱动因子与颜色映射
driver_colors = {
    1: ("MAP", "#FF0000"),
    2: ("MAT", "#00FF00"),
    3: ("Sro_Span_max", "#0000FF"),
    4: ("Sro_Span_min", "#FFFF00"),
    5: ("Sro_Span_mean", "#00FFFF"),
    6: ("Ddp_Span_max", "#FF00FF"),
    7: ("Ddp_Span_min", "#800000"),
    8: ("Ddp_Span_mean", "#008000"),
    9: ("Spei_Span_max", "#000080"),
    10: ("Spei_Span_min", "#808000"),
    11: ("Spei_Span_mean", "#008080"),
    12: ("Till", "#800080"),
    13: ("Irrigated", "#C0C0C0"),
    14: ("Sand", "#808080"),
    15: ("Clay", "#993366"),
    16: ("Slope", "#339966"),
    17: ("pH", "#999933"),
    18: ("Bulk density", "#336699"),
    19: ("SOC", "#A0522D"),
    20: ("C/N", "#D8BFD8"),
    21: ("N input rate", "#6A5ACD"),
    22: ("LAI_Span_max", "#FFA500"),
    23: ("LAI_Span_min", "#FFC0CB"),
    24: ("LAI_Span_mean", "#4B0082"),
    25: ("Rx1_Span_max", "#ADFF2F"),
    26: ("Rx1_Span_min", "#D2691E"),
    27: ("Rx1_Span_mean", "#DC143C"),
    28: ("Rx5_Span_max", "#00BFFF"),
    29: ("Rx5_Span_min", "#32CD32"),
    30: ("Rx5_Span_mean", "#8A2BE2"),
    31: ("Hddw_Span_max", "#FFD700"),
    32: ("Hddw_Span_min", "#F08080"),
    33: ("Hddw_Span_mean", "#00FA9A"),
    34: ("Cddw_Span_max", "#BA55D3"),
    35: ("Cddw_Span_min", "#F4A460"),
    36: ("Cddw_Span_mean", "#9370DB"),
    37: ("Hddm_Span_max", "#3CB371"),
    38: ("Hddm_Span_min", "#7B68EE"),
    39: ("Hddm_Span_mean", "#FF6347"),
    40: ("Cddm_Span_max", "#00CED1"),
    41: ("Cddm_Span_min", "#C71585"),
    42: ("Cddm_Span_mean", "#FF8C00")
}

# 加载大洲边界矢量数据
try:
    continents_gdf = gpd.read_file(continent_shapefile)
    print("成功加载大洲边界数据，包含以下大洲:", continents_gdf['CONTINENT'].unique())
except Exception as e:
    print(f"加载大洲边界数据失败: {e}")
    exit()

# 六大洲名称 (除南极洲外)
continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
continents_gdf = continents_gdf[continents_gdf['CONTINENT'].isin(continents)]

# 收集所有TIFF文件
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('_domin_int_pls1.tif')]
print(f"找到 {len(tif_files)} 个TIFF文件")

# 创建结果DataFrame
results = []

# 处理每个TIFF文件
for tif_file in tif_files:
    # 解析文件名获取变量信息
    parts = tif_file.split('_')
    if len(parts) < 3:
        print(f"跳过文件名格式不正确的文件: {tif_file}")
        continue

    crop_type = parts[0]  # 作物类型
    variable_type = '_'.join(parts[1:-2])  # 变量类型

    file_path = os.path.join(tif_folder, tif_file)
    try:
        with rasterio.open(file_path) as src:
            # 修改nodata值为0
            profile = src.profile
            profile.update(nodata=0)

            # 按大洲统计
            for continent in continents:
                # 获取该大洲的几何形状
                continent_geom = continents_gdf[continents_gdf['CONTINENT'] == continent]
                if len(continent_geom) == 0:
                    print(f"未找到大洲: {continent}")
                    continue

                # 裁剪TIFF到大洲范围
                try:
                    continent_union = continent_geom.geometry.unary_union
                    out_image, out_transform = mask(src, [continent_union], crop=True, nodata=0)
                    continent_data = out_image[0]

                    # 统计该大洲的值
                    continent_values = continent_data[continent_data != 0]  # 忽略nodata值
                    total_pixels = len(continent_values)

                    if total_pixels > 0:
                        unique, counts = np.unique(continent_values, return_counts=True)
                        percentages = (counts / total_pixels) * 100

                        # 保存结果
                        for driver, count, percent in zip(unique, counts, percentages):
                            if driver in driver_colors:
                                results.append({
                                    'File': tif_file,
                                    'Crop': crop_type,
                                    'Variable': variable_type,
                                    'Continent': continent,
                                    'Driver': int(driver),
                                    'DriverName': driver_colors[driver][0],
                                    'Percentage': percent,
                                    'TotalPixels': total_pixels,
                                    'PixelCount': count
                                })
                except Exception as e:
                    print(f"处理 {tif_file} 的 {continent} 时出错: {e}")
                    continue
    except Exception as e:
        print(f"无法读取文件 {tif_file}: {e}")
        continue

# 转换为DataFrame
if not results:
    print("没有处理任何数据，请检查输入文件")
    exit()

df = pd.DataFrame(results)

# 保存原始统计结果
df.to_csv(os.path.join(output_folder, 'driver_statistics_all.csv'), index=False)

# 为每个TIFF文件生成图表和统计结果
for tif_file in tif_files:
    # 筛选当前文件的数据
    file_data = df[df['File'] == tif_file]
    if len(file_data) == 0:
        continue

    # 创建数据透视表
    pivot_df = file_data.pivot_table(
        index='Continent',
        columns=['Driver', 'DriverName'],
        values='Percentage',
        aggfunc='sum'
    ).fillna(0)

    # 准备绘图数据
    plot_data = pivot_df.T.groupby(level='DriverName').sum().T

    # 获取驱动因子名称列表（仅名称部分）
    driver_names = [name for name, color in driver_colors.values()]

    # 确保只保留数据中存在的驱动因子
    existing_drivers = [name for name in driver_names if name in plot_data.columns]
    plot_data = plot_data[existing_drivers]

    # 获取对应的颜色
    colors = [color for name, color in driver_colors.values() if name in existing_drivers]

    # 创建图形（移除legend=True参数）
    plt.figure(figsize=(15, 8))
    ax = plot_data.plot(kind='bar', stacked=True, color=colors, width=0.8, legend=False)  # 关键修改在这里

    # 设置图表标题和标签
    parts = tif_file.split('_')
    title = f"{parts[0]} {parts[1]} - Dominant Drivers by Continent"
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Continent', fontsize=12)
    plt.ylabel('Percentage of Pixels (%)', fontsize=12)

    # 设置刻度标签（保持倾斜）
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    ax.tick_params(axis='y', labelsize=10)

    # 调整布局
    plt.subplots_adjust(bottom=0.2)

    # 保存图表
    output_name = os.path.splitext(tif_file)[0] + '_distribution.png'
    plt.savefig(os.path.join(output_folder, output_name), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存当前文件的统计结果
    file_data.to_csv(os.path.join(output_folder, f"{os.path.splitext(tif_file)[0]}_stats.csv"), index=False)

print("\n处理完成! 结果已保存到:", output_folder)