import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import ListedColormap
import uuid

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

# 加载大洲数据
try:
    continents_gdf = gpd.read_file(os.path.join(continent_shapefile))
    print("成功加载大洲数据，包含以下大洲:", continents_gdf['CONTINENT'].unique())
except Exception as e:
    print(f"加载大洲数据失败: {e}")
    exit()

# 六大洲名称
continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
continents_gdf = continents_gdf[continents_gdf['CONTINENT'].isin(continents)]

# 收集所有TIFF文件
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('_domin_int.tif')]
print(f"找到 {len(tif_files)} 个TIFF文件")

# 创建结果DataFrame
results = []
custom_variable_order = ['N2OEF', 'NH3EF', 'NOEF', 'LF', 'RF']

# 处理每个TIFF文件
for tif_file in tif_files:
    parts = tif_file.split('_')
    if len(parts) < 3:
        print(f"跳过文件名格式不正确的文件: {tif_file}")
        continue

    crop_type = parts[0]
    variable_type = '_'.join(parts[1:-2])

    file_path = os.path.join(tif_folder, tif_file)
    try:
        with rasterio.open(file_path) as src:
            profile = src.profile
            profile.update(nodata=0)

            for continent in continents:
                continent_geom = continents_gdf[continents_gdf['CONTINENT'] == continent]
                if len(continent_geom) == 0:
                    print(f"未找到大洲: {continent}")
                    continue

                continent_union = continent_geom.geometry.unary_union

                try:
                    out_image, out_transform = mask(src, [continent_union], crop=True, nodata=0)
                    continent_data = out_image[0]
                    continent_values = continent_data[continent_data != 0]
                    total_pixels = len(continent_values)

                    if total_pixels > 0:
                        unique, counts = np.unique(continent_values, return_counts=True)
                        for driver, count in zip(unique, counts):
                            if driver in driver_colors:
                                results.append({
                                    'Crop': crop_type,
                                    'Variable': variable_type,
                                    'Continent': continent,
                                    'Driver': int(driver),
                                    'Driver_Name': driver_colors[int(driver)][0],
                                    'Percentage': (count / total_pixels) * 100,
                                    'TotalPixels': total_pixels,
                                    'PixelCount': count,
                                    'Var_Order': custom_variable_order.index(variable_type) if variable_type in custom_variable_order else len(custom_variable_order)
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

# 保存统计结果
df.to_csv(os.path.join(output_folder, 'driver_statistics.csv'), index=False)

# 为每个变量生成堆叠条形图
driver_order = df.groupby('Driver')['Percentage'].sum().sort_values(ascending=False).index.tolist()

for crop in df['Crop'].unique():
    for variable in custom_variable_order:
        # 筛选数据
        var_data = df[(df['Crop'] == crop) & (df['Variable'] == variable)]
        if len(var_data) == 0:
            print(f"没有 {crop} {variable} 的数据，跳过")
            continue

        # 创建数据透视表
        pivot_df = var_data.pivot_table(index='Continent',
                                     columns='Driver',
                                     values='Percentage',
                                     aggfunc='mean').fillna(0)

        # 仅保留数据中存在的驱动因子
        existing_drivers = [d for d in driver_order if d in pivot_df.columns]
        pivot_df = pivot_df[existing_drivers]

        # 准备颜色
        colors = [driver_colors[d][1] for d in pivot_df.columns if d in driver_colors]

        # 创建图形
        plt.figure(figsize=(15, 8), dpi=300)
        ax = pivot_df.plot(kind='bar', stacked=True, color=colors, width=0.8, legend=False)

        plt.title(f'{crop} {variable} 主导驱动因子分布', fontsize=16, pad=20)
        plt.ylabel('像元百分比 (%)', fontsize=14)
        plt.xlabel('大洲', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim(0, 100)

        # 倾斜横坐标标签
        plt.xticks(rotation=45, ha='right')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        plt.savefig(os.path.join(output_folder, f'{crop}_{variable}_driver_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

print("\n处理完成！结果已保存到:", output_folder)