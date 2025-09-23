import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import ListedColormap

# 设置路径和参数
tif_folder = r'D:\domin_tif0907'
continent_shapefile = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\大洲边界矢量数据\大洲矢量修正版.shp'
output_folder = './results_stacked0919'
os.makedirs(output_folder, exist_ok=True)

# 驱动因子与颜色映射
driver_colors = {
    1: ("MAP", "#FF0000"),
    2: ("MAT", "#00FF00"),
    3: ("Sand", "#0000FF"),
    4: ("Clay", "#FFFF00"),
    5: ("Slope", "#FF00FF"),
    6: ("pH", "#00FFFF"),
    7: ("Bulk density", "#FF8000"),
    8: ("SOC", "#8000FF"),
    9: ("CN ratio", "#0080FF"),
    10: ("N input rate", "#FF0080"),
    11: ("Tillage", "#80FF00"),
    12: ("Irrigation", "#00FF80"),
    13: ("Sro span max", "#800000"),
    14: ("Sro span min", "#008000"),
    15: ("Sro key mean", "#000080"),
    16: ("LAI span max", "#808000"),
    17: ("LAI span min", "#800080"),
    18: ("LAI key mean", "#008080"),
    19: ("Rx1 span max", "#FF4000"),
    20: ("Rx1 span min", "#0040FF"),
    21: ("Rx1 key mean", "#FF0040"),
    22: ("Rx5 span max", "#40FF00"),
    23: ("Rx5 span min", "#0040FF"),
    24: ("Rx5 key mean", "#FF00BF"),
    25: ("Ddp span max", "#BF00FF"),
    26: ("Ddp span min", "#00BFFF"),
    27: ("Ddp key mean", "#FFBF00"),
    28: ("Spei span max", "#00FFBF"),
    29: ("Spei span min", "#BF00BF"),
    30: ("Spei key mean", "#00BFBF"),
    31: ("SMs span max", "#BFBF00"),
    32: ("SMs span min", "#4000FF"),
    33: ("SMs key mean", "#FF0040"),
    34: ("SMrz span max", "#0040FF"),
    35: ("SMrz span min", "#FF00BF"),
    36: ("SMrz key mean", "#BF00FF"),
    37: ("Hddw span max", "#00BFFF"),
    38: ("Hddw span min", "#FFBF00"),
    39: ("Hddw key mean", "#00FFBF"),
    40: ("Cddw span max", "#BF00BF"),
    41: ("Cddw span min", "#00BFBF"),
    42: ("Cddw key mean", "#BFBF00"),
    43: ("Hddm span max", "#FF1493"),
    44: ("Hddm span min", "#8B0000"),
    45: ("Hddm key mean", "#006400"),
    46: ("Cddm span max", "#00008B"),
    47: ("Cddm span min", "#8B008B"),
    48: ("Cddm key mean", "#8B4513"),
    49: ("Fertilizer type", "#008B8B")
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
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('factor.tif')]
print(f"找到 {len(tif_files)} 个TIFF文件")

# 创建结果DataFrame
results = []
global_stats = []  # 新增：用于存储全球统计数据

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

            # 读取全局数据
            global_data = src.read(1)
            global_values = global_data[global_data != 0]  # 忽略nodata值
            global_total_pixels = len(global_values)

            # 按大洲统计
            continent_results = []
            for continent in continents:
                continent_geom = continents_gdf[continents_gdf['CONTINENT'] == continent]
                if len(continent_geom) == 0:
                    print(f"未找到大洲: {continent}")
                    continue

                try:
                    continent_union = continent_geom.geometry.unary_union
                    out_image, out_transform = mask(src, [continent_union], crop=True, nodata=0)
                    continent_data = out_image[0]
                    continent_values = continent_data[continent_data != 0]
                    total_pixels = len(continent_values)

                    if total_pixels > 0:
                        unique, counts = np.unique(continent_values, return_counts=True)
                        percentages = (counts / total_pixels) * 100

                        for driver, count, percent in zip(unique, counts, percentages):
                            if driver in driver_colors:
                                continent_results.append({
                                    'Continent': continent,
                                    'Driver': int(driver),
                                    'Percentage': percent,
                                    'PixelCount': count,
                                    'TotalPixels': total_pixels
                                })
                except Exception as e:
                    print(f"处理 {tif_file} 的 {continent} 时出错: {e}")
                    continue

            # 计算全球统计数据（基于各洲统计数据，确保一致性）
            if continent_results:
                # 创建DataFrame
                continent_df = pd.DataFrame(continent_results)

                # 计算全球总计
                global_pixels = continent_df.groupby('Driver')['PixelCount'].sum()
                global_total = global_pixels.sum()

                # 计算全球百分比
                global_percent = (global_pixels / global_total) * 100

                # 添加全球数据
                for driver, percent in global_percent.items():
                    if driver in driver_colors:
                        global_stats.append({
                            'File': tif_file,
                            'Crop': crop_type,
                            'Variable': variable_type,
                            'Continent': 'Global',
                            'Driver': driver,
                            'DriverName': driver_colors[driver][0],
                            'Percentage': percent,
                            'TotalPixels': global_total,
                            'PixelCount': global_pixels[driver]
                        })

                # 添加大洲数据
                for _, row in continent_df.iterrows():
                    results.append({
                        'File': tif_file,
                        'Crop': crop_type,
                        'Variable': variable_type,
                        'Continent': row['Continent'],
                        'Driver': row['Driver'],
                        'DriverName': driver_colors[row['Driver']][0],
                        'Percentage': row['Percentage'],
                        'TotalPixels': row['TotalPixels'],
                        'PixelCount': row['PixelCount']
                    })
    except Exception as e:
        print(f"无法读取文件 {tif_file}: {e}")
        continue

# 合并大洲和全球统计数据
df = pd.DataFrame(results + global_stats)  # 修改：合并两个列表

# 保存原始统计结果
df.to_csv(os.path.join(output_folder, 'driver_statistics_all0919.csv'), index=False)

# 修改后的绘图部分代码
for tif_file in tif_files:
    # 筛选当前文件的数据
    file_data = df[df['File'] == tif_file]
    if len(file_data) == 0:
        continue

    # 定义正确的continent顺序（Global在最后）
    continent_order = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Global']

    # 创建分类类型确保正确排序
    file_data['Continent'] = pd.Categorical(file_data['Continent'],
                                            categories=continent_order,
                                            ordered=True)

    # 创建数据透视表
    pivot_df = file_data.pivot_table(
        index='Continent',
        columns=['Driver', 'DriverName'],
        values='Percentage',
        aggfunc='sum'
    ).fillna(0).loc[continent_order]  # 确保按指定顺序排列

    # 准备绘图数据
    plot_data = pivot_df.T.groupby(level='DriverName').sum().T

    # 获取驱动因子名称列表（仅名称部分）
    driver_names = [name for name, color in driver_colors.values()]

    # 确保只保留数据中存在的驱动因子
    existing_drivers = [name for name in driver_names if name in plot_data.columns]
    plot_data = plot_data[existing_drivers]

    # 获取对应的颜色
    colors = [color for name, color in driver_colors.values() if name in existing_drivers]

    # 创建图形
    plt.figure(figsize=(8, 16))
    ax = plot_data.plot(kind='bar', stacked=True, color=colors, width=0.8, legend=False)

    # 设置图表标题和标签
    parts = tif_file.split('_')
    title = f"{parts[0]} {parts[1]} - Dominant Drivers by Continent"
    plt.title(title, fontsize=12, pad=20)
    plt.xlabel('Continent', fontsize=12)
    plt.ylabel('Percentage of Pixels (%)', fontsize=12)

    # 设置刻度标签（保持倾斜）
    ax.set_xticks(range(len(continent_order)))  # 明确设置刻度位置
    ax.set_xticklabels(continent_order, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=10)

    # 添加图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    # plt.legend(handles, existing_drivers, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局
    plt.subplots_adjust(bottom=0.2, right=0.7)

    # 保存图表
    output_name = os.path.splitext(tif_file)[0] + '_distribution.png'
    plt.savefig(os.path.join(output_folder, output_name), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存当前文件的统计结果
    file_data.to_csv(os.path.join(output_folder, f"{os.path.splitext(tif_file)[0]}_stats0912.csv"), index=False)

print("\n处理完成! 结果已保存到:", output_folder)