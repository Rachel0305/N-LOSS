# 六大洲，10个因变量纵坐标标签，横坐标为百分比
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import ListedColormap

# 设置路径和参数
tif_folder = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers'  # 替换为你的TIFF文件所在文件夹
continent_shapefile = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\大洲边界矢量数据\大洲矢量修正版.shp'  # 替换为你的大洲边界矢量数据路径
output_folder = './results0623'  # 输出结果文件夹
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

# 六大洲名称 (根据您的矢量数据中的英文名)
continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
continents_gdf = continents_gdf[continents_gdf['CONTINENT'].isin(continents)]

# 收集所有TIFF文件
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('_domin_int.tif')]
print(f"找到 {len(tif_files)} 个TIFF文件")

# 创建结果DataFrame
results = []

# 处理每个TIFF文件
for tif_file in tif_files:
    # 解析文件名获取因变量信息
    parts = tif_file.split('_')
    if len(parts) < 3:
        print(f"跳过文件名格式不正确的文件: {tif_file}")
        continue

    crop_type = parts[0]  # 作物类型
    variable_type = '_'.join(parts[1:-2])  # 变量类型

    # 在文件处理循环之前定义变量顺序（放在代码开头部分）
    # custom_variable_order = ['Wheat N2OEF', 'Wheat NH3EF', 'Wheat NOEF', 'Wheat LF', 'Wheat RF', 'Maize N2OEF', 'Maize NH3EF', 'Maize NOEF', 'Maize LF', 'Maize RF']
    custom_variable_order = ['N2OEF', 'NH3EF', 'NOEF', 'LF', 'RF']
    # 读取TIFF文件
    file_path = os.path.join(tif_folder, tif_file)
    try:
        with rasterio.open(file_path) as src:
            # 修改nodata值为0或其他合适的整数值
            profile = src.profile
            profile.update(nodata=0)

            # 按大洲统计
            for continent in continents:
                # 获取该大洲的几何形状
                continent_geom = continents_gdf[continents_gdf['CONTINENT'] == continent]
                if len(continent_geom) == 0:
                    print(f"未找到大洲: {continent}")
                    continue

                # 使用union_all()替代已弃用的unary_union
                continent_union = continent_geom.geometry.unary_union

                try:
                    # 裁剪TIFF到大洲范围
                    out_image, out_transform = mask(src, [continent_union], crop=True, nodata=0)
                    continent_data = out_image[0]

                    # 统计该大洲的值
                    continent_values = continent_data[continent_data != 0]  # 忽略nodata值
                    total_pixels = len(continent_values)

                    if total_pixels > 0:
                        unique, counts = np.unique(continent_values, return_counts=True)
                        percentages = (counts / total_pixels) * 100

                        # 保存结果（按照变量顺序检查）
                        if variable_type in custom_variable_order:  # 只处理指定顺序的变量
                            for driver, count in zip(unique, counts):
                                if driver in driver_colors:
                                    results.append({
                                        'Crop': crop_type,
                                        'Variable': variable_type,
                                        'Continent': continent,
                                        'Driver': int(driver),
                                        'Percentage': (count / total_pixels) * 100,
                                        'TotalPixels': total_pixels,
                                        'PixelCount': count,
                                        # 添加排序权重列
                                        'Var_Order': custom_variable_order.index(variable_type)
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

# 调试信息
print("\n=== 数据处理结果 ===")
print("DataFrame形状:", df.shape)
print("列名:", df.columns.tolist())
print("\n前5行数据:")
print(df.head())
print("\n唯一大洲列表:", df['Continent'].unique())

# 保存原始结果
df.to_csv(os.path.join(output_folder, 'driver_statistics_raw.csv'), index=False)

# 为每个大洲创建堆叠图
variables = df['Variable'].unique()
crops = df['Crop'].unique()

# 创建驱动因子排序 (仅使用数据中存在的驱动因子)
driver_order = df.groupby('Driver')['Percentage'].sum().sort_values(ascending=False).index.tolist()

for continent in continents:
    # 筛选该大洲的数据
    continent_data = df[df['Continent'] == continent]
    if len(continent_data) == 0:
        print(f"没有 {continent} 的数据，跳过")
        continue

    # 创建数据透视表
    pivot_df = continent_data.pivot_table(index=['Crop', 'Variable'],
                                          columns='Driver',
                                          values='Percentage',
                                          aggfunc='mean').fillna(0)

    # 仅保留数据中存在的驱动因子
    existing_drivers = [d for d in driver_order if d in pivot_df.columns]
    pivot_df = pivot_df[existing_drivers]

    # 确保颜色与驱动因子对应
    colors = [driver_colors[d][1] for d in pivot_df.columns if d in driver_colors]
    legend_labels = [driver_colors[d][0] for d in pivot_df.columns if d in driver_colors]

    # 创建图形
    plt.figure(figsize=(20, 10), dpi=300)
    ax = plt.gca()  # 获取当前坐标轴

    # 设置固定的边距
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)  # 调整这些值来控制边距

    # 绘制堆叠条形图
    ax = pivot_df.plot(kind='barh', stacked=True, color=colors, width=0.8, legend=False, ax=ax)

    # 设置图表标题和标签
    plt.title(f'Dominant Drivers Distribution in {continent}', fontsize=16, pad=20)
    plt.xlabel('Percentage of Pixels (%)', fontsize=18)
    plt.ylabel('Crop and Variable', fontsize=18)
    # 新增：设置刻度标签字体大小
    ax.tick_params(axis='x', labelsize=16)  # x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=16)  # y轴刻度字体大小

    # 固定y轴标签大小和间距
    ax.set_yticklabels([f"{row[0]} - {row[1]}" for row in pivot_df.index],
                       fontsize=18,
                       fontfamily='sans-serif')
    ax.tick_params(axis='y', which='major', pad=5)  # 控制标签与条形的距离
    # 固定x轴范围（可选）
    ax.set_xlim(0, 100)  # 强制x轴从0到100%

    # 创建图例
    # legend_labels = [driver_colors[d][0] for d in pivot_df.columns if d in driver_colors]
    # plt.legend(legend_labels, title='Drivers', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表（bbox_inches='tight'可以防止标签被截断）
    plt.savefig(os.path.join(output_folder, f"{continent}_driver_distribution.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

print("\n处理完成! 结果已保存到:", output_folder)
