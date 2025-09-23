import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal, osr
import geopandas as gpd
from matplotlib.patches import Patch


def parse_legend_file(legend_file):
    """解析颜色映射表文件"""
    color_dict = {}
    with open(legend_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'RGB=' in line:
                parts = line.strip().split('|')
                key = int(parts[0].split(':')[0].strip())
                rgb_str = parts[0].split('RGB=')[1].split(')')[0].strip('(')
                rgb = tuple(map(int, rgb_str.split(',')))
                color_dict[key] = (rgb[0] / 255., rgb[1] / 255., rgb[2] / 255.)
    return color_dict


def create_colormap(color_dict):
    """创建颜色映射"""
    keys = sorted(color_dict.keys())
    colors = [color_dict[k] for k in keys]
    bounds = [k - 0.5 for k in keys] + [keys[-1] + 0.5]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, keys


def get_raster_info(raster_file):
    """获取栅格文件的空间参考和范围信息"""
    ds = gdal.Open(raster_file)
    if ds is None:
        raise ValueError(f"无法打开栅格文件: {raster_file}")
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + geotransform[1] * ds.RasterXSize
    ymin = ymax + geotransform[5] * ds.RasterYSize
    ds = None
    return projection, geotransform, (xmin, ymin, xmax, ymax)


def process_and_export(input_raster_dir, legend_file, template_raster, countries_shp, output_dir, dpi=2000):
    """处理并导出栅格为PNG"""
    # 打印调试信息
    print(f"当前工作目录: {os.getcwd()}")
    print(f"输入目录内容: {os.listdir(input_raster_dir)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 检查所有输入文件是否存在
    required_files = [legend_file, template_raster, countries_shp]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}")

    color_dict = parse_legend_file(legend_file)
    print(f"成功解析颜色映射表，包含 {len(color_dict)} 种颜色")

    try:
        projection, geotransform, extent = get_raster_info(template_raster)
        print(f"成功读取模板栅格: {template_raster}")
    except Exception as e:
        print(f"读取模板栅格失败: {str(e)}")
        return

    try:
        countries = gpd.read_file(countries_shp)
        print(f"成功读取国家边界文件: {countries_shp}")
    except Exception as e:
        print(f"读取国家边界文件失败: {str(e)}")
        return

    # 查找所有匹配的栅格文件
    raster_files = [f for f in os.listdir(input_raster_dir) if f.endswith('_domin.tif')]
    print(f"找到 {len(raster_files)} 个匹配的栅格文件: {raster_files}")

    if not raster_files:
        print("警告: 没有找到匹配的栅格文件 (*_domin.tif)")
        return

    for raster_file in raster_files:
        try:
            raster_path = os.path.join(input_raster_dir, raster_file)
            print(f"正在处理: {raster_path}")

            ds = gdal.Open(raster_path)
            if ds is None:
                print(f"无法打开栅格文件: {raster_path}")
                continue

            band = ds.GetRasterBand(1)
            data = band.ReadAsArray()
            ds = None

            fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi / 100)

            # 先绘制国家边界（作为底图）
            countries.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

            # 再绘制栅格数据（叠加在上面）
            cmap, norm, _ = create_colormap(color_dict)
            im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent,
                           interpolation='nearest')  # 设置一定透明度

            title = os.path.splitext(raster_file)[0].replace('_', ' ')
            ax.set_title(title, fontsize=12, pad=10)
            ax.axis('off')

            plt.tight_layout()
            output_file = os.path.join(output_dir, os.path.splitext(raster_file)[0] + '.png')
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"成功导出: {output_file}")

        except Exception as e:
            print(f"处理文件 {raster_file} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    # 设置参数
    input_raster_dir = r'D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers'  # 当前目录
    legend_file = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\domin_legend1.txt"  # 颜色映射表文件
    template_raster = 'Wheat_Harvested_area_2020.tif'
    countries_shp = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\preds\World_countries.shp"  # 国家边界文件
    output_dir = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers\domin_pngs"  # 输出文件夹
    dpi = 2000

    print("=== 开始处理 ===")
    process_and_export(input_raster_dir, legend_file, template_raster, countries_shp, output_dir, dpi)
    print("=== 处理完成 ===")