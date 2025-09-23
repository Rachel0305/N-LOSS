import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from osgeo import gdal, osr
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show

# 设置工作目录
input_folder = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\domin_drivers"  # 替换为您的文件夹路径
os.chdir(input_folder)

# 参考文件路径
reference_tif = "Wheat_Harvested_area_2020.tif"
world_shp = r"D:\CNPK\N_CODE_PY\up_results\升尺度存档0608\wheat_domin\World_countries.shp"

# 单个自变量颜色映射表
color_dict = {
    1: "#1F77B4", 2: "#2C7BB6", 3: "#8C510A", 4: "#A16216", 5: "#7F7F7F",
    6: "#B87333", 7: "#875C26", 8: "#C88746", 9: "#2CA02C", 10: "#4DAF4A",
    11: "#FF7F0E", 12: "#FF9933", 13: "#5E9ED6", 14: "#87B7E0", 15: "#4590B9",
    16: "#4D9221", 17: "#66A63A", 18: "#40821B", 19: "#D62728", 20: "#ED4142",
    21: "#E13334", 22: "#E64959", 23: "#F2656F", 24: "#EB5764", 25: "#1764AB",
    26: "#3273BF", 27: "#5087C7", 28: "#74A9CF", 29: "#97BBCD", 30: "#64A0C8",
    31: "#D94021", 32: "#F05A3B", 33: "#E44D2E", 34: "#C83042", 35: "#E05062",
    36: "#D44052", 37:'#CC3319', 38:'#ED664A', 39:'#DC4D31', 40:'#BA2B3D', 41:'#DB5E6E', 42:'#CE4555'
}
# 大类自变量颜色映射表
color_dict = {
    1: "#1F77B4", 2: "#1F77B4", 3: "#8C510A", 4: "#8C510A", 5: "#7F7F7F",
    6: "#8C510A", 7: "#8C510A", 8: "#8C510A", 9: "#11BBBB", 10: "#11BBBB",
    11: "#FF7F0E", 12: "#FF7F0E", 13: "#1F77B4", 14: "#1F77B4", 15: "#1F77B4",
    16: "#4D9221", 17: "#4D9221", 18: "#4D9221", 19: "#D62728", 20: "#D62728",
    21: "#D62728", 22: "#D62728", 23: "#D62728", 24: "#D62728", 25: "#1F77B4",
    26: "#1F77B4", 27: "#1F77B4", 28: "#1F77B4", 29: "#1F77B4", 30: "#1F77B4",
    31: "#D62728", 32: "#D62728", 33: "#D62728", 34: "#D62728", 35: "#D62728",
    36: "#D62728", 37:'#D62728', 38:'#D62728', 39:'#D62728', 40:'#D62728', 41:'#D62728', 42:'#D62728'
}

# 创建颜色映射对象
cmap = mcolors.ListedColormap([color_dict[i] for i in sorted(color_dict.keys())])
norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, 37.5, 1), ncolors=36)

# 获取参考影像的元数据
with rasterio.open(reference_tif) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height
    ref_bounds = ref.bounds

# 读取世界国家底图
world = gpd.read_file(world_shp)
world = world.to_crs(ref_crs)

# 处理每个TIFF文件
for tif_file in os.listdir(input_folder):
    if tif_file.endswith("_domin.tif"):
        print(f"Processing {tif_file}...")

        # 读取并预处理TIFF文件
        with rasterio.open(tif_file) as src:
            # 重投影和重采样到参考影像的坐标系和分辨率
            dst_crs = ref_crs
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            # 创建目标数组
            data = np.empty((height, width), dtype=np.float32)

            # 执行重投影
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

            # 裁剪到参考影像的范围
            data = data[:ref_height, :ref_width]

            # 创建掩膜（仅显示有小麦收获面积的区域）
            with rasterio.open(reference_tif) as ref:
                mask_data = ref.read(1)
                data[mask_data == 0] = np.nan

        # 创建图例标签
        # labels = {
        #     1: "MAP", 2: "MAT", 3: "Sand", 4: "Clay", 5: "slope",
        #     6: "pH", 7: "Bulk density", 8: "SOC", 9: "C/N", 10: "N input rate",
        #     11: "Till", 12: "Irrigated", 13: "sro_Span_max", 14: "sro_Span_min", 15: "sro_Key_mean",
        #     16: "LAI_Span_max", 17: "LAI_Span_min", 18: "LAI_Key_mean", 19: "rx1_Span_max", 20: "rx1_Span_min",
        #     21: "rx1_Key_mean", 22: "rx5_Span_max", 23: "rx5_Span_min", 24: "rx5_Key_mean", 25: "Ddp_Span_max",
        #     26: "Ddp_Span_min", 27: "Ddp_Key_mean", 28: "spei_Span_max", 29: "spei_Span_min", 30: "spei_Key_mean",
        #     31: "Hddw_Span_max", 32: "Hddw_Span_min", 33: "Hddw_Key_mean", 34: "Cddw_Span_max", 35: "Cddw_Span_min",
        #     36: "Cddw_Key_mean", 37: "Hddm_Span_max", 38: "Hddm_Span_min", 39: "Hddm_Key_mean", 40: "Cddm_Span_max", 41: "Cddm_Span_min",
        #     42: "Cddm_Key_mean"
        # }

        # 创建图例
        # patches = [mpatches.Patch(color=color_dict[i], label=labels[i]) for i in sorted(color_dict.keys())]

        # 绘制地图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 显示栅格数据
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=[ref_bounds.left, ref_bounds.right,
                                                           ref_bounds.bottom, ref_bounds.top])

        # 叠加世界国家边界
        world.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black', alpha=0.2)

        # 添加图例
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # 设置标题
        title = tif_file.replace("_domin.tif", "").replace("_", "_cate_")
        plt.title(title)

        # 调整布局
        plt.tight_layout()

        # 保存PNG
        output_png = tif_file.replace(".tif", "_cate.png")
        plt.savefig(output_png, dpi=2000, bbox_inches='tight')
        plt.close()

        print(f"Saved {output_png}")

print("All processing completed!")