import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

# 打开 HDF5 文件
h5_file = r"D:\CNPK相关\氮损失\田间管理\施氮量\Harvested_area_1961-2020.h5"
output_dir = "output_tif_files/"

with h5py.File(h5_file, 'r') as f:
    # 查看文件的关键结构
    print(list(f.keys()))  # 查看主数据集名称
    dataset_name = 'your_dataset'  # 替换为实际数据集名称
    data = f[dataset_name][:]  # 读取数据

    # 假设数据是 (年份数, 高度, 宽度) 的 3D 数组
    years = f[dataset_name].attrs['years']  # 读取年份属性（假设存在）
    geo_transform = f[dataset_name].attrs['geo_transform']  # 获取地理信息
    crs = f[dataset_name].attrs['crs']  # 获取投影信息

    for i, year in enumerate(years):
        # 提取对应年份数据
        yearly_data = data[i, :, :]

        # 设置 GeoTIFF 的元数据
        transform = from_origin(
            geo_transform[0],  # 左上角经度
            geo_transform[1],  # 左上角纬度
            geo_transform[2],  # 像元大小（经度方向）
            geo_transform[3]  # 像元大小（纬度方向）
        )

        # 保存为 TIF 文件
        output_path = f"{output_dir}{year}.tif"
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=yearly_data.shape[0],
                width=yearly_data.shape[1],
                count=1,  # 单波段
                dtype=yearly_data.dtype,
                crs=crs,
                transform=transform,
        ) as dst:
            dst.write(yearly_data, 1)

        print(f"Saved {year}.tif to {output_dir}")
