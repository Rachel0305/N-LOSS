import os
import numpy as np
from osgeo import gdal

# 配置参数
input_dir = r"D:\CNPK\氮损失\生物多样性\LAI"  # 输入栅格目录
output_dir = r"D:\CNPK\N_CODE_PY\N_ML\LAI_monthly"  # 输出目录
years = [2019, 2020]  # 处理年份


def process_monthly_lai():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        for month in range(1, 13):
            # 生成两个半月文件名
            files = [
                f"GIMMS_LAI4g_V1.2_{year}{month:02d}01.tif",
                f"GIMMS_LAI4g_V1.2_{year}{month:02d}02.tif"
            ]
            file_paths = [os.path.join(input_dir, f) for f in files]

            # 检查文件是否存在
            if not all(os.path.exists(fp) for fp in file_paths):
                print(f"缺失文件: {year}-{month:02d}")
                continue

            # 读取栅格数据
            arrays = []
            geo_info = None
            for fp in file_paths:
                ds = gdal.Open(fp)
                band = ds.GetRasterBand(1)

                # 获取空间参考信息
                if not geo_info:
                    geo_info = {
                        'projection': ds.GetProjection(),
                        'geotransform': ds.GetGeoTransform(),
                        'cols': ds.RasterXSize,
                        'rows': ds.RasterYSize
                    }

                # 读取数据并处理无效值
                arr = band.ReadAsArray().astype(float)
                nodata = band.GetNoDataValue()
                arr[arr == nodata] = np.nan
                arrays.append(arr)
                ds = None  # 关闭数据集

            # 计算月平均值
            mean_arr = np.nanmean(arrays, axis=0) / 1000.0

            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            output_path = os.path.join(output_dir, f"LAI_{year}-{month:02d}.tif")

            # 设置输出参数
            out_ds = driver.Create(
                output_path,
                geo_info['cols'],
                geo_info['rows'],
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'PREDICTOR=2']
            )

            # 写入地理信息
            out_ds.SetGeoTransform(geo_info['geotransform'])
            out_ds.SetProjection(geo_info['projection'])

            # 写入数据并设置NoData
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(mean_arr)
            out_band.SetNoDataValue(-9999)

            # 关闭数据集
            out_ds = None
            print(f"已生成：{output_path}")


if __name__ == "__main__":
    process_monthly_lai()