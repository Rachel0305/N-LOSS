# import pandas as pd
# import rasterio
# import os
# import numpy as np
# from tqdm import tqdm  # 进度条工具（可选）
#
# # 配置参数
# EXCEL_PATH = "Dataset_upd_ro_sro_slp.xlsx"
# LAI_DIR = r"D:\CNPK\氮损失\生物多样性\LAI"  # LAI文件存储目录
# OUTPUT_PATH = "Dataset_upd_LAI.xlsx"
#
# # 读取Excel数据
# df = pd.read_excel(EXCEL_PATH)
#
# # 为每个月创建存储列（LAI_01 ~ LAI_12）
# for month in range(1, 13):
#     df[f"LAI_{month:02d}"] = np.nan
#
#
# def extract_lai(lon, lat, year, month):
#     """提取指定年份月份的两个LAI值并返回平均值"""
#     lai_values = []
#
#     # 生成两个文件名（每月两个影像）
#     for half in ["01", "02"]:
#         filename = f"GIMMS_LAI4g_V1.2_{year}{month:02d}{half}.tif"
#         filepath = os.path.join(LAI_DIR, filename)
#
#         if not os.path.exists(filepath):
#             return np.nan  # 文件不存在时返回NaN
#
#         try:
#             with rasterio.open(filepath) as src:
#                 # 将地理坐标转换为像素坐标
#                 row, col = src.index(lon, lat)
#                 # 提取像素值（注意波段索引从1开始）
#                 value = src.read(1)[row, col]
#                 # 处理无效值（假设使用-9999作为NoData）
#                 lai_values.append(value if value != -9999 else np.nan)
#         except Exception as e:
#             print(f"Error reading {filename}: {str(e)}")
#             return np.nan
#
#     # 返回两个半月值的平均值
#     return np.nanmean(lai_values)
#
#
# # 遍历处理每条记录（使用tqdm显示进度）
# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     year = int(row["year"])
#     lon = row["Longitude"]
#     lat = row["Latitude"]
#
#     # 提取12个月的LAI数据
#     for month in range(1, 13):
#         lai_value = extract_lai(lon, lat, year, month)
#         df.at[idx, f"LAI_{month:02d}"] = lai_value
#
# # 保存结果
# df.to_excel(OUTPUT_PATH, index=False)
# print(f"处理完成！结果已保存至 {OUTPUT_PATH}")
import os
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm


def extract_lai_value(tif_path, lon, lat, band_num=1):
    """从指定波段提取栅格值"""
    try:
        ds = gdal.Open(tif_path)
        if ds is None:
            return np.nan
        band = ds.GetRasterBand(band_num)
        nodata = band.GetNoDataValue()
        transform = ds.GetGeoTransform()
        x_pixel = int((lon - transform[0]) / transform[1])
        y_pixel = int((lat - transform[3]) / transform[5])
        if x_pixel < 0 or y_pixel < 0 or x_pixel >= ds.RasterXSize or y_pixel >= ds.RasterYSize:
            return np.nan
        value = band.ReadAsArray(x_pixel, y_pixel, 1, 1)[0, 0]
        return value if value != nodata else np.nan
    except Exception as e:
        print(f"Error processing {tif_path}: {str(e)}")
        return np.nan


def process_year_data(df, year, tif_folder):
    time_series = {f'{year}_{month:02d}': [] for month in range(1, 13)}
    tif_files = sorted([f for f in os.listdir(tif_folder) if f.startswith(f'GIMMS_LAI4g_V1.2_{year}')])

    for tif_file in tqdm(tif_files, desc=f'Processing {year}'):
        date_str = tif_file.split('_')[-1].split('.')[0]
        month = int(date_str[4:6])
        tif_path = os.path.join(tif_folder, tif_file)
        df['temp'] = df.apply(lambda row: extract_lai_value(tif_path, row['Longitude'], row['Latitude']), axis=1)
        time_series[f'{year}_{month:02d}'].append(df['temp'])

    ts_df = pd.DataFrame()
    for month in range(1, 13):
        month_key = f'{year}_{month:02d}'
        if len(time_series[month_key]) == 2:
            ts_df[month_key] = (pd.concat(time_series[month_key], axis=1)
                                .mean(axis=1)
                                .interpolate(method='linear', limit_direction='both'))

    ts_df = ts_df.interpolate(axis=1, method='linear').ffill(axis=1).bfill(axis=1)
    return pd.concat([df, ts_df], axis=1)


if __name__ == "__main__":
    tif_folder = r"D:\CNPK\氮损失\生物多样性\LAI"
    excel_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phase_mon_Hddm_Cddm_add.xlsx'
    output_path = 'Dataset_with_LAI.xlsx'

    df = pd.read_excel(excel_path)
    final_df = pd.DataFrame()

    for year in range(1982, 2021):
        year_df = df[df['year'] == year].copy()
        if not year_df.empty:
            processed_df = process_year_data(year_df, year, tif_folder)
            final_df = pd.concat([final_df, processed_df], ignore_index=True)

    final_df.to_excel(output_path, index=False)