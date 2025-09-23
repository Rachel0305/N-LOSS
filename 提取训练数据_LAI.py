import os
import pandas as pd
import rasterio
from rasterio.windows import Window
import numpy as np
from collections import defaultdict
from datetime import datetime


def get_lai_files(lai_dir):
    """获取所有 LAI TIF 文件，并按年份分类"""
    lai_files = {}
    for file in os.listdir(lai_dir):
        if file.endswith(".tif") and "GlobMapLAIV3" in file:
            parts = file.split(".")
            try:
                date_part = parts[1]  # 提取 AYYYYDDD 部分
                year = int(date_part[1:5])  # 提取年份 YYYY
                doy = int(date_part[5:])  # 提取 DOY（日序号）
            except ValueError:
                print(f"跳过无法解析的文件: {file}")
                continue

            if year not in lai_files:
                lai_files[year] = []
            lai_files[year].append((doy, os.path.join(lai_dir, file)))
    print(lai_files)

    # 按时间排序
    for year in lai_files:
        lai_files[year].sort()
    return lai_files


def extract_lai_from_tif(tif_files, lon, lat):
    """从 TIF 文件列表中提取某经纬度的 LAI 值，并转换为月均值"""
    monthly_lai = defaultdict(list)  # 用于存储每月的 LAI 值

    for doy, tif in tif_files:
        with rasterio.open(tif) as src:
            transform = src.transform
            col, row = ~transform * (lon, lat)
            col, row = int(col), int(row)
            try:
                value = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                if value is not None:
                    month = (doy - 1) // 30 + 1  # 估算影像所在的月份
                    monthly_lai[month].append(value)
            except:
                continue  # 跳过无效数据

    # 计算每月平均值
    monthly_avg_lai = {month: np.mean(values) if values else None for month, values in monthly_lai.items()}

    return [monthly_avg_lai.get(i, None) for i in range(1, 13)]  # 按 1~12 月返回数据


def main(lai_dir, excel_path, output_path):
    # 读取 Excel 数据
    df = pd.read_excel(excel_path)

    if not {'year', 'Longitude', 'Latitude'}.issubset(df.columns):
        raise ValueError("Excel 文件必须包含 'year', 'Longitude', 'Latitude' 列")

    # 获取 LAI 数据
    lai_files = get_lai_files(lai_dir)

    # 添加 LAI 列
    lai_columns = [f'LAI_Month_{i+1}' for i in range(12)]
    df[lai_columns] = None

    for index, row in df.iterrows():
        year, lon, lat = int(row['year']), row['Longitude'], row['Latitude']

        if year in lai_files:
            lai_values = extract_lai_from_tif(lai_files[year], lon, lat)
            df.loc[index, lai_columns] = lai_values

    # 保存结果
    df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存到 {output_path}")



# 使用示例
lai_directory = "D:/CNPK/氮损失/生物多样性/LAI"  # 修改为实际的 LAI TIF 文件夹路径
excel_file = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phase_mon_Hddm_Cddm_add.xlsx"  # 修改为实际的 Excel 文件路径
output_file = "Dataset_with_LAI.xlsx"
main(lai_directory, excel_file, output_file)
