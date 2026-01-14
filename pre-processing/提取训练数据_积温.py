import pandas as pd
import rasterio
from pyproj import Transformer
import os
import numpy as np

# 配置参数
excel_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_phase_mon_Hddm_Cddm_add.xlsx'  # Excel文件路径（支持.xlsx）
tif_dir = r'D:\CNPK\氮损失\气候\ERA5_Accu_CDD_wheat-001\ERA5_Accu_CDD_wheat'  # TIFF文件目录
output_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phase_mon_Hddm_Cddm_add.xlsx'  # 输出文件路径

# 读取Excel数据（使用openpyxl引擎）
df = pd.read_excel(excel_path, engine='openpyxl')

# 添加12个月的新列
months = ['Cddw_Month_1', 'Cddw_Month_2', 'Cddw_Month_3', 'Cddw_Month_4', 'Cddw_Month_5', 'Cddw_Month_6',
          'Cddw_Month_7', 'Cddw_Month_8', 'Cddw_Month_9', 'Cddw_Month_10', 'Cddw_Month_11', 'Cddw_Month_12']
for month in months:
    df[month] = np.nan

# 按年份分组处理
for year, year_df in df.groupby('year'):
    if year < 1981 or year > 2020:
        print(f"跳过超出范围的年份: {year}")
        continue

    print(f"正在处理年份: {year}")

    for month_idx in range(1, 13):
        month_str = f"{month_idx:02d}"
        tif_file = os.path.join(tif_dir, f"Cdd_wheat{year}-{month_str}.tif")

        if not os.path.exists(tif_file):
            print(f"文件不存在: {tif_file}")
            continue

        try:
            with rasterio.open(tif_file) as dataset:
                # 读取栅格数据和元数据
                data = dataset.read(1)
                nodata = dataset.nodata
                transform = dataset.transform
                crs = dataset.crs

                # 创建坐标转换器
                transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

                # 处理该年份所有记录
                for idx in year_df.index:
                    lon = df.at[idx, 'Longitude']
                    lat = df.at[idx, 'Latitude']

                    # 坐标转换
                    x, y = transformer.transform(lon, lat)

                    # 获取行列号
                    row, col = dataset.index(x, y)

                    # 提取值并处理异常
                    if 0 <= row < dataset.height and 0 <= col < dataset.width:
                        value = data[row, col]
                        df.at[idx, months[month_idx - 1]] = value if value != nodata else np.nan
                    else:
                        df.at[idx, months[month_idx - 1]] = np.nan

        except Exception as e:
            print(f"处理文件 {tif_file} 时出错: {str(e)}")
            continue

# 保存结果（使用openpyxl引擎）
df.to_excel(output_path, index=False, engine='openpyxl')
print("处理完成，结果已保存到:", output_path)
