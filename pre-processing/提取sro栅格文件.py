import xarray as xr
import rioxarray
import pandas as pd
import os

# 1. 读取NetCDF文件
ds = xr.open_dataset('ERA5_2014_2020.nc')

# 2. 选择2019-2020年的数据
# 假设时间变量名为'time'，根据实际情况调整
selected_data = ds.sel(valid_time=slice('2019-01-01', '2020-12-31'))

# 3. 创建输出目录
output_dir = 'output_tifs'
os.makedirs(output_dir, exist_ok=True)

# 4. 遍历每个月的数据并保存为TIFF
for i in range(len(selected_data.valid_time)):
    # 获取当前时间步的数据
    time_step = selected_data.isel(valid_time=i)

    # 获取时间值并转换为指定格式
    time_val = pd.to_datetime(str(time_step.valid_time.values))
    time_str = time_val.strftime('%m/01/%Y %H:%M:%S')

    # 假设数据变量名为'variable'，根据实际情况调整
    # 如果有多个变量，需要遍历变量或指定特定变量
    data_var = time_step['sro']  # 替换为你的变量名

    # 确保数据有CRS信息（根据你的数据调整）
    if not data_var.rio.crs:
        data_var.rio.write_crs("EPSG:4326", inplace=True)  # 假设是WGS84

    # 构造输出文件名
    output_filename = f"{output_dir}/ERA5_sro_{time_val.strftime('%Y_%m')}.tif"

    # 保存为GeoTIFF
    data_var.rio.to_raster(output_filename)

    print(f"Saved: {output_filename} with time {time_str}")

# 关闭数据集
ds.close()
print("处理完成！")
