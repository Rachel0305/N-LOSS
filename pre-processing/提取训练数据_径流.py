import pandas as pd
import xarray as xr
import numpy as np

# 读取 NetCDF 文件
nc_file = r"D:\CNPK\ERA5_2014_2020.nc"
nc_data = xr.open_dataset(nc_file)

# **确保 `valid_time` 是 datetime64 类型**
nc_data["valid_time"] = pd.to_datetime(nc_data["valid_time"].values)

# 读取 Excel 数据
excel_file = "Dataset_upd_ro_sro5_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_phase_mon_Hddm_Cddm_add.xlsx"
df = pd.read_excel(excel_file)

# 检查 NetCDF 变量名
if "sro" not in nc_data:
    raise ValueError("❌ NetCDF 文件中没有变量 'sro'，请检查变量名！")

# 存储提取的月度数据
monthly_data = []

for index, row in df.iterrows():
    lat, lon, year = row["Latitude"], row["Longitude"], row["year"]

    # **检查 lat/lon 是否符合 NetCDF 数据范围**
    if lon < -0.05:
        lon += 360  # 如果 lon 是 -180~180，转换为 0~360

    if not (nc_data["latitude"].min() <= lat <= nc_data["latitude"].max()):
        print(f"⚠️ 纬度 {lat} 超出 NetCDF 范围 [{nc_data['latitude'].min()} ~ {nc_data['latitude'].max()}]")
        continue

    if not (nc_data["longitude"].min() <= lon <= nc_data["longitude"].max()):
        print(f"⚠️ 经度 {lon} 超出 NetCDF 范围 [{nc_data['longitude'].min()} ~ {nc_data['longitude'].max()}]")
        continue

    # 提取该年份的 12 个月数据
    year_data = []

    for month in range(1, 13):
        valid_time_value = pd.Timestamp(year, month, 1, 0, 0, 0)  # `YYYY-MM-01 00:00:00`

        # **检查时间是否存在**
        if valid_time_value not in nc_data["valid_time"].values:
            print(f"⚠️ 时间 {valid_time_value} 不在 NetCDF 数据中")
            year_data.append(None)
            continue

        # **获取最近的 lat/lon 数据**
        try:
            day_data = nc_data.sel(latitude=lat, longitude=lon, method="nearest").sel(valid_time=valid_time_value)
            sro_value = day_data["sro"].values.item()  # 获取单个值
        except Exception as e:
            print(f"❌ 错误: {e}")
            sro_value = None  # 数据缺失

        year_data.append(sro_value)

    # 添加到总列表
    monthly_data.append(year_data)

# **转换为 DataFrame 并添加到原 Excel**
df[[f"sro6_Month_{i}" for i in range(1, 13)]] = pd.DataFrame(monthly_data)

# **保存 Excel**
output_file = "Dataset_upd_ro_sro6_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_phase_mon_Hddm_Cddm_add.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ 数据提取完成，已保存到 {output_file}")
