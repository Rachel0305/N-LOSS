import pandas as pd
import xarray as xr

# 1. 读取 NetCDF 文件
nc_file = r"D:\CNPK\cdd.nc4"
nc_data = xr.open_dataset(nc_file)

# 检查时间格式并转换
if not isinstance(nc_data["time"].values[0], pd.Timestamp):
    nc_data["time"] = pd.to_datetime(nc_data["time"].values)

# 调试：打印时间、经纬度范围
print("NetCDF 时间示例：", nc_data["time"].values[:5])
print("纬度范围：", nc_data["latitude"].min().item(), "~", nc_data["latitude"].max().item())
print("经度范围：", nc_data["longitude"].min().item(), "~", nc_data["longitude"].max().item())
print("变量列表：", list(nc_data.data_vars))

# 2. 读取 Excel 数据
excel_file = "Dataset_upd_ro_sro5_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_phase_mon_Hddm_Cddm_add.xlsx"
df = pd.read_excel(excel_file)

# 清理 year 列
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# 3. 提取数据
monthly_data = []
for index, row in df.iterrows():
    lat, lon, year = row["Latitude"], row["Longitude"], row["year"]
    year_data = []

    for month in range(1, 13):
        # 生成更合理的日期（如15日）
        date_str = f"{year}-{month:02d}-28"  # 格式改为 YYYY-MM-DD
        try:
            time_value = pd.Timestamp(date_str)
        except Exception as e:
            print(f"日期解析失败：{date_str}，错误：{e}")
            year_data.append(None)
            continue

        try:
            # 提取数据并验证
            point_data = nc_data.sel(latitude=lat, longitude=lon, method="nearest")
            day_data = point_data.sel(time=time_value, method="nearest")
            cdd_value = day_data["consecutive_dry_days_index_per_time_period"].item()  # 使用 .item()
            print(f"匹配成功：时间={time_value} -> 实际时间={day_data['time'].values}, cdd={cdd_value}")
        except Exception as e:
            print(f"提取失败：时间={time_value}, 经纬度=({lat}, {lon}), 错误：{e}")
            cdd_value = None

        year_data.append(cdd_value)

    monthly_data.append(year_data)

# 4. 保存结果
df[[f"Dpp_Month_{i}" for i in range(1, 13)]] = monthly_data
output_file = "Dataset_upd_ro_sro_slp_LAI_fd_Ddp.xlsx"
df.to_excel(output_file, index=False)
print("处理完成")