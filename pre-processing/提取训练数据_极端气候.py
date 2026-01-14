import pandas as pd
import xarray as xr

# 读取 NetCDF 文件
nc_file = r"D:\CNPK\rx1day.nc"
nc_data = xr.open_dataset(nc_file)
# nc_data["valid_time"] = pd.to_datetime(nc_data["valid_time"].values)
print(nc_data.dims)  # 查看维度名称
print(nc_data.data_vars)  # 查看变量名称
print(nc_data.time.values[:10])  # 查看前10个时间点

# 确保时间是 pandas.Timestamp 格式
# if not isinstance(nc_data["valid_time"].values[0], pd.Timestamp):
#     nc_data["valid_time"] = pd.to_datetime(nc_data["valid_time"].values, format="%m/%d/%Y %H:%M:%S")
#     nc_data["time"] = pd.to_datetime(nc_data["time"].values)
# 确保时间是 pandas.Timestamp 格式
if not isinstance(nc_data["time"].values[0], pd.Timestamp):
    nc_data["time"] = pd.to_datetime(nc_data["time"].values)

# 读取 Excel 数据
excel_file = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phase_mon_Hddm_Cddm_add.xlsx"
df = pd.read_excel(excel_file)
# 检查 year 列是否有 NaN 或无效值
if df["year"].isnull().any():
    print("year 列包含缺失值：")
    print(df[df["year"].isnull()])

# 检查 year 列是否为数字
if not pd.api.types.is_numeric_dtype(df["year"]):
    print("year 列包含非数字值：")
    print(df[~df["year"].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))])

# 提取 12 个月结霜期数据
monthly_data = []
for index, row in df.iterrows():
    lat, lon, year = row["Latitude"], row["Longitude"], row["year"]
    print(lat, lon, year)
    # if lon < -0.05:
    #     lon += 360  # 处理 lon 取值范围问题

    # 提取该年份的 12 个月数据
    year_data = []
    for month in range(1, 13):
        # 构造时间字符串并转换为 pandas.Timestamp
        date_str = f"{month:02d}/16/{year} 00:00:00"
        valid_time_value = pd.Timestamp(date_str)

        try:
            # 匹配最近的时间和空间点
            day_data = nc_data.sel(lat=lat, lon=lon, method="nearest").sel(time=valid_time_value, method="nearest")
            fd_value = day_data["rx1day"].values
        except KeyError:
            fd_value = None  # 如果数据缺失

        year_data.append(fd_value)

    # 添加到总列表
    monthly_data.append(year_data)

# 将 12 个月的数据添加为新列
df[[f"rx1_Month_{i}" for i in range(1, 13)]] = pd.DataFrame(monthly_data)

# # 3. 提取每年数据
# yearly_data = []
# for index, row in df.iterrows():
#     lat, lon, year = row["Latitude"], row["Longitude"], row["year"]
#
#     try:
#         # 在 NetCDF 文件中找到符合年份的所有时间点
#         year_data = nc_data.sel(valid_time=nc_data["valid_time"].dt.year == year)
#
#         # 如果该年份存在多条记录，计算其平均值（也可以使用其他汇总方式）
#         if not year_data["valid_time"].size:
#             print(f"未找到 year={year} 的数据，设置为 None")
#             value = None
#         else:
#             # 按经纬度匹配最近点
#             data_for_location = year_data.sel(lat=lat, lon=lon, method="nearest")
#             value = data_for_location["ro"].mean().values  # 替换为 NetCDF 中的实际变量名
#     except Exception as e:
#         print(f"处理 year={year}, Latitude={lat}, Longitude={lon} 时出错: {e}")
#         value = None
#
#     yearly_data.append(value)
#
# # 将提取的数据添加到 Excel 表格中
# df["sdii_Year"] = yearly_data

# 保存更新的 Excel 文件
output_file = "Dataset_upd_ro_sro_slp_LAI_fd1_rx1.xlsx"
df.to_excel(output_file, index=False)

print(f"数据提取完成，结果已保存到 {output_file}")
