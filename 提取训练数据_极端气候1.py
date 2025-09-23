# cdd=最长干旱期
import pandas as pd
import xarray as xr

# 1. 读取 NetCDF 文件
nc_file = r"D:\CNPK\cdd.nc4"
nc_data = xr.open_dataset(nc_file)

# 检查时间格式，并转换为 pandas.Timestamp
if not isinstance(nc_data["time"].values[0], pd.Timestamp):
    nc_data["time"] = pd.to_datetime(nc_data["time"].values)

# 2. 读取 Excel 数据
excel_file = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5.xlsx"
df = pd.read_excel(excel_file)

# 检查 year 列是否包含无效值
if df["year"].isnull().any():
    print("year 列包含缺失值：")
    print(df[df["year"].isnull()])

if not pd.api.types.is_numeric_dtype(df["year"]):
    print("year 列包含非数字值：")
    print(df[~df["year"].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))])

# 删除无效年份的记录
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# 3. 提取 12 个月的降雨数据
monthly_data = []
for index, row in df.iterrows():
    lat, lon, year = row["Latitude"], row["Longitude"], row["year"]

    # 提取该年份的 12 个月的数据
    year_data = []
    for month in range(1, 13):
        date_str = f"{month:02d}/28/{int(year)} 00:00:00"
        try:
            time_value = pd.Timestamp(date_str)
        except Exception as e:
            print(f"日期解析失败：{date_str}，错误信息：{e}")
            time_value = None

        if time_value is not None:
            try:
                # 匹配最近的时间和空间点
                day_data = nc_data.sel(lat=lat, lon=lon, method="nearest").sel(time=time_value, method="nearest")
                cdd_value = day_data["consecutive_dry_days_index_per_time_period"].values
            except KeyError:
                cdd_value = None  # 如果数据缺失

            year_data.append(cdd_value)
        else:
            year_data.append(None)

    monthly_data.append(year_data)

# 将 12 个月的数据添加为新列
df[[f"cdd_Month_{i}" for i in range(1, 13)]] = pd.DataFrame(monthly_data)

# 保存更新的 Excel 文件
output_file = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_cdd.xlsx"
df.to_excel(output_file, index=False)

print(f"数据提取完成，结果已保存到 {output_file}")
