import pandas as pd
import xarray as xr
from pathlib import Path
import warnings


def process_nc_files(excel_path, nc_folder, output_path):
    # 读取Excel数据并预处理
    df = pd.read_excel(excel_path)
    df[[f"sm_Month_{i}" for i in range(1, 13)]] = None  # 初始化存储列

    # 验证年份数据完整性
    if df["year"].isnull().any():
        raise ValueError("Excel文件中存在缺失的年份值")
    if not pd.api.types.is_integer_dtype(df["year"]):
        raise TypeError("年份列必须为整数类型")

    # 创建NC文件路径模板
    # nc_template = nc_folder / "SMs_{year}_GLEAM_v4.2a_MO.nc"

    # 处理每个年份数据
    for year in range(1982, 2021):
        # 构建NC文件路径
        nc_file = f"SMs_{year}_GLEAM_v4.2a_MO.nc"
        # if not nc_file.exists():
        #     warnings.warn(f"NC文件 {nc_file.name} 不存在，跳过该年份")
        #     continue

        # 筛选当前年份的Excel数据
        year_df = df[df["year"] == year].copy()
        if year_df.empty:
            print(f"Excel中没有 {year} 年数据，跳过")
            continue

        # 处理单个NC文件
        with xr.open_dataset(nc_file) as ds:
            # 标准化时间格式
            ds = process_time_dimension(ds)

            # 处理每个观测点
            monthly_values = []
            for _, row in year_df.iterrows():
                # 调整经度坐标系 (假设NC文件使用0-360经度)
                # adj_lon = row["Longitude"] % 360

                # 提取各月数据
                monthly_values.append(
                    # [get_monthly_value(ds, row["Latitude"], adj_lon, year, month)
                    [get_monthly_value(ds, row["Latitude"], row["Longitude"], year, month)
                     for month in range(1, 13)]
                )

            # 更新DataFrame
            df.loc[year_df.index, [f"sm_Month_{i}" for i in range(1, 13)]] = monthly_values

    # 保存结果
    df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存至 {output_path}")


def process_time_dimension(ds):
    """标准化时间维度格式"""
    if not isinstance(ds.time.values[0], pd.Timestamp):
        try:
            # 处理原始时间格式 "01/31/2000 00:00:00"
            ds["time"] = pd.to_datetime(ds.time.values, format="%m/%d/%Y %H:%M:%S")
        except:
            ds["time"] = xr.decode_cf(ds).time
    return ds


def get_monthly_value(ds, lat, lon, year, month):
    """获取指定位置和时间的SMs值"""
    try:
        # 生成精确的月末时间戳
        target_time = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

        # 四维插值查询 (lat, lon, time)
        return ds.sel(
            lat=lat,
            lon=lon,
            time=target_time,
            method="nearest"
        )["SMs"].values.item()
    except Exception as e:
        print(f"提取失败 {year}-{month:02d} ({lat:.2f}, {lon:.2f}): {str(e)}")
        return None


if __name__ == "__main__":
    # 配置路径
    excel_file = Path("Dataset_upd_ro_sro_slp_LAI_fd_rx5_spei.xlsx")
    nc_folder = Path(r"D:\CNPK")
    output_file = Path("Dataset_upd_sm.xlsx")

    # 执行处理
    process_nc_files(excel_file, nc_folder, output_file)