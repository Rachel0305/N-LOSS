import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import rowcol
import os

# ========================
# 参数配置
# ========================
input_path = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm.csv"  # 输入表格路径
output_path = "Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv"  # 输出表格路径
raster_dir = r"D:\CNPK\氮损失\田间管理\产量\GlobalCropYield5min1982_2015_V3\GlobalCropYield5min_V3\Maize"  # 栅格文件存储目录
base_year = 5  # 基准间隔年（5年间隔）


# ========================
# 核心功能函数
# ========================
def get_target_year(year):
    """计算目标基准年份"""
    return (year // base_year) * base_year


def find_valid_raster(crop, target_year):
    """寻找可用的栅格文件（支持年份回溯）"""
    for offset in range(0, 10):  # 最多回溯10年
        check_year = target_year - offset
        raster_path = os.path.join(raster_dir, f"{crop}{check_year}.tif")
        if os.path.exists(raster_path):
            return check_year, raster_path
    return None, None


def extract_raster_value(raster_path, lon, lat):
    """从栅格中提取指定坐标的值"""
    try:
        with rasterio.open(raster_path) as src:
            # 将地理坐标转换为像素坐标
            py, px = src.index(lon, lat)

            # 检查坐标是否在有效范围内
            if (0 <= px < src.width) and (0 <= py < src.height):
                value = src.read(1)[py, px]
                return value if value != src.nodata else np.nan
            return np.nan
    except Exception as e:
        print(f"Error reading {raster_path}: {str(e)}")
        return np.nan


# ========================
# 主处理流程
# ========================
def main():
    # 读取输入数据
    df = pd.read_csv(input_path)

    # 新增yield列
    df["yield"] = np.nan

    # 逐行处理数据
    for idx, row in df.iterrows():
        # 计算目标年份
        target_year = get_target_year(row["year"])

        # 查找可用栅格文件
        valid_year, raster_path = find_valid_raster(
            row["Crop type"],
            target_year
        )

        # 提取栅格值
        if raster_path:
            value = extract_raster_value(
                raster_path,
                row["Longitude"],
                row["Latitude"]
            )

            # 记录结果和实际使用的年份
            df.at[idx, "yield"] = value
            df.at[idx, "used_year"] = valid_year  # 可选：记录实际使用的年份

    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"处理完成！结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
