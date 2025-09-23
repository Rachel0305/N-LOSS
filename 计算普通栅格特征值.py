# 计算cdd、hdd、lai map mat sro的特征值
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os


def load_raster_data(file_path, target_profile=None):
    """安全加载栅格数据，可选重采样到目标空间参考"""
    try:
        with rasterio.open(file_path) as src:
            if target_profile and not (src.transform.almost_equals(target_profile['transform']) and src.shape == (
            target_profile['height'], target_profile['width'])):
                # 需要重采样的情况
                data = np.empty((target_profile['height'], target_profile['width']), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_profile['transform'],
                    dst_crs=target_profile['crs'],
                    resampling=Resampling.nearest
                )
            else:
                data = src.read(1)

            # 处理无效值
            data = np.where(data == src.nodata, np.nan, data.astype(np.float32))
            return data
    except Exception as e:
        print(f"无法读取文件 {file_path}: {str(e)}")
        return None


def process_variable(var_name, plant_path, harvest_path, months_to_process, temp_dir="temp"):
    """处理单个变量的核心函数"""
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)

    # 步骤1：确定基准空间参考（以第一个存在的月份文件为准）
    target_profile = None
    for year, month in months_to_process:
        test_file = f"{var_name}{year}_{month:02d}.tif"
        if os.path.exists(test_file):
            with rasterio.open(test_file) as src:
                target_profile = src.profile.copy()
                target_profile.update(dtype=np.float32, nodata=np.nan)
            break

    if target_profile is None:
        raise ValueError(f"找不到任何 {var_name} 的月份数据文件")

    # 步骤2：加载并对齐物候数据
    print(
        f"正在处理变量: {var_name} | 空间参考: {target_profile['crs']} {target_profile['width']}x{target_profile['height']}")

    plant = load_raster_data(plant_path, target_profile)
    harvest = load_raster_data(harvest_path, target_profile)

    if plant is None or harvest is None:
        raise ValueError("物候数据加载失败")

    # 步骤3：初始化统计矩阵
    stats = {
        'max': np.full(plant.shape, -np.inf, dtype=np.float32),
        'min': np.full(plant.shape, np.inf, dtype=np.float32),
        'sum': np.zeros(plant.shape, dtype=np.float32),
        'count': np.zeros(plant.shape, dtype=np.int32)
    }

    # 步骤4：处理每个月份的数据
    for year, month in months_to_process:
        # 动态构建文件名（关键读取步骤）
        input_file = f"{var_name}{year}_{month:02d}.tif"
        if not os.path.exists(input_file):
            print(f"  跳过缺失文件: {input_file}")
            continue

        print(f"  处理 {year}_{month:02d}...", end='\r')

        # 读取当前月份数据（核心读取操作）
        current_data = load_raster_data(input_file)
        if current_data is None:
            continue

        # 计算有效掩膜
        if year == 2019:
            month_mask = (harvest < plant) & (month >= plant) & (month <= 12)
        else:
            month_mask = ((harvest < plant) & (month <= harvest)) | (
                        (harvest >= plant) & (month >= plant) & (month <= harvest))

        valid_mask = ~np.isnan(plant) & ~np.isnan(harvest) & ~np.isnan(current_data)
        final_mask = month_mask & valid_mask

        # 更新统计量
        stats['max'] = np.fmax(stats['max'], np.where(final_mask, current_data, -np.inf))
        stats['min'] = np.fmin(stats['min'], np.where(final_mask, current_data, np.inf))
        stats['sum'] += np.where(final_mask, current_data, 0)
        stats['count'] += final_mask.astype(np.int32)

    # 步骤5：计算结果
    results = {
        'max': stats['max'],
        'min': stats['min'],
        'mean': np.divide(stats['sum'], stats['count'], where=stats['count'] > 0)
    }
    results['max'][stats['count'] == 0] = np.nan
    results['min'][stats['count'] == 0] = np.nan

    # 步骤6：保存结果
    for stat_name, data in results.items():
        output_file = f"{var_name}m_Span_{stat_name}.tif"
        with rasterio.open(output_file, 'w', **target_profile) as dst:
            dst.write(data.astype(np.float32), 1)
        print(f"生成结果: {output_file}")


if __name__ == "__main__":
    # 配置参数
    config = {
        "plant_path": "Maize_plant.tif",
        "harvest_path": "Maize_harvest.tif",
        "variables": ["ERA5_sro_"],  # 示例变量
        "months": [(2019, m) for m in range(1, 13)] + [(2020, m) for m in range(1, 13)]
    }

    # 执行处理
    for var in config["variables"]:
        print(f"\n{'=' * 40}\n处理变量: {var}\n{'=' * 40}")
        try:
            process_variable(
                var_name=var,
                plant_path=config["plant_path"],
                harvest_path=config["harvest_path"],
                months_to_process=config["months"]
            )
        except Exception as e:
            print(f"!! 处理失败: {str(e)}")