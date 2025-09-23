import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import pandas as pd
from affine import Affine


def get_target_profile(nc_file):
    """生成完整的输出文件元数据"""
    with xr.open_dataset(nc_file) as ds:
        latitude = np.sort(ds.lat.values)
        longitude = np.sort(ds.lon.values)

        return {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': 'np.nan',
            'width': len(longitude),
            'height': len(latitude),
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': Affine.from_gdal(
                longitude[0],
                (longitude[-1] - longitude[0]) / len(longitude),
                0,
                latitude[-1],
                0,
                -(latitude[-1] - latitude[0]) / len(latitude)
            )
        }


def resample_raster(source_path, target_profile):
    """栅格重采样函数"""
    with rasterio.open(source_path) as src:
        data = np.zeros((target_profile['height'], target_profile['width']), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            resampling=Resampling.nearest
        )
        return data


def process_phenology(plant_path, harvest_path, target_profile):
    """处理物候数据"""
    plant = resample_raster(plant_path, target_profile)
    harvest = resample_raster(harvest_path, target_profile)
    plant = np.where((plant >= 1) & (plant <= 12), plant, np.nan)
    harvest = np.where((harvest >= 1) & (harvest <= 12), harvest, np.nan)
    return plant, harvest


def process_spei_data(nc_file, target_profile):
    """处理spei数据（无缩放）"""
    with xr.open_dataset(nc_file) as ds:
        times = pd.to_datetime(ds.time.values)
        time_mask = (times.year == 2018)  | (times.year == 2019)

        spei_data = ds.spei.sel(time=time_mask).load()

        # 维度处理
        if 'scale' in spei_data.dims:
            spei_data = spei_data.isel(scale=0)  # 选择第一个scale
        spei_data = spei_data.squeeze(drop=True)

        # 确保维度顺序
        if set(spei_data.dims) == {'time', 'lat', 'lon'}:
            spei_data = spei_data.transpose('time', 'lat', 'lon')

        return spei_data.astype(np.float32), times[time_mask]


def calculate_features(plant, harvest, spei_data, times):
    """特征计算（无缩放）"""
    max_val = np.full_like(plant, -np.inf, dtype=np.float32)
    min_val = np.full_like(plant, np.inf, dtype=np.float32)
    sum_val = np.zeros_like(plant, dtype=np.float32)
    count = np.zeros_like(plant, dtype=np.int32)

    for t in range(len(times)):
        year = times[t].year
        month = times[t].month
        layer = spei_data[t].values

        # 生成跨年掩膜
        if year == 2018:
            mask = (harvest < plant) & (month >= plant)
        else:
            mask = ((harvest < plant) & (month <= harvest)) | ((harvest >= plant) & (month >= plant))

        valid = ~np.isnan(plant) & ~np.isnan(harvest) & ~np.isnan(layer)
        final_mask = mask & valid

        # 更新统计量
        max_val = np.fmax(max_val, np.where(final_mask, layer, -np.inf))
        min_val = np.fmin(min_val, np.where(final_mask, layer, np.inf))
        sum_val += np.where(final_mask, layer, 0)
        count += final_mask.astype(np.int32)

    return (
        np.where(count > 0, max_val, np.nan),
        np.where(count > 0, min_val, np.nan),
        np.divide(sum_val, count, where=count > 0)
    )


def save_results(data_dict, profile):
    """保存结果"""
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,
        nodata=np.nan,
        compress='lzw',
        tiled=True
    )

    for suffix, data in data_dict.items():
        with rasterio.open(f"spei_Span_{suffix}.tif", 'w', **output_profile) as dst:
            dst.write(data.astype(np.float32), 1)


if __name__ == "__main__":
    NC_FILE = "spei_MON_3_6_12_scale_era5_land_only_0p09_deg_1981_2019.nc"
    PLANT_PATH = "Maize_plant.tif"
    HARVEST_PATH = "Maize_harvest.tif"

    target_profile = get_target_profile(NC_FILE)
    print(f"目标空间参考: {target_profile['width']}x{target_profile['height']}")

    plant, harvest = process_phenology(PLANT_PATH, HARVEST_PATH, target_profile)
    print(f"有效物候格点数: {np.sum(~np.isnan(plant))}")

    spei_data, times = process_spei_data(NC_FILE, target_profile)
    print(f"处理时间段: {times[0]} 至 {times[-1]}")

    max_val, min_val, mean_val = calculate_features(plant, harvest, spei_data, times)

    save_results({'max': max_val, 'min': min_val, 'mean': mean_val}, target_profile)
    print("处理完成！结果文件已保存")