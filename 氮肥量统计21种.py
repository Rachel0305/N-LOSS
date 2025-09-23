import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os
import glob


def process_raster_sum(input_dir, reference_file):
    """
    处理多个栅格文件的加和计算

    参数:
        input_dir: 包含输入栅格文件的目录
        reference_file: 参考栅格文件路径(用于设置输出参数)
    """

    # 1. 获取所有以"2020_nd0.tif"结尾的输入文件
    input_files = sorted(glob.glob(os.path.join(input_dir, "*2020_nd0.tif")))

    # 检查是否找到21个文件
    if len(input_files) != 21:
        raise ValueError(f"在目录 {input_dir} 中找到 {len(input_files)} 个文件，但预期是21个")

    # 2. 读取参考文件获取元数据
    with rasterio.open(reference_file) as ref:
        ref_profile = ref.profile.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = ref.shape
        ref_nodata = ref.nodata

    # 3. 准备输出数组
    output_sum = np.zeros(ref_shape, dtype=np.float32)
    output_count = np.zeros(ref_shape, dtype=np.uint8)

    # 4. 处理每个输入文件
    for file_path in input_files:
        with rasterio.open(file_path) as src:
            print(f"正在处理文件: {os.path.basename(file_path)}")

            # 检查是否需要重投影或重采样
            if src.crs != ref_crs or src.transform != ref_transform or src.shape != ref_shape:
                # 创建临时数组存储重投影后的数据
                temp_data = np.zeros(ref_shape, dtype=np.float32)

                # 执行重投影和重采样
                reproject(
                    source=rasterio.band(src, 1),
                    destination=temp_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )

                # 获取nodata值
                src_nodata = src.nodata if src.nodata is not None else ref_nodata
                src_data = np.where(temp_data == src_nodata, np.nan, temp_data)
            else:
                src_nodata = src.nodata if src.nodata is not None else ref_nodata
                src_data = np.where(src.read(1) == src_nodata, np.nan, src.read(1))

            # 更新计数和求和
            valid_mask = ~np.isnan(src_data)
            output_sum[valid_mask] += np.nan_to_num(src_data[valid_mask], nan=0)
            output_count[valid_mask] += 1

    # 5. 应用处理规则
    # 规则1: 至少有一个有效值 -> 将nodata视为0
    # 规则2: 全部为nodata -> 保持nodata
    result = np.where(output_count > 0, output_sum, ref_nodata)

    # 6. 保存结果
    ref_profile.update(dtype=rasterio.float32, nodata=ref_nodata, count=1)

    output_file = os.path.join(input_dir, "Wheat N input rate.tif")
    with rasterio.open(output_file, 'w', **ref_profile) as dst:
        dst.write(result, 1)

    print(f"\n处理完成，共处理了 {len(input_files)} 个文件")
    print(f"结果已保存到: {output_file}")
    return output_file


# 示例使用
if __name__ == "__main__":
    # 设置输入目录和参考文件路径
    input_dir = r"D:\CNPK\氮损失\田间管理\maize N input 21"  # 替换为实际包含栅格文件的目录
    reference_file = "Maize_Harvested_area_2020.tif"  # 参考文件

    # 确保参考文件存在
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"参考文件 {reference_file} 不存在")

    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录 {input_dir} 不存在")

    # 执行处理
    result_path = process_raster_sum(input_dir, reference_file)