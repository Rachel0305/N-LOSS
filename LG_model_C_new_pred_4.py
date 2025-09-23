import os
import joblib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import glob
import tempfile
import shutil
from tqdm import tqdm


def get_valid_nodata(dtype):
    """根据数据类型返回有效的nodata值"""
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.iinfo(dtype).max  # 如uint8返回255
    else:
        return -9999  # 默认值


def resample_to_target(src_path, target_profile, output_path):
    """
    将源栅格重采样至目标分辨率
    :param src_path: 输入栅格路径
    :param target_profile: 目标栅格的profile字典
    :param output_path: 输出路径
    """
    with rasterio.open(src_path) as src:
        # 动态设置nodata值
        nodata = get_valid_nodata(src.dtypes[0])

        # 更新输出参数
        kwargs = target_profile.copy()
        kwargs.update({
            'count': 1,
            'dtype': src.dtypes[0],
            'nodata': nodata
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=Resampling.bilinear
            )


def align_input_rasters(input_dir, target_raster_path, temp_dir):
    """
    对齐所有输入栅格到目标分辨率
    :param input_dir: 输入栅格目录
    :param target_raster_path: 参考栅格路径
    :param temp_dir: 临时文件目录
    :return: 对齐后的栅格路径列表（按Column_序号排序）
    """
    with rasterio.open(target_raster_path) as target_raster:
        target_profile = target_raster.profile

    input_files = sorted(
        glob.glob(os.path.join(input_dir, "Column_*.tif")),
        key=lambda x: int(os.path.basename(x).split('_')[1])
    )

    if not input_files:
        raise FileNotFoundError(f"目录 {input_dir} 中未找到Column_*.tif文件")

    aligned_rasters = []
    os.makedirs(temp_dir, exist_ok=True)

    for input_file in tqdm(input_files, desc="对齐输入栅格"):
        output_file = os.path.join(temp_dir, os.path.basename(input_file))
        resample_to_target(input_file, target_profile, output_file)
        aligned_rasters.append(output_file)

    return aligned_rasters


def predict_global_nitrogen(model_path, input_rasters, mask_raster_path, output_path):
    """
    执行全球尺度预测
    :param model_path: 模型路径(.joblib)
    :param input_rasters: 对齐后的输入栅格列表
    :param mask_raster_path: 掩膜栅格路径
    :param output_path: 输出栅格路径
    """
    model = joblib.load(model_path)

    with rasterio.open(mask_raster_path) as mask_raster:
        mask_data = mask_raster.read(1)
        profile = mask_raster.profile
        profile.update(dtype=np.float32, nodata=-9999)

        with rasterio.open(output_path, 'w', **profile) as dst:
            output_data = np.full(mask_data.shape, -9999, dtype=np.float32)
            valid_mask = (mask_data != mask_raster.nodata) & (mask_data > 0)
            rows, cols = np.where(valid_mask)

            if len(rows) == 0:
                raise ValueError("掩膜中没有有效像素")

            X = np.zeros((len(rows), len(input_rasters)), dtype=np.float32)
            for i, raster_path in enumerate(tqdm(input_rasters, desc="加载栅格数据")):
                with rasterio.open(raster_path) as src:
                    X[:, i] = src.read(1)[rows, cols]

            output_data[rows, cols] = model.predict(X)
            dst.write(output_data, 1)


def main():
    """主工作流"""
    # ============== 修改以下路径 ==============
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\model.pkl"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib"
    # MODEL_PATH = r":\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0727_GLO_所有变量特征_去掉异常值\Wheat\NUE\model.joblib"  # √
    MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0727_GLO_所有变量特征_去掉异常值\Maize\NUE\model.joblib"  # √

    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib"
    # 0616
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Wheat_N2OEF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_N2OEF_LightGBM.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Maize_LF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征maize\Maize\LF\model.pkl"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征maize\Maize\NUE\model.pkl"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\NUE\model.pkl"

    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Maize_LF_model.joblib"
    # MODEL_PATH = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0511_GLO_所有变量特征\Maize_LF_model.joblib"


    INPUT_DIR = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_nh3_predvars_test0608"
    MASK_RASTER_PATH = "Maize_Harvested_area_2020.tif"
    OUTPUT_PATH = "Maize_NUE0730.tif"
    # ========================================

    temp_dir = tempfile.mkdtemp()
    try:
        print("步骤1/3: 对齐输入栅格...")
        aligned_rasters = align_input_rasters(INPUT_DIR, MASK_RASTER_PATH, temp_dir)

        print("\n步骤2/3: 运行预测模型...")
        predict_global_nitrogen(MODEL_PATH, aligned_rasters, MASK_RASTER_PATH, OUTPUT_PATH)

        print(f"\n步骤3/3: 预测完成！结果已保存至: {OUTPUT_PATH}")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()