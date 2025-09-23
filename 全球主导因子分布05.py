import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. 定义变量和颜色映射
variable_info ={
    "MAP": {"order": 1, "color": "#1F77B4", "filename": "Column_1_MAP.tif"},
    "sro_Span_mean": {"order": 2, "color": "#1F77B4", "filename": "Column_2_sro_Span_mean.tif"},
    "LAIm_Span_mean": {"order": 3, "color": "#4D9221", "filename": "Column_3_LAIm_Span_mean.tif"},
    "Ddpm_Span_mean": {"order": 4, "color": "#1F77B4", "filename": "Column_4_Ddpm_Span_mean.tif"},
    "LAIm_Span_max": {"order": 5, "color": "#4D9221", "filename": "Column_5_LAIm_Span_max.tif"},
    "Ddpm_Span_max": {"order": 6, "color": "#1F77B4", "filename": "Column_6_Ddpm_Span_max.tif"},
    "Sand": {"order": 7, "color": "#8C510A", "filename": "Column_7_Sand.tif"},
    "Ddpm_Span_min": {"order": 8, "color": "#1F77B4", "filename": "Column_8_Ddpm_Span_min.tif"},
    "rx1m_Span_min": {"order": 9, "color": "#D62728", "filename": "Column_9_rx1m_Span_min.tif"},
    "LAIm_Span_min": {"order": 10, "color": "#4D9221", "filename": "Column_10_LAIm_Span_min.tif"},
    "C/N": {"order": 11, "color": "#2CA02C", "filename": "Column_11_CN.tif"},
    "MAT": {"order": 12, "color": "#1F77B4", "filename": "Column_12_MAT.tif"},
    "Clay": {"order": 13, "color": "#8C510A", "filename": "Column_13_Clay.tif"},
    "pH": {"order": 14, "color": "#8C510A", "filename": "Column_14_pH.tif"},
    "SOC": {"order": 15, "color": "#8C510A", "filename": "Column_15_SOC.tif"},
    "Bulk Density": {"order": 16, "color": "#8C510A", "filename": "Column_16_Bulk Density.tif"},
    "sro_Span_max": {"order": 17, "color": "#1F77B4", "filename": "Column_17_sro_Span_max.tif"},
    "Slope": {"order": 18, "color": "#7F7F7F", "filename": "Column_18_Slope.tif"},
    "Hddm_Span_max": {"order": 19, "color": "#D62728", "filename": "Column_19_Hddm_Span_max.tif"},
    "Hddm_Span_mean": {"order": 20, "color": "#D62728", "filename": "Column_20_Hddm_Span_mean.tif"}
}

# 按顺序排列变量
variables = sorted(variable_info.keys(), key=lambda x: variable_info[x]["order"])
colors = [variable_info[var]["color"] for var in variables]
filenames = [variable_info[var]["filename"] for var in variables]

# 2. 加载参考栅格
reference_raster = "Maize_Harvested_area_2020.tif"


def load_reference_raster():
    with rasterio.open(reference_raster) as src:
        profile = src.profile.copy()
        mask_data = src.read(1)
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds
        res = src.res
    return profile, mask_data, transform, crs, width, height, bounds, res


profile, mask_data, transform, crs, width, height, bounds, res = load_reference_raster()


# 3. 改进的预处理和重投影函数
def preprocess_and_reproject(filepath):
    with rasterio.open(filepath) as src:
        # 读取数据
        data = src.read(1)

        # 更彻底地处理异常值
        data = data.astype(np.float32)

        # 替换无限大值为NaN
        data[~np.isfinite(data)] = np.nan

        # 替换超出float32范围的值
        max_val = np.finfo(np.float32).max
        min_val = np.finfo(np.float32).min
        data[data > max_val] = max_val
        data[data < min_val] = min_val

        # 创建目标数组
        dst_data = np.empty((height, width), dtype=np.float32)
        dst_data.fill(np.nan)

        # 重投影到参考栅格
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear
        )

        # 应用掩膜
        dst_data[mask_data == 0] = np.nan

    return dst_data


# 4. 改进的加载所有预测因子函数
def load_all_predictors(predictor_dir):
    predictors = {}
    for var, filename in zip(variables, filenames):
        filepath = os.path.join(predictor_dir, filename)
        if os.path.exists(filepath):
            print(f"正在处理: {filename}")
            data = preprocess_and_reproject(filepath)

            # 检查数据质量
            valid_pixels = np.isfinite(data) & (mask_data > 0)
            if np.sum(valid_pixels) == 0:
                raise ValueError(f"文件 {filename} 中没有有效数据")

            predictors[var] = data
        else:
            raise FileNotFoundError(f"无法找到文件: {filepath}")
    return predictors


# 5. 改进的部分相关系数计算
def calculate_partial_correlations(model, X):
    """计算随机森林模型的部分相关系数"""
    n_samples, n_features = X.shape
    partial_corrs = np.zeros((n_samples, n_features))

    # 首先检查输入数据
    if not np.all(np.isfinite(X)):
        raise ValueError("输入数据包含非有限值(inf或NaN)")

    # 计算原始预测
    y_pred = model.predict(X)

    for j in range(n_features):
        # 创建扰动后的数据
        X_perturbed = X.copy()
        np.random.shuffle(X_perturbed[:, j])

        # 计算扰动后预测
        y_perturbed = model.predict(X_perturbed)

        # 部分相关系数可以近似为预测差异
        partial_corrs[:, j] = np.abs(y_pred - y_perturbed)

    return partial_corrs


# 6. 改进的主函数
def main(predictor_dir, model_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载预测因子
        print("正在加载和预处理预测因子...")
        predictors = load_all_predictors(predictor_dir)

        # 准备数据数组
        valid_pixels = np.where((mask_data > 0))
        n_pixels = len(valid_pixels[0])
        n_vars = len(variables)

        # 初始化数据矩阵
        X = np.zeros((n_pixels, n_vars), dtype=np.float32)

        # 填充数据矩阵
        for i, var in enumerate(variables):
            X[:, i] = predictors[var][valid_pixels]

        # 检查最终数据矩阵
        if not np.all(np.isfinite(X)):
            # 处理剩余的NaN值
            print("警告: 数据中仍然存在NaN值，将用列均值替换...")
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X[nan_indices] = np.take(col_means, nan_indices[1])

        # 加载随机森林模型
        print("正在加载随机森林模型...")
        model = joblib.load(model_path)

        # 计算预测结果
        print("正在计算预测结果...")
        y_pred = model.predict(X)

        # 创建预测结果栅格
        prediction_raster = np.full_like(mask_data, fill_value=np.nan, dtype=np.float32)
        prediction_raster[valid_pixels] = y_pred

        # 保存预测结果为tif
        prediction_tif = os.path.join(output_dir, "Maize_RF_pred.tif")
        with rasterio.open(prediction_tif, 'w', **profile) as dst:
            dst.write(prediction_raster, 1)

        # 绘制预测结果图
        plt.figure(figsize=(15, 10))
        plt.imshow(prediction_raster, cmap='viridis')
        plt.colorbar(label="预测值")
        plt.title("预测结果分布图")
        prediction_png = os.path.join(output_dir, "Maize_RF_pred.png")
        plt.savefig(prediction_png, bbox_inches='tight', dpi=300)
        plt.close()

        # 计算部分相关系数
        print("正在计算部分相关系数...")
        partial_corrs = calculate_partial_correlations(model, X)

        # 确定主导驱动因子
        print("确定主导驱动因子...")
        dominant_indices = np.argmax(partial_corrs, axis=1)

        # 创建结果栅格
        dominant_raster = np.full_like(mask_data, fill_value=np.nan, dtype=np.float32)
        dominant_raster[valid_pixels] = dominant_indices

        # 保存结果为tif
        dominant_tif = os.path.join(output_dir, "Maize_RF_domin.tif")
        with rasterio.open(dominant_tif, 'w', **profile) as dst:
            dst.write(dominant_raster, 1)

        # 创建图例和绘图
        print("正在生成可视化结果...")
        plt.figure(figsize=(15, 10))

        # 创建自定义颜色映射
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(len(variables) + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(dominant_raster, cmap=cmap, norm=norm)
        plt.colorbar(ticks=np.arange(len(variables)) + 0.5,
                     label="主导驱动因子",
                     boundaries=bounds,
                     values=np.arange(len(variables)) + 0.5)

        # 创建图例
        legend_elements = []
        color_groups = {}
        for var in variables:
            color = variable_info[var]["color"]
            if color not in color_groups:
                color_groups[color] = [var]
            else:
                color_groups[color].append(var)

        for color, vars in color_groups.items():
            label = ", ".join(vars)
            legend_elements.append(Patch(facecolor=color, label=label))

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("全球尺度氮预测主导驱动因子分布")

        dominant_png = os.path.join(output_dir, "Maize_RF_domin.png")
        plt.savefig(dominant_png, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"处理完成! 结果已保存到: {output_dir}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    # predictor_dir = r"D:\Nloss\CODE_PY\domin\wheat_n2o_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_nh3_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_no_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_lf_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_n2o_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_lf_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_rf_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_n2o_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_nh3_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_no_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_lf_predvars_test0608"  # 替换为预测因子tif文件所在目录
    predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_rf_predvars_test0608"  # 替换为预测因子tif文件所在目录
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\model.pkl"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\RF_models0509_GLO_前20个变量特征\Wheat_LF_RF.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_N2OEF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Maize_NH3EF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Maize_NOEF_model.joblib"  # 替换为模型文件路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib"  # 替换为模型文件路径
    model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_models0509_GLO_前20个变量特征\Maize_RF_XGBoost.joblib"  # 替换为模型文件路径
    output_dir = "Maize_RF_domin0608"  # 替换为输出目录

    main(predictor_dir, model_path, output_dir)