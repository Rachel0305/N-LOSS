import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. 定义变量和颜色映射（保持不变）
variable_info = {
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

# 2. 加载参考栅格（保持不变）
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


# 3. 改进的预处理和重投影函数（保持不变）
def preprocess_and_reproject(filepath):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        data = data.astype(np.float32)
        data[~np.isfinite(data)] = np.nan
        max_val = np.finfo(np.float32).max
        min_val = np.finfo(np.float32).min
        data[data > max_val] = max_val
        data[data < min_val] = min_val
        dst_data = np.empty((height, width), dtype=np.float32)
        dst_data.fill(np.nan)
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear
        )
        dst_data[mask_data == 0] = np.nan
    return dst_data


# 4. 改进的加载所有预测因子函数（保持不变）
def load_all_predictors(predictor_dir):
    predictors = {}
    for var, filename in zip(variables, filenames):
        filepath = os.path.join(predictor_dir, filename)
        if os.path.exists(filepath):
            print(f"正在处理: {filename}")
            data = preprocess_and_reproject(filepath)
            valid_pixels = np.isfinite(data) & (mask_data > 0)
            if np.sum(valid_pixels) == 0:
                raise ValueError(f"文件 {filename} 中没有有效数据")
            predictors[var] = data
        else:
            raise FileNotFoundError(f"无法找到文件: {filepath}")
    return predictors


# 5. 修改后的部分相关系数计算函数
def calculate_partial_correlations(model, X):
    """计算部分相关系数绝对值，并返回每个像元的系数矩阵"""
    n_samples, n_features = X.shape
    partial_corrs = np.zeros((n_samples, n_features), dtype=np.float32)

    # 获取原始预测值
    y_pred = model.predict(X)

    for j in range(n_features):
        # 对每个特征进行控制变量的部分相关系数计算
        X_controlled = X.copy()
        other_features = [i for i in range(n_features) if i != j]

        # 拟合控制模型（排除当前特征）
        control_model = RandomForestRegressor(n_estimators=100, random_state=42)
        control_model.fit(X_controlled[:, other_features], y_pred)
        y_control = control_model.predict(X_controlled[:, other_features])

        # 计算当前特征的残差
        residual_x = X[:, j] - np.mean(X[:, j])
        residual_y = y_pred - y_control

        # 计算部分相关系数
        valid_mask = np.isfinite(residual_x) & np.isfinite(residual_y)
        if np.sum(valid_mask) > 1:
            corr, _ = pearsonr(residual_x[valid_mask], residual_y[valid_mask])
            partial_corrs[:, j] = np.abs(corr)
        else:
            partial_corrs[:, j] = 0.0

    return partial_corrs


# 6. 主函数（稍作调整以保存偏相关系数）
def main(predictor_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("正在加载和预处理预测因子...")
        predictors = load_all_predictors(predictor_dir)
        valid_pixels = np.where((mask_data > 0))
        n_pixels = len(valid_pixels[0])
        n_vars = len(variables)

        X = np.zeros((n_pixels, n_vars), dtype=np.float32)
        for i, var in enumerate(variables):
            X[:, i] = predictors[var][valid_pixels]

        if not np.all(np.isfinite(X)):
            print("警告: 数据中仍然存在NaN值，将用列均值替换...")
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X[nan_indices] = np.take(col_means, nan_indices[1])

        print("正在加载随机森林模型...")
        model = joblib.load(model_path)

        print("正在计算预测结果...")
        y_pred = model.predict(X)

        prediction_raster = np.full_like(mask_data, fill_value=np.nan, dtype=np.float32)
        prediction_raster[valid_pixels] = y_pred

        prediction_tif = os.path.join(output_dir, "Maize_RF_pred.tif")
        with rasterio.open(prediction_tif, 'w', **profile) as dst:
            dst.write(prediction_raster, 1)

        plt.figure(figsize=(15, 10))
        plt.imshow(prediction_raster, cmap='viridis')
        plt.colorbar(label="预测值")
        plt.title("预测结果分布图")
        prediction_png = os.path.join(output_dir, "Maize_RF_pred.png")
        plt.savefig(prediction_png, bbox_inches='tight', dpi=300)
        plt.close()

        print("正在计算部分相关系数...")
        partial_corrs = calculate_partial_correlations(model, X)

        print("确定主导驱动因子...")
        dominant_indices = np.argmax(partial_corrs, axis=1)

        # 保存偏相关系数绝对值的平均值（用于桑基图）
        mean_partial_corrs = np.nanmean(partial_corrs, axis=0)
        corr_df = pd.DataFrame({
            'Variable': variables,
            'Mean_Abs_Partial_Corr': mean_partial_corrs
        })
        corr_df.to_csv(os.path.join(output_dir, "mean_partial_correlations.csv"), index=False)

        dominant_raster = np.full_like(mask_data, fill_value=np.nan, dtype=np.float32)
        dominant_raster[valid_pixels] = dominant_indices

        dominant_tif = os.path.join(output_dir, "Maize_RF_domin.tif")
        with rasterio.open(dominant_tif, 'w', **profile) as dst:
            dst.write(dominant_raster, 1)

        print("正在生成可视化结果...")
        plt.figure(figsize=(15, 10))
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(len(variables) + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(dominant_raster, cmap=cmap, norm=norm)
        plt.colorbar(ticks=np.arange(len(variables)) + 0.5,
                     label="主导驱动因子",
                     boundaries=bounds,
                     values=np.arange(len(variables)) + 0.5)

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
    predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_rf_predvars_test0608"
    model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_models0509_GLO_前20个变量特征\Maize_RF_XGBoost.joblib"
    output_dir = "Maize_RF_domin0608"
    main(predictor_dir, model_path, output_dir)