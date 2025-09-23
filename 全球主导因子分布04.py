# 全球主导因子分布only
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
variable_info = {
    "MAP": {"order": 1, "color": "#1F77B4", "filename": "Column_1_MAP.tif"},
    "MAT": {"order": 2, "color": "#2C7BB6", "filename": "Column_2_MAT.tif"},
    "Sand": {"order": 3, "color": "#8C510A", "filename": "Column_3_Sand.tif"},
    "Clay": {"order": 4, "color": "#A16216", "filename": "Column_4_Clay.tif"},
    "Slope": {"order": 5, "color": "#7F7F7F", "filename": "Column_5_Slope.tif"},
    "pH": {"order": 6, "color": "#B87333", "filename": "Column_6_pH.tif"},
    "Bulk density": {"order": 7, "color": "#875C26", "filename": "Column_7_Bulk density.tif"},
    "SOC": {"order": 8, "color": "#C88746", "filename": "Column_8_SOC.tif"},
    "C/N": {"order": 9, "color": "#2CA02C", "filename": "Column_9_CN.tif"},
    "N input rate": {"order": 10, "color": "#4DAF4A", "filename": "Column_10_Wheat N input rate.tif"},
    "Till": {"order": 11, "color": "#FF7F0E", "filename": "Column_11_wheat_Till.tif"},
    "Irrigated": {"order": 12, "color": "#FF9933", "filename": "Column_12_Irrigated.tif"},
    "sro_Span_max": {"order": 13, "color": "#5E9ED6", "filename": "Column_13_sro_Span_max.tif"},
    "sro_Span_min": {"order": 14, "color": "#87B7E0", "filename": "Column_14_sro_Span_min.tif"},
    "sro_Span_mean": {"order": 15, "color": "#4590B9", "filename": "Column_15_sro_Span_mean.tif"},
    "LAIw_Span_max": {"order": 16, "color": "#4D9221", "filename": "Column_16_LAIw_Span_max.tif"},
    "LAIw_Span_min": {"order": 17, "color": "#66A63A", "filename": "Column_17_LAIw_Span_min.tif"},
    "LAIw_Span_mean": {"order": 18, "color": "#40821B", "filename": "Column_18_LAIw_Span_mean.tif"},
    "rx1w_Span_max": {"order": 19, "color": "#D62728", "filename": "Column_19_rx1w_Span_max.tif"},
    "rx1w_Span_min": {"order": 20, "color": "#ED4142", "filename": "Column_20_rx1w_Span_min.tif"},
    "rx1w_Span_mean": {"order": 21, "color": "#E13334", "filename": "Column_21_rx1w_Span_mean.tif"},
    "rx5w_Span_max": {"order": 22, "color": "#E64959", "filename": "Column_22_rx5w_Span_max.tif"},
    "rx5w_Span_min": {"order": 23, "color": "#F2656F", "filename": "Column_23_rx5w_Span_min.tif"},
    "rx5w_Span_mean": {"order": 24, "color": "#EB5764", "filename": "Column_24_rx5w_Span_mean.tif"},
    "Ddpw_SSpan_max": {"order": 25, "color": "#1764AB", "filename": "Column_25_Ddpw_Span_max.tif"},
    "Ddpw_Span_min": {"order": 26, "color": "#3273BF", "filename": "Column_26_Ddpw_Span_min.tif"},
    "Ddpw_Span_mean": {"order": 27, "color": "#5087C7", "filename": "Column_27_Ddpw_Span_mean.tif"},
    "Speiw_Span_max": {"order": 28, "color": "#74A9CF", "filename": "Column_28_Speiw_Span_max.tif"},
    "Speiw_Span_min": {"order": 29, "color": "#97BBCD", "filename": "Column_29_Speiw_Span_min.tif"},
    "Speiw_Span_mean": {"order": 30, "color": "#64A0C8", "filename": "Column_30_Speiw_Span_mean.tif"},
    "Hddw_Span_max": {"order": 31, "color": "#D94021", "filename": "Column_37_Hddm_Span_max.tif"},
    "Hddw_Span_min": {"order": 32, "color": "#F05A3B", "filename": "Column_38_Hddm_Span_min.tif"},
    "Hddw_Span_mean": {"order": 33, "color": "#E44D2E", "filename": "Column_39_Hddm_Span_mean.tif"},
    "Cddw_Span_max": {"order": 34, "color": "#C83042", "filename": "Column_40_Cddm_Span_max.tif"},
    "Cddw_Span_min": {"order": 35, "color": "#E05062", "filename": "Column_41_Cddm_Span_min.tif"},
    "Cddw_Span_mean": {"order": 36, "color": "#D44052", "filename": "Column_42_Cddm_Span_mean.tif"},
    "Hddm_Span_max": {"order": 37, "color": "#CC3319", "filename": "Column_37_Hddm_Span_max.tif"},
    "Hddm_Span_min": {"order": 38, "color": "#ED664A", "filename": "Column_38_Hddm_Span_min.tif"},
    "Hddm_Span_mean": {"order": 39, "color": "#DC4D31", "filename": "Column_39_Hddm_Span_mean.tif"},
    "Cddm_Span_max": {"order": 40, "color": "#BA2B3D", "filename": "Column_40_Cddm_Span_max.tif"},
    "Cddm_Span_min": {"order": 41, "color": "#DB5E6E", "filename": "Column_41_Cddm_Span_min.tif"},
    "Cddm_Span_mean": {"order": 42, "color": "#CE4555", "filename": "Column_42_Cddm_Span_mean.tif"}
}

# 按顺序排列变量
variables = sorted(variable_info.keys(), key=lambda x: variable_info[x]["order"])
colors = [variable_info[var]["color"] for var in variables]
filenames = [variable_info[var]["filename"] for var in variables]

# 2. 加载参考栅格
reference_raster = "Wheat_Harvested_area_2020.tif"


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
        output_tif = os.path.join(output_dir, "dominant_drivers.tif")
        with rasterio.open(output_tif, 'w', **profile) as dst:
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

        output_png = os.path.join(output_dir, "dominant_drivers.png")
        plt.savefig(output_png, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"处理完成! 结果已保存到: {output_dir}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    # predictor_dir = r"D:\Nloss\CODE_PY\domin\wheat_n2o_predvars_test0602"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_nh3_predvars_test0602"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_no_predvars_test0602"  # 替换为预测因子tif文件所在目录
    # predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\maize_lf_predvars_test0602"  # 替换为预测因子tif文件所在目录
    predictor_dir = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_nh3_predvars_test0608"  # 替换为预测因子tif文件所在目录
    model_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0727_GLO_所有变量特征\Wheat\N2OEF\model.joblib'  # 替换为模型文件路径
    output_dir = "/Wheat_N2OEF_domin0728"  # 替换为输出目录

    main(predictor_dir, model_path, output_dir)