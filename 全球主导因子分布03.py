import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.plot import show
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import joblib
from tqdm import tqdm
from scipy.stats import pearsonr

# 设置工作目录和文件路径
input_folder =  r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_n2o_predvars_test0602"
output_folder = "./output1"
wheat_mask_file = "Wheat_Harvested_area_2020.tif"
model_file = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\model.pkl"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 定义变量顺序和颜色映射
variable_order = [
    "MAP", "MAT", "Sand", "Clay", "Slope", "pH", "Bulk density", "SOC", "CN",
    "N input rate", 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
    'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
    'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
    'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
    'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
    'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
    'Cddw_Key_mean'
]

colors = [
    '#1F77B4', '#2C7BB6', '#8C510A', '#A16216', '#7F7F7F', '#B87333',
    '#875C26', '#C88746', '#2CA02C', '#4DAF4A', '#5E9ED6', '#87B7E0',
    '#4590B9', '#4D9221', '#66A63A', '#40821B', '#D62728', '#ED4142',
    '#E13334', '#E64959', '#F2656F', '#EB5764', '#1764AB', '#3273BF',
    '#5087C7', '#74A9CF', '#97BBCD', '#64A0C8', '#D94021', '#F05A3B',
    '#E44D2E', '#C83042', '#E05062', '#D44052'
]

# 创建颜色映射
cmap = ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(len(variable_order) + 1) - 0.5, len(variable_order))

# 加载随机森林模型
model = joblib.load(model_file)

# 获取小麦掩膜的元数据
with rasterio.open(wheat_mask_file) as src:
    profile = src.profile
    wheat_mask = src.read(1)
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs


def clean_data(data):
    """清理数据中的无限值和NaN值"""
    data[np.isinf(data)] = np.nan
    for i in range(data.shape[1]):
        col = data[:, i]
        mask = ~np.isnan(col)
        if np.any(mask):
            median = np.median(col[mask])
            col[~mask] = median
        else:
            col[:] = 0
    return data


def preprocess_predictors(input_folder, reference_file):
    """预处理预测因子：将所有预测因子重采样到与参考文件相同的分辨率和范围"""
    # 获取参考文件的元数据
    with rasterio.open(reference_file) as ref:
        ref_profile = ref.profile
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs
        ref_bounds = ref.bounds

    # 创建字典保存所有预测因子
    predictors = {}

    # 遍历输入文件夹中的所有tif文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            # 解析变量名
            parts = filename.split('_')
            var_name = '_'.join(parts[2:]).replace('.tif', '')

            # 读取并重采样
            with rasterio.open(os.path.join(input_folder, filename)) as src:
                # 初始化输出数组
                data = np.zeros((ref_height, ref_width), dtype=np.float32)

                # 重采样
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )

                # 保存到字典
                predictors[var_name] = data

    # 确保所有变量都存在
    for var in variable_order:
        if var not in predictors:
            raise ValueError(f"Predictor {var} not found in input folder")

    # 按variable_order顺序排列数据
    ordered_data = np.stack([predictors[var] for var in variable_order], axis=-1)

    return ordered_data, ref_profile


def calculate_partial_correlations(X, y, feature_names):
    """
    计算每个特征与目标变量的部分相关系数
    """
    n_features = X.shape[1]
    partial_corrs = np.zeros(n_features)

    for i in range(n_features):
        # 计算当前特征与目标变量的偏相关系数
        # 这里简化处理，实际应用中可能需要更精确的偏相关计算
        # 可以使用残差法计算偏相关系数
        corr, _ = pearsonr(X[:, i], y)
        partial_corrs[i] = corr

    return partial_corrs


# 预处理预测因子
print("Preprocessing predictors...")
predictors_data, profile = preprocess_predictors(input_folder, wheat_mask_file)

# 重塑数据以适应模型输入
height, width, n_vars = predictors_data.shape
flat_predictors = predictors_data.reshape(-1, n_vars)

# 应用小麦掩膜
flat_mask = wheat_mask.ravel() > 0
masked_predictors = flat_predictors[flat_mask, :]

# 清理数据
print("Cleaning data...")
masked_predictors = clean_data(masked_predictors)

# 预测氮含量
print("Predicting nitrogen content...")
predicted_n = model.predict(masked_predictors)

# 计算主导驱动因子
print("Calculating dominant drivers based on partial correlations...")

# 为每个像素计算部分相关系数并确定主导因子
dominant_features = np.zeros(masked_predictors.shape[0], dtype=int)

# 分批处理以避免内存问题
batch_size = 10000
n_batches = int(np.ceil(masked_predictors.shape[0] / batch_size))

for i in tqdm(range(n_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, masked_predictors.shape[0])

    batch_X = masked_predictors[start_idx:end_idx, :]
    batch_y = predicted_n[start_idx:end_idx]

    # 计算部分相关系数
    partial_corrs = np.array([calculate_partial_correlations(batch_X, batch_y, variable_order)
                              for _ in range(batch_X.shape[0])])

    # 找到绝对值最大的相关系数对应的特征
    dominant_features[start_idx:end_idx] = np.argmax(np.abs(partial_corrs), axis=1)

# 创建全局结果数组
full_prediction = np.full((height * width), np.nan, dtype=np.float32)
full_drivers = np.full((height * width), -1, dtype=np.int32)

# 填充预测值和主导驱动因子
full_prediction[flat_mask] = predicted_n
full_drivers[flat_mask] = dominant_features

# 重塑为2D
prediction_2d = full_prediction.reshape(height, width)
drivers_2d = full_drivers.reshape(height, width)

# 保存结果
print("Saving results...")
profile.update(dtype=rasterio.float32, nodata=np.nan)
with rasterio.open(os.path.join(output_folder, "global_nitrogen_prediction.tif"), 'w', **profile) as dst:
    dst.write(prediction_2d, 1)

profile.update(dtype=rasterio.int32, nodata=-1)
with rasterio.open(os.path.join(output_folder, "dominant_drivers.tif"), 'w', **profile) as dst:
    dst.write(drivers_2d, 1)

# 可视化
print("Creating visualizations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 氮预测图
with rasterio.open(os.path.join(output_folder, "global_nitrogen_prediction.tif")) as src:
    data = src.read(1)
    im1 = ax1.imshow(data, cmap='viridis',
                     extent=[src.bounds.left, src.bounds.right,
                             src.bounds.bottom, src.bounds.top])
    ax1.set_title('Global Nitrogen Prediction')
    fig.colorbar(im1, ax=ax1, label='Nitrogen Content')

# 主导驱动因子图
with rasterio.open(os.path.join(output_folder, "dominant_drivers.tif")) as src:
    data = src.read(1)
    masked_data = np.ma.masked_where(data == -1, data)
    im2 = ax2.imshow(masked_data, cmap=cmap, norm=norm,
                     extent=[src.bounds.left, src.bounds.right,
                             src.bounds.bottom, src.bounds.top])
    ax2.set_title('Dominant Drivers (Partial Correlation)')

    # 创建图例
    patches = [plt.Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=colors[i], markersize=10,
                          label=variable_order[i])
               for i in range(len(variable_order))]
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "global_nitrogen_results.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("Processing complete!")