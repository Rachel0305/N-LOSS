import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sympy import false

# ---------------- 全局样式 ----------------
plt.rcParams.update({'font.size': 16})

# 颜色映射表
COLOR_MAP = {
    'N2OEF': {'new': '#FB000D', 'old': '#14D100'},
    'NH3EF': {'new': '#FF7600', 'old': '#01939A'},
    'NOEF': {'new': '#FFD700', 'old': '#3E13AF'},
    'LF': {'new': '#00AF64', 'old': '#FF4900'},
    'RF': {'new': '#1240AB', 'old': '#FFAA00'},
    'NUE': {'new': '#CE0071', 'old': '#9BED00'}
}

# 有效肥料类型
VALID_FERTILIZER_TYPES = ['Mineral', 'Min & Org', 'Organic', 'EEFs']

# 这三个列名是“原始分类列”的标准名
RAW_CATEGORICAL_COLS = ['Fertilizer type', 'Tillage', 'Irrigation']


def load_model_and_scaler(model_dir, target):
    """加载模型、归一化器及其训练用特征名"""
    try:
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # 训练使用的特征名
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names_path = os.path.join(model_dir, f'{target}', 'feature_names.csv')
            if os.path.exists(feature_names_path):
                feature_names = pd.read_csv(feature_names_path)['feature_names'].tolist()
            else:
                raise ValueError("无法获取模型训练时使用的特征名称")

        if not hasattr(model, 'predict'):
            raise ValueError("加载的对象不是有效的模型（缺少predict方法）")

        return model, scaler, feature_names
    except Exception as e:
        raise ValueError(f"加载模型失败: {str(e)}")


def filter_fertilizer_types(data: pd.DataFrame) -> pd.DataFrame:
    """筛选有效的肥料类型"""
    if 'Fertilizer type' in data.columns:
        original_count = len(data)
        data = data[data['Fertilizer type'].isin(VALID_FERTILIZER_TYPES)].copy()
        filtered_count = len(data)
        print(f"肥料类型筛选: 从 {original_count} 条记录筛选到 {filtered_count} 条记录")
    return data


def remove_3sigma_outliers(data: pd.DataFrame, target: str, min_samples: int = 30) -> pd.DataFrame:
    """
    只针对当前对比的因变量（target）去除3σ异常值；
    若删后不足 min_samples，则直接返回原数据
    """
    clean_data = data.copy()

    if target not in clean_data.columns:
        print(f"⚠️ {target} 不在数据列中，跳过异常值剔除")
        return clean_data

    series = clean_data[target]
    if not pd.api.types.is_numeric_dtype(series):
        print(f"⚠️ {target} 不是数值列，跳过异常值剔除")
        return clean_data

    mean_val = series.mean()
    std_val = series.std()
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val
    outliers = clean_data[(series < lower_bound) | (series > upper_bound)].index

    if len(outliers) > 0:
        print(f"  {target}: 标记 {len(outliers)} 个异常值")

    clean_data = clean_data.drop(index=outliers)

    if len(clean_data) < min_samples:
        print(f"⚠️ {target} 异常值去除后样本不足 {min_samples} 条，放弃去异常值，返回原数据")
        return data.copy()
    else:
        print(f"{target}: 去除异常值后样本量 {len(clean_data)} (剔除了 {len(outliers)} 条)")
        return clean_data


def preprocess_features(X: pd.DataFrame,
                        model_expected_features: list,
                        scaler: StandardScaler | None = None,
                        fit_scaler: bool = False):
    """
    将原始 X 预处理成“模型训练时的列集合与顺序（model_expected_features）”：
    1) 只要 X 里存在原始分类列（'Fertilizer type','Tillage','Irrigation'），就进行 one-hot；
    2) 合并数值 + one-hot 后的列；
    3) 按模型期望列补齐不存在的列（填 0），并重排列顺序；
    4) 可选：用给定 scaler 做 transform。
    """
    X_work = X.copy()

    # 1) 原始分类列：以 X 中是否存在为准（而非以 features 列表判断）
    cats_in_X = [c for c in RAW_CATEGORICAL_COLS if c in X_work.columns]

    # 2) one-hot（对存在于 X 的分类列做编码）
    if cats_in_X:
        X_encoded = pd.get_dummies(X_work[cats_in_X], columns=cats_in_X, drop_first=True)
        # 数值列 = 原始 X 去掉分类列后的剩余
        numeric_cols = [c for c in X_work.columns if c not in cats_in_X]
        X_numeric = X_work[numeric_cols].copy()
        X_processed = pd.concat([X_numeric, X_encoded], axis=1)
    else:
        # 没有原始分类列时，直接使用原始 X
        X_processed = X_work

    # 3) 补齐 & 重排为模型期望列
    for col in model_expected_features:
        if col not in X_processed.columns:
            X_processed[col] = 0
    X_processed = X_processed[model_expected_features]

    # 4) 归一化（若提供了 scaler）
    if scaler is not None:
        if fit_scaler:
            X_scaled = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns, index=X_processed.index)
        else:
            X_scaled = pd.DataFrame(scaler.transform(X_processed), columns=X_processed.columns, index=X_processed.index)
        return X_scaled, scaler
    else:
        return X_processed, None


def evaluate_models(data_path,
                    target_column,
                    crop_type,
                    model1_dir,
                    model2_dir,
                    test_size=0.3,
                    n_repeats=100,
                    seed_range=range(1, 101),
                    output_dir='results',
                    # 兼容可选“外部指定的模型期望特征名”（通常不需要提供）
                    model1_expected_features=None,
                    model2_expected_features=None,
                    **kwargs):
    """
    对新旧两套已训练模型做重复抽样验证并可视化。
    注意：默认使用模型随存的 feature_names_in_ / feature_names.csv，
    若你确实要强行指定模型期望特征列，可通过 model1_expected_features / model2_expected_features 传入。
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 读取数据 + 作物过滤 + 肥料过滤
    try:
        data = pd.read_csv(data_path, low_memory=False)
        if 'Crop type' in data.columns:
            data = data[data['Crop type'] == crop_type].copy()
        data = filter_fertilizer_types(data)
    except Exception as e:
        raise ValueError(f"加载数据文件失败: {str(e)}")

    # 加载两个模型及其 scaler、训练用特征名
    model1, scaler1, model1_features_from_model = load_model_and_scaler(model1_dir, target_column)
    model2, scaler2, model2_features_from_model = load_model_and_scaler(model2_dir, target_column)

    # 决定最终“模型期望特征名”
    feat1 = model1_expected_features if model1_expected_features is not None else model1_features_from_model
    feat2 = model2_expected_features if model2_expected_features is not None else model2_features_from_model

    # 需要存在于数据中的基本列（目标列 + 预处理会用到的原始分类列）
    required_base_cols = [target_column] + [c for c in RAW_CATEGORICAL_COLS if c in data.columns]

    # 去缺失
    base_exist_cols = [c for c in required_base_cols if c in data.columns]
    complete_rows = data[base_exist_cols].notna().all(axis=1)
    valid_data = data[complete_rows].copy()
    print(f"去除缺失值后样本量: {len(valid_data)}")

    # 只针对当前因变量去异常值
    clean_data = remove_3sigma_outliers(valid_data, target_column)

    # 如果样本太少，跳过
    if len(clean_data) < 30:
        raise ValueError(f"{crop_type}-{target_column}: 有效样本不足 30，无法评估")

    # 训练-验证分割 & 评估
    all_metrics = []
    all_validation_data = []

    for i, seed in enumerate(seed_range[:n_repeats]):
        if len(clean_data) < 30:  # double check
            continue
        X = clean_data.drop(columns=[target_column])
        y = clean_data[target_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

        # 用“模型期望列”来预处理验证集
        X_val_processed1, _ = preprocess_features(X_val, feat1, scaler1, fit_scaler=False)
        X_val_processed2, _ = preprocess_features(X_val, feat2, scaler2, fit_scaler=False)

        # 预测
        model1_val_pred = model1.predict(X_val_processed1)
        model2_val_pred = model2.predict(X_val_processed2)

        def calculate_metrics(y_true, y_pred, model_name, seed):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
            return {
                'Model': model_name, 'Seed': seed, 'R²': r2, 'RMSE': rmse,
                'Slope': slope, 'Intercept': intercept, 'Sample Size': len(y_pred)
            }

        all_metrics.append(calculate_metrics(y_val, model1_val_pred, "New Model", seed))
        all_metrics.append(calculate_metrics(y_val, model2_val_pred, "Old Model", seed))

        all_validation_data.append(pd.DataFrame({
            'Seed': seed,
            'True_Value': y_val.to_numpy(),
            'New_Model_Prediction': model1_val_pred,
            'Old_Model_Prediction': model2_val_pred
        }))

    # 汇总
    results_df = pd.DataFrame(all_metrics)
    summary_stats = results_df.groupby('Model').agg({
        'R²': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'Slope': ['mean', 'std'],
        'Intercept': ['mean', 'std'],
        'Sample Size': 'mean'
    }).round(4)
    summary_stats.columns = ['_'.join(col) for col in summary_stats.columns]
    summary_stats.reset_index(inplace=True)

    all_validation_data_df = pd.concat(all_validation_data, ignore_index=True)

    # 在 evaluate_models 函数末尾，可视化之前添加 T 检验
    # ---------------- 配对 T 检验 ----------------
    paired_metrics = []

    # 按 Seed 组合，确保配对
    seeds = results_df['Seed'].unique()
    r2_new_list = []
    r2_old_list = []
    rmse_new_list = []
    rmse_old_list = []

    for s in seeds:
        tmp_new = results_df[(results_df['Model'] == 'New Model') & (results_df['Seed'] == s)]['R²'].values[0]
        tmp_old = results_df[(results_df['Model'] == 'Old Model') & (results_df['Seed'] == s)]['R²'].values[0]
        r2_new_list.append(tmp_new)
        r2_old_list.append(tmp_old)

        tmp_rmse_new = results_df[(results_df['Model'] == 'New Model') & (results_df['Seed'] == s)]['RMSE'].values[0]
        tmp_rmse_old = results_df[(results_df['Model'] == 'Old Model') & (results_df['Seed'] == s)]['RMSE'].values[0]
        rmse_new_list.append(tmp_rmse_new)
        rmse_old_list.append(tmp_rmse_old)

    # R² 配对 T 检验
    r2_t_stat, r2_p_val = stats.ttest_rel(r2_new_list, r2_old_list)
    r2_diff_mean = np.mean(np.array(r2_new_list) - np.array(r2_old_list))

    # RMSE 配对 T 检验
    rmse_t_stat, rmse_p_val = stats.ttest_rel(rmse_old_list, rmse_new_list)  # 注意 RMSE 越小越好，比较 old-new
    rmse_diff_mean = np.mean(np.array(rmse_old_list) - np.array(rmse_new_list))

    # 整理结果表格
    ttest_results_df = pd.DataFrame({
        'Metric': ['R²', 'RMSE'],
        'Mean_Difference': [r2_diff_mean, rmse_diff_mean],
        'T_statistic': [r2_t_stat, rmse_t_stat],
        'P_value': [r2_p_val, rmse_p_val],
        'Significant_0.05': [r2_p_val < 0.05, rmse_p_val < 0.05],
        'Significant_0.01': [r2_p_val < 0.01, rmse_p_val < 0.01]
    })

    # 保存
    ttest_path = os.path.join(output_dir, f'{crop_type}_{target_column}_paired_ttest_{timestamp}.csv')
    ttest_results_df.to_csv(ttest_path, index=False)
    print(f"配对 T 检验结果已保存: {ttest_path}")

    # ---------- 可视化 ----------
    new_color = COLOR_MAP[target_column]['new']
    old_color = COLOR_MAP[target_column]['old']

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    # ===========================
    # 左：R² & RMSE 柱状图
    # ===========================
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(2)  # 两个位置：R² 和 RMSE
    width = 0.35  # 柱宽

    # 拆分新旧模型数据
    new = summary_stats[summary_stats['Model'] == 'New Model']
    old = summary_stats[summary_stats['Model'] == 'Old Model']

    # 取出均值与标准差，便于后面使用（减少重复索引）
    new_r2_mean = new['R²_mean'].values[0]
    new_r2_std = new['R²_std'].values[0]
    old_r2_mean = old['R²_mean'].values[0]
    old_r2_std = old['R²_std'].values[0]

    new_rmse_mean = new['RMSE_mean'].values[0]
    new_rmse_std = new['RMSE_std'].values[0]
    old_rmse_mean = old['RMSE_mean'].values[0]
    old_rmse_std = old['RMSE_std'].values[0]

    # ---- R² 柱状 ----
    new_r2_bar = ax1.bar(x[0] - width / 2, new_r2_mean, width, yerr=new_r2_std,
                         color=new_color, capsize=8, label='NEW', alpha=0.8,
                         edgecolor='black', linewidth=1.5)
    old_r2_bar = ax1.bar(x[0] + width / 2, old_r2_mean, width, yerr=old_r2_std,
                         color=old_color, capsize=8, label='OLD', alpha=0.8,
                         edgecolor='black', linewidth=1.5)

    # ---- R² 均值 ± 标准差 标签 ----
    ax1.text(x[0] - width / 2,
             new_r2_mean + new_r2_std + 0.01,
             f"{(new_r2_mean + new_r2_std):.3f}",
             ha='center', va='bottom', fontsize=14, color='black')
    ax1.text(x[0] - width / 2,
             new_r2_mean - new_r2_std - 0.01,
             f"{(new_r2_mean - new_r2_std):.3f}",
             ha='center', va='top', fontsize=14, color='white',)

    ax1.text(x[0] + width / 2,
             old_r2_mean + old_r2_std + 0.01,
             f"{(old_r2_mean + old_r2_std):.3f}",
             ha='center', va='bottom', fontsize=14, color='black')
    ax1.text(x[0] + width / 2,
             old_r2_mean - old_r2_std - 0.01,
             f"{(old_r2_mean - old_r2_std):.3f}",
             ha='center', va='top', fontsize=14, color='white',)

    ax1.set_ylabel('R²', fontsize=18)
    # ax1.set_xlabel('RMSE', fontsize=18)  # 保持你原本的写法（如需显示可以取消注释）

    # ---- RMSE 柱状（右 y 轴） ----
    ax1b = ax1.twinx()
    new_rmse_bar = ax1b.bar(x[1] - width / 2, new_rmse_mean, width, yerr=new_rmse_std,
                            color=new_color, capsize=8, alpha=0.8,
                            edgecolor='black', linewidth=1.5)
    old_rmse_bar = ax1b.bar(x[1] + width / 2, old_rmse_mean, width, yerr=old_rmse_std,
                            color=old_color, capsize=8, alpha=0.8,
                            edgecolor='black', linewidth=1.5)

    # ---- RMSE 均值 ± 标准差 标签 ----
    # 上端：均值 + std（黑字，无背景）
    ax1b.text(x[1] - width / 2,
              new_rmse_mean + new_rmse_std + 0.01,
              f"{(new_rmse_mean + new_rmse_std):.3f}",
              ha='center', va='bottom', fontsize=14, color='black')

    # 下端：均值 - std（白字，**去掉黑色方框**，保持无 bbox）
    ax1b.text(x[1] - width / 2,
              new_rmse_mean - new_rmse_std - 0.01,
              f"{(new_rmse_mean - new_rmse_std):.3f}",
              ha='center', va='top', fontsize=14, color='white')

    ax1b.text(x[1] + width / 2,
              old_rmse_mean + old_rmse_std + 0.01,
              f"{(old_rmse_mean + old_rmse_std):.3f}",
              ha='center', va='bottom', fontsize=14, color='black',)

    ax1b.text(x[1] + width / 2,
              old_rmse_mean - old_rmse_std - 0.01,
              f"{(old_rmse_mean - old_rmse_std):.3f}",
              ha='center', va='top', fontsize=14, color='white')

    ax1b.set_ylabel('RMSE', fontsize=18)

    # ---- 坐标轴与图例 ----
    ax1.set_xticks(x)
    ax1.set_xticklabels(['R²', 'RMSE'], fontsize=16)
    ax1.set_title(f'{crop_type} {target_column}: R² and RMSE', fontsize=20, pad=20)
    # ax1.legend([new_r2_bar, old_r2_bar], ['NEW', 'OLD'], loc='upper right',
    #            fontsize=16, frameon=True, fancybox=True, shadow=True)

    # ax1.legend([new_r2_bar, old_r2_bar], ['NEW', 'OLD'],
    #            loc='upper right', fontsize=16,
    #            frameon=True, fancybox=True, shadow=True)

    # 右：散点 + 100 根回归线
    ax2 = fig.add_subplot(gs[1])
    x_all = all_validation_data_df['True_Value']
    y_all_new = all_validation_data_df['New_Model_Prediction']
    y_all_old = all_validation_data_df['Old_Model_Prediction']

    min_val = min(x_all.min(), y_all_new.min(), y_all_old.min())
    max_val = max(x_all.max(), y_all_new.max(), y_all_old.max())
    margin = 0.05 * (max_val - min_val) if np.isfinite(max_val - min_val) else 1.0
    plot_min = min_val - margin
    plot_max = max_val + margin
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_ylim(plot_min, plot_max)

    ax2.scatter(x_all, y_all_new, alpha=0.5, c=new_color, edgecolors='none', s=30, label='NEW')
    ax2.scatter(x_all, y_all_old, alpha=0.5, c=old_color, edgecolors='none', s=30, marker='x', label='OLD')

    def plot_all_regressions_with_mean(x, y, ax, color, model_name):
        slopes = results_df[results_df['Model'] == model_name]['Slope']
        intercepts = results_df[results_df['Model'] == model_name]['Intercept']
        mean_slope = slopes.mean()
        mean_intercept = intercepts.mean()
        slope_std = slopes.std()
        x_vals = np.linspace(plot_min, plot_max, 100)
        for slope, intercept in zip(slopes, intercepts):
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color=color, alpha=0.05, linestyle='--', linewidth=0.5, zorder=1)
        y_mean = mean_intercept + mean_slope * x_vals
        ax.plot(x_vals, y_mean, color=color, linewidth=3, zorder=3)
        return mean_slope, slope_std

    mean_slope_new, slope_std_new = plot_all_regressions_with_mean(x_all, y_all_new, ax2, new_color, 'New Model')
    mean_slope_old, slope_std_old = plot_all_regressions_with_mean(x_all, y_all_old, ax2, old_color, 'Old Model')

    ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.7, zorder=0, linewidth=2, label='1:1 Line')
    ax2.set_xlabel('Actual', fontsize=18)
    ax2.set_ylabel('Predicted', fontsize=18)
    ax2.set_title(f'{crop_type} {target_column}: Predicted vs Actual (100 samples)', fontsize=20,  pad=20)

    # ax2.text(0.98, 0.02, f'N = {len(x_all):,}', transform=ax2.transAxes, fontsize=16,
    #          va='bottom', ha='right')

    ax2.text(0.02, 0.98, f'Slope$_{{NEW}}$ = {mean_slope_new:.3f}±{slope_std_new:.3f}',
             transform=ax2.transAxes, fontsize=16,
             va='top', ha='left')
    ax2.text(0.02, 0.90, f'Slope$_{{OLD}}$ = {mean_slope_old:.3f}±{slope_std_old:.3f}',
             transform=ax2.transAxes, fontsize=16,
             va='top', ha='left')

    ax2.legend(loc='lower right', fontsize=12, frameon=False, fancybox=False, shadow=False)

    # 保存输出
    plot_path = os.path.join(output_dir, f'{crop_type}_{target_column}_comparison_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    metrics_path = os.path.join(output_dir, f'{crop_type}_{target_column}_comparison_metrics_{timestamp}.csv')
    summary_path = os.path.join(output_dir, f'{crop_type}_{target_column}_summary_stats_{timestamp}.csv')
    validation_path = os.path.join(output_dir, f'{crop_type}_{target_column}_validation_predictions_{timestamp}.csv')

    results_df.to_csv(metrics_path, index=False)
    summary_stats.to_csv(summary_path, index=False)
    all_validation_data_df.to_csv(validation_path, index=False)

    return results_df, summary_stats, plot_path, metrics_path, summary_path, validation_path


def get_features_for_crop(crop_type, base_features):
    """根据作物类型返回作物相关原始特征（这里返回的是 '原始列名'，不是 one-hot 后的列名）"""
    if crop_type == 'Wheat':
        crop_specific_features = [
            'Hddw span max', 'Hddw span min', 'Hddw key mean',
            'Cddw span max', 'Cddw span min', 'Cddw key mean'
        ]
    else:  # Maize
        crop_specific_features = [
            'Hddm span max', 'Hddm span min', 'Hddm key mean',
            'Cddm span max', 'Cddm span min', 'Cddm key mean'
        ]

    # 在新版流程里，这个函数仅用于“你自己外部指定想用的原始特征集合”；
    # 真正用于模型输入的列仍以模型保存的 feature_names 为准。
    return list(dict.fromkeys(base_features + crop_specific_features))


def run_all_comparisons():
    """运行所有 12 个组合（6 目标 × 2 作物）"""
    data_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_SMs_yield_SMrz.csv'

    # 基础原始特征（此列表仅作参考或可选传入；实际按模型保存的列为准）
    base_features = [
        'MAP', 'MAT', 'Sand', 'Clay', 'Slope', 'pH', 'Bulk density', 'SOC', 'C/N',
        'N input rate', 'Tillage', 'Irrigation',
        'Sro span max', 'Sro span min', 'Sro key mean',
        'LAI span max', 'LAI span min', 'LAI key mean',
        'Rx1 span max', 'Rx1 span min', 'Rx1 key mean',
        'Rx5 span max', 'Rx5 span min', 'Rx5 key mean',
        'Ddp span max', 'Ddp span min', 'Ddp key mean',
        'Spei span max', 'Spei span min', 'Spei key mean',
        'SMs span max', 'SMs span min', 'SMs key mean',
        'SMrz span max', 'SMrz span min', 'SMrz key mean',
        'Fertilizer type'
    ]

    # 旧方案“外部设定”的原始特征（可选）
    old_features = ["MAP", "MAT", "Aridity index", "Sand", "Clay", "pH",
                    "Bulk density", "SOC", "C/N", "N input rate", "Fertilizer type"]

    targets = ['N2OEF', 'NH3EF', 'NOEF', 'LF', 'RF', 'NUE']
    crops = ['Maize']

    output_base_dir = 'Model_comparison_results0901'

    model1_base_dir = r'D:\CNPK\N_CODE_PY\N_ML_V2\RF_model_NEW_with_fert_till_irrig_sm'
    model2_base_dir = 'RF_model_OLD_fertilizer_norm_bestpar'

    for crop in crops:
        for target in targets:
            print(f"\n开始处理: {crop} - {target}")

            # 可选：你仍然可以准备一个“想用的原始特征”列表（并不会直接作为模型输入列）
            model1_orig_features = get_features_for_crop(crop, base_features)

            model1_path = os.path.join(model1_base_dir, crop, target)
            model2_path = os.path.join(model2_base_dir, crop, target)

            if not os.path.exists(os.path.join(model1_path, 'model.joblib')):
                print(f"警告: 新方案模型文件不存在: {model1_path}")
                continue
            if not os.path.exists(os.path.join(model2_path, 'model.joblib')):
                print(f"警告: 旧方案模型文件不存在: {model2_path}")
                continue

            try:
                # 现在 evaluate_models 支持可选的 expected 特征名，
                # 但通常你无需传，默认会使用模型随存的列名，最稳妥。
                results_df, summary_stats, plot_path, metrics_path, summary_path, validation_path = evaluate_models(
                    data_path=data_path,
                    target_column=target,
                    crop_type=crop,
                    model1_dir=model1_path,
                    model2_dir=model2_path,
                    # 如果你确实想覆盖模型保存的列，可解开下面两行：
                    # model1_expected_features=model1_features_you_want,
                    # model2_expected_features=old_features,
                    output_dir=os.path.join(output_base_dir, f'{crop}_{target}'),
                    n_repeats=100,
                    seed_range=range(1, 101)
                )
                print(f"完成: {crop} - {target}")
                print(f"结果图: {plot_path}")
            except Exception as e:
                print(f"处理 {crop} - {target} 时出错: {str(e)}")
                continue


if __name__ == "__main__":
    run_all_comparisons()
