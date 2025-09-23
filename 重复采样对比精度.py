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


def load_model(model_path):
    """加载PKL或Joblib格式的模型文件"""
    try:
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = joblib.load(f)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            raise ValueError("仅支持.pkl或.joblib格式的模型文件")

        if not hasattr(model, 'predict'):
            raise ValueError("加载的对象不是有效的模型（缺少predict方法）")

        return model
    except Exception as e:
        raise ValueError(f"加载模型失败: {str(e)}")


def evaluate_models(data_path, target_column,
                    model1_path, model1_features,
                    model2_path, model2_features,
                    test_size=0.3, n_repeats=100, seed_range=range(1, 101),
                    output_dir='results'):
    """
    评估两个模型在公共验证集上的表现，进行多次重复采样，并保存结果

    参数:
    n_repeats: 重复采样次数
    seed_range: 随机种子范围
    output_dir: 结果保存目录
    """

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 加载数据并筛选Wheat作物类型
    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"成功加载数据，共 {len(data)} 行")

        # 筛选Wheat作物类型
        if 'Crop type' in data.columns:
            data = data[data['Crop type'] == 'Wheat'].copy()
            print(f"筛选后Wheat数据: {len(data)} 行")
        else:
            print("警告: 数据中未找到'Crop type'列，将使用全部数据")

    except Exception as e:
        raise ValueError(f"加载数据文件失败: {str(e)}")

    # 2. 检查目标列和特征列
    required_columns = list(set(model1_features + model2_features + [target_column]))
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"缺失必需列: {missing_columns}")

    # 3. 检查并处理缺失值
    complete_rows = data[required_columns].notna().all(axis=1)
    valid_data = data[complete_rows].copy()
    print(f"处理缺失值后有效数据: {len(valid_data)} 行")

    # 4. 剔除目标列中偏离超过3个标准差的异常值并保存异常值数据
    mean_target = valid_data[target_column].mean()
    std_target = valid_data[target_column].std()
    lower_bound = mean_target - 3 * std_target
    upper_bound = mean_target + 3 * std_target

    # 提取异常值数据
    outliers = valid_data[
        (valid_data[target_column] < lower_bound) | (valid_data[target_column] > upper_bound)
    ].copy()
    if not outliers.empty:
        outliers_csv_path = os.path.join(output_dir, f'Wheat_RF_outliers_{timestamp}.csv')
        outliers.to_csv(outliers_csv_path, index=False)
        print(f"已保存 {len(outliers)} 行异常值数据到: {outliers_csv_path}")

    # 筛选正常值数据
    clean_data = valid_data[
        (valid_data[target_column] >= lower_bound) & (valid_data[target_column] <= upper_bound)
    ].copy()
    # clean_data = valid_data.copy()
    print(f"剔除 {target_column} 异常值（超出均值±3标准差）后剩余数据: {len(clean_data)} 行 "
          f"(均值={mean_target:.2f}, 标准差={std_target:.2f}, 范围=[{lower_bound:.2f}, {upper_bound:.2f}])")

    # 5. 确保特征列是数值类型
    for feature in model1_features + model2_features:
        if not pd.api.types.is_numeric_dtype(clean_data[feature]):
            try:
                clean_data[feature] = pd.to_numeric(clean_data[feature], errors='raise')
            except ValueError:
                raise ValueError(f"特征列 '{feature}' 包含非数值数据且无法转换为数值类型")

    # 6. 加载模型
    try:
        print("\n加载模型中...")
        model1 = load_model(model1_path)
        print(f"成功加载模型1: {type(model1).__name__}")
        model2 = load_model(model2_path)
        print(f"成功加载模型2: {type(model2).__name__}")
    except Exception as e:
        raise ValueError(f"模型加载失败: {str(e)}")

    # 7. 准备存储多次采样结果
    all_metrics = []
    all_validation_data = []

    # 8. 进行多次重复采样
    print(f"\n进行 {n_repeats} 次重复采样...")
    for i, seed in enumerate(seed_range[:n_repeats]):
        print(f"第 {i+1}/{n_repeats} 次采样，种子: {seed}")

        # 划分训练集和验证集
        X = clean_data.drop(columns=[target_column])
        y = clean_data[target_column]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # 获取预测结果
        try:
            model1_val_pred = model1.predict(X_val[model1_features])
            model2_val_pred = model2.predict(X_val[model2_features])
        except Exception as e:
            raise ValueError(f"第 {i+1} 次采样预测时出错: {str(e)}")

        # 计算评估指标
        def calculate_metrics(y_true, y_pred, model_name, seed):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)  # 真实值作为 x，预测值作为 y

            return {
                'Model': model_name,
                'Seed': seed,
                'R²': r2,
                'RMSE': rmse,
                'Slope': slope,
                'Intercept': intercept,
                'Sample Size': len(y_pred)
            }

        metrics_model1 = calculate_metrics(y_val, model1_val_pred, "New Model", seed)
        metrics_model2 = calculate_metrics(y_val, model2_val_pred, "Old Model", seed)
        all_metrics.extend([metrics_model1, metrics_model2])

        # 保存验证集真值和预测值
        validation_data = pd.DataFrame({
            'Seed': seed,
            'True_Value': y_val,
            'New_Model_Prediction': model1_val_pred,
            'Old_Model_Prediction': model2_val_pred
        })
        for feature in model1_features + model2_features:
            if feature not in validation_data.columns:
                validation_data[feature] = X_val[feature]
        all_validation_data.append(validation_data)

    # 9. 创建结果DataFrame
    results_df = pd.DataFrame(all_metrics)

    # 计算均值和标准差
    summary_stats = results_df.groupby('Model').agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'Slope': ['mean', 'std'],
        'Intercept': ['mean', 'std'],
        'Sample Size': 'mean'
    }).round(4)

    # 重命名列名
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.reset_index(inplace=True)

    # 保存精度结果到CSV
    results_csv_path = os.path.join(output_dir, f'Wheat_RF_comparison_metrics_{timestamp}.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n已保存每次采样精度结果到: {results_csv_path}")

    # 保存统计结果到CSV
    summary_csv_path = os.path.join(output_dir, f'Wheat_RF_summary_stats_{timestamp}.csv')
    summary_stats.to_csv(summary_csv_path, index=False)
    print(f"已保存统计结果（均值±标准差）到: {summary_csv_path}")

    # 保存验证集真值和预测值
    all_validation_data_df = pd.concat(all_validation_data, ignore_index=True)
    validation_csv_path = os.path.join(output_dir, f'Wheat_RF_validation_predictions_{timestamp}.csv')
    all_validation_data_df.to_csv(validation_csv_path, index=False)
    print(f"已保存验证集真值和预测值到: {validation_csv_path}")

    # 10. 可视化结果
    # 创建画布，使用 GridSpec 精确控制布局
    fig = plt.figure(figsize=(16, 8))  # 画布宽度为 12，高度为 5
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.25)  # 子图间距为 0.15

    # 条形图
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(1)
    metrics_to_plot = ['R²', 'RMSE']
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    new_model_means = [summary_stats[summary_stats['Model'] == 'New Model'][f'{metric}_mean'].values[0] for metric in metrics_to_plot]
    new_model_stds = [summary_stats[summary_stats['Model'] == 'New Model'][f'{metric}_std'].values[0] for metric in metrics_to_plot]
    old_model_means = [summary_stats[summary_stats['Model'] == 'Old Model'][f'{metric}_mean'].values[0] for metric in metrics_to_plot]
    old_model_stds = [summary_stats[summary_stats['Model'] == 'Old Model'][f'{metric}_std'].values[0] for metric in metrics_to_plot]

    ax1.bar(x - width/2, new_model_means, width, yerr=new_model_stds, label='New Model', color='#FFB600', capsize=5)
    ax1.bar(x + width/2, old_model_means, width, yerr=old_model_stds, label='Old Model', color='#FFD773', capsize=5)

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Wheat RF: Mean ± Std Across Repeats')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot)

    # 散点图
    ax2 = fig.add_subplot(gs[1])
    ax2.set_box_aspect(1)
    last_seed_data = all_validation_data[-1]
    y_val = last_seed_data['True_Value']
    model1_val_pred = last_seed_data['New_Model_Prediction']
    model2_val_pred = last_seed_data['Old_Model_Prediction']

    # 计算最后一次采样的 Slope 和 Intercept（真实值作为 x，预测值作为 y）
    slope1, intercept1, _, _, _ = stats.linregress(y_val, model1_val_pred)
    slope2, intercept2, _, _, _ = stats.linregress(y_val, model2_val_pred)

    margin = 0.05
    x_min = np.min(y_val)
    x_max = np.max(y_val)
    y_min = min(np.min(model1_val_pred), np.min(model2_val_pred))
    y_max = max(np.max(model1_val_pred), np.max(model2_val_pred))
    x_range = x_max - x_min
    y_range = y_max - y_min

    plot_min = min(x_min - margin * x_range, y_min - margin * y_range)
    plot_max = max(x_max + margin * x_range, y_max + margin * y_range)
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_ylim(plot_min, plot_max)

    # 绘制1:1线
    line_1_1, = ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='1:1 Line')

    # 绘制New Model拟合线
    line_new, = ax2.plot([plot_min, plot_max],
                         [intercept1 + slope1 * plot_min, intercept1 + slope1 * plot_max],
                         '-', color='#FFB600', linewidth=2, label=f'New Model (Slope={slope1:.3f})')

    # 绘制Old Model拟合线
    line_old, = ax2.plot([plot_min, plot_max],
                         [intercept2 + slope2 * plot_min, intercept2 + slope2 * plot_max],
                         '-', color='#FFD773', linewidth=2, label=f'Old Model (Slope={slope2:.3f})')
    # 绘制散点（真实值在 x 轴，预测值在 y 轴）
    ax2.scatter(y_val, model1_val_pred, alpha=0.5, c='#FFB600', edgecolors='gray', s=100)
    ax2.scatter(y_val, model2_val_pred, alpha=0.5, c='#FFD773', edgecolors='gray', s=100)

    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Wheat RF: Predicted vs Actual (Seed {seed_range[-1]})')
    # ax2.legend(loc='lower right')
    # 将散点图图例移到右侧，上方
    fig.legend(handles=[line_1_1, line_new, line_old], loc='center left', bbox_to_anchor=(0.72, 0.55),
               fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    # 添加详细图例到右侧
    legend_text = (
        f"New Model (Mean ± Std):\n"
        f"R² = {new_model_means[0]:.3f} ± {new_model_stds[0]:.3f}\n"
        f"RMSE = {new_model_means[1]:.3f} ± {new_model_stds[1]:.3f}\n"
        f"Slope = {summary_stats[summary_stats['Model'] == 'New Model']['Slope_mean'].values[0]:.3f} ± {summary_stats[summary_stats['Model'] == 'New Model']['Slope_std'].values[0]:.3f}\n\n"
        f"Old Model (Mean ± Std):\n"
        f"R² = {old_model_means[0]:.3f} ± {old_model_stds[0]:.3f}\n"
        f"RMSE = {old_model_means[1]:.3f} ± {old_model_stds[1]:.3f}\n"
        f"Slope = {summary_stats[summary_stats['Model'] == 'Old Model']['Slope_mean'].values[0]:.3f} ± {summary_stats[summary_stats['Model'] == 'Old Model']['Slope_std'].values[0]:.3f}\n"
        f"N = {int(summary_stats['Sample Size_mean'].iloc[0])}"
    )
    fig.text(0.73, 0.34, legend_text, fontsize=10, va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 调整布局，确保图例不被裁剪
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    # 保存图像
    plot_path = os.path.join(output_dir, f'Wheat_RF_comparison_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"已保存结果图到: {plot_path}")
    plt.show()

    return results_df, summary_stats, plot_path, results_csv_path, summary_csv_path, validation_csv_path


if __name__ == "__main__":
    try:
        data_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv'
        target_column = 'RF'

        # 模型1信息
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0612_去掉异常值wheat\Wheat\N2OEF\model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_N2OEF_LightGBM.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Wheat_LF_RF.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_LF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0612_去掉异常值wheat\Wheat\LF\model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_LF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_LF_LightGBM.joblib'
        model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Wheat_NUE_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_NUE_model.joblib'

        model1_features = [
            #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            # 'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            # 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            # 'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            # 'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            # 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
            # 'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
            # 'Cddw_Key_mean'
            # 'spei_Key_mean', 'MAT', 'Hddw_Key_mean', 'spei_Span_min', 'Ddp_Key_mean',
            # 'Hddw_Span_max', 'LAI_Span_max', 'rx1_Span_max', 'rx1_Key_mean',
            # 'rx1_Span_min', 'sro_Span_max', 'rx5_Key_mean', 'Ddp_Span_max',
            # 'Bulk density', 'pH', 'rx5_Span_min', 'rx5_Span_max', 'LAI_Span_min',
            # 'MAP', 'Cddw_Span_max'
            #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #  'N input rate',  'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            #  'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            #  'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            #  'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            #  'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
            #  'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
            #  'Cddw_Key_mean', 'Hddm_Span_max', 'Hddm_Span_min', 'Hddm_Key_mean',
            #  'Cddm_Span_max', 'Cddm_Span_min', 'Cddm_Key_mean'
             'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
             'N input rate', 'Till', 'Irrigated', 'spei_Key_mean', 'Hddw_Key_mean', 'spei_Span_min',
             'Ddp_Key_mean','Hddw_Span_max','LAI_Span_max', 'rx1_Span_max','rx1_Key_mean', 'rx1_Span_min',
             'sro_Span_max', 'rx5_Key_mean', 'Ddp_Span_max', 'rx5_Span_min','rx5_Span_max',
             'LAI_Span_min', 'Cddw_Span_max', 'Cddw_Key_mean', 'Cddw_Span_max', 'Hddw_Key_mean', 'spei_Key_mean', 'Ddp_Key_mean', 'sro_Span_max'
            #  'rx5_Key_mean','SOC','spei_Span_max','LAI_Span_min','LAI_Key_mean','rx1_Key_mean','Hddw_Span_max',
            #  'Ddp_Span_max','rx1_Span_max','rx5_Span_max','slope','Bulk density','C/N','sro_Span_min'
            #   'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #   'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            #   'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            #   'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            #   'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            #   'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
            #   'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
            #   'Cddw_Key_mean'
            #   'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #  'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            #  'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            #  'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            #  'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            #  'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddm_Span_max',
            #  'Hddm_Span_min', 'Hddm_Key_mean', 'Cddm_Span_max', 'Cddm_Span_min',
            #  'Cddm_Key_mean'
            #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #   'N input rate',  'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            #   'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            #   'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            #   'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            #   'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
            #   'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
            #   'Cddw_Key_mean', 'Hddm_Span_max', 'Hddm_Span_min', 'Hddm_Key_mean',
            #   'Cddm_Span_max', 'Cddm_Span_min', 'Cddm_Key_mean'
            # 'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            # 'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            # 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            # 'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            # 'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            # 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddm_Span_max',
            # 'Hddm_Span_min', 'Hddm_Key_mean', 'Cddm_Span_max', 'Cddm_Span_min',
            # 'Cddm_Key_mean'
            #  'MAP','sro_Key_mean','LAI_Key_mean','Ddp_Key_mean','LAI_Span_max','Ddp_Span_max','Sand','Ddp_Span_min',
            #  'rx1_Span_min','LAI_Span_min','C/N','MAT','Clay','pH','SOC','Bulk density','sro_Span_max','slope',
            #  'Hddm_Span_max','Hddm_Key_mean'
        ]

        model2_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_old_results0612_GLO\Wheat\RF\model.joblib'
        model2_features = ["MAP", "MAT", "Aridity index", "Sand", "Clay", "pH", "Bulk density", "SOC", "C/N",
                           "N input rate"]

        print("开始模型评估...")
        results_df, summary_stats, plot_path, metrics_path, summary_path, validation_path = evaluate_models(
            data_path=data_path,
            target_column=target_column,
            model1_path=model1_path,
            model1_features=model1_features,
            model2_path=model2_path,
            model2_features=model2_features,
            output_dir='Wheat_model_comparison_results',
            n_repeats=100,
            seed_range=range(1, 101)
        )

        print("\n模型性能统计（均值±标准差）:")
        print(summary_stats.to_markdown(tablefmt="grid", index=False))
        print("\n每次采样结果（部分）：")
        print(results_df.head(10).to_markdown(tablefmt="grid", index=False))
        print(f"\n结果图已保存到: {plot_path}")
        print(f"每次采样精度结果已保存到: {metrics_path}")
        print(f"统计结果已保存到: {summary_path}")
        print(f"验证集真值和预测值已保存到: {validation_path}")

    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        print("\n详细诊断建议:")
        print("1. 检查数据文件路径是否正确")
        print("2. 确认所有指定的列名在数据中存在")
        print("3. 检查模型文件是否是有效的scikit-learn模型")
        print("4. 使用 data.isna().sum() 检查数据中的缺失值")
        print("5. 确保所有特征列都是数值类型")
        print("6. 检查模型使用的特征列是否与训练时的特征一致")