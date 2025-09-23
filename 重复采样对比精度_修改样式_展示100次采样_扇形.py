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
import seaborn as sns  # 若前面未引入，请添加


def load_model(model_path):
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

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        data = pd.read_csv(data_path, low_memory=False)
        if 'Crop type' in data.columns:
            data = data[data['Crop type'] == 'Wheat'].copy()
    except Exception as e:
        raise ValueError(f"加载数据文件失败: {str(e)}")

    required_columns = list(set(model1_features + model2_features + [target_column]))
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"缺失必需列: {missing_columns}")

    complete_rows = data[required_columns].notna().all(axis=1)
    valid_data = data[complete_rows].copy()

    mean_target = valid_data[target_column].mean()
    std_target = valid_data[target_column].std()
    lower_bound = mean_target - 3 * std_target
    upper_bound = mean_target + 3 * std_target
    clean_data = valid_data[
        (valid_data[target_column] >= lower_bound) & (valid_data[target_column] <= upper_bound)
    ].copy()

    for feature in model1_features + model2_features:
        if not pd.api.types.is_numeric_dtype(clean_data[feature]):
            clean_data[feature] = pd.to_numeric(clean_data[feature], errors='raise')

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    all_metrics = []
    all_validation_data = []

    for i, seed in enumerate(seed_range[:n_repeats]):
        X = clean_data.drop(columns=[target_column])
        y = clean_data[target_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

        model1_val_pred = model1.predict(X_val[model1_features])
        model2_val_pred = model2.predict(X_val[model2_features])

        def calculate_metrics(y_true, y_pred, model_name, seed):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
            return {
                'Model': model_name, 'Seed': seed, 'R²': r2, 'RMSE': rmse,
                'Slope': slope, 'Intercept': intercept, 'Sample Size': len(y_pred)
            }
        # 迭代采样并预测
        for i, seed in enumerate(seed_range[:n_repeats]):
            X = clean_data.drop(columns=[target_column])
            y = clean_data[target_column]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

            model1_val_pred = model1.predict(X_val[model1_features])
            model2_val_pred = model2.predict(X_val[model2_features])

            all_metrics.append(calculate_metrics(y_val, model1_val_pred, "New Model", seed))
            all_metrics.append(calculate_metrics(y_val, model2_val_pred, "Old Model", seed))

            all_validation_data.append(pd.DataFrame({
                'Seed': seed,
                'True_Value': y_val,
                'New_Model_Prediction': model1_val_pred,
                'Old_Model_Prediction': model2_val_pred
            }))

        results_df = pd.DataFrame(all_metrics)
        summary_stats = results_df.groupby('Model').agg({
            'R²': ['mean', 'std'], 'RMSE': ['mean', 'std'],
            'Slope': ['mean', 'std'], 'Intercept': ['mean', 'std'],
            'Sample Size': 'mean'
        }).round(4)
        summary_stats.columns = ['_'.join(col) for col in summary_stats.columns]
        summary_stats.reset_index(inplace=True)

        all_validation_data_df = pd.concat(all_validation_data, ignore_index=True)
        # ---------- 可视化 ----------
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

        # ---------- 左侧条形图 ----------
        ax1 = fig.add_subplot(gs[0])
        x = np.arange(2)
        width = 0.35

        new = summary_stats[summary_stats['Model'] == 'New Model']
        old = summary_stats[summary_stats['Model'] == 'Old Model']

        ax1.bar(x[0] - width/2, new['R²_mean'], width, yerr=new['R²_std'], color='#FFB600', capsize=5)
        ax1.bar(x[0] + width/2, old['R²_mean'], width, yerr=old['R²_std'], color='#FFD773', capsize=5)
        ax1.set_ylabel('R²')
        ax1.set_ylim(0, 1)

        ax1b = ax1.twinx()
        ax1b.bar(x[1] - width/2, new['RMSE_mean'], width, yerr=new['RMSE_std'], color='#FFB600', capsize=5)
        ax1b.bar(x[1] + width/2, old['RMSE_mean'], width, yerr=old['RMSE_std'], color='#FFD773', capsize=5)
        ax1b.set_ylabel('RMSE')

        ax1.set_xticks(x)
        ax1.set_xticklabels(['R²', 'RMSE'])
        ax1.set_title('Wheat NUE: R² and RMSE')


        # ---------- 右侧散点图（100次） ----------
        ax2 = fig.add_subplot(gs[1])
        x_all = all_validation_data_df['True_Value']
        y_all_new = all_validation_data_df['New_Model_Prediction']
        y_all_old = all_validation_data_df['Old_Model_Prediction']

        min_val = min(x_all.min(), y_all_new.min(), y_all_old.min())
        max_val = max(x_all.max(), y_all_new.max(), y_all_old.max())
        margin = 0.05 * (max_val - min_val)
        plot_min = min_val - margin
        plot_max = max_val + margin
        ax2.set_xlim(plot_min, plot_max)
        ax2.set_ylim(plot_min, plot_max)

        ax2.scatter(x_all, y_all_new, alpha=0.3, c='#FFB600', edgecolors='none', s=20)
        ax2.scatter(x_all, y_all_old, alpha=0.3, c='#FFD773', edgecolors='none', s=20)


        def plot_regression_with_fan(x, y, ax, color, model_name):
            # 获取100次采样的所有斜率和截距
            slopes = results_df[results_df['Model'] == model_name]['Slope']
            intercepts = results_df[results_df['Model'] == model_name]['Intercept']

            # 计算平均斜率和截距
            mean_slope = slopes.mean()
            mean_intercept = intercepts.mean()

            # 找出斜率最大和最小的采样
            max_slope_idx = slopes.idxmax()
            min_slope_idx = slopes.idxmin()

            max_slope = slopes[max_slope_idx]
            max_intercept = intercepts[max_slope_idx]
            min_slope = slopes[min_slope_idx]
            min_intercept = intercepts[min_slope_idx]

            # 生成x值范围
            x_vals = np.linspace(plot_min, plot_max, 100)

            # 绘制最大和最小斜率的扇形区域
            y_max = max_intercept + max_slope * x_vals
            y_min = min_intercept + min_slope * x_vals

            ax.fill_between(x_vals, y_min, y_max, color=color, alpha=0.4,
                            label=f'{model_name} Range')

            # 绘制平均斜率的拟合直线
            y_mean = mean_intercept + mean_slope * x_vals
            ax.plot(x_vals, y_mean, color=color, linewidth=2,
                    label=f'{model_name} Mean Fit')


        plot_regression_with_fan(x_all, y_all_new, ax2, '#FFB600', 'New Model')
        plot_regression_with_fan(x_all, y_all_old, ax2, '#FFD773', 'Old Model')

        ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='1:1 Line')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'Wheat NUE: Predicted vs Actual (100 samples)')
        # ax2.legend()

        plot_path = os.path.join(output_dir, f'Wheat_NUE_comparison_plot_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        results_df.to_csv(os.path.join(output_dir, f'Wheat_NUE_comparison_metrics_{timestamp}.csv'), index=False)
        summary_stats.to_csv(os.path.join(output_dir, f'Wheat_NUE_summary_stats_{timestamp}.csv'), index=False)
        all_validation_data_df.to_csv(os.path.join(output_dir, f'Wheat_NUE_validation_predictions_{timestamp}.csv'),
                                      index=False)
        metrics_path = os.path.join(output_dir, f'Wheat_NUE_comparison_metrics_{timestamp}.csv')
        summary_path = os.path.join(output_dir, f'Wheat_NUE_summary_stats_{timestamp}.csv')
        validation_path = os.path.join(output_dir, f'Wheat_NUE_validation_predictions_{timestamp}.csv')

        return results_df, summary_stats, plot_path, metrics_path, summary_path, validation_path


if __name__ == "__main__":
    try:
        data_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv'
        target_column = 'NUE'

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
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib'
        model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Wheat_NUE_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_NUE_model.joblib'

        model1_features = [
             'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
            'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
            'Cddw_Key_mean'
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
            #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #  'N input rate', 'Till', 'Irrigated', 'spei_Key_mean', 'Hddw_Key_mean', 'spei_Span_min',
            #  'Ddp_Key_mean','Hddw_Span_max','LAI_Span_max', 'rx1_Span_max','rx1_Key_mean', 'rx1_Span_min',
            #  'sro_Span_max', 'rx5_Key_mean', 'Ddp_Span_max', 'rx5_Span_min','rx5_Span_max',
            #  'LAI_Span_min', 'Cddw_Span_max'
            #  'Cddw_Key_mean', 'Cddw_Span_max', 'Hddw_Key_mean', 'spei_Key_mean', 'Ddp_Key_mean', 'sro_Span_max',
            #  'rx5_Key_mean','SOC','spei_Span_max','LAI_Span_min','LAI_Key_mean','rx1_Key_mean','Hddw_Span_max',
            #  'Ddp_Span_max','rx1_Span_max','rx5_Span_max','slope','Bulk density','C/N','sro_Span_min'
            #   'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            #   'N input rate', 'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
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

        model2_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_old_results0612_GLO\Wheat\NUE\model.joblib'
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