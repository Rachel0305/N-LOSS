import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
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
                    test_size=0.3, random_state=31,
                    output_dir='results'):
    """
    评估两个模型在公共验证集上的表现并保存结果

    参数:
    output_dir: 结果保存目录
    """

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 加载数据并筛选Maize作物类型
    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"成功加载数据，共 {len(data)} 行")

        # 筛选Maize作物类型且NOEF有效的数据
        if 'Crop type' in data.columns:
            data = data[data['Crop type'] == 'Maize'].copy()
            print(f"筛选后Maize数据: {len(data)} 行")
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
    print(f"原始数据: {len(data)} 行, 有效数据: {len(valid_data)} 行")

    # 4. 确保特征列是数值类型
    for feature in model1_features + model2_features:
        if not pd.api.types.is_numeric_dtype(valid_data[feature]):
            try:
                valid_data[feature] = pd.to_numeric(valid_data[feature], errors='raise')
            except ValueError:
                raise ValueError(f"特征列 '{feature}' 包含非数值数据且无法转换为数值类型")

    # 5. 划分训练集和验证集
    X = valid_data.drop(columns=[target_column])
    y = valid_data[target_column]
    print(len(target_column))
    print(len(X))
    print(len(y))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n验证集大小: {len(X_val)} 行")

    # 6. 加载模型
    try:
        print("\n加载模型中...")
        model1 = load_model(model1_path)
        print(f"成功加载模型1: {type(model1).__name__}")
        model2 = load_model(model2_path)
        print(f"成功加载模型2: {type(model2).__name__}")
    except Exception as e:
        raise ValueError(f"模型加载失败: {str(e)}")

    # 7. 获取预测结果
    try:
        print("\n进行预测...")
        print(f"模型1使用特征: {model1_features}")
        model1_val_pred = model1.predict(X_val[model1_features])
        print(f"模型2使用特征: {model2_features}")
        model2_val_pred = model2.predict(X_val[model2_features])
    except Exception as e:
        raise ValueError(f"预测时出错: {str(e)}")

    # 8. 计算评估指标
    def calculate_metrics(y_true, y_pred, model_name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)

        return {
            'Model': model_name,
            'R²': round(r2, 4),
            'RMSE': round(rmse, 4),
            'Slope': round(slope, 4),
            'Intercept': round(intercept, 4),
            'Sample Size': len(y_pred)
        }

    metrics_model1 = calculate_metrics(y_val, model1_val_pred, "New Model")
    metrics_model2 = calculate_metrics(y_val, model2_val_pred, "Old Model")

    # 9. 创建结果DataFrame
    results = pd.DataFrame([metrics_model1, metrics_model2])

    # 保存精度结果到CSV
    results_csv_path = os.path.join(output_dir, f'Maize_NUE_comparison_metrics_{timestamp}.csv')
    results.to_csv(results_csv_path, index=False)
    print(f"\n已保存精度结果到: {results_csv_path}")

    # 10. 保存验证集真值和预测值
    validation_data = pd.DataFrame({
        'True_Value': y_val,
        'New_Model_Prediction': model1_val_pred,
        'Old_Model_Prediction': model2_val_pred
    })

    # 添加特征数据（如果需要）
    for feature in model1_features + model2_features:
        if feature not in validation_data.columns:
            validation_data[feature] = X_val[feature]

    validation_csv_path = os.path.join(output_dir, f'Maize_NUE_validation_predictions_{timestamp}.csv')
    validation_data.to_csv(validation_csv_path, index=False)
    print(f"已保存验证集真值和预测值到: {validation_csv_path}")

    # 11. 可视化结果
    plt.figure(figsize=(12, 6))  # 整体画布尺寸

    # 散点图 - 第一个正方形子图
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_box_aspect(1)  # 强制1:1宽高比

    # 计算坐标范围（扩展到5%的边距）
    margin = 0.05
    x_min = min(np.min(model1_val_pred), np.min(model2_val_pred))
    x_max = max(np.max(model1_val_pred), np.max(model2_val_pred))
    y_min = min(np.min(y_val), np.min(y_val))
    y_max = max(np.max(y_val), np.max(y_val))
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 设置相同的坐标范围
    plot_min = min(x_min - margin * x_range, y_min - margin * y_range)
    plot_max = max(x_max + margin * x_range, y_max + margin * y_range)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)

    # 绘制1:1线（黑色实线）
    ax1.plot([plot_min, plot_max], [plot_min, plot_max],
         color='black',   # 黑色
         linestyle='--',  # 虚线
         alpha=0.5,       # 半透明（0为全透明，1为不透明）
         label='1:1 Line')

    # 绘制New Model拟合线（橙色实线）
    ax1.plot([plot_min, plot_max],
             [metrics_model1['Intercept'] + metrics_model1['Slope'] * plot_min,
              metrics_model1['Intercept'] + metrics_model1['Slope'] * plot_max],
             '-', color='#A68900', linewidth=2, label='New Model Fit')

    # 绘制Old Model拟合线（黄色实线）
    ax1.plot([plot_min, plot_max],
             [metrics_model2['Intercept'] + metrics_model2['Slope'] * plot_min,
              metrics_model2['Intercept'] + metrics_model2['Slope'] * plot_max],
             '-', color='#FFDE40', linewidth=2, label='Old Model Fit')

    # 绘制散点
    ax1.scatter(model1_val_pred, y_val, alpha=0.5, label='New Model', c='#A68900', edgecolors='gray', s=100)
    ax1.scatter(model2_val_pred, y_val, alpha=0.5, label='Old Model', c='#FFDE40', edgecolors='gray', s=100)

    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Maize NUE: Predicted vs Actual')

    # 添加指标图例（右上角）
    legend_text = (
        "$\mathbf{New\ Model:}$\n"
        f"R² = {metrics_model1['R²']:.3f}\n"
        f"RMSE = {metrics_model1['RMSE']:.3f}\n"
        f"Slope = {metrics_model1['Slope']:.3f}\n\n"
        "$\mathbf{Old\ Model:}$\n"
        f"R² = {metrics_model2['R²']:.3f}\n"
        f"RMSE = {metrics_model2['RMSE']:.3f}\n"
        f"Slope = {metrics_model2['Slope']:.3f}\n"
        f"N = {metrics_model2['Sample Size']}"
    )

    plt.text(0.98, 0.35, legend_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
             )

    # 保留原始图例（显示线条和散点的对应关系）
    # ax1.legend(loc='lower left')

    # 条形图 - 第二个正方形子图
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_box_aspect(1)  # 关键修改：强制1:1宽高比

    results_display = results.copy()
    results_display.set_index('Model').drop(['Slope', 'Intercept', 'Sample Size'], axis=1).plot(
        kind='bar', ax=ax2, color=['#A68900', '#FFDE40'])
    ax2.set_title('Maize NUE Comparison')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    # 调整条形图y轴范围使其更正方形友好
    current_ylim = ax2.get_ylim()
    new_ylim = (current_ylim[0], current_ylim[1] + 0.2 * (current_ylim[1] - current_ylim[0]))
    ax2.set_ylim(new_ylim)

    # 在条形图上添加数值标签
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.4f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 5),
                     textcoords='offset points')

    # 调整子图间距
    plt.subplots_adjust(wspace=0.4)  # 增加子图间距

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(output_dir, f'Maize_NUE_comparison_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"已保存结果图到: {plot_path}")
    plt.show()

    return results, plot_path, results_csv_path, validation_csv_path


# 使用示例
if __name__ == "__main__":
    try:
        data_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv'
        target_column = 'NUE'

        # 模型1信息
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0612_去掉异常值wheat\Wheat\N2OEF\model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Wheat_LF_RF.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Wheat_NUE_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib'
        # model1_path = r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib'
        model1_path = r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_NUE_model.joblib'

        model1_features = [
           #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
           # 'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
           # 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
           # 'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
           # 'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
           # 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
           # 'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
           # 'Cddw_Key_mean'
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
           #  'LAI_Span_min', 'Cddw_Span_max',
           #  'Cddw_Key_mean', 'Cddw_Span_max', 'Hddw_Key_mean', 'spei_Key_mean', 'Ddp_Key_mean', 'sro_Span_max',
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
            'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
            'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
            'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
            'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
            'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
            'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddm_Span_max',
            'Hddm_Span_min', 'Hddm_Key_mean', 'Cddm_Span_max', 'Cddm_Span_min',
            'Cddm_Key_mean'
           #  'MAP','sro_Key_mean','LAI_Key_mean','Ddp_Key_mean','LAI_Span_max','Ddp_Span_max','Sand','Ddp_Span_min',
           #  'rx1_Span_min','LAI_Span_min','C/N','MAT','Clay','pH','SOC','Bulk density','sro_Span_max','slope',
           #  'Hddm_Span_max','Hddm_Key_mean'
                           ]

        # 模型2信息
        model2_path = r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_old_results0612_GLO\Maize\NUE\model.joblib'
        model2_features = ["MAP", "MAT", "Aridity index", "Sand", "Clay", "pH", "Bulk density", "SOC", "C/N",
                           "N input rate"]

        # 运行评估
        print("开始模型评估...")
        results, plot_path, metrics_path, validation_path = evaluate_models(
            data_path=data_path,
            target_column=target_column,
            model1_path=model1_path,
            model1_features=model1_features,
            model2_path=model2_path,
            model2_features=model2_features,
            output_dir='Maize_model_comparison_results'  # 指定结果保存目录
        )

        print("\n模型性能比较:")
        print(results.to_markdown(tablefmt="grid", index=False))
        print(f"\n结果图已保存到: {plot_path}")
        print(f"精度结果已保存到: {metrics_path}")
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