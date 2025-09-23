import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime


def load_and_preprocess_data(config):
    """数据加载和预处理（添加异常值处理）"""
    data = pd.read_csv(config['data_path'])
    crop_data = data[data[config['crop_col']] == config['crop_type']].copy()

    # 新增：去除因变量异常值（偏离3个标准差）
    target_var = config['target_var']
    mean = crop_data[target_var].mean()
    std = crop_data[target_var].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # 记录原始数据量
    original_count = len(crop_data)

    # 过滤异常值
    # crop_data = crop_data[
    #     (crop_data[target_var] >= lower_bound) &
    #     (crop_data[target_var] <= upper_bound)
    #     ].copy()
    crop_data = crop_data.copy()

    # 打印过滤信息
    filtered_count = len(crop_data)
    print(f"原始数据量: {original_count}")
    print(f"过滤后数据量: {filtered_count}")
    print(f"移除异常值数量: {original_count - filtered_count}")
    print(f"异常值阈值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # 标准化数值变量
    scaler = StandardScaler()
    crop_data[config['numeric_vars']] = scaler.fit_transform(crop_data[config['numeric_vars']])

    return crop_data[config['numeric_vars'] + [target_var]]  # 返回包含因变量的数据


def load_model(model_path):
    """加载预训练模型"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def perform_shap_analysis(model, X, config):
    """修正后的SHAP分析主函数"""
    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)

    # 分离特征和目标变量
    X_features = X.drop(columns=[config['target_var']])
    y = X[config['target_var']]

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_features)

    # 生成并保存SHAP点图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_features, plot_type="dot", show=False)
    plt.title(f"SHAP Values for {config['target_var']} ({config['crop_type']})")
    plt.savefig(
        f"{config['output_dir']}/shap_summary_{config['timestamp']}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 计算特征重要性
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 处理多分类情况

    importance_df = pd.DataFrame({
        'feature': X_features.columns.tolist(),
        'shap_importance': np.abs(shap_values).mean(axis=0),
        'shap_mean': shap_values.mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    # 新增：绘制变量重要性条形图
    plt.figure(figsize=(10, 6))
    importance_df.sort_values('shap_importance', ascending=True).plot(
        kind='barh',
        x='feature',
        y='shap_importance',
        color='steelblue',
        legend=False
    )
    plt.title(f"Feature Importance for {config['target_var']} ({config['crop_type']})")
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(
        f"{config['output_dir']}/feature_importance_bar_{config['timestamp']}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 保存颜色分界点信息
    color_thresholds = pd.DataFrame({
        col: {
            'low': np.percentile(X_features[col], 30),
            'high': np.percentile(X_features[col], 70)
        } for col in X_features.columns
    }).T
    color_thresholds.to_csv(f"{config['output_dir']}/color_thresholds_{config['timestamp']}.csv")

    return importance_df


if __name__ == "__main__":
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 配置参数
    CONFIG = {
        'data_path': 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_N2OEF_LightGBM.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_LF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Wheat_NUE_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib',
        'model_path': r'D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib',
        # 'model_path': r'D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_NUE_model.joblib',
        'target_var': 'RF',
        'crop_type': 'Maize',
        'crop_col': 'Crop type',
        'numeric_vars': [
        # 'spei_Key_mean', 'MAT', 'Hddw_Key_mean', 'spei_Span_min', 'Ddp_Key_mean',
        # 'Hddw_Span_max', 'LAI_Span_max', 'rx1_Span_max', 'rx1_Key_mean',
        # 'rx1_Span_min', 'sro_Span_max', 'rx5_Key_mean', 'Ddp_Span_max',
        # 'Bulk density', 'pH', 'rx5_Span_min', 'rx5_Span_max', 'LAI_Span_min',
        # 'MAP', 'Cddw_Span_max'
        # 'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density',
        # 'SOC', 'C/N', 'N input rate', 'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min',
        # 'sro_Key_mean', 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean',
        # 'rx1_Span_max', 'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max',
        # 'rx5_Span_min', 'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min',
        # 'Ddp_Key_mean', 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean',
        # 'Hddw_Span_max', 'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max',
        # 'Cddw_Span_min', 'Cddw_Key_mean', 'Hddm_Span_max', 'Hddm_Span_min',
        # 'Hddm_Key_mean', 'Cddm_Span_max', 'Cddm_Span_min', 'Cddm_Key_mean'
        #  'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
        #  'N input rate', 'Till', 'Irrigated', 'spei_Key_mean', 'Hddw_Key_mean', 'spei_Span_min',
        #  'Ddp_Key_mean','Hddw_Span_max','LAI_Span_max', 'rx1_Span_max','rx1_Key_mean', 'rx1_Span_min',
        #  'sro_Span_max', 'rx5_Key_mean', 'Ddp_Span_max', 'rx5_Span_min','rx5_Span_max',
        #  'LAI_Span_min', 'Cddw_Span_max'
        # 'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
        # 'N input rate', 'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
        # 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
        # 'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
        # 'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
        # 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
        # 'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
        # 'Cddw_Key_mean'
        #     'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
        #      'N input rate', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
        #      'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
        #      'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
        #      'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
        #      'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddm_Span_max',
        #      'Hddm_Span_min', 'Hddm_Key_mean', 'Cddm_Span_max', 'Cddm_Span_min',
        #      'Cddm_Key_mean'
        # 'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
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
         'MAP','sro_Key_mean','LAI_Key_mean','Ddp_Key_mean','LAI_Span_max','Ddp_Span_max','Sand','Ddp_Span_min',
         'rx1_Span_min','LAI_Span_min','C/N','MAT','Clay','pH','SOC','Bulk density','sro_Span_max','slope',
         'Hddm_Span_max','Hddm_Key_mean'
        ],  # 数值变量列表
        'output_dir': 'shap_results_Maize_RF',
        'timestamp': timestamp
    }

    # 执行分析
    data = load_and_preprocess_data(CONFIG)
    model = load_model(CONFIG['model_path'])

    # 分离特征和目标变量
    X = data.drop(columns=[CONFIG['target_var']])
    y = data[CONFIG['target_var']]

    importance_df = perform_shap_analysis(model, pd.concat([X, y], axis=1), CONFIG)

    # 保存结果
    importance_df.to_csv(
        f"{CONFIG['output_dir']}/feature_importance_{CONFIG['timestamp']}.csv",
        index=False
    )
    print(f"分析完成！结果保存在: {CONFIG['output_dir']}")