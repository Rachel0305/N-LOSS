import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 配置参数(这个没加till，irritated)
# --------------------------------------------
CONFIG = {
    'static_features': [
    # "MAP", "MAT", "Aridity index", "Sand", "Clay",
    # "pH", "Bulk density", "SOC", "C/N", "N input rate"
    # "MAP", "MAT", "Sand", "Clay", "slope",
    # "pH", "Bulk density", "SOC", "C/N", "N input rate",
    # 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
    # 'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean',
    # 'rx1_Span_max', 'rx1_Span_min', 'rx1_Key_mean',
    # 'rx5_Span_max', 'rx5_Span_min', 'rx5_Key_mean',
    # 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
    # 'spei_Span_max', 'spei_Span_min', 'spei_Key_mean',
    # 'Hddw_Span_max', 'Hddw_Span_min', 'Hddw_Key_mean',
    # 'Cddw_Span_max', 'Cddw_Span_min', 'Cddw_Key_mean',
    'MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C/N',
     'N input rate',  'Till', 'Irrigated', 'sro_Span_max', 'sro_Span_min', 'sro_Key_mean',
     'LAI_Span_max', 'LAI_Span_min', 'LAI_Key_mean', 'rx1_Span_max',
     'rx1_Span_min', 'rx1_Key_mean', 'rx5_Span_max', 'rx5_Span_min',
     'rx5_Key_mean', 'Ddp_Span_max', 'Ddp_Span_min', 'Ddp_Key_mean',
     'spei_Span_max', 'spei_Span_min', 'spei_Key_mean', 'Hddw_Span_max',
     'Hddw_Span_min', 'Hddw_Key_mean', 'Cddw_Span_max', 'Cddw_Span_min',
     'Cddw_Key_mean', 'Hddm_Span_max', 'Hddm_Span_min', 'Hddm_Key_mean',
     'Cddm_Span_max', 'Cddm_Span_min', 'Cddm_Key_mean'
    ],
    'targets': ['N2OEF', 'NH3EF', 'NOEF', 'LF', 'RF', 'NUE'],
    'crop_types': ['Maize'],
    'test_size': 0.3,
    'random_state': 42
}


# --------------------------------------------
# 数据加载与预处理
# --------------------------------------------
def load_and_preprocess(file_path):
    """数据加载与预处理"""
    # 读取数据
    data = pd.read_csv(file_path)

    # 筛选有效作物类型
    data = data[data['Crop type'].isin(CONFIG['crop_types'])]

    # 数值型数据转换
    for col in CONFIG['static_features'] + CONFIG['targets']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    print(f"总样本量: {len(data)}")
    return data


# --------------------------------------------
# 模型训练类
# --------------------------------------------
class CropSpecificModel:
    def __init__(self, crop_type):
        self.crop_type = crop_type
        self.models = {}
        self.results = {}

    def train_all_targets(self, data):
        """训练指定作物的所有目标变量模型"""
        # 筛选作物数据
        crop_data = data[data['Crop type'] == self.crop_type]
        print(f"\n{self.crop_type} 总样本量: {len(crop_data)}")

        for target in CONFIG['targets']:
            print(f"\n正在训练: {self.crop_type} - {target}")

            # 数据清洗
            valid_data = self._clean_data(crop_data, target)

            # if len(valid_data) < 20:
            #     print(f"样本不足({len(valid_data)})，跳过训练")
            #     continue

            # 训练模型
            model, metrics, importance, pred_df = self._train_single_model(valid_data, target)

            # 保存结果时添加pred_df
            self.results[target] = {
                'model': model,
                'metrics': metrics,
                'importance': importance,
                'pred_df': pred_df  # 新增预测数据
            }

    def _clean_data(self, data, target):
        """数据清洗"""
        # 选择有效特征列
        required_cols = CONFIG['static_features'] + [target, 'Crop type']
        df = data[required_cols].copy()

        # 过滤有效观测
        df_clean = df.dropna(subset=required_cols)

        # 去除负值
        df_clean = df_clean[df_clean[target] >= 0]

        print(f"有效样本量: {len(df_clean)}")
        return df_clean

    def _train_single_model(self, data, target):
        """单个模型训练流程"""
        # 数据拆分
        X = data[CONFIG['static_features']]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state']
        )

        # 超参数搜索
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=CONFIG['random_state']),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # 最佳模型
        best_model = grid_search.best_estimator_

        # 模型评估（修改此处接收pred_df）
        metrics, pred_df = self._evaluate_model(best_model, X_train, X_test, y_train, y_test)

        # 特征重要性
        importance = self._get_feature_importance(best_model, X_train.columns)

        # 保存预测数据到结果中
        return best_model, metrics, importance, pred_df

    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """模型评估"""
        # 生成预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算回归参数
        lr = LinearRegression().fit(y_test.values.reshape(-1, 1), y_test_pred)

        # 生成预测结果数据框
        pred_df = pd.DataFrame({
            'True': y_test,
            'Predicted': y_test_pred
        })

        return {
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Train R2': r2_score(y_train, y_train_pred),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Test R2': r2_score(y_test, y_test_pred),
            'Slope': lr.coef_[0],
            'Intercept': lr.intercept_,
            'Sample Size': len(y_test)
        }, pred_df  # 返回指标和预测数据

    def _get_feature_importance(self, model, feature_names):
        """获取特征重要性"""
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

    def save_results(self, output_dir):
        """保存所有结果"""
        # 创建作物专属目录
        crop_dir = os.path.join(output_dir, self.crop_type)
        os.makedirs(crop_dir, exist_ok=True)

        # 保存每个目标变量的结果
        for target, result in self.results.items():
            # 创建目标变量子目录
            target_dir = os.path.join(crop_dir, target)
            os.makedirs(target_dir, exist_ok=True)

            # 保存模型
            joblib.dump(result['model'], os.path.join(target_dir, 'model.joblib'))

            # 保存评估指标
            pd.DataFrame([result['metrics']]).to_csv(
                os.path.join(target_dir, 'metrics.csv'), index=False)

            # 保存特征重要性
            result['importance'].to_csv(
                os.path.join(target_dir, 'feature_importance.csv'), index=False)

            # 新增：保存真实值与预测值表格
            result['pred_df'].to_csv(
                os.path.join(target_dir, 'true_vs_predicted.csv'),  # 文件名可自定义
                index=False  # 不保存索引列
            )

            # 生成可视化图表
            self._plot_results(result, target_dir, target)

    def _plot_results(self, result, target_dir, target):
        """生成可视化图表"""
        # 特征重要性图
        plt.figure(figsize=(10, 10))
        sns.barplot(x='Importance', y='Feature', data=result['importance'], palette='viridis')
        plt.title(f'{self.crop_type} - {target} Feature Importance')
        plt.savefig(os.path.join(target_dir, 'feature_importance.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 预测值与真实值的散点图
        pred_df = result['pred_df']
        metrics = result['metrics']
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # 绘制散点
        sns.scatterplot(x='True', y='Predicted', data=pred_df, alpha=0.6,
                        color='#2c7bb6', s=150, edgecolor='w', ax=ax)

        # 设置坐标范围和样式
        plot_min = 0
        data_max = max(pred_df['True'].max(), pred_df['Predicted'].max())
        plot_max = data_max + data_max * 0.1
        ax.set(xlim=(plot_min, plot_max), ylim=(plot_min, plot_max), aspect='equal')

        # 绘制参考线和回归线
        ax.plot([plot_min, plot_max], [plot_min, plot_max], '--', color='gray', alpha=0.8)
        boundary = np.array([plot_min, plot_max])
        ax.plot(boundary, metrics['Slope'] * boundary + metrics['Intercept'],
                color='#d7191c', linewidth=2.5)

        # 设置标题和标签
        plt.title(f"{self.crop_type} - {target}_pred VS true", fontsize=20, pad=15)
        ax.set_xlabel("True Value", fontsize=12, labelpad=10)
        ax.set_ylabel("Predicted Value", fontsize=12, labelpad=10)
        ax.tick_params(labelsize=12)

        # 保存图表
        plt.savefig(os.path.join(target_dir, f'{self.crop_type}_{target}_scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# --------------------------------------------
# 主程序
# --------------------------------------------
if __name__ == "__main__":
    # 初始化输出目录
    output_dir = "RF_model_C_new_results0727_GLO_所有变量特征"
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    # data = load_and_preprocess('Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hdd_Cdd_phase_mon.csv')
    # data = load_and_preprocess('Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_CHINA.csv')
    data = load_and_preprocess('Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv')

    # 按作物类型训练模型
    for crop in CONFIG['crop_types']:
        print(f"\n{'=' * 30}\n开始处理: {crop}\n{'=' * 30}")

        # 初始化模型训练器
        trainer = CropSpecificModel(crop)

        # 训练所有目标变量
        trainer.train_all_targets(data)

        # 保存结果
        trainer.save_results(output_dir)

    print("\n所有模型训练完成！结果已保存至:", os.path.abspath(output_dir))