import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# 读取原始数据文件
df_original = pd.read_csv(r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\true_vs_predicted.csv')
print("原始数据列名:", df_original.columns.tolist())
true_values_original = df_original['True']
predicted_values_original = df_original['Predicted']

# 读取对比数据文件（请修改为您的对比数据文件路径）
df_compare = pd.read_csv(r'D:\CNPK\N_CODE_PY\N_ML\RF_model_C_old_results0612_GLO\Wheat\N2OEF\true_vs_predicted_del.csv')  # 请修改为实际路径
print("对比数据列名:", df_compare.columns.tolist())
true_values_compare = df_compare['TRUE']
predicted_values_compare = df_compare['Predicted']

# 计算原始数据的统计量
n_original = len(true_values_original)
slope_original, intercept_original, r_value_original, _, _ = stats.linregress(true_values_original, predicted_values_original)
r_squared_original = r_value_original**2
rmse_original = np.sqrt(np.mean((true_values_original - predicted_values_original)**2))

# 计算对比数据的统计量
n_compare = len(true_values_compare)
slope_compare, intercept_compare, r_value_compare, _, _ = stats.linregress(true_values_compare, predicted_values_compare)
r_squared_compare = r_value_compare**2
rmse_compare = np.sqrt(np.mean((true_values_compare - predicted_values_compare)**2))

# 回归模型及置信带计算（使用原始数据）
X_original = sm.add_constant(true_values_original)
model_original = sm.OLS(predicted_values_original, X_original)
results_original = model_original.fit()

# 获取预测值和置信区间
predictions_original = results_original.get_prediction()
pred_ci_original = predictions_original.conf_int(alpha=0.05)
pred_ci_lower_original = pred_ci_original[:, 0]
pred_ci_upper_original = pred_ci_original[:, 1]

# 创建画布
plt.figure(figsize=(6, 6), dpi=300)
ax = plt.gca()

# 添加图名
plt.suptitle('Wheat_N2O_LG Values Comparison',
             y=0.92,
             fontsize=14,
             fontweight='bold',
             fontfamily='Arial')

# 计算数据范围（包含置信区间和两组数据）
x_min = max(0, min(true_values_original.min(), pred_ci_lower_original.min(), true_values_compare.min()))
x_max = max(true_values_original.max(), pred_ci_upper_original.max(), true_values_compare.max())
margin = 0.05 * (x_max - x_min)

# 设置坐标轴范围（仅设置一次）
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(x_min - margin, x_max + margin)

# 绘制置信区间（使用fill_between）
sorted_idx = np.argsort(true_values_original)
sorted_x = true_values_original[sorted_idx]
sorted_lower = pred_ci_lower_original[sorted_idx]
sorted_upper = pred_ci_upper_original[sorted_idx]

# 扩展置信区间到坐标轴边界
x_min, x_max = ax.get_xlim()  # 获取当前坐标轴范围
extended_x = np.linspace(x_min, x_max, 500)  # 生成密集的x值

# 使用回归模型预测扩展后的置信区间
extended_X = sm.add_constant(extended_x)
extended_pred = results_original.get_prediction(extended_X)
extended_ci = extended_pred.conf_int(alpha=0.05)
extended_lower = extended_ci[:, 0]
extended_upper = extended_ci[:, 1]

# 绘制延伸到边界的置信区间
ax.fill_between(extended_x,
               extended_lower,
               extended_upper,
               color='#E6A1BA',
               alpha=0.3,
               zorder=2,
               edgecolor='none',
               label='95% CI (Original)')

# 绘制原始数据散点图
ax.scatter(true_values_original, predicted_values_original,
           c='#CC79A7', alpha=0.6, edgecolors='gray', s=100, zorder=3,
           label='Original Data')

# 绘制对比数据散点图
ax.scatter(true_values_compare, predicted_values_compare,
           c='#56B4E9', alpha=0.6, edgecolors='gray', s=100, zorder=3,
           label='Comparison Data')

# 1:1参考线
ax.plot([x_min - margin, x_max + margin],
        [x_min - margin, x_max + margin],
        '--', color='gray', lw=1.5, zorder=1)

# 回归线（原始数据）
reg_x = np.array([x_min - margin, x_max + margin])
reg_y_original = intercept_original + slope_original * reg_x
ax.plot(reg_x, reg_y_original, 'r-', lw=2, zorder=4, label='Regression (Original)')

# 回归线（对比数据，可选）
reg_y_compare = intercept_compare + slope_compare * reg_x
ax.plot(reg_x, reg_y_compare, 'b-', lw=2, zorder=4, label='Regression (Compare)')

# 坐标轴标签
ax.set_xlabel('Observed Values (%)', fontsize=12, labelpad=10)
ax.set_ylabel('Predicted Values (%)', fontsize=12, labelpad=10)
ax.tick_params(axis='both', which='major', length=6, width=1)

# 统计标注
stats_text = (f"Original: $R^2$ = {r_squared_original:.3f}, RMSE = {rmse_original:.2f}\n"
              f"Compare: $R^2$ = {r_squared_compare:.3f}, RMSE = {rmse_compare:.2f}\n"
              f"N (Original) = {n_original}, N (Compare) = {n_compare}")
ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=11,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'})

# 添加图例
ax.legend(loc='upper left', framealpha=0.7)

# 保存图像
plt.savefig('Wheat_N2O_OVP_comparison0612.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            format='png')
plt.tight_layout()