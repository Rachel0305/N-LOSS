import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 模拟数据
np.random.seed(42)
x1 = np.linspace(10, 100, 100).reshape(-1, 1)  # 城市A：小尺度，自变量
y1 = x1.flatten() + np.random.normal(0, 5, 100)  # 城市A：因变量，一维数组
x2 = np.linspace(50, 500, 100).reshape(-1, 1)  # 城市B：大尺度，自变量
y2 = 0.7 * x2.flatten() + 50 + np.random.normal(0, 15, 100)  # 城市B：因变量，一维数组

# 回归模型
model1 = LinearRegression().fit(x1, y1)
model2 = LinearRegression().fit(x2, y2)

# 计算R²和RMSE
y1_pred = model1.predict(x1)
y2_pred = model2.predict(x2)
r2_1 = r2_score(y1, y1_pred)
r2_2 = r2_score(y2, y2_pred)
rmse_1 = np.sqrt(mean_squared_error(y1, y1_pred))
rmse_2 = np.sqrt(mean_squared_error(y2, y2_pred))

# 提取标量值并打印
print(f"城市A: R²={r2_1:.3f}, RMSE={rmse_1:.3f}, 斜率={float(model1.coef_[0]):.3f}, 截距={float(model1.intercept_):.3f}")
print(f"城市B: R²={r2_2:.3f}, RMSE={rmse_2:.3f}, 斜率={float(model2.coef_[0]):.3f}, 截距={float(model2.intercept_):.3f}")

# 绘制散点图和拟合线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x1, y1, label="数据")
plt.plot(x1, y1_pred, color="red", label="拟合线")
plt.plot([10, 100], [10, 100], "k--", label="1:1线")
plt.title("城市A")
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x2, y2, label="数据")
plt.plot(x2, y2_pred, color="red", label="拟合线")
plt.plot([50, 500], [50, 500], "k--", label="1:1线")
plt.title("城市B")
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.legend()
plt.tight_layout()
plt.show()