import joblib
import pandas as pd

# 1. 加载模型
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Maize_NOEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\model.pkl")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Wheat_NH3EF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Wheat_LF_RF.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_N2OEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NH3EF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_NOEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_N2OEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_所有变量特征\Maize_NH3EF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0509_GLO_所有变量特征\Maize_NOEF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_LF_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Maize_RF_XGBoost.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Maize_NUE_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\pred_models\Wheat_NUE_model.joblib")
# model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_N2OEF_LightGBM.joblib")
model = joblib.load(r"D:\CNPK\N_CODE_PY\N_ML_V2\RF_model0810\Wheat\N2OEF\model.joblib")

# 2. 获取模型存储的变量顺序
if hasattr(model, 'feature_names_in_'):
    feature_order = model.feature_names_in_
elif hasattr(model, 'feature_names'):
    feature_order = model.feature_names
elif hasattr(model, 'steps'):  # Pipeline 模型
    final_estimator = model.steps[-1][1]
    if hasattr(final_estimator, 'feature_names_in_'):
        feature_order = final_estimator.feature_names_in_
else:
    # 如果没有存储顺序，需手动指定（训练时的列顺序）
    feature_order = ['A', 'B', 'C', 'D']

print("模型要求的变量顺序：", feature_order)

# 3. 准备新数据（假设列名不同）
new_data = pd.DataFrame({
    'X': [1],  # 对应 'A'
    'Y': [2],  # 对应 'B'
    'Z': [3],  # 对应 'C'
    'W': [4]   # 对应 'D'
})

# 4. 列名映射（新列名 → 模型列名）
column_mapping = {'X': 'A', 'Y': 'B', 'Z': 'C', 'W': 'D'}

# 5. 重命名 + 排序
new_data = new_data.rename(columns=column_mapping)
new_data = new_data[feature_order]

# 6. 预测
predictions = model.predict(new_data)
print("预测结果：", predictions)