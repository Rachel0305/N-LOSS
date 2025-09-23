import joblib


def get_model_feature_order(model_path):
    """
    从模型中提取变量顺序（特征名称）

    参数:
        model_path (str): 模型文件路径（.joblib 或 .pkl）

    返回:
        list: 模型要求的变量顺序（特征名称列表）
    """
    # 1. 加载模型
    model = joblib.load(model_path)

    # 2. 尝试从不同属性中提取特征顺序
    feature_order = None

    # 检查 sklearn 的 feature_names_in_（≥1.0 版本）
    if hasattr(model, 'feature_names_in_'):
        feature_order = list(model.feature_names_in_)

    # 检查 XGBoost/LightGBM 的 feature_names
    elif hasattr(model, 'feature_names'):
        feature_order = list(model.feature_names)

    # 如果是 Pipeline，检查最终 estimator 的特征顺序
    elif hasattr(model, 'steps'):
        final_step = model.steps[-1][1]  # 获取最后一个步骤（通常是模型）
        if hasattr(final_step, 'feature_names_in_'):
            feature_order = list(final_step.feature_names_in_)

    # 如果仍未找到，抛出错误
    if feature_order is None:
        raise ValueError(
            "无法自动提取特征顺序！此模型可能未存储特征名称。\n"
            "请手动记录训练时的列顺序，或升级 sklearn 到 ≥1.0 版本。"
        )

    return feature_order


# 使用示例
if __name__ == "__main__":
    # 替换为你的模型路径
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0529_GLO_所有变量特征wheat\Wheat\N2OEF\model.pkl"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_model_C_new_results0529_GLO_前20个变量特征\Wheat_NOEF_model.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_models0509_GLO_前20个变量特征\Maize_RF_XGBoost.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0508_GLO_所有变量特征\Wheat_N2OEF_model.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Wheat_N2OEF_LightGBM.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Maize_LF_LightGBM.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Wheat_RF_model.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\XGB_model_C_new_results0529_GLO_仿照0414PPT03\Maize_LF_model.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\LG_models0509_GLO_前20个变量特征\Maize_LF_LightGBM.joblib"
    # model_path = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0509_GLO_所有变量特征\Maize\LF\model.pkl"
    model_path = r"D:\CNPK\N_CODE_PY\N_ML\RF_model_C_new_results0508_GLO\Maize\LF\model.pkl"


    try:
        # 获取模型要求的变量顺序
        feature_order = get_model_feature_order(model_path)
        print("模型要求的变量顺序：")
        print("\n".join(feature_order))

        # 可选：保存到文件（方便后续使用）
        with open("model_feature_order.txt", "w") as f:
            f.write("\n".join(feature_order))
        print("\n变量顺序已保存到 model_feature_order.txt")

    except Exception as e:
        print(f"错误：{e}")