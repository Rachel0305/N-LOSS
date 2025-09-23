import os


def print_filenames(folder_path):
    """
    打印文件夹内所有文件的文件名

    参数:
        folder_path (str): 要扫描的文件夹路径
    """
    # 遍历文件夹中的所有条目
    for entry in os.listdir(folder_path):
        # 获取条目的完整路径
        full_path = os.path.join(folder_path, entry)

        # 检查是否是文件（不是文件夹）
        if os.path.isfile(full_path):
            print(entry)


# 使用示例 - 替换为你的文件夹路径
folder_path = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_n2o_predvars_test0616"  # 当前目录
print_filenames(folder_path)