import pandas as pd
import ast


def calculate_features(df):
    """特征计算主函数"""
    # 转换月份列格式
    df['Span_Months'] = df['Span_Months'].apply(ast.literal_eval)
    df['Key_Months'] = df['Key_Months'].apply(ast.literal_eval)

    # variables = ['sro', 'LAI', 'rx1', 'rx5', 'Ddp', 'spei', 'Hddm', 'Cddm', 'Hddw', 'Cddw', 'sm']
    variables = ['sm', 'SMrz']


    for var in variables:
        # 生成月份列名列表（验证列是否存在）
        month_columns = [f"{var}_Month_{m}" for m in range(1, 13)]
        valid_columns = [col for col in month_columns if col in df.columns]

        # 计算Span Months的特征
        df[f'{var}_Span_max'] = df.apply(lambda row:
                                         row[[f"{var}_Month_{m}" for m in row['Span_Months'] if
                                              f"{var}_Month_{m}" in df.columns]].max(),
                                         axis=1)

        df[f'{var}_Span_min'] = df.apply(lambda row:
                                         row[[f"{var}_Month_{m}" for m in row['Span_Months'] if
                                              f"{var}_Month_{m}" in df.columns]].min(),
                                         axis=1)

        # 计算Key Months的特征
        df[f'{var}_Key_mean'] = df.apply(lambda row:
                                         row[[f"{var}_Month_{m}" for m in row['Key_Months'] if
                                              f"{var}_Month_{m}" in df.columns]].mean(),
                                         axis=1)

    return df


# 文件路径配置（根据实际情况修改）
input_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_SMs_yield_SMrz0901.csv'  # 输入文件路径
output_path = 'Dataset_output_fea0901.csv'  # 输出文件路径

if __name__ == "__main__":
    # 读取CSV数据（支持多种编码格式）
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except:
        df = pd.read_csv(input_path, encoding='gbk')

    # 执行特征计算
    result_df = calculate_features(df)

    # 导出结果（保留原始索引，使用UTF-8编码）
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"数据处理完成！结果已保存至：{output_path}")