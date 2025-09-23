import pandas as pd


def get_month_sequence(start, end):
    """生成从播种到收获的月份序列（包含起止）"""
    if start <= end:
        return list(range(start, end + 1))
    else:
        return list(range(start, 13)) + list(range(1, end + 1))


# 读取数据
df = pd.read_excel('new_maize_phase.xlsx')

# 生成月份序列和特征列名
month_sequences = []
feature_columns = []

for idx, row in df.iterrows():
    # 选择对应的作物列
    if row['Crop type'] == 'Wheat':
        start = int(row['Whe_plan_CJ'])
        end = int(row['Whe_harv_CJ'])
    elif row['Crop type'] == 'Maize':
        start = int(row['Mai_plan'])
        end = int(row['Mai_harv'])
    else:
        month_sequences.append([])
        feature_columns.append([])
        continue

    # 生成月份序列
    months = get_month_sequence(start, end)
    month_sequences.append(months)

    # 生成特征列名（按月份排序）
    sorted_months = sorted(months, key=lambda x: x if x >= start else x + 12)
    feature_columns.append([f'ro_Month_{m}' for m in sorted_months])

# 添加新列
df['Span_Months'] = month_sequences
df['Feature_Columns'] = feature_columns

# 保存结果
df.to_excel('new_maize_phase_month.xlsx', index=False)

print("处理完成！生成的字段包含：")
print("- Span_Months: 跨越的月份列表（按时间顺序）")
print("- Feature_Columns: 对应的特征列名列表")