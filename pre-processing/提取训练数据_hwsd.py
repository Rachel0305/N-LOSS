import pandas as pd

# 读取SMU数据
smu_data = pd.read_excel(r"C:\Users\DELL\Downloads\HWSD2_DB\HWSD2_SMU.xlsx")
# 读取LAYERS数据
layers_data = pd.read_excel(r"C:\Users\DELL\Downloads\HWSD2_DB\HWSD2_LAYERS.xlsx")
# 初始化一个空的DataFrame来存储结果
select_data_all = pd.DataFrame()

# 遍历SMU数据
for index, row in smu_data.iterrows():
    smu_id = row['HWSD2_SMU_ID']
    fao90 = row['FAO90']
    share = row['SHARE']
    print(index, flush=True)

    # 根据条件筛选LAYERS数据
    lookup = layers_data[
        (layers_data['HWSD2_SMU_ID'] == smu_id) &
        (layers_data['FAO90'] == fao90) &
        (layers_data['SHARE'] == share)
        ]
    # 追加筛选后的数据
    select_data_all = pd.concat([select_data_all, lookup], ignore_index=True)

# 写出结果
with pd.ExcelWriter(r'C:\Users\DELL\Downloads\HWSD2_DB\HWSD2_LAYERS_alllayers.xlsx') as writer:
    select_data_all.to_excel(writer, sheet_name='All_LAYERS', index=False)
    for d in range(1, 8):
        sheet_name = f'D{d}'
        select_data = select_data_all[select_data_all['LAYER'] == sheet_name]
        select_data.to_excel(writer, sheet_name=sheet_name, index=False)

print("finished")
