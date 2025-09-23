import pandas as pd

# 1. 读取CSV文件（假设文件名为'crop_data.csv'）
df = pd.read_csv('Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv')
# 2. 统计各组样本数
group_counts = df.groupby(['Crop type', 'Fertilizer type']).size()
print("各组样本数统计：")
print(group_counts)

# 3. 查看各组产量的均值/方差
yield_stats = df.groupby(['Crop type', 'Fertilizer type'])['N2OEF'].agg(['mean', 'std'])
print("\n各组N2OEF统计（均值/标准差）：")
print(yield_stats)


# 若某些组的Yield均值差异大（如小麦_化肥 vs 水稻_有机肥），分层更重要。