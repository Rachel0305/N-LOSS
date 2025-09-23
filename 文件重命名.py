import os
import glob

input_folder = r"D:\CNPK\N_CODE_PY\N_ML\pred_models\wheat_n2o_predvars_test0602"
# 文件名映射
old_to_new = {
    "Column_0_pre_mean.tif": "Column_1_MAP.tif",
    "Column_1_tmp_mean.tif": "Column_2_MAT.tif",
    "Column_2_Sand.tif": "Column_3_Sand.tif",
    "Column_3_Clay.tif": "Column_4_Clay.tif",
    "Column_4_Slope.tif": "Column_5_Slope.tif",
    "Column_5_Ph.tif": "Column_6_pH.tif",
    "Column_6_Bulk.tif": "Column_7_Bulk density.tif",
    "Column_7_Soc.tif": "Column_8_SOC.tif",
    "Column_8_CN.tif": "Column_9_CN.tif",
    "Column_9_Wheat_NinSum.tif": "Column_10_N input rate.tif",
    "Column_10_sro_Span_max.tif": "Column_11_sro_Span_max.tif",
    "Column_11_sro_Span_min.tif": "Column_12_sro_Span_min.tif",
    "Column_12_sro_Span_mean.tif": "Column_13_sro_Key_mean.tif",
    "Column_13_LAIw_Span_max.tif": "Column_14_LAI_Span_max.tif",
    "Column_14_LAIw_Span_min.tif": "Column_15_LAI_Span_min.tif",
    "Column_15_LAIw_Span_mean.tif": "Column_16_LAI_Key_mean.tif",
    "Column_16_rx1w_Span_max.tif": "Column_17_rx1_Span_max.tif",
    "Column_17_rx1w_Span_min.tif": "Column_18_rx1_Span_min.tif",
    "Column_18_rx1w_Span_mean.tif": "Column_19_rx1_Key_mean.tif",
    "Column_19_rx5w_Span_max.tif": "Column_20_rx5_Span_max.tif",
    "Column_20_rx5w_Span_min.tif": "Column_21_rx5_Span_min.tif",
    "Column_21_rx5w_Span_mean.tif": "Column_22_rx5_Key_mean.tif",
    "Column_22_Ddpw_Span_max.tif": "Column_23_Ddp_Span_max.tif",
    "Column_23_Ddpw_Span_min.tif": "Column_24_Ddp_Span_min.tif",
    "Column_24_Ddpw_Span_mean.tif": "Column_25_Ddp_Key_mean.tif",
    "Column_25_speiw_Span_max.tif": "Column_26_spei_Span_max.tif",
    "Column_26_speiw_Span_min.tif": "Column_27_spei_Span_min.tif",
    "Column_27_speiw_Span_mean.tif": "Column_28_spei_Key_mean.tif",
    "Column_28_Hddw_Span_max.tif": "Column_29_Hddw_Span_max.tif",
    "Column_29_Hddw_Span_min.tif": "Column_30_Hddw_Span_min.tif",
    "Column_30_Hddw_Span_mean.tif": "Column_31_Hddw_Key_mean.tif",
    "Column_31_Cddw_Span_max.tif": "Column_32_Cddw_Span_max.tif",
    "Column_32_Cddw_Span_min.tif": "Column_33_Cddw_Span_min.tif",
    "Column_33_Cddw_Span_mean.tif": "Column_34_Cddw_Key_mean.tif"
}

# 批量重命名
for old_name, new_name in old_to_new.items():
    old_path = os.path.join(input_folder, old_name)
    new_path = os.path.join(input_folder, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"已将 {old_name} 重命名为 {new_name}")
    else:
        print(f"文件 {old_name} 不存在，请检查")

print("重命名完成！")