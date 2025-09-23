import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def remove_outliers(data, y_var, n_std=3):
    """移除超出n个标准差的异常值"""
    mean = data[y_var].mean()
    std = data[y_var].std()
    cutoff = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    return data[(data[y_var] >= lower) & (data[y_var] <= upper)].copy()


def read_data(file_path):
    return pd.read_csv(file_path)


def create_output_dir(dir_name='output_plots0801'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def plot_scatter(data, x_var, y_var, crop_type, output_dir):
    plt.figure(figsize=(10, 6))
    crop_data = data[data['Crop type'] == crop_type]

    # 移除异常值
    crop_data_clean = remove_outliers(crop_data, y_var)

    sns.scatterplot(x=x_var, y=y_var, data=crop_data_clean, alpha=0.7, color='blue')

    plt.title(f'{crop_type} - {y_var} vs {x_var} (Scatter)')
    plt.xlabel(x_var, fontsize=10)
    plt.ylabel(y_var, fontsize=10)

    # 添加半透明网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 调整刻度间距
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(20))

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    filename = f"{output_dir}/Scatter_{crop_type}_{y_var}_vs_{x_var}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_line(data, x_var, y_var, crop_type, output_dir):
    plt.figure(figsize=(10, 6))
    crop_data = data[data['Crop type'] == crop_type]

    # 移除异常值并按x_var排序
    crop_data_clean = remove_outliers(crop_data, y_var).sort_values(by=x_var)

    sns.lineplot(x=x_var, y=y_var, data=crop_data_clean,
                 color='red', marker='o', ci=None)

    plt.title(f'{crop_type} - {y_var} vs {x_var} (Line)')
    plt.xlabel(x_var, fontsize=10)
    plt.ylabel(y_var, fontsize=10)

    # 添加半透明网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 调整刻度间距
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(30))

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    filename = f"{output_dir}/Line_{crop_type}_{y_var}_vs_{x_var}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    file_path = 'Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_sm_yield.csv'
    data = read_data(file_path)
    output_dir = create_output_dir()

    variable_combinations = {
        'Wheat': {
            # # 'N2OEF': ['MAP', 'Bulk density', 'spei_Key_mean', 'Ddp_Key_mean', 'Hddw_Span_max'],
            # # 'NH3EF': ['Sand', 'N input rate', 'LAI_Span_max', 'Clay', 'SOC'],
            # 'NOEF': ['sro_Span_max'],
            # # 'LF': ['C_N'],
            # # 'RF': ['Ddp_Span_min', 'N input rate', 'Sand', 'rx5_Key_mean', 'slope'],
            # 'NUE': ['sro_Key_mean']
            'N2OEF': ['rx5_Span_max'],
            # 'NH3EF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'NOEF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'LF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'RF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'NUE': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean']
        },
        'Maize': {
            # 'N2OEF': ['sro_Key_mean'],
            # # 'NH3EF': ['Clay', 'LAI_Key_mean', 'N input rate', 'spei_Span_min', 'MAP'],
            # # 'NOEF': ['LAI_Span_min', 'rx5_Span_max', 'slope', 'Hddm_Span_max', 'N input rate'],
            # 'LF': ['sro_Span_min'],
            # 'RF': ['sro_Key_mean'],
            # 'NUE': ['sro_Span_max']
            # 'N2OEF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'NH3EF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'NOEF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'LF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'RF': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean'],
            # 'NUE': ['MAP', 'MAT', 'Sand', 'Clay', 'slope', 'pH', 'Bulk density', 'SOC', 'C_N', 'N input rate', 'Till', 'Irrigated', 'LAI_Key_mean']
        }
    }

    for crop, y_vars in variable_combinations.items():
        for y_var, x_vars in y_vars.items():
            for x_var in x_vars:
                plot_scatter(data, x_var, y_var, crop, output_dir)
                plot_line(data, x_var, y_var, crop, output_dir)

    print(f"所有图表已保存到 {output_dir} 目录")
    print(f"共生成 {len(variable_combinations)} 张图片")


if __name__ == '__main__':
    main()