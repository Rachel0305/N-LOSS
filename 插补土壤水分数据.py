import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


def spatial_temporal_fill(input_path, output_path):
    """基于时空最近邻的土壤湿度缺失值填充（修正距离计算版本）"""
    # 读取数据并校验列名
    df = pd.read_csv(input_path)
    required_columns = {'year', 'Longitude', 'Latitude'} | {f'SMrz_Month_{i}' for i in range(1, 13)}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"缺少必要列: {missing}")

    # 分离完整/缺失数据
    sm_columns = [f'SMrz_Month_{i}' for i in range(1, 13)]
    complete_mask = df[sm_columns].notnull().all(axis=1)

    if complete_mask.sum() == 0:
        raise ValueError("数据集中不存在完整记录，无法进行插值")

    complete_df = df[complete_mask].copy()
    missing_df = df[~complete_mask].copy()

    # 构建空间索引
    coord_complete = np.deg2rad(complete_df[['Latitude', 'Longitude']])
    tree = BallTree(coord_complete, metric='haversine')
    EARTH_RADIUS = 6371  # 地球半径，单位：公里

    # 对每个缺失记录进行插值
    for idx, row in missing_df.iterrows():
        current_coord = np.deg2rad([[row['Latitude'], row['Longitude']]])
        current_year = row['year']

        # 阶段1：优先查找50公里内点
        indices = tree.query_radius(current_coord, r=50 / EARTH_RADIUS)
        candidates = complete_df.iloc[indices[0]] if len(indices[0]) > 0 else None

        # 阶段2：若范围无数据，查找最近点
        if candidates is None or len(candidates) == 0:
            dist, idx_nearest = tree.query(current_coord, k=1)
            candidates = complete_df.iloc[idx_nearest[0]]
            candidates['distance'] = dist[0] * EARTH_RADIUS
        else:
            # 精确计算候选点实际距离
            lat_current, lon_current = row['Latitude'], row['Longitude']
            lat_candidates = candidates['Latitude'].values
            lon_candidates = candidates['Longitude'].values

            # Haversine距离计算
            dlat = np.radians(lat_candidates - lat_current)
            dlon = np.radians(lon_candidates - lon_current)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat_current)) * np.cos(np.radians(lat_candidates)) * np.sin(
                dlon / 2) ** 2
            distances = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
            candidates['distance'] = distances

        # 时空排序
        candidates['year_diff'] = np.abs(candidates['year'] - current_year)
        candidates = candidates.sort_values(by=['distance', 'year_diff'])

        # 填充数据
        best = candidates.iloc[0]
        df.loc[idx, sm_columns] = best[sm_columns].values

    df.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    spatial_temporal_fill(
        input_path="Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phaseggcmi_mon_Hddm_Cddm_SMs_yield_SMrz.csv",
        output_path="filled_data0831.csv"
    )