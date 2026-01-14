import pandas as pd
from datetime import datetime, timedelta


def get_months(year, start_day, end_day):
    """根据年份计算两个年日数之间的跨越月份"""
    if pd.isna(year) or pd.isna(start_day) or pd.isna(end_day):
        return []

    try:
        year = int(year)
        start_day = int(start_day)
        end_day = int(end_day)
    except:
        return []

    months = []

    try:
        # 处理闰年（datetime会自动处理）
        base_date = datetime(year, 1, 1)
        start_date = base_date + timedelta(days=start_day - 1)

        # 处理跨年情况
        if start_day > end_day:
            # 分两段处理：当前年份的start_date到年底，以及下一年年初到end_date
            end_date = datetime(year + 1, 1, 1) + timedelta(days=end_day - 1)
        else:
            end_date = base_date + timedelta(days=end_day - 1)
    except:
        return []

    # 收集所有日期对应的月份
    current_date = start_date
    end_date_inclusive = end_date + timedelta(days=1)  # 包含最后一天

    while current_date < end_date_inclusive:
        months.append(current_date.month)
        current_date += timedelta(days=1)

    # 去重并保持时间顺序
    seen = set()
    unique_months = []
    for m in months:
        if m not in seen:
            seen.add(m)
            unique_months.append(m)

    # 处理跨年时的月份排序
    if start_date.year != end_date.year:
        # 分离年份的月份
        year1_months = [m for m in unique_months if m >= start_date.month]
        year2_months = [m for m in unique_months if m < start_date.month]
        return year1_months + year2_months
    return unique_months


# 读取数据
df = pd.read_excel('Dataset_upd_ro_sro_slp_LAI_fd_rx1_rx5_ddp_spei_Hddw_Cddw_phase_mon_Hddm_Cddm.xls')  # 替换为实际文件名

# # 定义各作物类型前缀
# crop_groups = [
#     'mai_ir', 'mai_rf',  # 玉米
#     'wwh_ir', 'wwh_rf',  # 冬小麦
#     'swh_ir', 'swh_rf'  # 春小麦
# ]
# 定义各作物类型前缀
crop_groups = [
    'mai_ir'  # 玉米
]

# 为每个作物类型计算月份
for prefix in crop_groups:
    pla_col = f'{prefix}_pla'  # 播种列
    mat_col = f'{prefix}_mat'  # 成熟列
    new_col = f'{prefix}_months'  # 结果列

    # 应用计算函数（传入年份列）
    df[new_col] = df.apply(
        lambda row: get_months(row['year'], row[pla_col], row[mat_col]),
        axis=1
    )

# 保存结果
df.to_excel('new_maize_phase.xlsx', index=False)
