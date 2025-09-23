# # 主导驱动因子图例 4*9
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# # 数据
# labels = [
#     "MAP", "MAT", "Sro_Span_max", "Sro_Span_min", "Sro_Span_mean", "Ddp_Span_max",
#     "Ddp_Span_min", "Ddp_Span_mean", "Spei_Span_max", "Spei_Span_min", "Spei_Span_mean", "Till",
#     "Irrigated", "Sand", "Clay", "Slope", "pH", "Bulk density",
#     "SOC", "C/N", "N input rate", "LAI_Span_max", "LAI_Span_min", "LAI_Span_mean",
#     "Rx1_Span_max", "Rx1_Span_min", "Rx1_Span_mean", "Rx5_Span_max", "Rx5_Span_min", "Rx5_Span_mean",
#     "Hdd_Span_max", "Hdd_Span_min", "Hdd_Span_mean", "Cdd_Span_max", "Cdd_Span_min", "Cdd_Span_mean"
# ]
# colors = [
#     "#1F77B4", "#2C7BB6", "#5E9ED6", "#87B7E0", "#4590B9", "#1764AB",
#     "#3273BF", "#5087C7", "#74A9CF", "#97BBCD", "#64A0C8", "#FF7F0E",
#     "#FF9933", "#8C510A", "#A16216", "#7F7F7F", "#B87333", "#875C26",
#     "#C88746", "#2CA02C", "#4DAF4A", "#4D9221", "#66A63A", "#40821B",
#     "#D62728", "#ED4142", "#E13334", "#E64959", "#F2656F", "#EB5764",
#     "#D94021", "#F05A3B", "#E44D2E", "#C83042", "#E05062", "#D44052"
# ]
#
# # 创建图表和轴
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_xlim(0, 4)
# ax.set_ylim(0, 9)
# ax.axis('off')
#
# # 添加长方形色块和标签，4列，每列9个，调整长宽比例避免文字重叠
# for i, (label, color) in enumerate(zip(labels, colors)):
#     col = i // 9
#     row = i % 9
#     rect = patches.Rectangle((col * 0.8 + 0.1, 8 - row), 0.2, 0.5, linewidth=1, edgecolor='black', facecolor=color)
#     ax.add_patch(rect)
#     ax.text(col * 0.8 +0.35, 8 - row + 0.3, label, va='center', ha='left')
#
# # 调整布局
# plt.tight_layout()
# plt.savefig('wheat_domin_legend.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close()

# 主导驱动因子图例 9*4
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# # 数据
# labels = [
#     "MAP", "MAT", "Sro_Span_max", "Sro_Span_min", "Sro_Span_mean", "Ddp_Span_max",
#     "Ddp_Span_min", "Ddp_Span_mean", "Spei_Span_max", "Spei_Span_min", "Spei_Span_mean", "Till",
#     "Irrigated", "Sand", "Clay", "Slope", "pH", "Bulk density",
#     "SOC", "C/N", "N input rate", "LAI_Span_max", "LAI_Span_min", "LAI_Span_mean",
#     "Rx1_Span_max", "Rx1_Span_min", "Rx1_Span_mean", "Rx5_Span_max", "Rx5_Span_min", "Rx5_Span_mean",
#     "Hddw_Span_max", "Hddw_Span_min", "Hddw_Span_mean", "Cddw_Span_max", "Cddw_Span_min", "Cddw_Span_mean",
#     "Hddm_Span_max", "Hddm_Span_min", "Hddm_Span_mean", "Cddm_Span_max", "Cddm_Span_min", "Cddm_Span_mean",
# ]
# colors = [
#     "#1F77B4", "#2C7BB6", "#5E9ED6", "#87B7E0", "#4590B9", "#1764AB",
#     "#3273BF", "#5087C7", "#74A9CF", "#97BBCD", "#64A0C8", "#FF7F0E",
#     "#FF9933", "#8C510A", "#A16216", "#7F7F7F", "#B87333", "#875C26",
#     "#C88746", "#2CA02C", "#4DAF4A", "#4D9221", "#66A63A", "#40821B",
#     "#D62728", "#ED4142", "#E13334", "#E64959", "#F2656F", "#EB5764",
#     "#D94021", "#F05A3B", "#E44D2E", "#C83042", "#E05062", "#D44052",
#     '#CC3319', '#ED664A', '#DC4D31', '#BA2B3D', '#DB5E6E', '#CE4555'
# ]
#
# # 创建图表和轴
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.set_xlim(0, 9)
# ax.set_ylim(0, 4)
# ax.axis('off')
#
# # 添加标题
# # ax.text(4.5, 3.7, "主导驱动因子图例", ha='center', va='top', fontsize=12, fontweight='bold')
#
# # 添加长方形色块和标签，4行9列
# for i, (label, color) in enumerate(zip(labels, colors)):
#     row = i // 9
#     col = i % 9
#     rect = patches.Rectangle((col * 0.9 + 0.1, 3 - row), 0.7, 0.5, linewidth=1, edgecolor='black', facecolor=color)
#     ax.add_patch(rect)
#     ax.text(col * 0.9 + 0.5, 3 - row - 0.25, label, va='center', ha='center')
#
# # 调整布局
# plt.tight_layout()
# plt.savefig('wheat_domin_legend.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close()

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# # 数据
# labels = [
#     "MAP", "MAT", "Sro_Span_max", "Sro_Span_min", "Sro_Span_mean", "Ddp_Span_max",
#     "Ddp_Span_min", "Ddp_Span_mean", "Spei_Span_max", "Spei_Span_min", "Spei_Span_mean", "Till",
#     "Irrigated", "Sand", "Clay", "Slope", "pH", "Bulk density",
#     "SOC", "C/N", "N input rate", "LAI_Span_max", "LAI_Span_min", "LAI_Span_mean",
#     "Rx1_Span_max", "Rx1_Span_min", "Rx1_Span_mean", "Rx5_Span_max", "Rx5_Span_min", "Rx5_Span_mean",
#     "Hddw_Span_max", "Hddw_Span_min", "Hddw_Span_mean", "Cddw_Span_max", "Cddw_Span_min", "Cddw_Span_mean",
#     "Hddm_Span_max", "Hddm_Span_min", "Hddm_Span_mean", "Cddm_Span_max", "Cddm_Span_min", "Cddm_Span_mean",
# ]
# colors = [
# '#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#FF00FF','#800000','#008000','#000080','#808000','#008080','#800080',
# '#C0C0C0','#808080','#993366','#339966','#999933','#336699','#A0522D','#D8BFD8','#6A5ACD','#FFA500','#FFC0CB','#4B0082',
# '#ADFF2F','#D2691E','#DC143C','#00BFFF','#32CD32','#8A2BE2','#FFD700','#F08080','#00FA9A','#BA55D3','#F4A460','#9370DB',
# '#3CB371','#7B68EE','#FF6347','#00CED1','#C71585','#FF8C00'
# ]
#
# # 创建图表和轴
# fig, ax = plt.subplots(figsize=(16, 5))
# ax.set_xlim(0, 14)
# ax.set_ylim(0, 3)
# ax.axis('off')
#
# # 添加标题
# ax.text(7, 2.7, "Domin Drivers", ha='center', va='top', fontsize=12, fontweight='bold')
#
# # 添加长方形色块和标签，3行14列
# for i, (label, color) in enumerate(zip(labels, colors)):
#     row = i // 14
#     col = i % 14
#     rect = patches.Rectangle((col * 1.0 + 0.1, 2 - row), 0.8, 0.3, linewidth=1, edgecolor='black', facecolor=color)
#     ax.add_patch(rect)
#     ax.text(col * 1.0 + 0.48, 2 - row - 0.15, label, va='center', ha='center', fontsize=8)
#
# # 调整布局
# plt.tight_layout()
# plt.savefig('domin_legend0907.png', dpi=600, bbox_inches='tight')
# # plt.show()
# plt.close()


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 数据
labels = [
    'MAP', 'MAT', 'Sand', 'Clay', 'Slope', 'pH', 'Bulk density', 'SOC', 'CN ratio',
        'N input rate', 'Tillage', 'Irrigation',
        'Sro span max', 'Sro span min', 'Sro key mean',
        'LAI span max', 'LAI span min', 'LAI key mean',
        'Rx1 span max', 'Rx1 span min', 'Rx1 key mean',
        'Rx5 span max', 'Rx5 span min', 'Rx5 key mean',
        'Ddp span max', 'Ddp span min', 'Ddp key mean',
        'Spei span max', 'Spei span min', 'Spei key mean',
        'SMs span max', 'SMs span min', 'SMs key mean',
        'SMrz span max', 'SMrz span min', 'SMrz key mean',
        'Hddw span max', 'Hddw span min', 'Hddw key mean',
        'Cddw span max', 'Cddw span min', 'Cddw key mean',
        'Hddm span max', 'Hddm span min', 'Hddm key mean',
        'Cddm span max', 'Cddm span min', 'Cddm key mean',
        'Fertilizer type'  # 加入肥料类型
]
colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000',
    '#8000FF', '#0080FF', '#FF0080', '#80FF00', '#00FF80', '#800000', '#008000',
    '#000080', '#808000', '#800080', '#008080', '#FF4000', '#0040FF', '#FF0040',
    '#ccac60', '#0040FF', '#FF00BF', '#BF00FF', '#00BFFF', '#FFBF00', '#00FFBF',
    '#BF00BF', '#00BFBF', '#BFBF00', '#4000FF', '#FF0040', '#0040FF', '#FF00BF',
    '#BF00FF', '#00BFFF', '#FFBF00', '#00FFBF', '#BF00BF', '#00BFBF', '#BFBF00',
    '#FF1493', '#8B0000', '#006400',  '#00008B', '#8B008B', '#8B4513', '#008B8B',
    '#FF8C00'
]

# 创建图表和轴 - 增大图形高度以容纳更大的行间距
fig, ax = plt.subplots(figsize=(14, 10))  # 增加高度从6到10
ax.set_xlim(0, 7)
ax.set_ylim(0, 9)  # 增加Y轴范围从7到9，提供更多垂直空间
ax.axis('off')

# 添加标题
ax.text(3.5, 9.5, "Domin Drivers", ha='center', va='top', fontsize=14, fontweight='bold')

# 添加长方形色块和标签，7行7列 - 增大行间距
for i, (label, color) in enumerate(zip(labels, colors)):
    row = i // 7
    col = i % 7
    # 增大行间距：将Y坐标乘以1.2倍，增加垂直间距
    y_position = 8 - row * 1.2  # 从6改为8，并乘以1.2增大间距

    # 色块
    rect = patches.Rectangle((col * 1.0 + 0.1, y_position + 0.4), 0.6, 0.6,
                             linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)

    # 标签 - 调整到色块下方更远的位置
    ax.text(col * 1.0 + 0.4, y_position + 0.2, label,  # 从-0.2改为-0.4，增大间距
            va='center', ha='center', fontsize=12, color='black')  # 稍微增大字体

# 调整布局
plt.tight_layout()
plt.savefig('domin_legend0907.png', dpi=600, bbox_inches='tight')
plt.close()