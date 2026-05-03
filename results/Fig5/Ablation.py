import pandas as pd
import re
from plotnine import *

# ==============================
# 1. 通用指标提取函数
# ==============================
def extract_metric(text, name):
    if pd.isna(text): return None
    # 清理非 ASCII 字符防止乱码
    text_clean = str(text).encode('ascii', 'ignore').decode('ascii')
    match = re.search(rf"{name}\s*=\s*(\d+\.\d+)", text_clean)
    return float(match.group(1)) if match else None

# ==============================
# 2. 处理 test3.csv (视觉大改版：Dark2 配色 + 多线型)
# ==============================
df3_raw = pd.read_csv("../test3.csv", names=["size", "metrics"])
df3_raw['size'] = pd.to_numeric(df3_raw['size'], errors='coerce')

metrics_list = ["F1", "AUC", "AUPRC", "MCC", "r"]
for m in metrics_list:
    df3_raw[m] = df3_raw['metrics'].apply(lambda x: extract_metric(x, m))

df3_clean = df3_raw.dropna(subset=['size']).sort_values('size')
df3_clean['size'] = pd.Categorical(df3_clean['size'].astype(int).astype(str), 
                                 categories=df3_clean['size'].astype(int).astype(str).unique())

df3_long = df3_clean.melt(id_vars=['size'], value_vars=metrics_list, var_name='Metric', value_name='Value')

# ⭐ 自定义 test3 的形状，确保每个指标形状都不同
test3_shapes = {"F1": "o", "AUC": "s", "AUPRC": "D", "MCC": "^", "r": "v"}
# ⭐ 自定义线型：MCC 虚线，r 点划线，其他实线
test3_lines = {"MCC": "dashed", "r": "dashdot", "F1": "solid", "AUC": "solid", "AUPRC": "dashed"}

p3 = (
    ggplot(df3_long, aes(x='size', y='Value', color='Metric', group='Metric', shape='Metric', linetype='Metric'))
    + geom_line(size=1.5, alpha=0.8)
    + geom_point(size=6)
    + scale_color_brewer(type='qual', palette='Dark2') # ⭐ 换成 Dark2，与 test10 区分
    + scale_shape_manual(values=test3_shapes)
    + scale_linetype_manual(values=test3_lines)
    + theme_bw()
    + labs(title="A549: Multi-Metric Evaluation", x="Center region size", y="Metric value")
    + theme(
        text=element_text(family="sans-serif", size=20),
        axis_text=element_text(color="black", size=20),
        axis_title=element_text(weight='bold', size=20),
        legend_key_height=35, # ⭐ 更大的图例间距
        plot_title=element_text(ha='center', size=20, weight='bold')
    )
)
p3.save("test3_distinct_style.pdf", width=8, height=6)

# ==============================
# 3. 处理 test10.csv (保持原样：Set1 + 分面)
# ==============================
df = pd.read_csv("../test4.csv", index_col=0)

import pandas as pd
import re
import plotnine as p9
from plotnine import *

# 1. 解析函数：同时提取 F1 和 AUPRC
def extract_metrics(text):
    F1 = re.search(r'F1=([\d\.]+)', str(text))
    auprc = re.search(r'AUPRC=([\d\.]+)', str(text))
    return float(F1.group(1)) if F1 else None, float(auprc.group(1)) if auprc else None

# 假设 df 是读取后的原始 DataFrame
plot_rows = []
for full_name, row in df.iterrows():
    train_cell, length = full_name.split('-')
    
    F1_list = []
    auprc_list = []
    
    for val in row:
        F1, auprc = extract_metrics(val)
        if F1 is not None: F1_list.append(F1)
        if auprc is not None: auprc_list.append(auprc)
    
    # 计算平均值并存入长表格式
    plot_rows.append({'TrainCell': train_cell, 'Length': int(length), 'Value': sum(F1_list)/len(F1_list), 'Metric': 'Average F1'})
    plot_rows.append({'TrainCell': train_cell, 'Length': int(length), 'Value': sum(auprc_list)/len(auprc_list), 'Metric': 'Average AUPRC'})

plot_df = pd.DataFrame(plot_rows)
plot_df['Length'] = pd.Categorical(plot_df['Length'], categories=[128, 256, 512, 1024])

# 2. 绘图：一左一右布局
p9.options.figure_size = (16, 6) # 增加宽度以适应两个图

p = (
    ggplot(plot_df, aes(x='Length', y='Value', color='TrainCell', group='TrainCell', shape='TrainCell'))
    + geom_line(size=1.2, alpha=0.8)
    + geom_point(size=6)
    # 分面核心：Metric 列决定左右分布，scales="free_y" 让 F1 和 AUPRC 各自优化 Y 轴范围
    + facet_wrap('~Metric', nrow=1, scales="free_y")
    + scale_color_brewer(type='qual', palette='Set1')
    + labs(
        x="Sequence length",
        y="Metric value",
        color="Training Cell Type",
        shape="Training Cell Type"
    )
    + theme_bw()
    + theme(
        text=element_text(family="Arial"),
        # 移除分面标题的灰色背景框
        strip_background=element_blank(),
        strip_text=element_text(size=20, weight='bold'),
        axis_title=element_text(size=20, weight='bold'),
        axis_text=element_text(size=20, color='black'),
        legend_text=element_text(size=18),
        legend_title=element_text(size=18),
        legend_entry_spacing_y=10, 
        # 2. 增加图例图标的高度，也能间接拉开文字距离
        legend_key_height=25,

        panel_spacing=0.1, # 增加两图之间的间距
        # 优化 X 轴标题空间
        axis_title_x=element_text(margin={'t': 15})
    )
)

print(p)
p.save("sequence_length_comparison.pdf", bbox_inches="tight")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("../test5.csv", index_col=0)

metrics = ["Acc", "F1", "AUC", "AUPRC", "MCC", "r"]

def parse_metrics(text):
    pairs = re.findall(r'(\w+)=([0-9.]+)', str(text))
    return {k: float(v) for k, v in pairs}

def get_metric(row, col, metric="AUC"):
    if row not in df.index:
        return np.nan
    parsed = parse_metrics(df.loc[row, col])
    return parsed.get(metric, np.nan)

train_cells = ["293T", "A549", "HepG2", "K562"]
test_cells = ["293T", "A549", "HepG2", "K562"]

# ======================
# 2. 构建矩阵（核心）
# ======================
g4former_auc = np.zeros((4,4))
cas_auc = np.zeros((4,4))
delta_auc = np.zeros((4,4))

g4former_AUPRC = np.zeros((4,4))
cas_AUPRC = np.zeros((4,4))
delta_AUPRC = np.zeros((4,4))

for i, train in enumerate(train_cells):
    for j, test in enumerate(test_cells):

        base_auc = get_metric(train, test, "AUC")
        cas_auc_val = get_metric(f"{train}-Without CSA", test, "AUC")

        base_AUPRC = get_metric(train, test, "AUPRC")
        cas_AUPRC_val = get_metric(f"{train}-Without CSA", test, "AUPRC")

        g4former_auc[i,j] = base_auc
        cas_auc[i,j] = cas_auc_val
        delta_auc[i,j] = base_auc - cas_auc_val

        g4former_AUPRC[i,j] = base_AUPRC
        cas_AUPRC[i,j] = cas_AUPRC_val
        delta_AUPRC[i,j] = round(base_AUPRC,3) - round(cas_AUPRC_val ,3)

# ======================
# 3. 设置绘图风格（Nature风）
# ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# ======================
# 8. AUPRC Heatmaps
# ======================
fig2 = plt.figure(figsize=(17, 5)) # 稍微增加画布宽度，给间距留空间
def add_diag_box(ax, num_species=4):
    for i in range(num_species):
        # 使用 clip_on=False 确保边框线条不会因为紧贴轴线而被切断
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2, clip_on=False))

ax1 = plt.subplot(1, 3, 1)
sns.heatmap(
    g4former_AUPRC,
    annot=True, fmt=".3f", cmap="Reds", vmin=0.7, vmax=0.95,
    xticklabels=test_cells, yticklabels=train_cells,
    cbar_kws={'label': 'AUPRC'} # 可选：增加 colorbar 标签
)
ax1.set_title("G4former (AUPRC)", pad=15) # pad 增加标题与图的间距
ax1.set_xlabel("Test cell")
ax1.set_ylabel("Train cell")
add_diag_box(ax1) # 添加红框


ax2 = plt.subplot(1, 3, 2)
sns.heatmap(
    cas_AUPRC,
    annot=True, fmt=".3f", cmap="Blues",
    xticklabels=test_cells, yticklabels=train_cells
)
ax2.set_title("G4former-Without CSA (AUPRC)", pad=15)
add_diag_box(ax2) # 添加红框

ax3 = plt.subplot(1, 3, 3)
sns.heatmap(
    delta_AUPRC,
    annot=True, fmt=".3f", cmap="coolwarm", center=0,
    xticklabels=test_cells, yticklabels=train_cells
)
ax3.set_title("Difference (AUPRC)", pad=15)
add_diag_box(ax3) # 添加红框

# --- 关键修改：调整间距 ---
plt.tight_layout() 
# wspace 控制子图宽度比例的间距，0.4 左右通常比较宽松
plt.subplots_adjust(wspace=0.25) 

plt.savefig("Figure_main_AUPRC_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.show()