import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16

df = pd.read_csv("../test2.csv", index_col=0)

expanded_rows = []

for row_name in df.index:
    for col_name in df.columns:

        text = str(df.loc[row_name, col_name])

        pairs = re.findall(r'(\w+)=([0-9.]+)', text)

        metrics_dict = {k: float(v) for k, v in pairs}

        metrics_dict["Train_Model"] = row_name
        metrics_dict["Test_Cell"] = col_name

        expanded_rows.append(metrics_dict)

metrics_df = pd.DataFrame(expanded_rows)
metrics_df = metrics_df.fillna(0)

metrics_df.to_csv("parsed_metrics.csv", index=False)

print(metrics_df.head())
# 需要的指标

metrics = ["Acc", "F1", "AUC", "AUPRC", "MCC", "r"]

# custom colors for models (better contrast)
model_colors = {
    "G4Beacon": "#8c564b",
    "DeepG4": "#17becf",
    "epiG4NN": "#1f77b4",
    "Seq": "#ff7f0e",
    "DNase": "#2ca02c",
    "WGBS": "#9467bd",
    "G4former": "#d62728"
}

# 解析字符串中的指标
def parse_metrics(text):
    result = {}
    pairs = re.findall(r'(\w+)=([0-9.]+)', text)
    for k, v in pairs:
        result[k] = float(v)
    return result


def get_values(row_name, col_name):
    text = df.loc[row_name, col_name]
    m = parse_metrics(text)
    return [m.get(k, 0) for k in metrics]


def radar_axes():
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    return angles


# 1×4 bar plot figure for all datasets
def plot_all_bar():

    train_cells = ["293T", "A549", "K562", "HepG2"]

    fig, axes = plt.subplots(1, 4, figsize=(18,5))

    for j, train_cell in enumerate(train_cells):

        ax1 = axes[j]

        models = {
            "Seq": f"{train_cell}-seq",
            "Seq+DNase": f"{train_cell}-DNase",
            "Seq+WGBS": f"{train_cell}-WGBS",
            "G4former": f"{train_cell}"
        }

        test_cells = ["293T", "A549", "HepG2", "K562"]

        results = []

        for model, row in models.items():

            auroc_list = []
            aupr_list = []

            for test in test_cells:

                v = get_values(row, test)

                auroc_list.append(v[2])
                aupr_list.append(v[3])

            results.append((auroc_list, aupr_list))

        methods = list(models.keys())

        aurocs = [np.mean(method[0]) for method in results]
        auprcs = [np.mean(method[1]) for method in results]
        print(aurocs)
        # print(auprcs)

        aurocs_err = [np.std(method[0]) for method in results]
        auprcs_err = [np.std(method[1]) for method in results]

        x = np.arange(len(methods))
        bar_width = 0.35

        ax1.bar(
            x - bar_width/2,
            aurocs,
            yerr=aurocs_err,
            width=bar_width,
            color='#FA8072',
            capsize=4,
            label="AUC"
        )

        ax2 = ax1.twinx()

        ax2.bar(
            x + bar_width/2,
            auprcs,
            yerr=auprcs_err,
            width=bar_width,
            color='#90EE90',
            capsize=4,
            label="AUPRC"
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=30)
        ax1.set_ylim(0.5,1)
        ax2.set_ylim(0.3,1)

        ax1.set_title(train_cell)

        if j == 0:
            ax1.set_ylabel("AUC")
            ax2.set_ylabel("AUPRC")

    plt.tight_layout()

    plt.savefig("Figure2_bar.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    


# Figure 2: bar plots
plot_all_bar()
