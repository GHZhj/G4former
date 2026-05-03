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



# Combine all dataset radar figures into one large figure
def plot_all_datasets():

    train_cells = ["293T", "A549", "HepG2", "K562"]
    test_cells = ["293T", "A549", "HepG2", "K562"]

    fig = plt.figure(figsize=(18,18))

    outer = fig.add_gridspec(2, 2)

    angles = radar_axes()

    for idx, train_cell in enumerate(train_cells):

        row = idx // 2
        col = idx % 2

        # sub grid for each dataset (2x2 radar)
        sub_gs = outer[row, col].subgridspec(2,2)

        models = {
            "G4Beacon": f"{train_cell}-G4Beacon",
            "DeepG4": f"{train_cell}-DeepG4",
            "epiG4NN": f"{train_cell}-epiG4NN",
            "G4former": f"{train_cell}"
        }

        for i, test in enumerate(test_cells):

            ax = fig.add_subplot(sub_gs[i//2, i%2], polar=True)
            # highlight the subplot where train_cell == test (self-test case)
            if test == train_cell:
                ax.set_facecolor("#e6f2ff")

            for model, row_name in models.items():

                values = get_values(row_name, test)
                values = values + [values[0]]

                color = model_colors.get(model, "gray")
                lw = 3 if model == "G4former" else 2

                ax.plot(angles, values, linewidth=lw, label=model, color=color)
                ax.fill(angles, values, alpha=0.08 if model == "G4former" else 0.01, color=color)

            ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
            # place metric labels following the circular arc direction
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([])

            for angle, label in zip(angles[:-1], metrics):
                rotation = np.degrees(angle) - 90
                ax.text(
                    angle,
                    1.08,
                    label,
                    rotation=rotation,
                    rotation_mode='anchor',
                    ha='center',
                    va='center',
                    fontsize=14
                )
            ax.set_ylim(0,1)
            ax.set_title(f"Test on {test}", fontsize=16)

        # dataset title
        fig.text(
            (col*0.5)+0.25,
            0.95 - row*0.48,
            f"{train_cell} dataset",
            ha="center",
            fontsize=20
        )
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        fontsize=16,
        frameon=False
    )

    plt.tight_layout(rect=[0,0,1,0.95])

    plt.savefig("all_datasets_radar.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def plot_feature_bar(train_cell):

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

            auroc = v[2]   # AUC
            aupr = v[3]    # AUPRC

            auroc_list.append(auroc)
            aupr_list.append(aupr)

        results.append((auroc_list, aupr_list))

    methods = list(models.keys())

    aurocs = [np.mean(method[0]) for method in results]
    auprcs = [np.mean(method[1]) for method in results]

    aurocs_err = [np.std(method[0]) for method in results]
    auprcs_err = [np.std(method[1]) for method in results]

    x = np.arange(len(methods))

    fig, ax1 = plt.subplots(figsize=(10, 7))

    bar_width = 0.35

    # AUROC
    ax1.bar(
        x - bar_width / 2,
        aurocs,
        yerr=aurocs_err,
        width=bar_width,
        label='AUROC',
        color='#FA8072',
        capsize=5
    )

    ax1.set_xlabel('Features', fontsize=16)
    ax1.set_ylabel('AUROC', color='#FA8072', fontsize=16)

    ax1.tick_params(axis='y', labelcolor='#FA8072', labelsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=14)

    ax1.set_ylim([0.5, 1])

    # AUPR
    ax2 = ax1.twinx()

    ax2.bar(
        x + bar_width / 2,
        auprcs,
        yerr=auprcs_err,
        width=bar_width,
        label='AUPR',
        color='#90EE90',
        capsize=5
    )

    ax2.set_ylabel('AUPR', color='#90EE90', fontsize=16)

    ax2.tick_params(axis='y', labelcolor='#90EE90', labelsize=12)

    ax2.set_ylim([0.3, 1])

    # scatter points
    for i, (aurocs_data, auprcs_data) in enumerate(results):

        ax1.scatter(
            [i - bar_width / 2] * len(aurocs_data),
            aurocs_data,
            color='black',
            s=50,
            label='AUROC Points' if i == 0 else ""
        )

        ax2.scatter(
            [i + bar_width / 2] * len(auprcs_data),
            auprcs_data,
            color='gray',
            s=50,
            label='AUPR Points' if i == 0 else ""
        )

    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    plt.title(f"{train_cell} Feature Contribution", fontsize=18)

    plt.tight_layout()

    output_file = f"{train_cell}_feature_AUROC_AUPR_bar.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")

    # plt.show()


# Figure 1: radar plots
# plot_all_radar()
 
plot_all_datasets()

