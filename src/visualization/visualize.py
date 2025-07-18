import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_function(RunFolderName):

    csv_path = os.path.join(RunFolderName, "results.csv")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # === Load data ===
    df = pd.read_csv(csv_path)

    
    # === Metrics to plot ===
    metrics = [
        "CV_HitsAt500", "CV_PrecisionAt200", "CV_HitsAt200", "CV_PrecisionAt500", "CV_TotalHits",
        "Test_HitsAt500", "Test_PrecisionAt200", "Test_HitsAt200", "Test_PrecisionAt500", "Test_TotalHits"
    ]
    
    # === Group and average by ModelType and ColumnName ===
    grouped = df.groupby(["ModelType", "ColumnName"])[metrics].mean().reset_index()
    models = grouped["ModelType"].unique()
    fingerprints = grouped["ColumnName"].unique()
    color_map = plt.cm.get_cmap("tab10", len(fingerprints))
    
    # === Create one plot per metric ===
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 6))
        bar_width = 0.8 / len(fingerprints)
        index_base = range(len(models))
    
        for i, fingerprint in enumerate(fingerprints):
            values = []
            for model in models:
                row = grouped[(grouped["ModelType"] == model) & (grouped["ColumnName"] == fingerprint)]
                val = row[metric].values[0] if not row.empty else 0
                values.append(val)
            bar_positions = [x + i * bar_width for x in index_base]
            ax.bar(bar_positions, values, bar_width, label=fingerprint, color=color_map(i))
    
        ax.set_title(f"{metric} by Model and Fingerprint")
        ax.set_xlabel("Model Type")
        ax.set_ylabel(metric)
        ax.set_xticks([x + bar_width * (len(fingerprints) / 2 - 0.5) for x in index_base])
        ax.set_xticklabels(models, rotation=45)
        ax.legend(title="Fingerprint", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()
