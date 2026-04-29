"""
Standalone script: overlay Hit@K curves of arbitrary prediction files on one plot.

Use this when you want to compare a hand-picked set of experiments / runs on the
same axes (no dependency on the pipeline or BestModelsResults.csv).

Edit `file_map` below with friendly labels -> prediction file paths (.parquet or .csv).
Each file must contain a 'label' (or 'LABEL') column and, if not already sorted,
a 'y_prob' column to sort by descending. Then run:

    python plot_ensemble_hit_rates.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# EDIT THESE
# ============================================================================

# label -> path to a predictions parquet/csv file
file_map = {
    # "Run A": r"D:\path\to\runA_predictions_sorted.parquet",
    # "Run B": r"D:\path\to\runB_predictions_sorted.parquet",
}

OUT_PATH = "ensemble_hit_rates.png"

# Hard-coded axis caps (same scale as the in-pipeline hit plots).
Y_MAX = 100
X_MAX = 1000
# Number of simulated random-permutation curves to overlay (faint grey fan
# behind the theoretical 'Random expected' line). 0 = theoretical only.
# Bumping this slows rendering when the test set is large.
N_RANDOM_RUNS = 10
# ============================================================================


def _load_predictions(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    df.columns = [c.lower() for c in df.columns]
    return df


def plot_ensemble_hit_rates(file_map, out_path=OUT_PATH, x_max=X_MAX, y_max=Y_MAX):
    if not file_map:
        raise ValueError("file_map is empty — add entries before running.")

    plt.figure(figsize=(9, 6), dpi=200)
    random_drawn = False

    for label, path in file_map.items():
        df = _load_predictions(path)
        if 'label' not in df.columns:
            raise ValueError(f"No 'label' column in {path}")

        # If the file isn't sorted by descending probability, sort it now.
        if 'y_prob' in df.columns:
            df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

        y = df['label'].astype(int).values
        n_total = len(y)
        n_pos = int(y.sum())
        if n_pos == 0:
            print(f"Skipping (no positives): {path}")
            continue

        ranks = np.arange(1, n_total + 1)
        cum_hits = np.cumsum(y)

        if not random_drawn:
            # Simulated random runs (drawn first so they sit behind everything).
            if N_RANDOM_RUNS > 0:
                rng = np.random.default_rng(42)
                sim_labels = np.zeros(n_total, dtype=np.int8)
                sim_labels[:n_pos] = 1
                for i in range(N_RANDOM_RUNS):
                    sim_curve = np.cumsum(rng.permutation(sim_labels))
                    kw = dict(color='lightgray', linewidth=0.8, alpha=0.3)
                    if i == 0:
                        kw['label'] = f'Random simulated ({N_RANDOM_RUNS} runs)'
                    plt.plot(ranks, sim_curve, **kw)
            random_curve = ranks * (n_pos / n_total)
            plt.plot(ranks, random_curve, color='gray', linestyle='--',
                     linewidth=1.5, label='Random expected')
            random_drawn = True

        plt.plot(ranks, cum_hits, linewidth=2, label=label)

    plt.xlabel('K')
    plt.ylabel('Hit@K')
    plt.title('Hit@K - ensemble comparison')
    plt.xlim(0, x_max)  # hard-coded cap (see X_MAX)
    plt.ylim(0, y_max)  # hard-coded cap (see Y_MAX)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_ensemble_hit_rates(file_map, out_path=OUT_PATH, x_max=X_MAX, y_max=Y_MAX)
