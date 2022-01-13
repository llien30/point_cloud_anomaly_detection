import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def vis_histgram(csv_file: str, save_dir: str) -> None:

    df = pd.read_csv(csv_file, index_col=0)
    label = df["1"]
    result = df["2"]
    normal_result = []
    abnormal_result = []
    for lbl, r in zip(label, result):
        if lbl[7] == "0":
            normal_result.append(r * 1000)
        else:
            abnormal_result.append(r * 1000)
    bin_max = max(max(normal_result), max(abnormal_result))
    bins = np.linspace(0, bin_max, 100)
    plt.hist(normal_result, bins, alpha=0.5, label="normal")
    plt.hist(abnormal_result, bins, alpha=0.5, label="abnormal")
    plt.legend(loc="upper left")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of samples")
    plt.savefig(os.path.join(save_dir, "histgram.png"))
    plt.close()
