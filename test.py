import argparse
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from addict import Dict
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from libs.checkpoint import resume
from libs.dataset import ShapeNeth5pyDataset
from libs.emd.emd_module import emdModule
from libs.foldingnet import FoldingNet, SkipFoldingNet, SkipValiationalFoldingNet
from libs.loss import ChamferLoss


def get_parameters():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(description="take config file path")

    parser.add_argument("config", type=str, help="path of a config file for testing")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="path of the file where the weight is saved",
    )
    parser.add_argument(
        "-c",
        "--chamfer",
        action="store_true",
        help="Whether to add a chamfer score or not",
    )
    parser.add_argument(
        "-e",
        "--emd",
        action="store_true",
        help="Whether to add a emd score or not",
    )
    parser.add_argument(
        "-k",
        "--kldiv",
        action="store_true",
        help="Whether to add a kldiv score or not",
    )
    parser.add_argument(
        "-f",
        "--feature_diff",
        action="store_true",
        help="Whether to add a feature diff score or not",
    )
    parser.add_argument(
        "--histgram",
        action="store_true",
        help="Visualize histgram or not",
    )
    parser.add_argument(
        "--save_points",
        action="store_true",
        help="Save points or not",
    )

    return parser.parse_args()


def rescale(input):
    input = np.array(input, dtype=float)
    _min = np.array(min(input))
    _max = np.array(max(input))
    with np.errstate(invalid="ignore"):
        re_scaled = (input - _min) / (_max - _min)
    return re_scaled


def vis_histgram(label: List, result: List, save_name: str) -> None:
    normal_result = []
    abnormal_result = []
    for lbl, r in zip(label, result):
        if lbl == 0:
            normal_result.append(r * 1000)
        else:
            abnormal_result.append(r * 1000)
    bin_max = max(max(normal_result), max(abnormal_result))
    bins = np.linspace(0, bin_max, 100)
    plt.hist(normal_result, bins, alpha=0.5, label="normal")
    plt.hist(abnormal_result, bins, alpha=0.5, label="abnormal")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.savefig(save_name)
    plt.close()


def main():
    # seedの固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    set_seed(0)

    args = get_parameters()

    # configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    CONFIG = Dict(config_dict)

    print(config_dict)

    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ShapeNeth5pyDataset(
        root_path=CONFIG.root_path,
        split="test",
        normal_class=CONFIG.normal_class,
        abnormal_class=CONFIG.abnormal_class,
        n_point=CONFIG.n_points,
        random_rotate=False,
        random_jitter=False,
        random_translate=False,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=CONFIG.test_batch_size, shuffle=False
    )
    if CONFIG.model == "FoldingNet":
        model = FoldingNet(CONFIG.n_points, CONFIG.feat_dims, CONFIG.shape)
    elif CONFIG.model == "SkipFoldingNet":
        model = SkipFoldingNet(CONFIG.n_points, CONFIG.feat_dims, CONFIG.shape)
    elif CONFIG.model == "SkipVariationalFoldingNet":
        model = SkipValiationalFoldingNet(
            CONFIG.n_points, CONFIG.feat_dims, CONFIG.shape
        )
    model.to(device)

    lr = 0.0001 * 16 / CONFIG.batch_size
    beta1, beta2 = 0.9, 0.999

    optimizer = torch.optim.Adam(
        model.parameters(), lr, [beta1, beta2], weight_decay=1e-6
    )

    epoch, model, optimizer = resume(args.checkpoint_path, model, optimizer)

    print(f"---------- Start testing for epoch{epoch} ----------")

    model.eval()
    pred = []
    labels = [""] * len(test_dataloader.dataset)
    names = [""] * len(test_dataloader.dataset)
    n = 0
    chamferloss = ChamferLoss()
    emd_loss = emdModule()

    chamfer_scores = []
    emd_scores = []
    kldiv_scores = []
    feature_diff_scores = []

    for samples in test_dataloader:
        data = samples["data"].float()
        label = samples["label"]
        name = samples["name"]

        mini_batch_size = data.size()[0]

        data = data.to(device)
        if CONFIG.model == "SkipVariationalFoldingNet":
            with torch.no_grad():
                output, folding1, mu, log_var = model(data)

                if args.kldiv or args.feature_diff:
                    _, _, fake_mu, fake_log_var = model(output)

            if args.chamfer:
                for d, o in zip(data, output):
                    d = d.reshape(1, 2048, -1)
                    o = o.reshape(1, 2048, -1)
                    cl = chamferloss(d, o)
                    chamfer_scores.append(cl)
            else:
                for _ in range(mini_batch_size):
                    chamfer_scores.append(0)

            if args.emd:
                for d, o in zip(data, output):
                    d = d.reshape(1, 2048, -1)
                    o = o.reshape(1, 2048, -1)
                    el, _ = emd_loss(d, o, 0.005, 50)
                    el = torch.sqrt(el).mean(1)
                    emd_scores.append(el)
            else:
                for _ in range(mini_batch_size):
                    emd_scores.append(0)

            if args.kldiv:
                for m, l in zip(mu, log_var):
                    kldiv = torch.mean(
                        -0.5 * torch.sum(1 + l - m ** 2 - l.exp(), dim=1), dim=0
                    )
                    # kldiv = torch.mean(
                    #     0.5
                    #     * torch.sum(
                    #         m ** 2 + l ** 2 - torch.log(l ** 2 + 1e-12) - 1, dim=1
                    #     ),
                    #     dim=0,
                    # )
                    kldiv_scores.append(kldiv)
                # for m, l, fm, fl in zip(mu, log_var, fake_mu, fake_log_var):
                #     P = torch.distributions.Normal(m, l)
                #     Q = torch.distributions.Normal(fm, fl)
                #     # kld_loss = torch.distributions.kl_divergence(G, P).mean()
                #     kldiv = torch.distributions.kl_divergence(P, Q).mean()
                #     # kldiv = torch.mean(
                #     #     -0.5 * torch.sum(1 + l - m ** 2 - l.exp(), dim=1), dim=0
                #     # )
                #     kldiv_scores.append(kldiv)
            else:
                for _ in range(mini_batch_size):
                    kldiv_scores.append(0)

            if args.feature_diff:
                for m, l, fm, fl in zip(mu, log_var, fake_mu, fake_log_var):
                    std = torch.exp(0.5 * l)
                    eps = torch.randn_like(std)
                    feat = eps * std + m
                    fake_std = torch.exp(0.5 * fl)
                    fake_eps = torch.randn_like(fake_std)
                    fake_feat = fake_eps * fake_std + fm

                    diff_feat = feat - fake_feat
                    diff_feat = diff_feat.reshape(-1)
                    feature_diff_score = np.mean(
                        np.power(diff_feat.to("cpu").numpy(), 2.0)
                    )
                    feature_diff_scores.append(feature_diff_score)
            else:
                for _ in range(mini_batch_size):
                    feature_diff_scores.append(0)

            if args.save_points:
                for i in range(mini_batch_size):
                    o = output[i]
                    d = data[i]

                    d = d.reshape(1, 2048, -1)
                    o = o.reshape(1, 2048, -1)

                    d = d.to("cpu").numpy()
                    o = o.to("cpu").numpy()

                    if not os.path.exists("./original"):
                        os.makedirs("./original")
                    if not os.path.exists("./reconstructed"):
                        os.makedirs("./reconstructed")
                    np.save(f"./original/{n+i}.npy", d)
                    np.save(f"./reconstructed/{n+i}.npy", o)

            labels[n : n + mini_batch_size] = label.reshape(mini_batch_size)
            names[n : n + mini_batch_size] = name

            n += mini_batch_size

    if args.chamfer:
        chamfer_scores = rescale(chamfer_scores)
    if args.emd:
        emd_scores = rescale(emd_scores)
    if args.kldiv:
        kldiv_scores = rescale(kldiv_scores)
    if args.feature_diff:
        feature_diff_scores = rescale(feature_diff_scores)

    for chamfer_score, emd_score, kldiv_score, feature_diff_score in zip(
        chamfer_scores, emd_scores, kldiv_scores, feature_diff_scores
    ):
        score = chamfer_score + emd_score + kldiv_score + feature_diff_score
        pred.append(score)

    pred = np.array(pred, dtype=float)
    _min = np.array(min(pred))
    _max = np.array(max(pred))

    re_scaled = (pred - _min) / (_max - _min)

    # re_scaled = rescale(pred)
    re_scaled = np.array(re_scaled, dtype=float)
    fpr, tpr, _ = roc_curve(labels, re_scaled)
    roc_auc = auc(fpr, tpr)

    if args.histgram:
        vis_histgram(labels, pred, "lamp_histgram.png")

    if args.save_points:
        df = pd.DataFrame(list(zip(names, labels, pred)))
        df.to_csv("result.csv")

    print(f"ROC:{roc_auc}")

    print("Done")


if __name__ == "__main__":
    main()
