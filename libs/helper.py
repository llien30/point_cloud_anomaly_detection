import os
import time
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from .emd.emd_module import emdModule
from .loss import ChamferLoss, mse_loss
from .meter import AverageMeter
from .vis_histgram import vis_histgram
from .visualize import vis_points_3d

matplotlib.use("Agg")


def train_foldingnet(
    loader: DataLoader,
    model: nn.Module,
    reconstruction_loss: str,
    optimizer: optim.Optimizer,
    weight: List,
    epoch: int,
    device: str,
    save_dir: str,
) -> Tuple[float, float, float, float, float]:

    loss_meter = AverageMeter("loss", ":.4e")
    inner_loss_meter = AverageMeter("inner_loss", ":.4e")
    out_loss_meter = AverageMeter("out_loss", ":.4e")
    feat_loss_meter = AverageMeter("feat_loss", ":.4e")
    inf_loss_meter = AverageMeter("inf_loss", ":4e")

    if reconstruction_loss == "CD":
        inner_criterion = ChamferLoss()
        out_criterion = ChamferLoss()
    elif reconstruction_loss == "EMD":
        # inner_criterion = emdModule()
        inner_criterion = ChamferLoss()
        out_criterion1 = ChamferLoss()
        out_criterion2 = emdModule()

    # switch to train mode
    model.train()
    softmax = nn.Softmax(dim=2)

    t_epoch_start = time.time()
    for samples in loader:
        points = samples["data"].float()
        points = points.to(device)

        # Forword Pass
        output, folding1, feature = model(points)
        _, _, fake_feature = model(output)

        # points = points.cpu().detach().numpy()
        # folding1 = folding1.cpu().detach().numpy()
        # output = output.cpu().detach().numpy()
        if reconstruction_loss == "CD":
            inner_loss = inner_criterion(points, folding1)
            out_loss = out_criterion(points, output)
        elif reconstruction_loss == "EMD":
            inner_loss = inner_criterion(points, folding1)
            if epoch < 100:
                out_loss = out_criterion1(points, output)
            else:
                out_loss, _ = out_criterion2(points, output, 0.005, 50)
                out_loss = torch.sqrt(out_loss).mean(1)
                out_loss = out_loss.mean()

        # print(inner_loss)
        # inner_loss = wasserstein_distance(points, folding1)
        # out_loss = wasserstein_distance(points, output)
        # inner_loss = torch.from_numpy(inner_loss.to(device))
        # out_loss = torch.from_numpy(out_loss.to(device))
        softmax_feat = softmax(feature)
        inf_loss = torch.sum(Categorical(probs=softmax_feat).entropy())

        feat_loss = mse_loss(feature, fake_feature)

        w_in = weight[0]
        w_out = weight[1]
        w_feat = weight[2]
        w_inf = weight[3]
        if reconstruction_loss == "CD":
            if w_in == 0 and w_inf == 0:
                loss = out_loss
            elif w_inf == 0:
                loss = w_in * inner_loss + w_out * out_loss
            elif w_in != 0:
                loss = w_in * inner_loss + w_out * out_loss + w_inf * inf_loss
            if w_feat != 0 and epoch > 100:
                loss += w_feat * feat_loss
        else:
            if w_inf == 0:
                loss = w_in * inner_loss
            else:
                loss = w_in * inner_loss + w_inf * inf_loss
            if w_feat != 0 and epoch > 100:
                loss += w_feat * feat_loss
            if epoch > 100:
                loss += w_out * out_loss

        inner_loss_meter.update(inner_loss.item())
        out_loss_meter.update(out_loss.item())
        feat_loss_meter.update(feat_loss.item())
        inf_loss_meter.update(inf_loss.item())
        loss_meter.update(loss.item())

        # Backword Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - t_epoch_start
    epoch_loss = loss_meter.avg
    epoch_inner_loss = inner_loss_meter.avg
    epoch_out_loss = out_loss_meter.avg
    epoch_feat_loss = feat_loss_meter.avg
    epoch_inf_loss = inf_loss_meter.avg

    if not os.path.exists(os.path.join(save_dir, "original")):
        os.makedirs(os.path.join(save_dir, "original"))
    if not os.path.exists(os.path.join(save_dir, "folding1")):
        os.makedirs(os.path.join(save_dir, "folding1"))
    if not os.path.exists(os.path.join(save_dir, "reconstructed")):
        os.makedirs(os.path.join(save_dir, "reconstructed"))

    vis_points_3d(
        points[0],
        save_dir + f"/original/{epoch}.png",
    )
    point = points[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/original/{epoch}.npy", point)
    vis_points_3d(
        folding1[0],
        save_dir + f"/folding1/{epoch}.png",
    )
    folding = folding1[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/folding1/{epoch}.npy", folding)
    vis_points_3d(
        output[0],
        save_dir + f"/reconstructed/{epoch}.png",
    )
    out_point = output[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/reconstructed/{epoch}.npy", out_point)

    return (
        epoch_loss,
        epoch_inner_loss,
        epoch_out_loss,
        epoch_feat_loss,
        epoch_inf_loss,
        epoch_time,
    )


def evaluate_foldingnet(
    loader: DataLoader,
    model: nn.Module,
    epoch: int,
    device: str,
    save_dir: str,
) -> Tuple[float, float, float, float, float, float]:

    # 保存ディレクトリの作成
    save_normal = os.path.join(save_dir, f"epoch{epoch}/normal")
    save_abnormal = os.path.join(save_dir, f"epoch{epoch}/abnormal")
    if not os.path.exists(save_normal):
        os.makedirs(save_normal)
    if not os.path.exists(save_abnormal):
        os.makedirs(save_abnormal)

    eval_criterion = ChamferLoss()

    # switch to evalutation mode
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred = []
    labels = [""] * len(loader.dataset)
    names = [""] * len(loader.dataset)
    points = [""] * len(loader.dataset)
    folding1s = [""] * len(loader.dataset)
    out_points = [""] * len(loader.dataset)
    n = 0

    feature_vec = []
    diff_vec = []
    chamferloss = []

    for samples in loader:
        data = samples["data"].float()
        label = samples["label"]
        name = samples["name"]

        mini_batch_size = data.size()[0]
        points[n : n + mini_batch_size] = data

        data = data.to(device)

        with torch.no_grad():
            output, folding1, real_feat = model(data)
            _, _, fake_feat = model(output)

        folding1s[n : n + mini_batch_size] = folding1
        out_points[n : n + mini_batch_size] = output

        n_input_points = data.shape[1]
        n_output_points = output.shape[1]
        for d, o in zip(data, output):
            d = d.reshape(1, n_input_points, -1)
            o = o.reshape(1, n_output_points, -1)
            cl = eval_criterion(d, o)
            cl = cl.tolist()
            chamferloss.append(cl)

        test_feat = real_feat - fake_feat
        test_feat = test_feat.to("cpu")

        real_feat = real_feat.to("cpu").squeeze()
        for rf in real_feat:
            rf = rf.tolist()
            feature_vec.append(rf)

        for tf in test_feat:
            tf = tf.squeeze()
            tf = tf.tolist()
            diff_vec.append(tf)

        for i in range(test_feat.shape[0]):
            vec = test_feat[i]
            vec = vec.reshape(-1)
            score = np.mean(np.power(vec.numpy(), 2.0))
            pred.append(score)
        labels[n : n + mini_batch_size] = label.reshape(mini_batch_size)
        names[n : n + mini_batch_size] = name

        n += mini_batch_size

    # T-SNE visualization
    # feature vector
    feat_reduced = TSNE(n_components=2).fit_transform(feature_vec)
    plt.scatter(feat_reduced[:, 0], feat_reduced[:, 1], c=labels)
    plt.savefig(os.path.join(save_dir, f"epoch{epoch}/feat_tsne.png"))
    plt.close()
    # diff vector
    diff_feat_reduced = TSNE(n_components=2).fit_transform(diff_vec)
    plt.scatter(diff_feat_reduced[:, 0], diff_feat_reduced[:, 1], c=labels)
    plt.savefig(os.path.join(save_dir, f"epoch{epoch}/diff_tsne.png"))
    plt.close()

    if not os.path.exists(os.path.join(save_dir, "result_point/original")):
        os.makedirs(os.path.join(save_dir, "result_point/original"))
    if not os.path.exists(os.path.join(save_dir, "result_point/reconstructed")):
        os.makedirs(os.path.join(save_dir, "result_point/reconstructed"))

    if epoch == 200:
        cnt = 0
        for point in points:
            point = point.to("cpu").detach().numpy()
            np.save(os.path.join(save_dir, f"result_point/original/{cnt}.npy"), point)
            cnt += 1
        cnt = 0
        for out_point in out_points:
            out_point = out_point.to("cpu").detach().numpy()
            np.save(
                os.path.join(save_dir, f"result_point/reconstructed/{cnt}.npy"),
                out_point,
            )
            cnt += 1

    # save result
    df = pd.DataFrame(list(zip(names, labels, pred, points, folding1s, out_points)))
    df.to_csv(os.path.join(save_dir, f"epoch{epoch}/result.csv"))
    vis_histgram(
        os.path.join(save_dir, f"epoch{epoch}/result.csv"),
        os.path.join(save_dir, f"epoch{epoch}"),
    )
    # pred = np.array(chamferloss)

    _min = min(pred)
    _max = max(pred)

    re_scaled = (pred - _min) / (_max - _min)
    re_scaled = np.array(re_scaled)
    fpr, tpr, _ = roc_curve(labels, re_scaled)
    roc_auc = auc(fpr, tpr)
    thresh = 0.2
    re_scaled[re_scaled >= thresh] = 1
    re_scaled[re_scaled < thresh] = 0

    acc = accuracy_score(labels, re_scaled)
    prec = precision_score(labels, re_scaled)
    rec = recall_score(labels, re_scaled)
    f1 = f1_score(labels, re_scaled)
    avg_prec = average_precision_score(labels, re_scaled)

    return roc_auc, acc, prec, rec, f1, avg_prec


def train_variational_foldingnet(
    loader: DataLoader,
    model: nn.Module,
    reconstruction_loss: str,
    optimizer: optim.Optimizer,
    weight: List,
    epoch: int,
    device: str,
    save_dir: str,
) -> Tuple[float, float, float, float, float]:

    loss_meter = AverageMeter("loss", ":.4e")
    inner_loss_meter = AverageMeter("inner_loss", ":.4e")
    out_loss_meter = AverageMeter("out_loss", ":.4e")
    kld_loss_meter = AverageMeter("kld_loss", ":4e")
    fake_kld_loss_meter = AverageMeter("fake_kld_loss", ":4e")

    if reconstruction_loss == "CD":
        inner_criterion = ChamferLoss()
        out_criterion = ChamferLoss()
    elif reconstruction_loss == "EMD":
        # inner_criterion = emdModule()
        inner_criterion = ChamferLoss()
        out_criterion = emdModule()

    # switch to train mode
    model.train()
    softmax = nn.Softmax(dim=2)

    t_epoch_start = time.time()
    for samples in loader:
        points = samples["data"].float()
        points = points.to(device)

        # Forword Pass
        output, folding1, mu, sigma = model(points)
        _, _, fake_mu, fake_sigma = model(output)

        if reconstruction_loss == "CD":
            inner_loss = inner_criterion(points, folding1)
            out_loss = out_criterion(points, output)
        elif reconstruction_loss == "EMD":
            inner_loss = inner_criterion(points, folding1)
            out_loss, _ = out_criterion(points, output, 0.005, 50)
            out_loss = torch.sqrt(out_loss).mean(1)
            out_loss = out_loss.mean()

        # softmax_feat = softmax(feature)
        # inf_loss = torch.sum(Categorical(probs=softmax_feat).entropy())

        """KL Divergence between N(mu, sigma^2) and N(0, 1)"""
        mu = torch.squeeze(mu)
        sigma = torch.squeeze(sigma)
        kld_loss = torch.mean(
            0.5
            * torch.sum(
                mu ** 2 + sigma ** 2 - torch.log(sigma ** 2 + 1e-12) - 1, dim=1
            ),
            dim=0,
        )
        # """KL Divergence between N(mu, sigma^2) and N(fake_mu, fake_sigma^2)"""
        # fake_mu = torch.squeeze(fake_mu)
        # fake_sigma = torch.squeeze(fake_sigma)
        # fake_kld_loss = torch.mean(
        #     0.5
        #     * torch.sum(
        #         torch.log(fake_sigma ** 2 + 1e-12)
        #         - torch.log(sigma ** 2 + 1e-12)
        #         + (sigma ** 2 + (mu - fake_mu) ** 2) / fake_sigma ** 2
        #         - 1,
        #         dim=1,
        #     ),
        #     dim=0,
        # )
        """KL Divergence between N(fake_mu, fake_sigma^2) and N(0, 1)"""
        fake_mu = torch.squeeze(fake_mu)
        fake_sigma = torch.squeeze(fake_sigma)
        fake_kld_loss = torch.mean(
            0.5
            * torch.sum(
                fake_mu ** 2 + fake_sigma ** 2 - torch.log(fake_sigma ** 2 + 1e-12) - 1,
                dim=1,
            ),
            dim=0,
        )

        # G = torch.distributions.Normal(0, 1)
        # P = torch.distributions.Normal(mu, log_var)
        # Q = torch.distributions.Normal(fake_mu, fake_log_var)
        # kld_loss = torch.distributions.kl_divergence(G, P).mean()
        # fake_kld_loss = torch.distributions.kl_divergence(G, Q).mean()

        w_in = weight[0]
        w_out = weight[1]
        w_kld = weight[2]
        w_fake_kld = weight[3]
        if reconstruction_loss == "CD":
            if w_in == 0 and w_fake_kld == 0:
                loss = w_out * out_loss + w_kld * kld_loss
            elif w_fake_kld == 0:
                loss = w_in * inner_loss + w_out * out_loss + w_kld * kld_loss
            else:
                loss = (
                    w_in * inner_loss
                    + w_out * out_loss
                    + w_kld * kld_loss
                    + w_fake_kld * fake_kld_loss
                )
        else:
            if w_in == 0:
                loss = w_out * out_loss + w_kld * kld_loss
            else:
                loss = w_in * inner_loss + w_out * out_loss + w_kld * kld_loss

        inner_loss_meter.update(inner_loss.item())
        out_loss_meter.update(out_loss.item())
        kld_loss_meter.update(kld_loss.item())
        fake_kld_loss_meter.update(fake_kld_loss.item())
        loss_meter.update(loss.item())

        # Backword Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - t_epoch_start
    epoch_loss = loss_meter.avg
    epoch_inner_loss = inner_loss_meter.avg
    epoch_out_loss = out_loss_meter.avg
    epoch_kld_loss = kld_loss_meter.avg
    epoch_fake_kld_loss = fake_kld_loss_meter.avg

    if not os.path.exists(os.path.join(save_dir, "original")):
        os.makedirs(os.path.join(save_dir, "original"))
    if not os.path.exists(os.path.join(save_dir, "folding1")):
        os.makedirs(os.path.join(save_dir, "folding1"))
    if not os.path.exists(os.path.join(save_dir, "reconstructed")):
        os.makedirs(os.path.join(save_dir, "reconstructed"))

    vis_points_3d(
        points[0],
        save_dir + f"/original/{epoch}.png",
    )
    point = points[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/original/{epoch}.npy", point)
    vis_points_3d(
        folding1[0],
        save_dir + f"/folding1/{epoch}.png",
    )
    folding = folding1[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/folding1/{epoch}.npy", folding)
    vis_points_3d(
        output[0],
        save_dir + f"/reconstructed/{epoch}.png",
    )
    out_point = output[0].to("cpu").detach().numpy()
    np.save(save_dir + f"/reconstructed/{epoch}.npy", out_point)

    return (
        epoch_loss,
        epoch_inner_loss,
        epoch_out_loss,
        epoch_kld_loss,
        epoch_fake_kld_loss,
        epoch_time,
    )
