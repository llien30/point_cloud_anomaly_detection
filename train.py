import argparse
import os
import random

import numpy as np
import torch
import wandb
import yaml
from addict import Dict

# from emd.emd_module import emdModule
from libs.checkpoint import save_checkpoint
from libs.dataset import ShapeNeth5pyDataset
from libs.foldingnet import SkipValiationalFoldingNet
from libs.helper import train_variational_foldingnet
from torch.utils.data import DataLoader


def get_parameters():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(description="take config file path")

    parser.add_argument("config", type=str, help="path of a config file for training")

    parser.add_argument("--no_wandb", action="store_true", help="Add --no_wandb option")

    return parser.parse_args()


def main():
    args = get_parameters()

    # configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    CONFIG = Dict(config_dict)
    print(config_dict)

    assert CONFIG.reconstruction_loss in [
        "CD",
        "EMD",
    ], "reconstruction loss must be CD(Chamfer Distance) or EMD(Earth mover's Distance)"

    if not args.no_wandb:
        wandb.init(
            config=CONFIG,
            name=CONFIG.name,
            project="ICIP2021",
        )

    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ShapeNeth5pyDataset(
        root_path=CONFIG.root_path,
        split="train",
        normal_class=CONFIG.normal_class,
        abnormal_class=[],
        n_point=CONFIG.n_points,
        random_rotate=CONFIG.rotate,
        random_jitter=CONFIG.jitter,
        random_translate=CONFIG.translate,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True
    )

    model = SkipValiationalFoldingNet(CONFIG.n_points, CONFIG.feat_dims, CONFIG.shape)

    model.to(device)
    if CONFIG.reconstruction_loss == "CD":
        # lr = 0.0001 * 100 / CONFIG.batch_size
        lr = 0.001
    elif CONFIG.reconstruction_loss == "EMD":
        lr = 0.0001 * 16 / CONFIG.batch_size
    beta1, beta2 = 0.9, 0.999

    optimizer = torch.optim.Adam(
        model.parameters(), lr, [beta1, beta2], weight_decay=1e-6
    )

    if not args.no_wandb:
        # Magic
        wandb.watch(model, log="all")

    print("---------- Start training ----------")

    for epoch in range(CONFIG.num_epochs):

        (
            epoch_loss,
            epoch_inner_loss,
            epoch_out_loss,
            epoch_kld_loss,
            epoch_fake_kld_loss,
            epoch_time,
        ) = train_variational_foldingnet(
            train_dataloader,
            model,
            CONFIG.reconstruction_loss,
            optimizer,
            CONFIG.weight,
            epoch,
            device,
            CONFIG.save_dir,
        )
        print(
            f"inner_loss:{epoch_inner_loss} || out_loss:{epoch_out_loss} || kld_loss:{epoch_kld_loss}"
        )

        if not os.path.exists(os.path.join(CONFIG.save_dir, "checkpoint")):
            os.makedirs(os.path.join(CONFIG.save_dir, "checkpoint"))

        save_checkpoint(
            os.path.join(CONFIG.save_dir, "checkpoint"),
            epoch,
            model,
            optimizer,
        )

        print(f"epoch{epoch} || LOSS:{epoch_loss} | time:{epoch_time}")

        wandb.log(
            {
                "train_time": epoch_time,
                "loss": epoch_loss,
                "inner_loss": epoch_inner_loss,
                "out_loss": epoch_out_loss,
                "kld_loss": epoch_kld_loss,
                "fake_kld_loss": epoch_fake_kld_loss,
            },
            step=epoch,
        )

    print("Done")


if __name__ == "__main__":
    main()
