import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> None:

    save_states = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(save_states, os.path.join(result_path, f"{epoch}.pth"))


def resume(
    resume_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[int, nn.Module, optim.Optimizer]:

    assert os.path.exists(resume_path), "there is no checkpoint at the result folder"

    print("loading checkpoint {}".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    print("Successfly loaded the weight of {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer
