"""
Copyright (c) 2021 Bahareh Tolooshams

train the model x = Dz for simulated data

:author: Bahareh Tolooshams
"""


import torch

import os
import json
import numpy as np
import tensorboardX
from datetime import datetime
from tqdm import tqdm, trange
import argparse

import sys

sys.path.append("src/")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--exp_name", type=str, help="experiment name", default="exp1",
    )
    parser.add_argument(
        "-n", "--network", type=str, help="network", default="AE",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        help="data path",
        default="../data/simulated_data.pt",
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "network": args.network,
        "data_path": args.data_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "shuffle": True,
        "batch_size": 64,
        "num_workers": 4,
        # related to the Network
        "m": 50,
        "p": 100,
        "lam": 1,
        "num_layers": 50,
        "twosided": False,
        "step": 0.1,
        # related to the optimizer
        "lr": 1e-3,
        "adam_eps": 1e-8,
        # related to DLLoss
        "lam_loss": 0.0,
        "rho_loss": 0.0,
        #
        "normalize": True,
        "num_epochs": 100,
        #
        "log_info_epoch_period": 5,
        "log_model_epoch_period": 5,
        "tqdm_prints_disable": False,
    }

    return params


def main():

    print("Train model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    # make folder for results
    out_path = os.path.join(
        "..", "results", "{}_{}".format(params["exp_name"], params["random_date"])
    )
    params["out_path"] = out_path
    os.makedirs(params["out_path"])
    os.makedirs(os.path.join(params["out_path"], "model"))
    os.makedirs(os.path.join(params["out_path"], "figures"))

    # dump params  ---------------------------------------------------------#
    with open(os.path.join(params["out_path"], "params.txt"), "w") as file:
        file.write(json.dumps(params, sort_keys=True, separators=("\n", ":")))

    params["device"] = torch.device(params["device"])
    device = params["device"]

    # board --------------------------------------------------------------#
    writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
    writer.add_text("params", str(params))
    writer.flush()

    # load data ------------------------------------------------------------#
    train_dataset = utils.datasets.get_simulated_dataset(
        data_path=params["data_path"], device=params["device"]
    )

    # make dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=params["shuffle"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # create model ---------------------------------------------------------#
    print("create model.")
    if params["network"] == "AE":
        net = model.AE(params)
        if params["normalize"]:
            net.normalize()
    else:
        print("Network is not implemented!")
        raise NotImplementedError

    # optimizer ------------------------------------------------------------#
    optimizer = torch.optim.Adam(
        net.parameters(), lr=params["lr"], eps=params["adam_eps"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = utils.loss.DLLoss1D(params)

    # train  ---------------------------------------------------------------#
    for epoch in tqdm(
        range(params["num_epochs"]), disable=params["tqdm_prints_disable"]
    ):
        net.train()

        for idx, (x, z) in tqdm(enumerate(train_loader), disable=True):
            optimizer.zero_grad()

            z = z.to(device)
            x = x.to(device)

            # forward ------------------------------------------------------#
            zT = net.encode(x)
            xhat = net.decode(zT)
            dhat = net.W
            loss = criterion(x, xhat, zT, dhat)

            # backward -----------------------------------------------------#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if params["normalize"]:
                net.normalize()

        if epoch % params["log_info_epoch_period"] == 0:
            writer.add_scalar("loss/train", loss.item(), epoch)
            writer.flush()

        if epoch % params["log_model_epoch_period"] == 0:
            torch.save(
                net, os.path.join(out_path, "model", "model_epoch{}.pt".format(epoch))
            )

    writer.close()


if __name__ == "__main__":
    main()
