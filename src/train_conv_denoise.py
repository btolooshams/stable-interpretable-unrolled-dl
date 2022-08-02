"""
Copyright (c) 2021 Bahareh Tolooshams

train the model x = Dz for denoising

:author: Bahareh Tolooshams
"""


import torch
import torch.nn.functional as F
import torchvision

import os
import json
import pickle
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
        "-e", "--exp_name", type=str, help="experiment name", default="bsd/exp1",
    )
    parser.add_argument(
        "-n", "--network", type=str, help="network", default="CAE",
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "network": args.network,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "shuffle": True,
        "batch_size": 1,
        "num_workers": 4,
        # data processing
        "data_normalize": False,
        "data_whiten": False,
        # related to the Network
        "init_model_path": None,
        "num_conv": 64,
        "dictionary_dim": 9,
        "stride": 4,
        "split_stride": 1,
        "patch_size": 129,
        "lam": 0.12,
        "step": 0.1,
        "num_layers": 15,
        "twosided": False,
        # related to the optimizer
        "lr": 1e-4,
        # "lr": 1e-2,
        "lr_step": 2000,
        "lr_decay": 1,
        "adam_eps": 1e-3,
        "adam_weight_decay": 0,
        # related to DLLoss
        "lam_loss": 0.00,
        "rho_loss": 0,
        "noise_std": 25,
        #
        "scale_dictionary_init": None,
        "normalize": False,
        "num_epochs": 400,
        #
        "train_val_split": 1,
        "log_info_epoch_period": 10,
        "log_model_epoch_period": 400,
        "log_fig_epoch_period": 100,
        "tqdm_prints_disable": False,
        #
        "random_split_manual_seed": 1099,
        "train_image_path": "../data/CBSD432",
        "test_image_path": "../data/BSD68",
    }

    return params


def main():

    print("Train model x = Dz for conv denoising.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    print("Exp: {}".format(params["exp_name"]))

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

    with open(os.path.join(params["out_path"], "params.pickle"), "wb") as file:
        pickle.dump(params, file)

    params["device"] = torch.device(params["device"])
    device = params["device"]

    # board --------------------------------------------------------------#
    writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
    writer.add_text("params", str(params))
    writer.flush()

    # load data ------------------------------------------------------------#
    train_dataset = torchvision.datasets.ImageFolder(
        root=params["train_image_path"],
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(
                    params["patch_size"],
                    padding=None,
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=params["test_image_path"],
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomCrop(
                    params["patch_size"],
                    padding=None,
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    # make dataloader ------------------------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=params["shuffle"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    # create model ---------------------------------------------------------#
    print("create model.")
    if params["init_model_path"]:
        print("initialize from a trained model.")
        net = torch.load(params["init_model_path"], map_location=params["device"])
    else:
        if params["network"] == "CAE":
            net = model.CAE(params)
        elif params["network"] == "CAElearnbias":
            net = model.CAElearnbias(params)
        else:
            print("Network is not implemented!")
            raise NotImplementedError

        if params["normalize"]:
            net.normalize()

        if params["scale_dictionary_init"]:
            print("scale norm of W.")
            net.W.data = params["scale_dictionary_init"] * net.W.data

    torch.save(net, os.path.join(out_path, "model", "model_init.pt"))

    # optimizer ------------------------------------------------------------#
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=params["lr"],
        eps=params["adam_eps"],
        weight_decay=params["adam_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params["lr_step"], gamma=params["lr_decay"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = utils.loss.DLLoss2D(params)

    # train  ---------------------------------------------------------------#
    for epoch in tqdm(
        range(params["num_epochs"]), disable=params["tqdm_prints_disable"]
    ):
        net.train()

        if epoch > 0:
            scheduler.step()

        for idx, (x, _) in tqdm(enumerate(train_loader), disable=True):
            optimizer.zero_grad()

            x = x.to(device)

            # forward ------------------------------------------------------#
            if params["noise_std"]:
                x_noisy = (
                    x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
                ).to(device)
                xhat, zT = net(x_noisy)
            else:
                xhat, zT = net(x)

            dhat = net.W
            loss = criterion(x, xhat, zT, dhat)

            # backward -----------------------------------------------------#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if params["normalize"]:
                net.normalize()

        if (epoch + 1) % params["log_info_epoch_period"] == 0:

            writer.add_scalar("loss/train", loss.item(), epoch)

            test_psnr = utils.utils.test_network(test_loader, net, params)
            writer.add_scalar("psnr/test", test_psnr, epoch)

            writer.flush()

            if (epoch + 1) % params["log_fig_epoch_period"] == 0:
                writer = utils.board.log_dictionary_conv(writer, net, epoch)
                if params["noise_std"]:
                    writer = utils.board.log_img(writer, x_noisy, xhat, epoch)
                else:
                    writer = utils.board.log_img(writer, x, xhat, epoch)
            writer.flush()

        if (epoch + 1) % params["log_model_epoch_period"] == 0:
            torch.save(
                net, os.path.join(out_path, "model", "model_epoch{}.pt".format(epoch))
            )

    writer.close()


if __name__ == "__main__":
    main()
