"""
Copyright (c) 2021 Bahareh Tolooshams

predict for the model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.gridspec as gridspec
import pickle


import sys

sys.path.append("src/")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-path",
        type=str,
        help="experiment path",
        default="../results/bsd/exp1_xxx",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "train_num": 500,
        "test_num": 68,
    }

    return params


def main():

    print("Predict and save results on conv model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    params_pickle = pickle.load(
        open(os.path.join(params["exp_path"], "params.pickle"), "rb")
    )
    for key in params.keys():
        params_pickle[key] = params[key]
    params = params_pickle

    print("Exp: {}".format(params["exp_path"]))

    model_path = os.path.join(
        params["exp_path"],
        "model",
        "model_epoch{}.pt".format(params["num_epochs"] - 1),
    )
    result_path = os.path.join(params["exp_path"], "trained_results.pt")

    device = params["device"]

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
        train_dataset, shuffle=False, batch_size=1,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1,
    )

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=params["device"])
    net.eval()

    trained_results = {}
    trained_results["D"] = torch.squeeze(net.W.data.clone(), dim=1)

    if params["network"] == "CAElearnbias":
        trained_results["bias"] = torch.squeeze(net.b.data.clone())
        print(trained_results["bias"])

    # compute the dim of input data and code
    xtmp, _ = train_dataset.__getitem__(0)
    xtmp = torch.unsqueeze(xtmp, dim=0)
    _, z = net(xtmp)
    z_shape = z.shape
    #
    X_train = torch.zeros(
        (params["train_num"], params["patch_size"], params["patch_size"]), device=device
    )
    Xhat_train = torch.zeros(
        (params["train_num"], params["patch_size"], params["patch_size"]), device=device
    )
    Z_train = torch.zeros(
        (params["train_num"], z_shape[1], z_shape[2], z_shape[3]), device=device
    )

    X_test = torch.zeros(
        (params["test_num"], params["patch_size"], params["patch_size"]), device=device
    )
    Xhat_test = torch.zeros(
        (params["test_num"], params["patch_size"], params["patch_size"]), device=device
    )
    Z_test = torch.zeros(
        (params["test_num"], z_shape[1], z_shape[2], z_shape[3]), device=device
    )

    # train
    print("predict train.")
    ctr = 0
    flag_ctr = True
    while flag_ctr:
        for idx, (x, _) in tqdm(enumerate(train_loader), disable=True):

            print(ctr)
            x = x.to(device)

            if params["noise_std"]:
                x_noisy = (
                    x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
                ).to(device)
                xhat, zT = net(x_noisy)
            else:
                xhat, zT = net(x)

            X_train[ctr] = torch.squeeze(x.clone())
            Xhat_train[ctr] = torch.squeeze(xhat.clone())
            Z_train[ctr] = torch.mean(zT, dim=0).clone()

            ctr += 1
            if ctr == params["train_num"]:
                flag_ctr = False
                break

    trained_results["X_train"] = X_train.clone()
    trained_results["Xhat_train"] = Xhat_train.clone()
    trained_results["Z_train"] = Z_train.clone()

    # test
    print("predict test.")
    ctr = 0
    flag_ctr = True
    while flag_ctr:
        for idx, (x, _) in tqdm(enumerate(test_loader), disable=True):
            x = x.to(device)

            if params["noise_std"]:
                x_noisy = (
                    x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
                ).to(device)
                xhat, zT = net(x_noisy)
            else:
                xhat, zT = net(x)

            X_test[ctr] = torch.squeeze(x.clone())
            Xhat_test[ctr] = torch.squeeze(xhat.clone())
            Z_test[ctr] = torch.mean(zT, dim=0).clone()

            ctr += 1
            if ctr == params["test_num"]:
                flag_ctr = False
                break

    trained_results["X_test"] = X_test.clone()
    trained_results["Xhat_test"] = Xhat_test.clone()
    trained_results["Z_test"] = Z_test.clone()

    torch.save(trained_results, result_path)


if __name__ == "__main__":
    main()
