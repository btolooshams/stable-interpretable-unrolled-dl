"""
Copyright (c) 2021 Bahareh Tolooshams

predict for the conv model x = Dz for color

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
        default="../results/cifar_color/exp1",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "train_num": 6000,
        "test_num": 100,
        "patch_size_org": 32,
    }

    return params


def main():

    print("Predict and save results on conv color model x = Dz.")

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
    if params["patch_size"] < params["patch_size_org"]:
        result_path = os.path.join(
            params["exp_path"], "trained_results_{}.pt".format(params["patch_size"])
        )
    else:
        result_path = os.path.join(params["exp_path"], "trained_results.pt")

    device = params["device"]

    # load data ------------------------------------------------------------#
    if params["dataset_name"] == "cifar":
        print("cifar!")
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_cifar_dataset(
            params["class_list"],
            blackandwhite=False,
            make_flat=False,
            whiten=params["data_whiten"],
        )
    else:
        print("Dataset is not implemented!")
        raise NotImplementedError

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_tr), torch.tensor(Y_tr, dtype=torch.long)
    )
    if params["overfit_to_only"]:
        train_num = params["overfit_to_only"]
    else:
        train_num = np.int32(X_tr.shape[0] * params["train_val_split"])
    val_num = X_tr.shape[0] - train_num
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_num, val_num],
        generator=torch.Generator().manual_seed(params["random_split_manual_seed"]),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_te), torch.tensor(Y_te, dtype=torch.long)
    )

    print("classes {} are in the dataset.".format(params["class_list"]))
    print(
        "total number of train/val/test data is {}/{}/{}".format(
            train_num, val_num, X_te.shape[0]
        )
    )

    # make dataloader ------------------------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=params["shuffle"],
        batch_size=1,
        num_workers=params["num_workers"],
    )

    if val_num:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=params["shuffle"],
            batch_size=1,
            num_workers=params["num_workers"],
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    # make transforms -------------------------------------------------------#
    if params["patch_size"] < params["patch_size_org"]:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomResizedCrop(params["patch_size"]),]
        )
    else:
        transform = None

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
        (
            params["train_num"],
            params["num_ch"],
            params["patch_size"],
            params["patch_size"],
        ),
        device=device,
    )
    Xhat_train = torch.zeros(
        (
            params["train_num"],
            params["num_ch"],
            params["patch_size"],
            params["patch_size"],
        ),
        device=device,
    )
    Z_train = torch.zeros(
        (params["train_num"], z_shape[1], z_shape[2], z_shape[3]), device=device
    )
    Y_train = torch.zeros(params["train_num"], device=device)

    X_test = torch.zeros(
        (
            params["test_num"],
            params["num_ch"],
            params["patch_size"],
            params["patch_size"],
        ),
        device=device,
    )
    Xhat_test = torch.zeros(
        (
            params["test_num"],
            params["num_ch"],
            params["patch_size"],
            params["patch_size"],
        ),
        device=device,
    )
    Z_test = torch.zeros(
        (params["test_num"], z_shape[1], z_shape[2], z_shape[3]), device=device
    )
    Y_test = torch.zeros(params["test_num"], device=device)

    # train
    print("predict train.")
    ctr = 0
    flag_ctr = True
    while flag_ctr:
        for idx, (x, y) in tqdm(enumerate(train_loader), disable=True):

            x = x.to(device)
            y = y.to(device)

            if transform:
                x = transform(x)

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
            Y_train[idx] = torch.squeeze(y.clone())

            ctr += 1
            if ctr == params["train_num"]:
                flag_ctr = False
                break

    trained_results["X_train"] = X_train.clone()
    trained_results["Xhat_train"] = Xhat_train.clone()
    trained_results["Z_train"] = Z_train.clone()
    trained_results["Y_train"] = Y_train.clone()

    # test
    print("predict test.")
    ctr = 0
    flag_ctr = True
    while flag_ctr:
        for idx, (x, _) in tqdm(enumerate(test_loader), disable=True):
            x = x.to(device)

            if transform:
                x = transform(x)

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
    trained_results["Y_test"] = Y_test.clone()

    torch.save(trained_results, result_path)


if __name__ == "__main__":
    main()
