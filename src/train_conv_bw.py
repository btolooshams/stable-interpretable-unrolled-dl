"""
Copyright (c) 2021 Bahareh Tolooshams

train the conv model x = Dz for bw

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
        "-e",
        "--exp_name",
        type=str,
        help="experiment name",
        default="cifar/exp1",

    )
    parser.add_argument(
        "-n", "--network", type=str, help="network", default="CAE",
    )
    parser.add_argument(
        "-d", "--dataset-name", type=str, help="name of dataset", default="cifar",
    )
    parser.add_argument(
        "-c",
        "--class-list",
        type=list,
        help="list of classes from the dataset",
        default=[0, 1],
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "network": args.network,
        "dataset_name": args.dataset_name,
        "class_list": args.class_list,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "shuffle": True,
        "batch_size": 32,
        "num_workers": 4,
        # data processing
        "data_normalize": False,
        "data_whiten": False,
        "blackandwhite": True,
        # related to the Network
        "init_model_path": None,
        "num_conv": 64,
        "dictionary_dim": 7,
        "stride": 1,
        "split_stride": 1,
        "patch_size": 32,
        "lam": 0.5,
        "step": 0.1,
        "num_layers": 15,
        "twosided": False,
        # related to the optimizer
        "lr": 1e-4,
        "lr_step": 2000,
        "lr_decay": 1,
        "adam_eps": 1e-15,
        "adam_weight_decay": 0,
        # related to DLLoss
        "lam_loss": 0.5,
        "rho_loss": 0,
        "noise_std": 0,
        #
        "scale_dictionary_init": None,
        "normalize": False,
        "num_epochs": 500,
        #
        "overfit_to_only": None,
        "train_val_split": 1,
        "log_info_epoch_period": 50,
        "log_model_epoch_period": 500,
        "log_fig_epoch_period": 50,
        "tqdm_prints_disable": False,
        #
        "random_split_manual_seed": 1099,
    }

    return params


def compute_psnr(x, xhat):
    psnr = []
    for i in range(x.shape[0]):
        mse = np.mean((x[i] - xhat[i]) ** 2)
        max_x = np.max(x[i])
        psnr.append(20 * np.log10(max_x) - 10 * np.log10(mse))
    return np.mean(psnr)


def test_network(data_loader, net, params, name="test"):

    net.eval()

    device = params["device"]

    psnr = []
    for idx, (x, _) in enumerate(data_loader):

        x = x.to(device)

        # forward ------------------------------------------------------#
        if params["noise_std"]:
            x_noisy = (
                x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
            ).to(device)
            xhat, _ = net(x_noisy)
        else:
            xhat, _ = net(x)

        xhat = torch.clamp(xhat, 0, 1)

        psnr.append(
            compute_psnr(
                x[:, 0].clone().detach().cpu().numpy(),
                xhat[:, 0].clone().detach().cpu().numpy(),
            )
        )

    psnr = np.mean(np.array(psnr))

    return psnr


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
    if params["dataset_name"] == "mnist":
        print("mnist!")
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_mnist_dataset(
            params["class_list"],
            make_flat=False,
            normalize=params["data_normalize"],
            whiten=params["data_whiten"],
        )
    elif params["dataset_name"] == "cifar":
        print("cifar!")
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_cifar_dataset(
            params["class_list"],
            params["blackandwhite"],
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
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    if val_num:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=params["shuffle"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
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

            test_psnr = test_network(test_loader, net, params)
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
