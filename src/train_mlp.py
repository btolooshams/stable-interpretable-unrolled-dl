"""
Copyright (c) 2021 Bahareh Tolooshams

train mlp

:author: Bahareh Tolooshams
"""


import torch
import torch.nn.functional as F

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
        "-e",
        "--exp_name",
        type=str,
        help="experiment name",
        default="mnist/exp1",
    )
    parser.add_argument(
        "-c",
        "--class-list",
        type=list,
        help="list of classes from the dataset",
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "class_list": args.class_list,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "shuffle": True,
        "batch_size": 64,
        "num_workers": 4,
        # related to the Network
        "out_dim": 5,
        "in_dim": 784,
        "hid_dim": 500,
        # related to the optimizer
        "lr": 1e-4,
        "adam_eps": 1e-15,
        "num_epochs": 300,
        #
        "train_val_split": 0.85,
        "log_info_epoch_period": 10,
        "log_model_epoch_period": 300,
        "log_fig_epoch_period": 20,
        "tqdm_prints_disable": False,
        "code_reshape": (25, 20),
        #
        "random_split_manual_seed": 43,
    }

    return params


def test_network_for_classification(data_loader, net, params):

    device = params["device"]

    net.eval()

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for idx, (x, y) in tqdm(enumerate(data_loader), disable=True):

            x = x.to(device)
            y = y.to(device)

            # forward ------------------------------------------------------#
            z = net.encode(x)
            yhat = net.classify(z)

            correct_indicators = yhat.max(1)[1].data == y
            num_correct += correct_indicators.sum().item()
            num_total += y.size()[0]

    acc = num_correct / num_total

    return acc


def main():

    print("Train model x = Dz for mnist.")

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

    params["device"] = torch.device(params["device"])
    device = params["device"]

    # board --------------------------------------------------------------#
    writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
    writer.add_text("params", str(params))
    writer.flush()

    # load data ------------------------------------------------------------#
    X_tr, Y_tr, X_te, Y_te = utils.datasets.get_mnist_dataset(
        params["class_list"], make_flat=True
    )
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_tr), torch.tensor(Y_tr, dtype=torch.long)
    )
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

    # make dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=params["shuffle"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

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
    print("create MLP model.")
    net = model.MLP(params)

    torch.save(net, os.path.join(out_path, "model", "model_init.pt"))

    # optimizer ------------------------------------------------------------#
    optimizer = torch.optim.Adam(
        net.parameters(), lr=params["lr"], eps=params["adam_eps"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = torch.nn.CrossEntropyLoss()

    # train  ---------------------------------------------------------------#
    for epoch in tqdm(
        range(params["num_epochs"]), disable=params["tqdm_prints_disable"]
    ):
        net.train()

        for idx, (x, y) in tqdm(enumerate(train_loader), disable=True):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            # forward ------------------------------------------------------#
            z = net.encode(x)
            yhat = net.classify(z)
            loss = criterion(yhat, y)

            # backward -----------------------------------------------------#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % params["log_info_epoch_period"] == 0:

            train_acc = test_network_for_classification(train_loader, net, params)
            val_acc = test_network_for_classification(val_loader, net, params)
            test_acc = test_network_for_classification(test_loader, net, params)

            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/test", test_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("loss/class-train", loss.item(), epoch)

            writer.flush()

            if (epoch + 1) % params["log_fig_epoch_period"] == 0:
                writer = utils.board.log_code(
                    writer, z, epoch, reshape=params["code_reshape"]
                )
                writer = utils.board.log_mlp_encoder(
                    writer, net, epoch, reshape=(28, 28)
                )
            writer.flush()

        if (epoch + 1) % params["log_model_epoch_period"] == 0:
            torch.save(
                net, os.path.join(out_path, "model", "model_epoch{}.pt".format(epoch))
            )

    writer.close()


if __name__ == "__main__":
    main()
