"""
Copyright (c) 2021 Bahareh Tolooshams

train the model x = Dz for cifar color

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
        "--exp_path",
        type=str,
        help="experiment path",
        default="../results/cifar_color/exp1_xxx",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "shuffle": True,
        "batch_size": 32,
        "num_workers": 4,
        "p": 900,
        "lr": 1e-3,
        "lr_step": 2000,
        "lr_decay": 1,
        "adam_eps": 1e-15,
        "adam_weight_decay": 0,
        "num_epochs": 500,
        "train_val_split": 1,
        "log_info_epoch_period": 10,
        "log_model_epoch_period": 250,
        "tqdm_prints_disable": False,
        #
        "random_split_manual_seed": 1099,
    }

    return params


def test_network_for_classification(data_loader, net, params):

    device = params["device"]

    net.eval()

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for idx, (z, y) in tqdm(enumerate(data_loader), disable=True):

            z = z.to(device)
            y = y.to(device)

            # forward ------------------------------------------------------#
            yhat = net(z)

            correct_indicators = yhat.max(1)[1].data == y
            num_correct += correct_indicators.sum().item()
            num_total += y.size()[0]

    acc = num_correct / num_total

    return acc


def main():

    print("Train model x = Dz for conv denoising.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    params_pickle = pickle.load(
        open(os.path.join(params["exp_path"], "params.pickle"), "rb")
    )
    for key in params.keys():
        params_pickle[key] = params[key]
    params = params_pickle
    params["num_class"] = len(params["class_list"])

    print("Exp: {}".format(params["exp_path"]))

    model_path = os.path.join(
        params["exp_path"],
        "model",
        "model_epoch{}.pt".format(params["num_epochs"] - 1),
    )

    # make folder for classification results
    class_path = os.path.join(
        params["exp_path"], "classification_{}".format(params["random_date"])
    )
    os.makedirs(class_path)
    os.makedirs(os.path.join(class_path, "model"))

    params["device"] = torch.device(params["device"])
    device = params["device"]

    # board --------------------------------------------------------------#
    writer = tensorboardX.SummaryWriter(os.path.join(class_path))
    writer.add_text("params", str(params))
    writer.flush()

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
    net = torch.load(model_path, map_location=device)
    net.eval()

    net_class = model.Classifier(params)

    # train data -----------------------------------------------------------#
    z_train = []
    y_train = []
    pooling = torch.nn.AvgPool2d(kernel_size=8)

    ctr = 0
    for idx, (x, y) in tqdm(enumerate(train_loader), disable=True):

        x = x.to(device)
        y = y.to(device)

        _, z = net(x)
        z_pool = pooling(z)
        z_flat = torch.flatten(z_pool, start_dim=1)

        z_train.append(z_flat.clone().detach().cpu())
        y_train.append(y.clone().detach().cpu())

        ctr += 1

    train_dataset_for_classification = torch.utils.data.TensorDataset(
        torch.Tensor(torch.cat(z_train, dim=0)),
        torch.tensor(torch.cat(y_train, dim=0), dtype=torch.long),
    )
    train_loader_for_classification = torch.utils.data.DataLoader(
        train_dataset_for_classification,
        shuffle=params["shuffle"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # test data -----------------------------------------------------------#
    z_test = []
    y_test = []
    for idx, (x, y) in tqdm(enumerate(test_loader), disable=True):

        x = x.to(device)
        y = y.to(device)

        _, z = net(x)
        z_pool = pooling(z)
        z_flat = torch.flatten(z_pool, start_dim=1)

        z_test.append(z_flat.clone().detach().cpu())
        y_test.append(y.clone().detach().cpu())

    test_dataset_for_classification = torch.utils.data.TensorDataset(
        torch.Tensor(torch.cat(z_test, dim=0)),
        torch.tensor(torch.cat(y_test, dim=0), dtype=torch.long),
    )
    test_loader_for_classification = torch.utils.data.DataLoader(
        test_dataset_for_classification,
        shuffle=False,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # optimizer ------------------------------------------------------------#
    optimizer = torch.optim.Adam(
        net_class.parameters(),
        lr=params["lr"],
        eps=params["adam_eps"],
        weight_decay=params["adam_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params["lr_step"], gamma=params["lr_decay"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = torch.nn.CrossEntropyLoss()

    # train  ---------------------------------------------------------------#
    for epoch in tqdm(
        range(params["num_epochs"]), disable=params["tqdm_prints_disable"]
    ):

        if epoch > 0:
            scheduler.step()

        for idx, (z, y) in tqdm(
            enumerate(train_loader_for_classification), disable=True
        ):
            optimizer.zero_grad()

            z = z.to(device)
            y = y.to(device)

            yhat = net_class(z)
            loss = criterion(yhat, y)

            # backward -----------------------------------------------------#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % params["log_info_epoch_period"] == 0:

            train_acc = test_network_for_classification(
                train_loader_for_classification, net_class, params
            )
            test_acc = test_network_for_classification(
                test_loader_for_classification, net_class, params
            )

            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/test", test_acc, epoch)
            if val_num:
                writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("loss/class-train", loss.item(), epoch)

            writer.flush()

        if (epoch + 1) % params["log_model_epoch_period"] == 0:
            torch.save(
                net, os.path.join(class_path, "model", "model_epoch{}.pt".format(epoch))
            )

    writer.close()


if __name__ == "__main__":
    main()
