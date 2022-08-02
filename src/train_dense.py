"""
Copyright (c) 2021 Bahareh Tolooshams

train the model x = Dz for bw

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
        "-e", "--exp_name", type=str, help="experiment name", default="mnist/exp1",
    )
    parser.add_argument(
        "-n", "--network", type=str, help="network", default="AE",
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
        "network": args.network,
        "class_list": args.class_list,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "dataset_name": "mnist",
        "shuffle": True,
        "batch_size": 32,
        "num_workers": 4,
        "overfit_to_only": None,
        # data processing
        "data_normalize": False,
        "data_whiten": False,
        "blackandwhite": True,
        # related to the Network
        "num_class": 5,
        "beta": 0,
        "init_model_path": None,
        "m": 784,
        "p": 500,
        "lam": 0.7,
        "step": 1,
        "num_layers": 15,
        "twosided": False,
        # related to the optimizer
        "lr": 1e-4,
        "lr_step": 200,
        "lr_decay": 0.1,
        "adam_eps": 1e-15,
        # related to DLLoss
        "lam_loss": 0.7,
        "rho_loss": 0,
        "noise_std": 0,
        #
        "normalize": False,
        "num_epochs": 200,
        #
        "train_val_split": 1,
        "log_info_epoch_period": 10,
        "log_model_epoch_period": 200,
        "log_fig_epoch_period": 25,
        "tqdm_prints_disable": False,
        "code_reshape": (25, 20),
        "data_reshape": (28, 28),
        #
        "random_split_manual_seed": 1099,
    }

    return params


def test_network_for_classification(data_loader, net, classifier, params):

    device = params["device"]

    net.eval()

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for idx, (x, y) in tqdm(enumerate(data_loader), disable=True):

            x = x.to(device)
            y = y.to(device)

            # forward ------------------------------------------------------#
            if params["noise_std"]:
                x_noisy = (
                    x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
                ).to(device)
                zT = net.encode(x_noisy)
            else:
                zT = net.encode(x)
            xhat = net.decode(zT)

            code = torch.squeeze(F.normalize(zT, dim=1), dim=-1)
            yhat = classifier(code)

            correct_indicators = yhat.max(1)[1].data == y
            num_correct += correct_indicators.sum().item()
            num_total += y.size()[0]

    acc = num_correct / num_total

    return acc


def main():

    print("Train model x = Dz for dense.")

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
            make_flat=True,
            normalize=params["data_normalize"],
            whiten=params["data_whiten"],
        )
    elif params["dataset_name"] == "cifar":
        print("cifar!")
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_cifar_dataset(
            params["class_list"], params["blackandwhite"], make_flat=True
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

    # make dataloader
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
        if params["network"] == "AE":
            net = model.AE(params)
        elif params["network"] == "AElearnbias":
            net = model.AElearnbias(params)
        else:
            print("Network is not implemented!")
            raise NotImplementedError

        if params["normalize"]:
            net.normalize()

    if params["beta"]:
        classifier = model.Classifier(params)

    torch.save(net, os.path.join(out_path, "model", "model_init.pt"))

    # optimizer ------------------------------------------------------------#
    if params["beta"]:
        net_params = []
        for param in net.parameters():
            net_params.append(param)
        for param in classifier.parameters():
            net_params.append(param)
        optimizer = torch.optim.Adam(
            net_params, lr=params["lr"], eps=params["adam_eps"]
        )
    else:
        optimizer = torch.optim.Adam(
            net.parameters(), lr=params["lr"], eps=params["adam_eps"]
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params["lr_step"], gamma=params["lr_decay"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = utils.loss.DLLoss1D(params)

    if params["beta"]:
        criterion_class = torch.nn.CrossEntropyLoss()

    # train  ---------------------------------------------------------------#
    for epoch in tqdm(
        range(params["num_epochs"]), disable=params["tqdm_prints_disable"]
    ):
        net.train()

        if epoch > 0:
            scheduler.step()

        for idx, (x, y) in tqdm(enumerate(train_loader), disable=True):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            if params["noise_std"]:
                x_noisy = (
                    x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
                ).to(device)
                zT = net.encode(x_noisy)
            else:
                zT = net.encode(x)
            # forward ------------------------------------------------------#

            xhat = net.decode(zT)
            dhat = net.W

            # classification
            if params["beta"]:
                code = torch.squeeze(F.normalize(zT, dim=1), dim=-1)
                yhat = classifier(code)

                loss_class = criterion_class(yhat, y)
            else:
                loss_class = 0.0

            loss_ae = criterion(x, xhat, zT, dhat)

            loss = (1 - params["beta"]) * loss_ae + params["beta"] * loss_class

            # backward -----------------------------------------------------#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if params["normalize"]:
                net.normalize()

        if (epoch + 1) % params["log_info_epoch_period"] == 0:

            if params["beta"]:
                train_acc = test_network_for_classification(
                    train_loader, net, classifier, params
                )
                if val_num:
                    val_acc = test_network_for_classification(
                        val_loader, net, classifier, params
                    )
                test_acc = test_network_for_classification(
                    test_loader, net, classifier, params
                )

                writer.add_scalar("acc/train", train_acc, epoch)
                writer.add_scalar("acc/test", test_acc, epoch)
                writer.add_scalar("acc/val", val_acc, epoch)
                writer.add_scalar("loss/class-train", loss_class.item(), epoch)

            writer.add_scalar("loss/ae-train", loss_ae.item(), epoch)
            writer.add_scalar("loss/total-train", loss.item(), epoch)
            writer.flush()

            if (epoch + 1) % params["log_fig_epoch_period"] == 0:
                writer = utils.board.log_code(
                    writer, zT, epoch, reshape=params["code_reshape"]
                )
                writer = utils.board.log_dictionary(
                    writer, net, epoch, reshape=params["data_reshape"],
                )
                writer = utils.board.log_img(
                    writer, x, xhat, epoch, reshape=params["data_reshape"],
                )
            writer.flush()

        if (epoch + 1) % params["log_model_epoch_period"] == 0:
            torch.save(
                net, os.path.join(out_path, "model", "model_epoch{}.pt".format(epoch))
            )
            if params["beta"]:
                torch.save(
                    classifier,
                    os.path.join(
                        out_path, "model", "classifier_epoch{}.pt".format(epoch)
                    ),
                )

    writer.close()


if __name__ == "__main__":
    main()
