"""
Copyright (c) 2021 Bahareh Tolooshams

predict for the dense model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torch.nn.functional as F
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
        default="../results/cifar/exp1",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "num_partial": 3000,
    }

    return params


def get_partial_data(X, Y, params):
    # get partial data
    c_indices = []
    X_c = []
    Y_c = []
    for c in range(len(params["class_list"])):
        c_indices.append(Y == params["class_list"][c])
        X_c.append(torch.Tensor(X[c_indices[c]]))
        Y_c.append(torch.Tensor(Y[c_indices[c]]))

    X_partial = []
    Y_partial = []
    for c in range(len(params["class_list"])):
        shuffled_indices = np.linspace(0, X_c[c].shape[0] - 1, X_c[c].shape[0])
        np.random.shuffle(shuffled_indices)
        X_partial.append(
            X_c[c][
                shuffled_indices[
                    : np.int32(params["num_partial"] / len(params["class_list"]))
                ]
            ]
        )
        Y_partial.append(
            Y_c[c][
                shuffled_indices[
                    : np.int32(params["num_partial"] / len(params["class_list"]))
                ]
            ]
        )

    X_partial = torch.cat(X_partial, dim=0)
    Y_partial = torch.cat(Y_partial, dim=0)

    return X_partial, Y_partial


def main():

    print("Predict and save results on model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    print(
        "WE ARE ONLY PREDICTING PARTIAL DATA OF SIZE {}!".format(params["num_partial"])
    )

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
    if params["beta"]:
        classifier_path = os.path.join(
            params["exp_path"],
            "model",
            "classifier_epoch{}.pt".format(params["num_epochs"] - 1),
        )
    result_path = os.path.join(params["exp_path"], "trained_results.pt")

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    # load dataset ------------------------------------------------------#
    if params["dataset_name"] == "mnist":
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_mnist_dataset(
            params["class_list"], make_flat=True
        )
    elif params["dataset_name"] == "cifar":
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_cifar_dataset(
            params["class_list"], params["blackandwhite"], make_flat=True
        )
    else:
        print("Dataset is not implemented!")
        raise NotImplementedError

    # get partial dataset -----------------------------------------------#
    X_tr, Y_tr = get_partial_data(X_tr, Y_tr, params)
    X_te, Y_te = get_partial_data(X_te, Y_te, params)
    params["train_val_split"] = 1

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
        train_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    net = torch.load(model_path, map_location=params["device"])
    net.eval()

    if params["beta"]:
        classifier = torch.load(classifier_path, map_location=params["device"])
        classifier.eval()

    trained_results = {}
    trained_results["D"] = net.W.data.clone()

    X_train = torch.zeros((net.m, train_num), device=device)  # m x n
    Z_train = torch.zeros((net.p, train_num), device=device)  # p x n
    Y_train = torch.zeros(train_num, device=device)  # n
    if params["beta"]:
        Yhat_train = torch.zeros(train_num, device=device)  # n

    X_test = torch.zeros((net.m, X_te.shape[0]), device=device)
    Xhat_test = torch.zeros((net.m, X_te.shape[0]), device=device)
    Z_test = torch.zeros((net.p, X_te.shape[0]), device=device)
    Y_test = torch.zeros((X_te.shape[0]), device=device)  # n
    if params["beta"]:
        Yhat_test = torch.zeros((X_te.shape[0]), device=device)  # n

    # train
    for idx, (x, y) in tqdm(enumerate(train_loader), disable=True):
        x = x.to(device)
        zT = net.encode(x)

        if params["beta"]:
            code = torch.squeeze(F.normalize(zT, dim=1), dim=-1)
            yhat = classifier(code)

        X_train[:, idx] = torch.squeeze(x.clone())
        Z_train[:, idx] = torch.squeeze(zT.clone())
        Z_train[:, idx] = torch.squeeze(zT.clone())
        Y_train[idx] = torch.squeeze(y.clone())
        if params["beta"]:
            Yhat_train[idx] = torch.squeeze(yhat.max(1)[1].data.clone())

    trained_results["X_train"] = X_train.clone()
    trained_results["Z_train"] = Z_train.clone()
    trained_results["Y_train"] = Y_train.clone()
    if params["beta"]:
        trained_results["Yhat_train"] = Yhat_train.clone()

    # test
    for idx, (x, y) in tqdm(enumerate(test_loader), disable=True):
        x = x.to(device)
        zT = net.encode(x)
        xhat = net.decode(zT)

        if params["beta"]:
            code = torch.squeeze(F.normalize(zT, dim=1), dim=-1)
            yhat = classifier(code)

        X_test[:, idx] = torch.squeeze(x.clone())
        Xhat_test[:, idx] = torch.squeeze(xhat.clone())
        Z_test[:, idx] = torch.squeeze(zT.clone())
        Y_test[idx] = torch.squeeze(y.clone())

        if params["beta"]:
            Yhat_test[idx] = torch.squeeze(yhat.max(1)[1].data.clone())

    trained_results["X_test"] = X_test.clone()
    trained_results["Xhat_test"] = Xhat_test.clone()
    trained_results["Z_test"] = Z_test.clone()
    trained_results["Y_test"] = Y_test.clone()
    if params["beta"]:
        trained_results["Yhat_test"] = Yhat_test.clone()

    torch.save(trained_results, result_path)


if __name__ == "__main__":
    main()
