"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torchvision
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

sys.path.append("../")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-path",
        type=str,
        help="experiment path",
        default="../../results/exp1",
        )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reshape": (28, 28),
    }

    return params


def main():

    print(
        "Visualzie image interpolation based on code similarity on dense model x = Dz."
    )

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
    if params["beta"]:
        classifier_path = os.path.join(
            params["exp_path"],
            "model",
            "classifier_epoch{}.pt".format(params["num_epochs"] - 1),
        )
    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/image_interpolation_based_on_code_similarity_for_newimage".format(
        params["exp_path"]
    )
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    net = torch.load(model_path, map_location=params["device"])

    _, _, X_cifar, _ = utils.datasets.get_cifar_dataset(
        [0], True, make_flat=False
    )

    X_cifar_samples = X_cifar[:50]
    X_cifar_samples = torch.Tensor(X_cifar_samples)

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomResizedCrop(params["reshape"]),]
    )
    X_cifar_samples = transform(X_cifar_samples)
    X_cifar_samples = X_cifar_samples.reshape(-1, 784, 1)
    X_cifar_samples_hat, Z_cifar_samples = net(X_cifar_samples)

    X_cifar_samples = torch.squeeze(X_cifar_samples, dim=-1).T
    X_cifar_samples_hat = torch.squeeze(X_cifar_samples_hat, dim=-1).T.detach()
    Z_cifar_samples = torch.squeeze(Z_cifar_samples, dim=-1).T.detach()


    # load -----------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    X_train = trained_results["X_train"]
    Z_train = trained_results["Z_train"].detach()
    Y_train = trained_results["Y_train"]
    if params["beta"]:
        Yhat_train = trained_results["Yhat_train"].detach()

    # visualize data ------------------------------------------------------#
    if 1:
        for random_image_index in range(X_cifar_samples.shape[-1]):
            x_new = X_cifar_samples[:, random_image_index]
            xhat_new = X_cifar_samples_hat[:, random_image_index]
            z_new = Z_cifar_samples[:, random_image_index]
            utils.visualizations.visualize_dense_most_similar_trainig_examples_based_on_code_similarity(
                Z_train,
                z_new,
                X_train,
                Y_train,
                x_new,
                xhat_new,
                params,
                save_path=os.path.join(
                    fig_path,
                    "cifar_{}_similar_training_examples_based_on_code_testimage{}".format(
                        class_str, random_image_index
                    ),
                ),
                reshape=params["reshape"],
            )



if __name__ == "__main__":
    main()
