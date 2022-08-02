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

import utils


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
        "num": 6000,
        "rho": 1e-3,
        "data_dim": 784,
    }

    return params


def main():

    print("Visualzie image interpolation based on G and Z on model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    num = params["num"]
    rho = params["rho"]

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
    fig_path = "{}/figures/image_interpolation_based_on_GandZ_for_newimage/num{}_rho{}".format(
        params["exp_path"], num, rho
    )
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    net = torch.load(model_path, map_location=params["device"])

    _, _, X_cifar, _ = utils.datasets.get_cifar_dataset([0], True, make_flat=False)

    X_cifar_samples = X_cifar[:50]
    X_cifar_samples = torch.Tensor(X_cifar_samples)

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomResizedCrop(params["reshape"]),]
    )
    X_cifar_samples = transform(X_cifar_samples)
    X_cifar_samples = X_cifar_samples.reshape(-1, param["data_dim"], 1)
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

    #### train
    c_train_indices = []
    Z_train_c = []
    X_train_c = []
    Y_train_c = []
    for c in range(len(params["class_list"])):
        c_train_indices.append(Y_train == params["class_list"][c])
        Z_train_c.append(Z_train[:, c_train_indices[c]])
        X_train_c.append(X_train[:, c_train_indices[c]])
        Y_train_c.append(Y_train[c_train_indices[c]])

    X_train_partial = []
    Z_train_partial = []
    Y_train_partial = []
    for c in range(len(params["class_list"])):
        shuffled_indices = np.linspace(
            0, X_train_c[c].shape[-1] - 1, X_train_c[c].shape[-1]
        )
        np.random.shuffle(shuffled_indices)
        X_train_partial.append(
            X_train_c[c][
                :, shuffled_indices[: np.int32(num / len(params["class_list"]))]
            ]
        )
        Z_train_partial.append(
            Z_train_c[c][
                :, shuffled_indices[: np.int32(num / len(params["class_list"]))]
            ]
        )
        Y_train_partial.append(
            Y_train_c[c][shuffled_indices[: np.int32(num / len(params["class_list"]))]]
        )

    X_train_partial = torch.cat(X_train_partial, dim=1)
    Z_train_partial = torch.cat(Z_train_partial, dim=1)
    Y_train_partial = torch.cat(Y_train_partial, dim=0)

    Z_train_partial_normalized = torch.nn.functional.normalize(
        Z_train_partial, dim=0
    ).clone()

    G_partial = torch.matmul(
        Z_train_partial_normalized.T, Z_train_partial_normalized
    ) + rho * torch.eye(Z_train_partial_normalized.shape[-1])

    u, s, vh = np.linalg.svd(G_partial)

    G_partial_inverse = u.T @ np.diag(1 / s) @ vh.T
    G_partial_inverse = torch.Tensor(G_partial_inverse)

    for random_image_index in range(X_cifar_samples.shape[-1]):
        x_new = X_cifar_samples[:, random_image_index]
        xhat_new = X_cifar_samples_hat[:, random_image_index]
        z_new = Z_cifar_samples[:, random_image_index]
        utils.visualizations.visualize_most_similar_trainig_examples_based_on_GandZ_nohist(
            G_partial_inverse,
            Z_train_partial,
            z_new,
            X_train_partial,
            Y_train_partial,
            x_new,
            xhat_new,
            params,
            save_path=os.path.join(
                fig_path,
                "cifar_{}_similar_training_examples_based_on_beta_testimage{}_nohist.pdf".format(
                    class_str, random_image_index
                ),
            ),
            reshape=params["reshape"],
        )

        utils.visualizations.visualize_most_similar_trainig_examples_based_on_GandZ_nohist_donotnormalize(
            G_partial_inverse,
            Z_train_partial,
            z_new,
            X_train_partial,
            Y_train_partial,
            x_new,
            xhat_new,
            params,
            save_path=os.path.join(
                fig_path,
                "cifar_{}_notnormalized_similar_training_examples_based_on_beta_testimage{}_nohist.pdf".format(
                    class_str, random_image_index
                ),
            ),
            reshape=params["reshape"],
        )


if __name__ == "__main__":
    main()
