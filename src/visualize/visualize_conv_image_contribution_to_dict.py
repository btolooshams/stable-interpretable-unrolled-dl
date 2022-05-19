"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

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
        default="../../results/bsd/cae_noisestd25_conv64_kernel7_stride5_2021_12_11_12_52_02",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    return params


def main():

    print("Visualzie image contribution to conv dictinoary on model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    num = 100
    rho = 1e-3

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
    fig_path = "{}/figures/image_contribution_to_dict/num{}_rho{}".format(
        params["exp_path"], num, rho
    )
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    # load -----------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    X_train = trained_results["X_train"]
    Z_train = trained_results["Z_train"].detach()

    X_test = trained_results["X_test"]
    Xhat_test = trained_results["Xhat_test"].detach()
    Z_test = trained_results["Z_test"].detach()

    D = trained_results["D"]

    # visualize data ------------------------------------------------------#

    shuffled_indices = np.linspace(0, X_train.shape[-1] - 1, X_train.shape[-1])
    np.random.shuffle(shuffled_indices)

    X_train_partial = X_train[shuffled_indices[:num]]
    Z_train_partial = Z_train[shuffled_indices[:num]]

    Z_train_partial = torch.reshape(Z_train_partial, (num, -1)).T

    G_partial = torch.matmul(Z_train_partial.T, Z_train_partial) + rho * torch.eye(
        Z_train_partial.shape[-1]
    )

    u, s, vh = np.linalg.svd(G_partial)

    G_inverse = u.T @ np.diag(1 / s) @ vh.T
    G_inverse = torch.Tensor(G_inverse)

    if 0:
        utils.visualizations.visualize_eigenvalues_of_G(
            G_partial,
            save_path=os.path.join(
                fig_path,
                "eigenvalues_G_train.png",
            ),
        )

        utils.visualizations.visualize_Ginverse_matrix(
            G_inverse,
            save_path=os.path.join(
                fig_path,
                "partial{}_Ginverse_rho{}.png".format(num, rho),
            ),
        )

        utils.visualizations.visualize_Ginversew_matrix(
            G_inverse,
            Z_train_partial,
            save_path=os.path.join(
                fig_path,
                "partial{}_Ginversew_rho{}.png".format(num, rho),
            ),
        )

        utils.visualizations.visualize_XGinverse_matrix(
            G_inverse,
            X_train_partial,
            save_path=os.path.join(
                fig_path,
                "partial{}_XGinverse_rho{}.png".format(num, rho),
            ),
        )

    for j in range(16):
        utils.visualizations.visualize_contribution_of_images_for_dict_j_using_Ginversew(
            D,
            j,
            G_inverse,
            Z_train_partial,
            X_train_partial,
            Y_train_partial,
            params,
            save_path=os.path.join(
                fig_path,
                " image_contribution_on_dict_{}.png".format(j),
            ),
            reshape=params["reshape"],
            D_reshape=params["reshape"],
        )


if __name__ == "__main__":
    main()
