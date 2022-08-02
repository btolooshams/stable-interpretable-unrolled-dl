"""
Copyright (c) 2021 Bahareh Tolooshams

visualize dense image contribution to dictionary for the model x = Dz

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
        "num": 6000,
        "rho": 1e-3,
    }

    return params


def main():

    print("Visualize image contribution to dense dictinoary on model x = Dz.")

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
    fig_path = "{}/figures/image_contribution_to_dict/num{}_rho{}".format(
        params["exp_path"], num, rho
    )
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    # load -----------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    X_train = trained_results["X_train"]
    Z_train = trained_results["Z_train"].detach()
    Y_train = trained_results["Y_train"]
    if params["beta"]:
        Yhat_train = trained_results["Yhat_train"].detach()

    X_test = trained_results["X_test"]
    Xhat_test = trained_results["Xhat_test"].detach()
    Z_test = trained_results["Z_test"].detach()
    Y_test = trained_results["Y_test"]
    if params["beta"]:
        Yhat_test = trained_results["Yhat_test"].detach()

    D = trained_results["D"]

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

    #### test
    c_test_indices = []
    Z_test_c = []
    X_test_c = []
    Xhat_test_c = []
    for c in range(len(params["class_list"])):
        c_test_indices.append(Y_test == params["class_list"][c])
        Z_test_c.append(Z_test[:, c_test_indices[c]])
        X_test_c.append(X_test[:, c_test_indices[c]])
        Xhat_test_c.append(Xhat_test[:, c_test_indices[c]])

    # visualize data ------------------------------------------------------#

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

    G_partial = torch.matmul(Z_train_partial.T, Z_train_partial) + rho * torch.eye(
        Z_train_partial.shape[-1]
    )

    u, s, vh = np.linalg.svd(G_partial)

    G_inverse = u.T @ np.diag(1 / s) @ vh.T
    G_inverse = torch.Tensor(G_inverse)

    if 0:
        utils.visualizations.visualize_eigenvalues_of_G(
            G_partial, save_path=os.path.join(fig_path, "eigenvalues_G_train.png",),
        )

        utils.visualizations.visualize_Ginverse_matrix(
            G_inverse,
            save_path=os.path.join(
                fig_path, "{}_partial{}_Ginverse_rho{}.png".format(class_str, num, rho),
            ),
        )

        utils.visualizations.visualize_Ginversew_matrix(
            G_inverse,
            Z_train_partial,
            save_path=os.path.join(
                fig_path,
                "{}_partial{}_Ginversew_rho{}.png".format(class_str, num, rho),
            ),
        )

        utils.visualizations.visualize_XGinverse_matrix(
            G_inverse,
            X_train_partial,
            save_path=os.path.join(
                fig_path,
                "{}_partial{}_XGinverse_rho{}.png".format(class_str, num, rho),
            ),
        )

    eps = 1e-11
    Z_energy = torch.mean(Z_train.pow(2), dim=-1).detach().cpu().numpy()
    Z_energy_normalized = Z_energy / (np.linalg.norm(Z_energy) + eps)
    Z_energy_normalized = np.nan_to_num(Z_energy_normalized)

    sorted_atoms = np.flip(np.argsort(Z_energy_normalized))

    if 1:
        for j in range(10):
            utils.visualizations.visualize_contribution_of_images_for_dict_j_using_Ginversew_nohist(
                D,
                sorted_atoms[j],
                G_inverse,
                Z_train_partial,
                X_train_partial,
                Y_train_partial,
                params,
                save_path=os.path.join(
                    fig_path,
                    "{}_image_contribution_on_dict_{}.pdf".format(
                        class_str, sorted_atoms[j]
                    ),
                ),
                reshape=params["reshape"],
                D_reshape=params["reshape"],
            )


if __name__ == "__main__":
    main()
