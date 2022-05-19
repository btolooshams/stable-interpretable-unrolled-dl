"""
Copyright (c) 2021 Bahareh Tolooshams

visualize conv feature maps for the model x = Dz

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
    }

    return params


def main():

    print("Visualize feature maps on conv model x = Dz.")

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
    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/feature_maps".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    # load -----------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    X_train = trained_results["X_train"]
    Xhat_train = trained_results["Xhat_train"].detach()
    Z_train = trained_results["Z_train"].detach()

    X_test = trained_results["X_test"]
    Xhat_test = trained_results["Xhat_test"].detach()
    Z_test = trained_results["Z_test"].detach()

    # visualize data ------------------------------------------------------#

    if 1:
        Z_train_proc = torch.sqrt(torch.sum(Z_train.pow(2), dim=(-1, -2)))
        Z_test_proc = torch.sqrt(torch.sum(Z_test.pow(2), dim=(-1, -2)))

        Z_train_proc = Z_train_proc.T
        Z_test_proc = Z_test_proc.T

        utils.visualizations.visualize_code_matrix(
            Z_train_proc,
            save_path=os.path.join(fig_path, "feature_maps_train.png"),
            sorted_atom_index=[],
        )

        utils.visualizations.visualize_code_matrix(
            Z_test_proc,
            save_path=os.path.join(fig_path, "feature_maps_test.png"),
            sorted_atom_index=[],
        )

    if 1:
        for i in range(20):
            random_image_index = np.random.randint(X_train.shape[0])
            x = X_train[random_image_index]
            xhat = Xhat_train[random_image_index]
            z = Z_train[random_image_index]

            utils.visualizations.visualize_conv_feature_maps(
                z,
                save_path=os.path.join(
                    fig_path, "train_{}_feature_maps.png".format(random_image_index)
                ),
            )

            utils.visualizations.visualize_image(
                x,
                xhat,
                save_path=os.path.join(
                    fig_path, "train_{}_image.png".format(random_image_index)
                ),
            )


if __name__ == "__main__":
    main()
