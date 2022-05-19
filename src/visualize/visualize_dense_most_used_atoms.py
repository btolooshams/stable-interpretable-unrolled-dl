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

sys.path.append("../")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-path",
        type=str,
        help="experiment path",
        default="../../results/mnist/mnist_01234_p500_layers15_lamp7_step1_lamlossp7_2021_12_01_11_49_41",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reshape": (28, 28),
    }

    return params


def main():

    print("Visualzie mnist most used atoms on model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    params_pickle = pickle.load(
        open(os.path.join(params["exp_path"], "params.pickle"), "rb")
    )
    for key in params.keys():
        params_pickle[key] = params[key]
    params = params_pickle

    print("Exp: {}".format(params["exp_path"]))

    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/most_used_atoms".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    # load data --------------------------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])
    Z_train = trained_results["Z_train"].detach()
    Y_train = trained_results["Y_train"]
    D = trained_results["D"]

    #### train
    c_train_indices = []
    Z_train_c = []
    for c in range(len(params["class_list"])):
        c_train_indices.append(Y_train == params["class_list"][c])
        Z_train_c.append(Z_train[:, c_train_indices[c]])

    # visualize most used atoms ------------------------------------------#
    utils.visualizations.visualize_sorted_atoms(
        Z_train,
        D,
        save_path=os.path.join(
            fig_path,
            "{}_dictionary_sorted.png".format(
                class_str
            ),
        ),
        reshape=params["reshape"],
    )

    for c in range(len(params["class_list"])):
        utils.visualizations.visualize_most_used_atoms_energy(
            Z_train_c[c],
            D,
            save_path=os.path.join(
                fig_path,
                "{}_mostusedatoms_energy_train_{}.png".format(
                    class_str, params["class_list"][c]
                ),
            ),
            reshape=params["reshape"],
        )

        utils.visualizations.visualize_most_used_atoms_active(
            Z_train_c[c],
            D,
            save_path=os.path.join(
                fig_path,
                "{}_mostusedatoms_active_train_{}.png".format(
                    class_str, params["class_list"][c]
                ),
            ),
            reshape=params["reshape"],
        )


if __name__ == "__main__":
    main()
