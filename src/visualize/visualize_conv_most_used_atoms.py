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
        # default="../../results/bsd/cae_noisestd25_conv64_kernel7_stride5_2021_12_11_12_52_02",
        # default="../../results/bsd/caelearn_noisestd25_conv64_kernel7_stride5_2021_12_11_19_56_35",
        # default="../../results/bsd/caelearn_noisestd25_conv64_kernel8_stride8_patch8_batch128_lr2_2021_12_13_18_51_48",
        default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_17_20_18_06",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "eps": 0.1,
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

    # load data --------------------------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])
    D = trained_results["D"]
    Z_train = trained_results["Z_train"].detach()

    Z_train_proc = torch.sqrt(torch.sum(Z_train.pow(2), dim=(-1, -2)))
    Z_train_proc = Z_train_proc.T

    D_proc = torch.reshape(D, (D.shape[0], -1)).T

    reshape = (D.shape[-2], D.shape[-1])

    # visualize most used atoms ------------------------------------------#
    utils.visualizations.visualize_most_used_atoms_energy(
        Z_train_proc,
        D_proc,
        save_path=os.path.join(fig_path, "mostusedatoms_energy_train.png"),
        reshape=reshape,
    )

    utils.visualizations.visualize_most_used_atoms_active(
        Z_train_proc,
        D_proc,
        save_path=os.path.join(
            fig_path, "mostusedatoms_active_train_eps{}.png".format(params["eps"])
        ),
        eps=params["eps"],
        reshape=reshape,
    )


if __name__ == "__main__":
    main()
