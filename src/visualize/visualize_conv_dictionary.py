"""
Copyright (c) 2021 Bahareh Tolooshams

visualize for the model x = Dz

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
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_17_20_18_06",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp01_2021_12_19_11_32_33",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp05_2021_12_17_16_42_37",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv256_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_19_11_34_26",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd15_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_17_15_20_53",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp002_2021_12_20_16_46_54",
        # default="../../results/cifar_color/cifar_conv_0123456789_caelearnbias_noisestd0_conv100_kernel7_stride1_layers15_lamp0_stepp1_lamlossp01_2022_04_29_20_31_24",
        # default="../../results/cifar_color/cifar_conv_0123456789_caelearnbias_noisestd0_conv100_kernel7_stride1_layers15_lamp0_stepp1_lamlossp001_2022_04_30_22_07_29",
        # default="../../results/cifar_color/cifar_conv_0123456789_caelearnbias_noisestd0_conv100_kernel7_stride1_layers15_lamp0_stepp1_lamlossp005_2022_04_30_07_18_58",
        default="../../results/cifar_color/cifar_conv_0123456789_caelearnbias_noisestd0_conv100_kernel7_stride1_layers15_lamp0_stepp1_lamlossp05_2022_04_30_07_20_07",

    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    return params


def main():

    print("Visualzie convolutional dictionary on model x = Dz.")

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
    fig_path = "{}/figures".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    # load data -------------------------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])
    D = trained_results["D"]

    # visualize dictionary --------------------------------------------------#

    utils.visualizations.visualize_conv_dictionary(
        D, save_path=os.path.join(fig_path, "dict.png"),
    )

    if params["network"] == "CAElearnbias":
        bias = trained_results["bias"]

        D_sorted = D[np.argsort(bias)]
        utils.visualizations.visualize_conv_dictionary(
            D_sorted, save_path=os.path.join(fig_path, "dict_sorted.png"),
        )


if __name__ == "__main__":
    main()
