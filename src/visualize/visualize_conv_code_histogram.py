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
        default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_17_20_18_06",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp01_2021_12_19_11_32_33",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp05_2021_12_17_16_42_37",
        # ### default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd0_conv256_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_19_11_34_26",
        # default="../../results/cifar_color/cifar_conv_01234_caelearnbias_noisestd15_conv64_kernel7_stride1_layers15_lamp0_stepp1_lamlossp1_2021_12_17_15_20_53",

    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "threshold": 0.0152,
    }

    return params


def main():

    print(
        "Visualzie image interpolation based on code similarity on conv model x = Dz."
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
    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    if params["threshold"]:
        fig_path = "{}/figures/code_histogram_based_on_class/threshold{}".format(
            params["exp_path"], params["threshold"]
        )
    else:
        fig_path = "{}/figures/code_histogram_based_on_class".format(
            params["exp_path"]
        )
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    # load -----------------------------------------------#
    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    Z_train = trained_results["Z_train"].detach()
    Y_train = trained_results["Y_train"].detach()

    Z_test = trained_results["Z_test"].detach()
    Y_test = trained_results["Y_test"].detach()

    Z_train_proc = Z_train.clone()
    Z_test_proc = Z_test.clone()

    Z_train_proc = torch.sqrt(torch.sum(Z_train.pow(2), dim=(-1, -2)))
    Z_test_proc = torch.sqrt(torch.sum(Z_test.pow(2), dim=(-1, -2)))

    if params["network"] == "CAElearnbias":
        bias = trained_results["bias"]

        print(bias[np.argsort(bias)])

        Z_train_proc[:, bias < params["threshold"]] = 0
        Z_test_proc[:, bias < params["threshold"]] = 0

    Z_train_proc = Z_train_proc.T
    Z_test_proc = Z_test_proc.T

    # visualize data ------------------------------------------------------#
    #### train
    for c in range(len(params["class_list"])):
        c_train_indices = Y_train == params["class_list"][c]
        Z_train_c = Z_train_proc[:, c_train_indices]

        utils.visualizations.visualize_conv_code_histogram(Z_train_c,
            params,
            save_path=os.path.join(
                fig_path,
                "train_code_histogram_class_{}.png".format(
                    c
                ),
            ),
        )

    #### test
    for c in range(len(params["class_list"])):
        c_test_indices = Y_test == params["class_list"][c]
        Z_test_c = Z_test_proc[:, c_test_indices]

        utils.visualizations.visualize_conv_code_histogram(Z_test_c,
            params,
            save_path=os.path.join(
                fig_path,
                "test_code_histogram_class_{}.png".format(
                    c
                ),
            ),
        )


if __name__ == "__main__":
    main()
