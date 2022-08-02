"""
Copyright (c) 2021 Bahareh Tolooshams

visualize bias for the model x = Dz

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

    print("Visualize bias on model x = Dz.")

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
    b = trained_results["bias"]
    b = b.clone().detach().cpu().numpy()
    b[b < 0] = 0

    # visualize dictionary --------------------------------------------------#

    utils.visualizations.visualize_conv_bias(
        b, save_path=os.path.join(fig_path, "bias.png"),
    )


if __name__ == "__main__":
    main()
