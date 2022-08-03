"""
Copyright (c) 2021 Bahareh Tolooshams

generate data for model x = Dz

:author: Bahareh Tolooshams
"""


import torch

import os
import json
import numpy as np
import tensorboardX
from datetime import datetime
from tqdm import tqdm, trange
import argparse

import sys

sys.path.append("src/")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--data-path", type=str, help="data path", default="../data/simulated",
    )

    args = parser.parse_args()

    params = {
        "data_path": args.data_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        # "n": 50000,  # number of data
        "n": 500,  # number of data
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "code_dist": "subgaussian",
        "z_mean": 5.0,  # for subgaussian
        "z_std": 1.0,  # for subgaussian
        "c_min": 0,  # for uniform
        "c_max": 1,  # for uniform
        "orth_col": False,
        "num_distinct_supp_sets": 1,
        "manual_seed": 9,
    }

    return params


def main():

    print("Generate data for x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    dataset = utils.datasets.xDzDataset(params)

    if params["code_dist"] == "subgaussian":
        filename = os.path.join(
            params["data_path"],
            "data_n{}_m{}_p{}_s{}_{}_mean{}_std{}.pt".format(
                params["n"],
                params["m"],
                params["p"],
                params["s"],
                params["code_dist"],
                params["z_mean"],
                params["z_std"],
            ),
        )
    elif params["code_dist"] == "uniform":
        filename = os.path.join(
            paras["data_path"],
            "data_n{}_m{}_p{}_s{}_{}_min{}_max{}.pt".format(
                params["n"],
                params["m"],
                params["p"],
                params["s"],
                params["code_dist"],
                params["c_min"],
                params["c_max"],
            ),
        )
    else:
        print("Code distribution is not implemented!")
        raise NotImplementedError

    torch.save(dataset, filename)


if __name__ == "__main__":
    main()
