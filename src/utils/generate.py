"""
Copyright (c) 2021 Bahareh Tolooshams

generate dataset for x = Dz

:author: Bahareh Tolooshams
"""


import torch
import torch.nn.functional as F

import os
import numpy as np
from datetime import datetime
import argparse

import utils

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--num-data", type=int, help="number of data", default=500,
    )
    parser.add_argument(
        "-m", "--x-dim", type=int, help="x data dimension", default=50,
    )
    parser.add_argument(
        "-p", "--z-dim", type=int, help="z sparse code dimension", default=100,
    )
    parser.add_argument(
        "-s",
        "--sparsity",
        type=int,
        help="sparsity of the code (# non-zero elements)",
        default=5,
    )
    parser.add_argument(
        "-a", "--c-min", type=float, help="code minimum amplitude", default=1.0,
    )
    parser.add_argument(
        "-b", "--c-max", type=float, help="code maximum amplitude", default=2.0,
    )
    parser.add_argument(
        "-r",
        "--num-distinct-supp-sets",
        type=int,
        help="number of distinct supp sets",
        default=1,
    )
    parser.add_argument(
        "-k", "--manual-seed", type=bool, help="manual seed bool", default=False,
    )
    parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        help="out path",
        default="../../data/simulated_data/",
    )
    parser.add_argument(
        "-u", "--dataset-usage", type=str, help="train or test", default="train",
    )

    args = parser.parse_args()

    params = {
        "n": args.num_data,
        "m": args.x_dim,
        "p": args.z_dim,
        "s": args.sparsity,
        "c_min": args.c_min,
        "c_max": args.c_max,
        "num_distinct_supp_sets": args.num_distinct_supp_sets,
        "manual_seed": args.manual_seed,
        "out_path": args.out_path,
        "dataset_usage": args.dataset_usage,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }

    return params


def main():

    print("Generate dataset for x = Dz.")

    # init parameters
    params = init_params()

    print(
        "There are {} number of distinct support in the dataset!".format(
            params["num_distinct_supp_sets"]
        )
    )

    # create dataset
    dataset = utils.datasets.xDzDataset_uniform(params)

    # save dataset
    if not os.path.exists(params["out_path"]):
        os.makedirs(params["out_path"])
    save_path = os.path.join(
        params["out_path"],
        "{}_n{}_m{}_p{}_s{}_dissupp{}_{}.pt".format(
            params["dataset_usage"],
            params["n"],
            params["m"],
            params["p"],
            params["s"],
            params["num_distinct_supp_sets"],
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        ),
    )
    print("data saved at {}".format(save_path))
    torch.save(dataset, save_path)


if __name__ == "__main__":
    main()

# python utils/generate.py -u "test" -r 3
