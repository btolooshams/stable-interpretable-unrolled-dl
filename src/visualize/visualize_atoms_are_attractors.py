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
        default="../../results/exp1",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reshape": (32, 32),
    }

    return params


def main():

    print(
        "Visualzie atoms to show that they are fixed points of the mapping on model x = Dz."
    )

    # init parameters -------------------------------------------------------#
    params = init_params()

    loops = [1, 2, 5, 10, 20, 50, 100, 500]

    params_pickle = pickle.load(
        open(os.path.join(params["exp_path"], "params.pickle"), "rb")
    )
    params_pickle["exp_path"] = params["exp_path"]
    params_pickle["device"] = params["device"]
    params = params_pickle

    print("Exp: {}".format(params["exp_path"]))

    model_path = os.path.join(
        params["exp_path"],
        "model",
        "model_epoch{}.pt".format(params["num_epochs"] - 1),
    )
    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/atoms_are_fixedpoints".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    # load data ------------------------------------------------------------#
    print("classes {} are in the dataset.".format(params["class_list"]))

    net = torch.load(model_path, map_location=params["device"])
    net.eval()

    trained_results = torch.load(result_path, map_location=params["device"])

    # load data --------------------------------------------------------------#
    D = trained_results["D"].detach()

    for k in range(5):
        random_atom_index = np.random.randint(D.shape[-1])

        # test -------------------------------------------------------------------#
        x_list = []
        x_list.append(D[:, random_atom_index].detach().cpu().numpy())

        for num_loops in loops:
            print("loop", num_loops)

            x_new = torch.unsqueeze(torch.Tensor(D[:, random_atom_index]), dim=0)
            x_new = torch.unsqueeze(x_new, dim=-1)

            xhat_new = x_new.clone()
            xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()
            for k in range(num_loops):
                xhat_new = xhat_new.to(device)
                z_new = net.encode(xhat_new)
                xhat_new = net.decode(z_new)
                xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()

            xhat_new = torch.squeeze(xhat_new).detach().cpu().numpy()
            x_list.append(xhat_new)

        utils.visualizations.visualize_atoms_for_fixedpoint(
            x_list,
            loops,
            save_path=os.path.join(
                fig_path,
                "{}_atom{}_fixedpoint.png".format(class_str, random_atom_index),
            ),
            reshape=params["reshape"],
        )


if __name__ == "__main__":
    main()
