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
        # default="../results/mnist_01234_p500_layers15_lamp02_step1_lamlossp02_2021_12_01_11_15_32",
        # default="../results/mnist_01234_p500_layers15_lamp1_step1_lamlossp1_2021_12_01_11_14_01",
        # default="../results/mnist_01234_p500_layers15_lamp7_step1_lamlossp7_2021_12_01_11_49_41",
        default="../../results/mnist_01234_p500_layers15_lamp1_step1_lamlossp1_overfit5_2021_12_05_16_46_42",
        # default="../results/mnist_01234_p500_layers15_lamp02_step1_lamlossp02_overfit5_2021_12_05_17_06_32",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reshape": (32, 32),
    }

    return params


def main():

    print("Visualzie attractors of model x = Dz.")

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
    fig_path = "{}/figures/contractive".format(params["exp_path"])
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
    X_train = trained_results["X_train"]
    Z_train = trained_results["Z_train"].detach()
    Y_train = trained_results["Y_train"]

    X_test = trained_results["X_test"]

    D = trained_results["D"].detach()

    # test -------------------------------------------------------------------#
    random_image_index = np.random.randint(X_test.shape[-1])

    x_new = torch.unsqueeze(torch.Tensor(X_test[:, random_image_index]), dim=0)
    x_new = torch.unsqueeze(x_new, dim=-1)

    for num_loops in [1, 2, 3, 5, 100, 500]:

        xhat_new = x_new.clone()
        norm_value = torch.norm(xhat_new)
        xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()

        for k in range(num_loops):
            xhat_new = xhat_new.to(device)
            xhat_new = xhat_new.to(device) * norm_value
            z_new = net.encode(xhat_new)
            xhat_new = net.decode(z_new)
            xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()

        # similar training examples based on code similarity --------------#
        utils.visualizations.visualize_contraction(
            torch.squeeze(z_new).detach(),
            torch.squeeze(x_new),
            torch.squeeze(xhat_new).detach(),
            D,
            params,
            save_path=os.path.join(
                fig_path,
                "{}_contractive_test_testimage{}_loop{}.png".format(
                    class_str, random_image_index, num_loops,
                ),
            ),
            reshape=params["reshape"],
        )

    # train -------------------------------------------------------------------#
    random_image_index = np.random.randint(X_train.shape[-1])

    x_new = torch.unsqueeze(torch.Tensor(X_train[:, random_image_index]), dim=0)
    x_new = torch.unsqueeze(x_new, dim=-1)

    # x_new += torch.rand(x_new.shape) * 0.1

    for num_loops in [1, 2, 3, 5, 100, 500]:

        xhat_new = x_new.clone()
        norm_value = torch.norm(xhat_new)
        xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()

        for k in range(num_loops):
            xhat_new = xhat_new.to(device)
            xhat_new = xhat_new.to(device) * norm_value
            z_new = net.encode(xhat_new)
            xhat_new = net.decode(z_new)
            xhat_new = torch.nn.functional.normalize(xhat_new, dim=1).clone()

        # similar training examples based on code similarity --------------#
        utils.visualizations.visualize_contraction(
            torch.squeeze(z_new).detach(),
            torch.squeeze(x_new),
            torch.squeeze(xhat_new).detach(),
            D,
            params,
            save_path=os.path.join(
                fig_path,
                "{}_contractive_test_trainimage{}_loop{}.png".format(
                    class_str, random_image_index, num_loops,
                ),
            ),
            reshape=params["reshape"],
        )


if __name__ == "__main__":
    main()
