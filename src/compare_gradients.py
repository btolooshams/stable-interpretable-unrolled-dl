"""
Copyright (c) 2021 Bahareh Tolooshams

train x = Dz with choice of gradient

:author: Bahareh Tolooshams
"""
import torch
import torch.nn.functional as F
import torchvision

import os
import json
import pickle
import numpy as np
import tensorboardX
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm, trange
import argparse

import sys

sys.path.append("src/")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--exp_name", type=str, help="experiment name", default="gradients/exp1",
    )
    parser.add_argument(
        "-n", "--network", type=str, help="network", default="AE",
    )
    parser.add_argument(
        "-g", "--grad-method", type=str, help="gradient method", default="g_aels",
    )
    parser.add_argument(
        "-d",
        "--data-filename-path",
        type=str,
        help="data filename path",
        default="../data/simulated_data.pt",
    )
    parser.add_argument(
        "-l", "--num-layers", type=int, help="number of unfolded layers", default=100,
    )
    parser.add_argument(
        "-i",
        "--dict-init",
        type=int,
        help="initial closeness of dictionary",
        default=0.02,
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "network": args.network,
        "grad_method": args.grad_method,
        "data_filename_path": args.data_filename_path,
        "num_layers": args.num_layers,
        "dict_init": args.dict_init,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "m": 100,  # x dimension
        "p": 150,  # z dimension
        "lam": 0.2,
        "step": 0.2,
        "beta": 1,
        "num_distinct_supp_sets": 1,
        "threshold": 0.45,
        "normalize": True,
        "shuffle": False,
        "twosided": True,
        "batch_size": 1000,
        "init_close": True,
        "num_epochs": 10,
        "num_workers": 0,
        #
        "num_layers": 100,
        "min_num_layers": 1,
        "max_num_layers": 100,
        "netlasso_num_layers": 1000,
        "enable_manual_seed": True,
        "manual_seed": 39,
    }

    return params


def main():

    print("Compare gradients for model x = Dz on simulated data.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    if params["enable_manual_seed"]:
        torch.manual_seed(params["manual_seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    print("Exp: {}".format(params["exp_name"]))

    # make folder for results
    out_path = os.path.join(
        "..", "results", "{}_{}".format(params["exp_name"], params["random_date"])
    )
    params["out_path"] = out_path
    os.makedirs(params["out_path"])
    os.makedirs(os.path.join(params["out_path"], "model"))
    os.makedirs(os.path.join(params["out_path"], "figures"))

    # dump params  ---------------------------------------------------------#
    with open(os.path.join(params["out_path"], "params.txt"), "w") as file:
        file.write(json.dumps(params, sort_keys=True, separators=("\n", ":")))

    with open(os.path.join(params["out_path"], "params.pickle"), "wb") as file:
        pickle.dump(params, file)

    params["device"] = torch.device(params["device"])
    device = params["device"]

    # board --------------------------------------------------------------#
    writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
    writer.add_text("params", str(params))
    writer.flush()

    # load data ------------------------------------------------------------#
    if params["data_filename_path"]:
        print("load data.")
        dataset = torch.load(
            params["data_filename_path"], map_location=params["device"]
        )
    else:
        print("generate data.")
        dataset = utils.datasets.xDzDataset(params)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=params["shuffle"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # create model ---------------------------------------------------------#
    print("create model.")
    if params["network"] == "AE":
        net = model.AE(params)
    elif params["network"] == "AEdecay":
        net = model.AEdecay(params)
    elif params["network"] == "AEhard":
        net = model.AEhard(params)
    else:
        print("Network is not implemented!")
        raise NotImplementedError

    # initialize network
    with torch.no_grad():
        if params["init_close"]:
            net.W.data = dataset.D.clone()
            net.W.data += params["dict_init"] * torch.randn(
                net.W.data.shape, device=net.W.data.device
            )
            net.W.data = F.normalize(net.W.data, dim=0)
        net = net.to(params["device"])

    if params["normalize"]:
        net.normalize()

    # criterion
    criterion = utils.loss.SparseLoss(lam=params["lam"])

    D = dataset.D.clone()
    W = net.W.clone()

    params_lasso = params.copy()
    params_lasso["num_layers"] = params["netlasso_num_layers"]
    net_lasso = model.AE(params_lasso, W.clone())

    print("method: {}".format(params["grad_method"]))

    zT_zhat_err_list = []
    g_ghat_err_list = []
    g_gstar_err_list = []

    # train  ---------------------------------------------------------------#
    for num_layers in tqdm(
        range(params["min_num_layers"], params["max_num_layers"] + 1), disable=False
    ):
        net.num_layers = num_layers

        for idx, (x, zstar) in tqdm(enumerate(data_loader), disable=True):

            net_lasso.W.data = net.W.data.clone()
            net.zero_grad()
            net_lasso.zero_grad()

            zstar = zstar.to(device)
            x = x.to(device)

            # forward -----------------------------------------------------#
            # get zT
            # forward encoder
            zT = net.encode(x)
            if params["grad_method"] == "g_dec":
                zT = zT.clone().detach().requires_grad_(False)
            # decoder
            xhat = net.decode(zT)
            # compute loss
            loss_ae, loss_lasso = criterion(x, xhat, zT)
            if params["grad_method"] == "g_dec":
                loss = loss_lasso
            elif params["grad_method"] == "g_aelasso":
                loss = loss_lasso
            elif params["grad_method"] == "g_aels":
                loss = loss_ae
            net.zero_grad()
            loss.backward()
            g = net.W.grad.clone()

            # get zhat and ghat ---------------------------------------------#
            if num_layers == params["min_num_layers"]:
                # forward encoder
                zhat = net_lasso.encode(x)
                zhat = zhat.clone().detach().requires_grad_(False)
                # decoder
                xhat_lasso = net_lasso.decode(zhat)
                # compute loss
                _, loss_lasso = criterion(x, xhat_lasso, zhat)
                net_lasso.zero_grad()
                loss_lasso.backward()
                ghat = net_lasso.W.grad.clone()

            # get gstar
            ########################
            if num_layers == params["min_num_layers"]:
                # decoder
                xhat_star = net_lasso.decode(zstar)
                # compute loss
                loss, _ = criterion(x, xhat_star, zstar)
                loss_star = loss
                net_lasso.zero_grad()
                loss_star.backward()
                gstar = net_lasso.W.grad.clone()
            ########################

            # error -----------------------------------------------------#
            with torch.no_grad():
                # zT - zhat
                zT_zhat_err = torch.mean(
                    torch.sqrt(torch.sum((zT - zhat) ** 2, dim=1))
                ).item()
                writer.add_scalar("gradient/zT-zhat-err", zT_zhat_err, num_layers)

                g_ghat_err = torch.sqrt(torch.sum((g - ghat) ** 2)).item()
                writer.add_scalar("gradient/g-ghat-err", g_ghat_err, num_layers)

                g_gstar_err = torch.sqrt(torch.sum((g - gstar) ** 2)).item()
                writer.add_scalar("gradient/g-gstar-err", g_gstar_err, num_layers)

                ghat_gstar_err = torch.sqrt(torch.sum((ghat - gstar) ** 2)).item()

                zT_zhat_err_list.append(zT_zhat_err)
                g_ghat_err_list.append(g_ghat_err)
                g_gstar_err_list.append(g_gstar_err)

            writer.flush()

        result_dict = {
            "zT_zhat_err": zT_zhat_err_list,
            "g_ghat_err": g_ghat_err_list,
            "g_gstar_err": g_gstar_err_list,
            "ghat_gstar_err": ghat_gstar_err,
        }

        torch.save(
            result_dict,
            os.path.join(out_path, "{}_results.pt".format(params["grad_method"])),
        )

        writer.close()


if __name__ == "__main__":
    main()
