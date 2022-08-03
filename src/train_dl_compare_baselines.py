"""
Copyright (c) 2021 Bahareh Tolooshams

train x = Dz with choice of gradient compare baselines

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
        "-e",
        "--exp_name",
        type=str,
        help="experiment name",
        default="baseline_dl/exp1",
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
        default=0.005,
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
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "code_dist": "uniform",
        "c_min": 1.0,
        "c_max": 2.0,
        "lam": 0.2,
        "step": 0.2,
        "num_distinct_supp_sets": 1,
        "threshold": 0.1,
        "normalize": True,
        "shuffle": True,
        "twosided": True,
        "batch_size": 50,
        "init_close": True,
        "num_epochs": 1,
        "lr": 1e-3,
        "lr_step": 1000,
        "lr_decay": 1,
        "adam_eps": 1e-3,
        "adam_weight_decay": 0,
        "num_workers": 0,
        #
        "enable_manual_seed": True,
        "manual_seed": 39,
    }

    return params


def main():

    print("Train model x = Dz on simulated data.")

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

    train_loader = torch.utils.data.DataLoader(
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

    torch.save(net, os.path.join(out_path, "model", "model_init.pt"))

    # optimizer ------------------------------------------------------------#
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=params["lr"],
        eps=params["adam_eps"],
        weight_decay=params["adam_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params["lr_step"], gamma=params["lr_decay"]
    )

    # loss criterion  ------------------------------------------------------#
    criterion = utils.loss.SparseLoss(lam=params["lam"])

    D = dataset.D.clone()

    D_Dstar_err_relative_list = []

    # train  ---------------------------------------------------------------#
    ctr = -1
    if params["network"] == "AEdecay":
        net.beta = 1
    for epoch in tqdm(range(params["num_epochs"]), disable=True):
        net.train()

        if epoch > 0:
            scheduler.step()

        for idx, (x, zstar) in tqdm(enumerate(train_loader), disable=False):
            ctr += 1

            if params["network"] == "AEdecay":
                if (ctr + 1) % 100 == 0:
                    net.beta -= 0.005

            optimizer.zero_grad()

            net.train()

            optimizer.zero_grad()

            x = x.to(device)
            zstar = zstar.to(device)

            # forward ------------------------------------------------------#
            xhat, zT = net(x)
            if params["grad_method"] == "g_dec":
                zT = zT.clone().detach().requires_grad_(False)
            elif params["grad_method"] == "g_noodl":
                zT = zT.clone().detach().requires_grad_(False)
            # compute loss
            loss_ae, loss_lasso = criterion(x, xhat, zT)

            if params["grad_method"] == "g_dec":
                loss = loss_lasso
            elif params["grad_method"] == "g_aelasso":
                loss = loss_lasso
            elif params["grad_method"] == "g_aels":
                loss = loss_ae
            elif params["grad_method"] == "g_noodl":
                loss = loss_ae

            # backward -----------------------------------------------------#
            if params["grad_method"] == "g_noodl":
                g_noodl = torch.matmul(
                    torch.matmul(net.W, zT) - x, torch.sign(zT).transpose(1, 2)
                ) / float(x.shape[0])
                g_noodl = torch.mean(g_noodl, dim=0)
                net.W = torch.nn.Parameter(net.W - params["lr"] * g_noodl)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if params["normalize"]:
                net.normalize()

            # error -----------------------------------------------------#
            with torch.no_grad():
                if (ctr + 1) % 500 == 0:
                    fig = plt.figure()
                    plt.plot(
                        zstar[0, :].clone().detach().cpu().numpy(), "*g", label="zstar"
                    )
                    plt.plot(zT[0, :].clone().detach().cpu().numpy(), ".r", label="zT")
                    plt.legend()
                    writer.add_figure("baseline/code", fig, ctr)
                    plt.close()

                D_Dstar_err_relative = utils.utils.fro_distance_relative(
                    D.clone(), net.W.data.clone()
                )
                writer.add_scalar(
                    "baseline/D-Dstar-err-relative", D_Dstar_err_relative, ctr
                )

                writer.add_scalar("baseline/train-loss", loss.item(), ctr)

            D_Dstar_err_relative_list.append(D_Dstar_err_relative)

        writer.flush()

    result_dict = {
        "D_Dstar_err_relative": D_Dstar_err_relative_list,
    }

    torch.save(
        result_dict,
        os.path.join(out_path, "{}_results.pt".format(params["grad_method"])),
    )

    writer.close()


if __name__ == "__main__":
    main()
