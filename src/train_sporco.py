"""
Copyright (c) 2021 Bahareh Tolooshams

train sporco on x = Dz

:author: Bahareh Tolooshams
"""
import torch
import torch.nn.functional as F
import torchvision

import os
import json
import pickle
import numpy as np
from datetime import datetime
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
        default="sporco_dl/exp1",
    )
    parser.add_argument(
        "-d",
        "--data-filename-path",
        type=str,
        help="data filename path",
        default="../data/simulated_data.pt",
    )
    parser.add_argument(
        "-i",
        "--dict-init",
        type=int,
        help="initial closeness of dictionary",
        default=0.0228,
    )

    args = parser.parse_args()

    params = {
        "exp_name": args.exp_name,
        "data_filename_path": args.data_filename_path,
        "dict_init": args.dict_init,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "code_dist": "subgaussian",
        "z_mean": 5.0,
        "z_std": 1.0,
        "sporco_altmin_iter": 200,
        "sporco_rho": 1,
        "sporco_L": 1,
        "lam": 0.5,
        "num_distinct_supp_sets": 1,
        "normalize": True,
        "shuffle": True,
        "init_close": True,
        "num_epochs": 4,
        "lr": 1e-3,
        "num_workers": 0,
        #
        "enable_manual_seed": True,
        "manual_seed": 39,
    }

    return params


def main():

    print("Train sporco for model x = Dz on simulated data.")

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

    D = dataset.D.clone().detach().cpu().numpy()
    X = torch.squeeze(dataset.x).clone().detach().cpu().numpy().T
    Z = torch.squeeze(dataset.z).clone().detach().cpu().numpy().T

    # initialize network
    with torch.no_grad():
        if params["init_close"]:
            W_init = dataset.D.clone()
            W_init += params["dict_init"] * torch.randn(
                W_init.shape, device=W_init.device
            )
            W_init = F.normalize(W_init, dim=0)
    W_init = W_init.clone().detach().cpu().numpy()

    dl_opt = bpdndl.BPDNDictLearn.Options(
        {
            "Verbose": True,
            "AccurateDFid": False,
            "MaxMainIter": params["sporco_altmin_iter"],
            "CMOD": {"ZeroMean": False},
        },
    )

    # run dl
    d = bpdndl.BPDNDictLearn(W_init, X, params["lam"], dl_opt)

    W = d.solve()
    print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed("solve"))

    # set solver for solving for the codes now given the learned dictionary
    print("solve for code.")
    sc_opt = bpdn.BPDN.Options(
        {"Verbose": False, "MaxMainIter": 200, "RelStopTol": 5e-3, "AuxVarObj": False}
    )

    # # run csc
    # b = bpdn.BPDN(W, X, params["lam"], sc_opt)
    # Z_hat = b.solve()
    # print("BPDN solve time: %.2fs" % b.timer.elapsed("solve"))
    # z_zhat_err = np.mean(np.sqrt(np.sum((Z - Z_hat) ** 2, axis=0)))

    D_Dstar_err_relative = utils.utils.fro_distance_relative(
        torch.Tensor(D).clone(), torch.Tensor(W).clone()
    )
    print("D_Dtilde_err_relative", D_Dtilde_err_relative)

if __name__ == "__main__":
    main()
