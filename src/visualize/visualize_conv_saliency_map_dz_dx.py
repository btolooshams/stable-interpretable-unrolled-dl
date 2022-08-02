"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
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
        "num_images": 8,
        "num_codes": 5,
        "train_image_path": "../../data/CBSD432",
        "test_image_path": "../../data/BSD68",
        "threshold": 0.3,
    }

    return params


def compute_psnr(x, xhat):
    psnr = []
    for i in range(x.shape[0]):
        mse = np.mean((x[i] - xhat[i]) ** 2)
        max_x = np.max(x[i])
        psnr.append(20 * np.log10(max_x) - 10 * np.log10(mse))
    return np.mean(psnr)


def get_saliency_maps(data_loader, net, params):
    x_list = []
    xhat_list = []
    psnr_list = []
    saliency_list = dict()
    D_list = dict()
    z_list = dict()
    for index in range(params["num_codes"]):
        saliency_list["{}".format(index + 1)] = []
        D_list["{}".format(index + 1)] = []
        z_list["{}".format(index + 1)] = []

    D = net.W.data.clone().cpu().detach()

    if params["network"] == "CAElearnbias":
        bias = net.b.data.clone()

    ctr = 0
    for idx, (x, _) in tqdm(enumerate(data_loader), disable=True):
        ctr += 1

        x = x.to(params["device"])

        for code_index in range(params["num_codes"]):
            net.zero_grad()
            x.requires_grad_()

            optimizer = torch.optim.Adam([x])
            optimizer.zero_grad()

            xhat, zT = net(x)

            if code_index == 0:
                x_list.append(torch.squeeze(x).clone().detach().cpu().numpy())
                xhat_list.append(torch.squeeze(xhat).clone().detach().cpu().numpy())

                psnr_list.append(
                    compute_psnr(
                        x[:, 0].clone().detach().cpu().numpy(),
                        xhat[:, 0].clone().detach().cpu().numpy(),
                    )
                )

            if params["network"] == "CAElearnbias":

                print(bias[np.argsort(bias)])

                zT[:, bias < params["threshold"], :, :] = 0

            # print(zT[:, torch.argsort(zT, dim=1)[0][-1 - code_index]])
            zT = torch.sqrt(
                torch.sum(torch.mean(zT, dim=0, keepdims=True).pow(2), dim=(-1, -2))
            )
            zT_entry = zT[:, torch.argsort(zT, dim=1)[0][-1 - code_index]]

            # zT_entry = torch.mean(zT[:, torch.argsort(zT, dim=1)[0][-1:]])

            zT_entry.backward()

            z_list["{}".format(code_index + 1)].append(
                zT_entry.clone().detach().cpu().numpy()
            )

            saliency = x.grad.data.abs()

            saliency_list["{}".format(code_index + 1)].append(
                torch.squeeze(saliency).clone().detach().cpu().numpy()
            )
            D_list["{}".format(code_index + 1)].append(
                torch.squeeze(D[torch.argsort(zT, dim=1)[0][-1 - code_index]])
                .clone()
                .detach()
                .cpu()
                .numpy()
            )

        if ctr >= params["num_images"]:
            break
    return x_list, xhat_list, saliency_list, D_list, z_list, psnr_list


def main():

    print("Predict and save results on model x = Dz.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    params_pickle = pickle.load(
        open(os.path.join(params["exp_path"], "params.pickle"), "rb")
    )
    for key in params.keys():
        params_pickle[key] = params[key]

    params_pickle["dataset_name"] = "mnist"
    params_pickle["overfit_to_only"] = None

    params = params_pickle

    print("Exp: {}".format(params["exp_path"]))

    model_path = os.path.join(
        params["exp_path"],
        "model",
        "model_epoch{}.pt".format(params["num_epochs"] - 1),
    )

    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/saliency_code".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    device = params["device"]

    # load data ------------------------------------------------------------#
    train_dataset = torchvision.datasets.ImageFolder(
        root=params["train_image_path"],
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(
                    params["patch_size"],
                    padding=None,
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=params["test_image_path"],
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomCrop(
                    params["patch_size"],
                    padding=None,
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    # make dataloader ------------------------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, batch_size=1,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1,
    )

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=params["device"])
    net.eval()

    # train -----------------------------------------------#
    x_list, xhat_list, saliency_list, D_list, z_list, psnr_list = get_saliency_maps(
        train_loader, net, params
    )

    print("psnr", psnr_list)

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        saliency_list,
        z_list,
        save_path=os.path.join(fig_path, "saliency_map_code_train.png",),
        s_name="s",
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        D_list,
        z_list,
        save_path=os.path.join(fig_path, "saliency_dict_map_code_train.png",),
        s_name="d",
    )

    # test -----------------------------------------------#
    x_list, xhat_list, saliency_list, D_list, z_list, psnr_list = get_saliency_maps(
        test_loader, net, params
    )

    print("psnr", psnr_list)

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        saliency_list,
        z_list,
        save_path=os.path.join(fig_path, "{}_saliency_map_code_test.png",),
        s_name="s",
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        D_list,
        z_list,
        save_path=os.path.join(fig_path, "{}_saliency_dict_map_code_test.png",),
        s_name="d",
    )


if __name__ == "__main__":
    main()
