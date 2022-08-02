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
        default="../../results/exp1",
    )

    args = parser.parse_args()

    params = {
        "exp_path": args.exp_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reshape": (28, 28),
        "num_images": 8,
        "num_codes": 5,
    }

    return params


def get_saliency_maps(data_loader, net, params):
    x_list = []
    xhat_list = []
    saliency_list = dict()
    D_list = dict()
    z_list = dict()
    for index in range(params["num_codes"]):
        saliency_list["{}".format(index + 1)] = []
        D_list["{}".format(index + 1)] = []
        z_list["{}".format(index + 1)] = []

    D = net.W.data.clone().cpu().detach()

    ctr = 0
    for idx, (x, y) in tqdm(enumerate(data_loader), disable=True):
        ctr += 1

        x = x.to(params["device"])

        for code_index in range(params["num_codes"]):
            net.zero_grad()
            x.requires_grad_()

            optimizer = torch.optim.Adam([x])
            optimizer.zero_grad()

            zT = net.encode(x)
            xhat = net.decode(zT)

            if code_index == 0:
                x_list.append(
                    x.reshape(params["reshape"]).clone().detach().cpu().numpy()
                )
                xhat_list.append(
                    xhat.reshape(params["reshape"]).clone().detach().cpu().numpy()
                )

            zT_entry = torch.squeeze(
                zT[:, torch.argsort(zT, dim=1)[0][-1 - code_index]]
            )
            zT_entry.backward()

            z_list["{}".format(code_index + 1)].append(
                zT_entry.clone().detach().cpu().numpy()
            )

            saliency = x.grad.data.abs()

            # saliency *= x
            saliency_list["{}".format(code_index + 1)].append(
                saliency.reshape(params["reshape"]).clone().detach().cpu().numpy()
            )
            D_list["{}".format(code_index + 1)].append(
                D[:, torch.argsort(zT, dim=1)[0][-1 - code_index]]
                .reshape(params["reshape"])
                .clone()
                .detach()
                .cpu()
                .numpy()
            )

        if ctr >= params["num_images"]:
            break
    return x_list, xhat_list, saliency_list, D_list, z_list


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
    if params["beta"]:
        classifier_path = os.path.join(
            params["exp_path"],
            "model",
            "classifier_epoch{}.pt".format(params["num_epochs"] - 1),
        )
    result_path = os.path.join(params["exp_path"], "trained_results.pt")
    fig_path = "{}/figures/saliency_code".format(params["exp_path"])
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    device = params["device"]

    class_str = ""
    for c in params["class_list"]:
        class_str += str(c)

    # load dataset ------------------------------------------------------#
    if params["dataset_name"] == "mnist":
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_mnist_dataset(
            params["class_list"], make_flat=True
        )
    elif params["dataset_name"] == "cifar":
        X_tr, Y_tr, X_te, Y_te = utils.datasets.get_cifar_dataset(
            params["class_list"], params["blackandwhite"], make_flat=True
        )
    else:
        print("Dataset is not implemented!")
        raise NotImplementedError

    # get partial dataset -----------------------------------------------#
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_tr), torch.tensor(Y_tr, dtype=torch.long)
    )
    if params["overfit_to_only"]:
        train_num = params["overfit_to_only"]
    else:
        train_num = np.int32(X_tr.shape[0] * params["train_val_split"])
    val_num = X_tr.shape[0] - train_num

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_num, val_num],
        generator=torch.Generator().manual_seed(params["random_split_manual_seed"]),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_te), torch.tensor(Y_te, dtype=torch.long)
    )

    print("classes {} are in the dataset.".format(params["class_list"]))
    print(
        "total number of train/val/test data is {}/{}/{}".format(
            train_num, val_num, X_te.shape[0]
        )
    )

    # make dataloader -----------------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, num_workers=params["num_workers"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, num_workers=params["num_workers"],
    )

    net = torch.load(model_path, map_location=params["device"])
    net.eval()

    if params["beta"]:
        classifier = torch.load(classifier_path, map_location=params["device"])
        classifier.eval()

    # train -----------------------------------------------#
    x_list, xhat_list, saliency_list, D_list, z_list = get_saliency_maps(
        train_loader, net, params
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        saliency_list,
        z_list,
        save_path=os.path.join(
            fig_path, "{}_saliency_map_code_train.png".format(class_str,),
        ),
        s_name="s",
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        D_list,
        z_list,
        save_path=os.path.join(
            fig_path, "{}_saliency_dict_map_code_train.png".format(class_str,),
        ),
        s_name="d",
    )

    # test -----------------------------------------------#
    x_list, xhat_list, saliency_list, D_list, z_list = get_saliency_maps(
        test_loader, net, params
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        saliency_list,
        z_list,
        save_path=os.path.join(
            fig_path, "{}_saliency_map_code_test.png".format(class_str,),
        ),
        s_name="s",
    )

    utils.visualizations.visualize_saliency_map_of_code(
        x_list,
        xhat_list,
        D_list,
        z_list,
        save_path=os.path.join(
            fig_path, "{}_saliency_dict_map_code_test.png".format(class_str,),
        ),
        s_name="d",
    )


if __name__ == "__main__":
    main()
