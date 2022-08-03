"""
Copyright (c) 2021 Bahareh Tolooshams

find initial estimate of dicitonary using pairwise method (Arora et al.)

:author: Bahareh Tolooshams
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import numpy as np
import pickle
import os
import pickle
from datetime import datetime
from scipy.optimize import linear_sum_assignment as lsa
import argparse

import sys

sys.path.append("src/")

import utils, model


def dictionary_distance(d, dhat, permute=False, eps=1e-6):

    assert d.shape == dhat.shape
    m, p = d.shape

    d = d.clone().cpu()
    dhat = dhat.clone().cpu()

    d /= torch.norm(d, keepdim=True, dim=0)
    dhat /= torch.norm(dhat, keepdim=True, dim=0)

    d = torch.nan_to_num(d)
    dhat = torch.nan_to_num(dhat)

    cost = torch.zeros(p, p)

    for i in range(p):
        for j in range(p):
            a = 1 - torch.dot(d[:, i], dhat[:, j]).pow(2)
            cost[i, j] = torch.sqrt(a + eps)
    p = lsa(cost) if permute else (np.arange(p), np.arange(p))

    return cost[p].mean().item(), cost[p].median().item(), cost[p].max().item()


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()

    params = {
        "data_filename_path": "../data/simulated_data.pt",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "shuffle": True,
        "num_workers": 4,
        "batch_size2": 50000,
        "num_data1": 1000,
    }

    return params


def main():

    print("Find initial estimate of D on simulated data.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    device = params["device"]

    if params["data_filename_path"]:
        dataset = torch.load(
            params["data_filename_path"], map_location=params["device"]
        )
    else:
        dataset = utils.datasets.xDzDataset(params)

    X = dataset.x.clone().detach()
    Z = dataset.z.clone().detach()
    D = dataset.D.clone().detach()
    Dn = D / torch.norm(D, keepdim=True, dim=0)

    shuffled_indices = np.linspace(0, dataset.n - 1, dataset.n)
    np.random.shuffle(shuffled_indices)

    num_data = dataset.n

    dataset1 = torch.utils.data.TensorDataset(
        X[shuffled_indices[: params["num_data1"]]],
        torch.tensor(Z[shuffled_indices[: params["num_data1"]]]),
    )
    dataset2 = torch.utils.data.TensorDataset(
        X[shuffled_indices[params["num_data1"] :]],
        torch.tensor(Z[shuffled_indices[params["num_data1"] :]]),
    )

    data_loader1 = DataLoader(dataset1, shuffle=params["shuffle"], batch_size=2,)
    data_loader2 = DataLoader(
        dataset2, shuffle=params["shuffle"], batch_size=params["batch_size2"],
    )

    m = dataset.m
    p = dataset.p
    s = dataset.s

    print("data", dataset.n, "m", m, "p", p, "s", s)

    init_dict = torch.zeros(m, p).to(device)

    a1 = s / p
    a2 = s / (p * np.log(p))

    a1 = a1 * 1
    a2 = a2 * 1

    atom_ctr = 0
    while atom_ctr < p:
        print("while", atom_ctr)

        for idx, (u_and_v, uz_and_vz) in tqdm(enumerate(data_loader1), disable=True):
            # pick u and v from dataset1
            u = u_and_v[0]
            v = u_and_v[1]

            uz = uz_and_vz[0]
            vz = uz_and_vz[1]

            if torch.sum(torch.abs(uz * vz)) == 0:
                continue

            Muv = torch.zeros(m, m).to(device)

            for idx, (x, _) in tqdm(enumerate(data_loader2), disable=True):
                # from dataset2
                xu = torch.matmul(x.transpose(1, 2), u)
                xv = torch.matmul(x.transpose(1, 2), v)

                xxT = torch.matmul(x, x.transpose(1, 2))

                Muv += torch.sum(xu * xv * xxT, dim=0)

            Muv /= num_data - num_data1

            U, S, Vh = torch.linalg.svd(Muv)

            sig1 = S[0].item()
            sig2 = S[1].item()

            v1 = Vh[0]
            v1 = v1 / torch.norm(v1)

            if atom_ctr > p:
                break

            if sig1 >= a1 and sig2 < a2:
                if atom_ctr == 0:
                    init_dict[:, atom_ctr] = v1
                    atom_ctr += 1
                else:
                    dis = torch.min(
                        torch.sum((init_dict - torch.unsqueeze(v1, dim=1)) ** 2, dim=0)
                    )
                    dis_flip = torch.min(
                        torch.sum((init_dict + torch.unsqueeze(v1, dim=1)) ** 2, dim=0)
                    )
                    dis = torch.minimum(dis, dis_flip)

                    distance = torch.sqrt(dis).item()

                    scale_dist = 2
                    if distance > scale_dist * (1 / np.log(p)):
                        init_dict[:, atom_ctr] = v1
                        atom_ctr += 1

                        ttt = torch.sqrt(
                            torch.min(
                                torch.sum((Dn - torch.unsqueeze(v1, dim=1)) ** 2, dim=0)
                            )
                        )
                        ttt_flip = torch.sqrt(
                            torch.min(
                                torch.sum((Dn + torch.unsqueeze(v1, dim=1)) ** 2, dim=0)
                            )
                        )
                        ttt = torch.minimum(ttt, ttt_flip).item()
                        if ttt > 1 / np.log(p):
                            print("Oh! one added that is not close")
                            print(
                                "distance within init",
                                distance,
                                "distance from a true",
                                ttt,
                            )

                        if (atom_ctr + 1) % 10 == 0:
                            closeness_distance = utils.utils.fro_distance_permute_all(
                                init_dict, D
                            )[:-1]
                            print("closeness_distance", closeness_distance)

                            closeness_sine = utils.utils.sine_distance_permute_all(
                                init_dict, D
                            )[:-1]
                            print("closeness_sine", closeness_sine)

    closeness_distance = utils.fro_distance_permute_all(init_dict, D)[:-1]
    print("closeness_distance", closeness_distance)

    closeness_sine = utils.sine_distance_permute_all(init_dict, D)[:-1]
    print("closeness_sine", closeness_sine)
    print("done")

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    torch.save(init_dict, "../results/init/dict_init_{}.pt".format(random_date))
    torch.save(D, "../results/init/dict_true_{}.pt".format(random_date))


if __name__ == "__main__":
    main()
