"""
Copyright (c) 2021 Bahareh Tolooshams

utils

:author: Bahareh Tolooshams
"""

import random
from math import sqrt

import numpy as np

import torch, torchvision
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa
from torch.utils.data import Dataset
from scipy.linalg import qr


def fro_distance(A, B):

    assert A.shape == B.shape

    A = A.clone().cpu()
    B = B.clone().cpu()

    A /= torch.norm(A, keepdim=True, dim=0)
    B /= torch.norm(B, keepdim=True, dim=0)

    err = torch.sqrt(torch.sum((A - B) ** 2)).item()
    return err


def fro_distance_relative(A, B):

    assert A.shape == B.shape

    A = A.clone().cpu()
    B = B.clone().cpu()

    A /= torch.norm(A, keepdim=True, dim=0)
    B /= torch.norm(B, keepdim=True, dim=0)

    err = (
        torch.sqrt(torch.sum((A - B) ** 2)).item()
        / torch.sqrt(torch.sum((A) ** 2)).item()
    )
    return err


def fro_distance_permute(d, dhat, permute=True):

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
            a = torch.sum((d[:, i] - dhat[:, j]) ** 2).item()
            b = torch.sum((d[:, i] + dhat[:, j]) ** 2).item()
            cost[i, j] = torch.minimum(torch.tensor(a), torch.tensor(b)).item()

    rc = lsa(cost) if permute else (np.arange(p), np.arange(p))

    return torch.sqrt(cost[rc].sum()).item()


def fro_distance_permute_all(d, dhat, permute=True):

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
            a = torch.sum((d[:, i] - dhat[:, j]) ** 2).item()
            b = torch.sum((d[:, i] + dhat[:, j]) ** 2).item()
            cost[i, j] = torch.minimum(torch.tensor(a), torch.tensor(b)).item()

    rc = lsa(cost) if permute else (np.arange(p), np.arange(p))

    out = cost[rc]

    sorted_col = p[1]

    return (
        torch.sqrt(cost[rc]).mean().item(),
        torch.sqrt(cost[rc]).median().item(),
        torch.sqrt(cost[rc]).max().item(),
        sorted_col,
    )


def sine_distance(A, B):

    assert A.shape == B.shape

    A = A.clone().cpu()
    B = B.clone().cpu()

    A /= torch.norm(A, keepdim=True, dim=0)
    B /= torch.norm(B, keepdim=True, dim=0)

    cos = (A * B).double().sum(dim=0)
    return (
        (1 - cos ** 2).relu().sqrt().max().float().item(),
        (1 - cos ** 2).relu().sqrt().argmax().item(),
    )


def sine_distance_permute(d, dhat, permute=True, eps=1e-10):

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
    rc = lsa(cost) if permute else (np.arange(p), np.arange(p))

    return torch.sqrt(cost[rc]).max().item()


def sine_distance_permute_all(d, dhat, permute=True, eps=1e-10):

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
    rc = lsa(cost) if permute else (np.arange(p), np.arange(p))
    out = torch.sqrt(cost[rc])

    sorted_col = rc[1]

    return out.mean().item(), out.median().item(), out.max().item(), sorted_col


def compute_psnr(x, xhat):
    psnr = []
    for i in range(x.shape[0]):
        mse = np.mean((x[i] - xhat[i]) ** 2)
        max_x = np.max(x[i])
        psnr.append(20 * np.log10(max_x) - 10 * np.log10(mse))
    return np.mean(psnr)


def test_network(data_loader, net, params, name="test"):

    net.eval()

    device = params["device"]

    psnr = []
    for idx, (x, _) in enumerate(data_loader):

        x = x.to(device)

        # forward ------------------------------------------------------#
        if params["noise_std"]:
            x_noisy = (
                x + params["noise_std"] / 255 * torch.randn(x.shape, device=device)
            ).to(device)
            xhat, _ = net(x_noisy)
        else:
            xhat, _ = net(x)

        xhat = torch.clamp(xhat, 0, 1)

        psnr.append(
            utils.utils.compute_psnr(
                x[:, 0].clone().detach().cpu().numpy(),
                xhat[:, 0].clone().detach().cpu().numpy(),
            )
        )

    psnr = np.mean(np.array(psnr))

    return psnr


def test_network_for_classification(data_loader, net, params):

    device = params["device"]

    net.eval()

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for idx, (z, y) in tqdm(enumerate(data_loader), disable=True):

            z = z.to(device)
            y = y.to(device)

            # forward ------------------------------------------------------#
            yhat = net(z)

            correct_indicators = yhat.max(1)[1].data == y
            num_correct += correct_indicators.sum().item()
            num_total += y.size()[0]

    acc = num_correct / num_total

    return acc
