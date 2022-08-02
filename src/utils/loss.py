"""
Copyright (c) 2021 Bahareh Tolooshams

loss utils

:author: Bahareh Tolooshams
"""

import torch


class DLLoss1D(torch.nn.Module):
    def __init__(self, params):
        super(DLLoss1D, self).__init__()
        self.lam = params["lam_loss"]
        self.rho = params["rho_loss"]

    def forward(self, x, xhat, zhat, dhat, new_lam=None):
        if new_lam:
            lam = new_lam
        else:
            lam = self.lam

        rec = 0.5 * (x - xhat).pow(2).sum(dim=1).mean()

        if lam:
            l1z = zhat.abs().sum(dim=1).mean()
        else:
            l1z = 0.0

        if self.rho:
            l2d = dhat.pow(2).sum(dim=1).mean()
        else:
            l2d = 0.0

        return rec + lam * l1z + (self.rho / 2) * l2d


class DLLoss2D(torch.nn.Module):
    def __init__(self, params):
        super(DLLoss2D, self).__init__()
        self.lam = params["lam_loss"]
        self.rho = params["rho_loss"]

    def forward(self, x, xhat, zhat, dhat, new_lam=None):
        if new_lam:
            lam = new_lam
        else:
            lam = self.lam

        rec = 0.5 * (x - xhat).pow(2).sum(dim=(-1, -2)).mean()

        if lam:
            l1z = zhat.abs().sum(dim=(-1, -2)).mean()
        else:
            l1z = 0.0

        if self.rho:
            l2d = dhat.pow(2).sum(dim=(-1, -2)).mean()
        else:
            l2d = 0.0

        return rec + lam * l1z + (self.rho / 2) * l2d


class SparseLoss(torch.nn.Module):
    def __init__(self, lam=1):
        super(SparseLoss, self).__init__()
        self.lam = lam

    def forward(self, x, xhat, zhat):
        rec = 0.5 * (x - xhat).pow(2).sum(dim=1).mean()
        l1z = zhat.abs().sum(dim=1).mean()
        return rec, rec + self.lam * l1z


class Lossl2(torch.nn.Module):
    def __init__(self):
        super(Lossl2, self).__init__()

    def forward(self, x, xhat):
        return 0.5 * (x - xhat).pow(2).sum(dim=1).mean()
