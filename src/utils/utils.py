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
