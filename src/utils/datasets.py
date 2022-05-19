"""
Copyright (c) 2021 Bahareh Tolooshams

dataset

:author: Bahareh Tolooshams
"""

import numpy as np
import torch, torchvision


def get_simulated_dataset(data_path, device="cpu"):
    return torch.load(data_path, map_location=device)


def get_mnist_dataset(
    class_list=None,
    make_flat=False,
    normalize=False,
    whiten=False,
    datadir="../data/data_cache",
):

    train_dataset = torchvision.datasets.MNIST(
        root=datadir, train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.MNIST(
        root=datadir, train=False, download=True, transform=None
    )

    def to_xy(dataset):
        X = np.array(dataset.data) / 255.0  # [0, 1]
        Y = np.array(dataset.targets)
        return X, Y

    def get_subclasses(X, Y, class_list):
        Y_subclasses = []
        X_subclasses = []
        for c in class_list:
            Yc = Y[Y == c].copy()
            Xc = X[Y == c].copy()

            Y_subclasses.append(Yc)
            X_subclasses.append(Xc)

        X_subclasses = np.concatenate(X_subclasses, axis=0)
        Y_subclasses = np.concatenate(Y_subclasses, axis=0)

        return X_subclasses, Y_subclasses

    X_tr, Y_tr = to_xy(train_dataset)
    X_te, Y_te = to_xy(test_dataset)

    if class_list:
        X_tr, Y_tr = get_subclasses(X_tr, Y_tr, class_list)
        X_te, Y_te = get_subclasses(X_te, Y_te, class_list)

    if normalize:
        X_tr = X_tr - np.mean(X_tr, axis=(1, 2), keepdims=True) / np.std(
            X_tr, axis=(1, 2), keepdims=True
        )
        X_te = X_te - np.mean(X_te, axis=(1, 2), keepdims=True) / np.std(
            X_te, axis=(1, 2), keepdims=True
        )

    if whiten:
        print("ERROR WHITENING!")
        # raise NotImplementedError

        eps = 1e-5

        # normalize
        X_tr = X_tr - np.mean(X_tr, axis=(1, 2), keepdims=True) / np.std(
            X_tr, axis=(1, 2), keepdims=True
        )
        # flat
        X_tr = np.reshape(X_tr, (X_tr.shape[0], -1)).T
        # whiten
        xcov = np.cov(X_tr, rowvar=True)
        U, S, V = np.linalg.svd(xcov)
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
        zca_tr = np.dot(zca_matrix, X_tr)
        # reshape
        X_tr = np.reshape(zca_tr.T, (zca_tr.shape[-1], 28, 28))

        X_tr = (X_tr - np.min(X_tr)) / (np.max(X_tr) - np.min(X_tr))

        # normalize
        X_te = X_te - np.mean(X_te, axis=(1, 2), keepdims=True) / np.std(
            X_te, axis=(1, 2), keepdims=True
        )
        # flat
        X_te = np.reshape(X_te, (X_te.shape[0], -1)).T
        # whiten
        xcov = np.cov(X_te, rowvar=True, bias=True)
        U, S, V = np.linalg.svd(xcov)
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
        zca_te = np.dot(zca_matrix, X_te)
        # reshape
        X_te = np.reshape(zca_te.T, (zca_te.shape[-1], 28, 28))

        X_te = (X_te - np.min(X_te)) / (np.max(X_te) - np.min(X_te))

    X_tr = np.expand_dims(X_tr, axis=1)
    X_te = np.expand_dims(X_te, axis=1)

    if make_flat:
        X_tr = np.reshape(X_tr, (X_tr.shape[0], -1, 1))
        X_te = np.reshape(X_te, (X_te.shape[0], -1, 1))

    return X_tr, Y_tr, X_te, Y_te


def get_cifar_dataset(
    class_list=None,
    blackandwhite=True,
    make_flat=False,
    whiten=False,
    datadir="../data/data_cache",
):

    train_ds = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True)

    def to_xy(dataset):
        X = np.transpose(dataset.data, (0, 3, 1, 2)) / 255.0  # [0, 1]
        Y = np.array(dataset.targets)
        return X, Y

    def get_subclasses(X, Y, class_list):
        Y_subclasses = []
        X_subclasses = []
        for c in class_list:
            Yc = Y[Y == c].copy()
            Xc = X[Y == c].copy()

            Y_subclasses.append(Yc)
            X_subclasses.append(Xc)

        X_subclasses = np.concatenate(X_subclasses, axis=0)
        Y_subclasses = np.concatenate(Y_subclasses, axis=0)

        return X_subclasses, Y_subclasses

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)

    if class_list:
        X_tr, Y_tr = get_subclasses(X_tr, Y_tr, class_list)
        X_te, Y_te = get_subclasses(X_te, Y_te, class_list)

    if blackandwhite:
        X_tr = 0.2989 * X_tr[:, 0] + 0.5870 * X_tr[:, 1] + 0.1140 * X_tr[:, 2]
        X_te = 0.2989 * X_te[:, 0] + 0.5870 * X_te[:, 1] + 0.1140 * X_te[:, 2]

        X_tr = np.expand_dims(X_tr, axis=1)
        X_te = np.expand_dims(X_te, axis=1)

    if whiten:
        print("ERROR WHITENING!")
        # raise NotImplementedError

        eps = 1e-5

        # normalize
        X_tr = X_tr - np.mean(X_tr, axis=(1, 2, 3), keepdims=True) / np.std(
            X_tr, axis=(1, 2, 3), keepdims=True
        )
        # flat
        X_tr = np.reshape(X_tr, (X_tr.shape[0], -1)).T
        # whiten
        xcov = np.cov(X_tr, rowvar=True)
        U, S, V = np.linalg.svd(xcov)
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
        zca_tr = np.dot(zca_matrix, X_tr)
        # reshape
        X_tr = np.reshape(zca_tr.T, (zca_tr.shape[-1], 1, 32, 32))

        X_tr = (X_tr - np.min(X_tr)) / (np.max(X_tr) - np.min(X_tr))

        # normalize
        X_te = X_te - np.mean(X_te, axis=(1, 2, 3), keepdims=True) / np.std(
            X_te, axis=(1, 2, 3), keepdims=True
        )
        # flat
        X_te = np.reshape(X_te, (X_te.shape[0], -1)).T
        # whiten
        xcov = np.cov(X_te, rowvar=True, bias=True)
        U, S, V = np.linalg.svd(xcov)
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
        zca_te = np.dot(zca_matrix, X_te)
        # reshape
        X_te = np.reshape(zca_te.T, (zca_te.shape[-1], 1, 32, 32))

        X_te = (X_te - np.min(X_te)) / (np.max(X_te) - np.min(X_te))

    if make_flat:
        X_tr = np.reshape(X_tr, (X_tr.shape[0], -1, 1))
        X_te = np.reshape(X_te, (X_te.shape[0], -1, 1))

    return X_tr, Y_tr, X_te, Y_te
