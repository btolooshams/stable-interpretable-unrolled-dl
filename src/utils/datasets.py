"""
Copyright (c) 2021 Bahareh Tolooshams

dataset

:author: Bahareh Tolooshams
"""

import numpy as np
import torch, torchvision
import torch.nn.functional as F
import numpy as np


class xDzDataset(torch.utils.data.Dataset):
    def __init__(self, params, D=None):

        self.n = params["n"]
        self.m = params["m"]
        self.p = params["p"]
        self.s = params["s"]
        self.code_dist = params["code_dist"]
        if self.code_dist == "uniform":
            self.c_min = params["c_min"]
            self.c_max = params["c_max"]
        else:
            self.z_mean = params["z_mean"]
            self.z_std = params["z_std"]
        self.manual_seed = params["manual_seed"]
        self.num_distinct_supp_sets = params["num_distinct_supp_sets"]
        self.orth_col = params["orth_col"]
        self.device = params["device"]

        # generate code
        if self.code_dist == "uniform":
            self.z = self.generate_sparse_samples_uniform(
                self.n,
                self.p,
                self.s,
                self.c_min,
                self.c_max,
                self.num_distinct_supp_sets,
                device=self.device,
                seed=self.manual_seed,
            )
        elif self.code_dist == "subgaussian":
            self.z = self.generate_sparse_samples_subgaussian(
                self.n,
                self.p,
                self.s,
                self.z_mean,
                self.z_std,
                self.num_distinct_supp_sets,
                device=self.device,
                seed=self.manual_seed,
            )
        else:
            print("Code distribution is not implemented!")
            raise NotImplementedError

        # create filters
        if D is None:
            if self.orth_col:
                print("This line gives you error if m neq p!")
                D, _ = qr(np.random.randn(self.m, self.p))
                D = torch.Tensor(D, device=self.device)
            else:
                D = (1 / np.sqrt(self.m)) * torch.randn(
                    (self.m, self.p), device=self.device
                )
            D = F.normalize(D, p=2, dim=0)
            D *= 1
        self.D = D
        # generate data
        self.x = torch.matmul(self.D, self.z)

    def generate_sparse_samples_uniform(
        self, n, p, s, c_min, c_max, num_distinct_supp_sets=1, device="cpu", seed=None
    ):

        samples = torch.zeros((n, p, 1), device=device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        for j in range(num_distinct_supp_sets):
            example_set_size = int(n / num_distinct_supp_sets)
            for i in range(example_set_size):
                supp_set_size = int(p / num_distinct_supp_sets)
                ind = np.random.choice(supp_set_size, s, replace=False)
                ind = ind + j * supp_set_size
                i = i + j * example_set_size
                # [1, 2]
                samples[i][ind, 0] = (
                    torch.rand(s, device=device) * (c_max - c_min) + c_min
                ) * ((torch.rand(1, device=device) > 0.5) * 2 - 1)

        return samples

    def generate_sparse_samples_subgaussian(
        self, n, p, s, z_mean, z_std, num_distinct_supp_sets=1, device="cpu", seed=None
    ):
        samples = torch.zeros((n, p, 1), device=device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        for j in range(num_distinct_supp_sets):
            example_set_size = int(n / num_distinct_supp_sets)
            for i in range(example_set_size):
                supp_set_size = int(p / num_distinct_supp_sets)
                ind = np.random.choice(supp_set_size, s, replace=False)
                ind = ind + j * supp_set_size
                i = i + j * example_set_size
                samples[i][ind, 0] = (
                    torch.randn(s, device=device) * z_std + z_mean
                ) * ((torch.rand(1, device=device) > 0.5) * 2 - 1)

        return samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with torch.no_grad():
            return self.x[idx], self.z[idx]


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


def get_test_path_loader(batch_size, image_path, shuffle=False, num_workers=4):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=image_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def get_train_path_loader(
    batch_size, image_path, crop_dim=(128, 128), shuffle=True, num_workers=4
):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=image_path,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.RandomCrop(
                        crop_dim,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader
