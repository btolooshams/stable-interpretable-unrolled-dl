"""
Copyright (c) 2021 Bahareh Tolooshams

models

:author: Bahareh Tolooshams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Classifier(torch.nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()

        self.p = params["p"]
        self.num_class = params["num_class"]
        self.device = params["device"]

        self.classifier = torch.nn.Linear(self.p, self.num_class, device=self.device)  # (in_dim, out_dim)

    def forward(self, x):
        return self.classifier(x)


class MLP(torch.nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()

        self.hid_dim = params["hid_dim"]
        self.in_dim = params["in_dim"]
        self.out_dim = params["out_dim"]
        self.device = params["device"]

        self.E = torch.nn.Linear(self.in_dim, self.hid_dim, device=self.device)
        self.C = torch.nn.Linear(self.hid_dim, self.out_dim, device=self.device)

    def encode(self, x):
        return torch.relu(self.E(torch.squeeze(x, dim=-1)))

    def classify(self, x):
        return self.C(x)

    def forward(self, x):
        z = self.encode(x)
        y = self.classify(z)
        return y, z


# soft-thresholding with fixed lam
class AE(torch.nn.Module):
    def __init__(self, params, W=None):
        super(AE, self).__init__()

        self.m = params["m"]
        self.p = params["p"]
        self.device = params["device"]
        self.lam = params["lam"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn((self.m, self.p), device=self.device)
            W = F.normalize(W, p=2, dim=0)

        self.register_parameter("W", torch.nn.Parameter(W))
        self.register_buffer("step", torch.tensor(params["step"]))

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p=2, dim=0)

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.lam * self.step) * torch.sign(z)
        else:
            z = self.relu(z - self.lam * self.step)
        return z

    def encode(self, x):
        batch_size, device = x.shape[0], x.device
        zhat = torch.zeros(batch_size, self.p, 1, device=device)
        IplusWTW = torch.eye(self.p, device=device) - self.step * torch.matmul(
            torch.t(self.W), self.W
        )
        WTx = self.step * torch.matmul(torch.t(self.W), x)
        for k in range(self.num_layers):
            zhat = self.nonlin(torch.matmul(IplusWTW, zhat) + WTx)
        return zhat

    def decode(self, x):
        return torch.matmul(self.W, x)

    def forward(self, x):
        zT = self.encode(x)
        xhat = self.decode(zT)
        return xhat, zT


# soft-thresholding with learnable lam
class AElearnbias(torch.nn.Module):
    def __init__(self, params, W=None):
        super(AElearnbias, self).__init__()

        self.m = params["m"]
        self.p = params["p"]
        self.device = params["device"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn((self.m, self.p), device=self.device)
            W = F.normalize(W, p=2, dim=0)

        self.register_parameter("W", torch.nn.Parameter(W))
        self.register_buffer("step", torch.tensor(params["step"]))

        b = torch.nn.Parameter(
            torch.zeros((1), device=self.device) + params["lam"] * params["step"]
        )
        self.register_parameter("b", b)  # this is lam * step

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p=2, dim=0)

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.b) * torch.sign(z)
        else:
            z = self.relu(z - self.b)
        return z

    def encode(self, x):
        batch_size, device = x.shape[0], x.device
        zhat = torch.zeros(batch_size, self.p, 1, device=device)
        IplusWTW = torch.eye(self.p, device=device) - self.step * torch.matmul(
            torch.t(self.W), self.W
        )
        WTx = self.step * torch.matmul(torch.t(self.W), x)
        for k in range(self.num_layers):
            zhat = self.nonlin(torch.matmul(IplusWTW, zhat) + WTx)
        return zhat

    def decode(self, x):
        return torch.matmul(self.W, x)

    def forward(self, x):
        zT = self.encode(x)
        xhat = self.decode(zT)
        return xhat, zT


# convolutional - soft-thresholding with fixed lam
class CAE(torch.nn.Module):
    def __init__(self, params, W=None):
        super(CAE, self).__init__()

        self.device = params["device"]
        self.num_ch = params["num_ch"]
        self.lam = params["lam"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]
        self.num_conv = params["num_conv"]
        self.dictionary_dim = params["dictionary_dim"]
        self.stride = params["stride"]
        self.split_stride = params["split_stride"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn(
                (self.num_conv, self.num_ch, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            W = F.normalize(W, p="fro", dim=(-1, -2))
            W /= self.num_ch
        self.register_parameter("W", torch.nn.Parameter(W))
        self.register_buffer("step", torch.tensor(params["step"]))

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p="fro", dim=(-1, -2))
        self.W.data /= self.num_ch

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.lam * self.step) * torch.sign(z)
        else:
            z = self.relu(z - self.lam * self.step)
        return z

    def calc_pad_sizes(self, x):
        left_pad = self.split_stride
        right_pad = (
            0
            if (x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride)
        )
        top_pad = self.split_stride
        bot_pad = (
            0
            if (x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride)
        )
        right_pad += self.split_stride
        bot_pad += self.split_stride
        return left_pad, right_pad, top_pad, bot_pad

    def split_image(self, x):
        if self.split_stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = self.calc_pad_sizes(x)
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.split_stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=x.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        batch_size, device = x_batched_padded.shape[0], x_batched_padded.device
        p1, p2 = F.conv2d(x_batched_padded, self.W, stride=self.stride).shape[2:]

        zhat = torch.zeros(batch_size, self.num_conv, p1, p2, device=device)
        for k in range(self.num_layers):
            Wz = F.conv_transpose2d(zhat, self.W, stride=self.stride)
            res = Wz - x_batched_padded
            grad = F.conv2d(res, self.W, stride=self.stride)
            zhat = self.nonlin(zhat - grad * self.step)

        if self.split_stride > 1:
            xhat = (
                torch.masked_select(
                    F.conv_transpose2d(zhat, self.W, stride=self.stride),
                    valids_batched.bool(),
                ).reshape(x.shape[0], self.split_stride ** 2, *x.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            xhat = F.conv_transpose2d(zhat, self.W, stride=self.stride)

        return xhat, zhat

# convolutional - soft-thresholding with learnable lam
class CAElearnbias(torch.nn.Module):
    def __init__(self, params, W=None):
        super(CAElearnbias, self).__init__()

        self.device = params["device"]
        self.num_ch = params["num_ch"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]
        self.num_conv = params["num_conv"]
        self.dictionary_dim = params["dictionary_dim"]
        self.stride = params["stride"]
        self.split_stride = params["split_stride"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn(
                (self.num_conv, self.num_ch, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            W = F.normalize(W, p="fro", dim=(-1, -2))
            W /= self.num_ch
        self.register_parameter("W", torch.nn.Parameter(W))
        self.register_buffer("step", torch.tensor(params["step"]))

        b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device)
            + params["lam"] * params["step"]
        )
        self.register_parameter("b", b)  # this is lam * step

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p="fro", dim=(-1, -2))
        self.W.data /= self.num_ch

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.b) * torch.sign(z)
        else:
            z = self.relu(z - self.b)
        return z

    def calc_pad_sizes(self, x):
        left_pad = self.split_stride
        right_pad = (
            0
            if (x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride)
        )
        top_pad = self.split_stride
        bot_pad = (
            0
            if (x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride)
        )
        right_pad += self.split_stride
        bot_pad += self.split_stride
        return left_pad, right_pad, top_pad, bot_pad

    def split_image(self, x):
        if self.split_stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = self.calc_pad_sizes(x)
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.split_stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=x.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        batch_size, device = x_batched_padded.shape[0], x_batched_padded.device
        p1, p2 = F.conv2d(x_batched_padded, self.W, stride=self.stride).shape[2:]

        zhat = torch.zeros(batch_size, self.num_conv, p1, p2, device=device)
        for k in range(self.num_layers):
            Wz = F.conv_transpose2d(zhat, self.W, stride=self.stride)
            res = Wz - x_batched_padded
            grad = F.conv2d(res, self.W, stride=self.stride)
            zhat = self.nonlin(zhat - grad * self.step)

        if self.split_stride > 1:
            xhat = (
                torch.masked_select(
                    F.conv_transpose2d(zhat, self.W, stride=self.stride),
                    valids_batched.bool(),
                ).reshape(x.shape[0], self.split_stride ** 2, *x.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            xhat = F.conv_transpose2d(zhat, self.W, stride=self.stride)

        return xhat, zhat

# convolutional - soft-thresholding with learnable lam
class CAElearnbiasuntied(torch.nn.Module):
    def __init__(self, params, W=None):
        super(CAElearnbiasuntied, self).__init__()

        self.device = params["device"]
        self.num_ch = params["num_ch"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]
        self.num_conv = params["num_conv"]
        self.dictionary_dim = params["dictionary_dim"]
        self.stride = params["stride"]
        self.split_stride = params["split_stride"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn(
                (self.num_conv, self.num_ch, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            W = F.normalize(W, p="fro", dim=(-1, -2))
            W /= self.num_ch
        self.register_parameter("W", torch.nn.Parameter(W))

        E = torch.clone(W)
        D = torch.clone(W)

        self.register_parameter("E", torch.nn.Parameter(E))
        self.register_parameter("D", torch.nn.Parameter(D))

        self.register_buffer("step", torch.tensor(params["step"]))

        b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device)
            + params["lam"] * params["step"]
        )
        self.register_parameter("b", b)  # this is lam * step

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p="fro", dim=(-1, -2))
        self.W.data /= self.num_ch

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.b) * torch.sign(z)
        else:
            z = self.relu(z - self.b)
        return z

    def calc_pad_sizes(self, x):
        left_pad = self.split_stride
        right_pad = (
            0
            if (x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride)
        )
        top_pad = self.split_stride
        bot_pad = (
            0
            if (x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride)
        )
        right_pad += self.split_stride
        bot_pad += self.split_stride
        return left_pad, right_pad, top_pad, bot_pad

    def split_image(self, x):
        if self.split_stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = self.calc_pad_sizes(x)
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.split_stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=x.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        batch_size, device = x_batched_padded.shape[0], x_batched_padded.device
        p1, p2 = F.conv2d(x_batched_padded, self.W, stride=self.stride).shape[2:]

        zhat = torch.zeros(batch_size, self.num_conv, p1, p2, device=device)
        for k in range(self.num_layers):
            Wz = F.conv_transpose2d(zhat, self.D, stride=self.stride)
            res = Wz - x_batched_padded
            grad = F.conv2d(res, self.E, stride=self.stride)
            zhat = self.nonlin(zhat - grad * self.step)

        if self.split_stride > 1:
            xhat = (
                torch.masked_select(
                    F.conv_transpose2d(zhat, self.W, stride=self.stride),
                    valids_batched.bool(),
                ).reshape(x.shape[0], self.split_stride ** 2, *x.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            xhat = F.conv_transpose2d(zhat, self.W, stride=self.stride)

        return xhat, zhat

# convolutional - soft-thresholding with learnable lam
class CAElearnbiasstep(torch.nn.Module):
    def __init__(self, params, W=None):
        super(CAElearnbiasstep, self).__init__()

        self.device = params["device"]
        self.num_ch = params["num_ch"]
        self.num_layers = params["num_layers"]
        self.twosided = params["twosided"]
        self.num_conv = params["num_conv"]
        self.dictionary_dim = params["dictionary_dim"]
        self.stride = params["stride"]
        self.split_stride = params["split_stride"]

        self.relu = torch.nn.ReLU()

        if W is None:
            W = torch.randn(
                (self.num_conv, self.num_ch, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            W = F.normalize(W, p="fro", dim=(-1, -2))
            W /= self.num_ch
        self.register_parameter("W", torch.nn.Parameter(W))
        step = torch.nn.Parameter(
            torch.zeros(1, device=self.device) + params["step"]
        )
        self.register_parameter("step", step)

        b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device)
            + params["lam"] * params["step"]
        )
        self.register_parameter("b", b)  # this is lam * step

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p="fro", dim=(-1, -2))
        self.W.data /= self.num_ch

    def nonlin(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.b) * torch.sign(z)
        else:
            z = self.relu(z - self.b)
        return z

    def calc_pad_sizes(self, x):
        left_pad = self.split_stride
        right_pad = (
            0
            if (x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[3] + left_pad - self.dictionary_dim) % self.split_stride)
        )
        top_pad = self.split_stride
        bot_pad = (
            0
            if (x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride == 0
            else self.split_stride
            - ((x.shape[2] + top_pad - self.dictionary_dim) % self.split_stride)
        )
        right_pad += self.split_stride
        bot_pad += self.split_stride
        return left_pad, right_pad, top_pad, bot_pad

    def split_image(self, x):
        if self.split_stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = self.calc_pad_sizes(x)
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.split_stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=x.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        batch_size, device = x_batched_padded.shape[0], x_batched_padded.device
        p1, p2 = F.conv2d(x_batched_padded, self.W, stride=self.stride).shape[2:]

        zhat = torch.zeros(batch_size, self.num_conv, p1, p2, device=device)
        for k in range(self.num_layers):
            Wz = F.conv_transpose2d(zhat, self.W, stride=self.stride)
            res = Wz - x_batched_padded
            grad = F.conv2d(res, self.W, stride=self.stride)
            zhat = self.nonlin(zhat - grad * self.step)

        if self.split_stride > 1:
            xhat = (
                torch.masked_select(
                    F.conv_transpose2d(zhat, self.W, stride=self.stride),
                    valids_batched.bool(),
                ).reshape(x.shape[0], self.split_stride ** 2, *x.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            xhat = F.conv_transpose2d(zhat, self.W, stride=self.stride)

        return xhat, zhat
