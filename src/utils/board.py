"""
Copyright (c) 2021 Bahareh Tolooshams

board

:author: Bahareh Tolooshams
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.gridspec as gridspec


def log_code(writer, z, epoch, reshape=None):
    # plot code

    i = np.random.randint(z.shape[0])
    code = torch.squeeze(z[i]).clone().detach().cpu().numpy()
    code = np.reshape(code, reshape)

    fig = plt.figure()
    ax1 = plt.subplot(111)
    plt.imshow(code, cmap="gray")
    plt.axis("off")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect("equal")
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("z", fig, epoch)
    plt.close()
    return writer


def log_dictionary(writer, net, epoch, reshape=None):
    # plot dict
    a = np.int(np.ceil(np.sqrt(net.p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(net.p):
        ax1 = plt.subplot(gs1[col])
        wi = net.W.data[:, col].clone().detach().cpu().numpy()
        if reshape:
            wi = np.reshape(wi, reshape)
        plt.imshow(wi, cmap="gray")
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("W", fig, epoch)
    plt.close()
    return writer


def log_dictionary_conv(writer, net, epoch):
    # plot dict
    a = np.int(np.ceil(np.sqrt(net.num_conv)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)

    W = net.W.data.clone().detach().cpu().numpy()
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
    for conv in range(net.num_conv):
        ax1 = plt.subplot(gs1[conv])
        wi = W[conv]
        wi = np.transpose(wi, (1, 2, 0))
        if wi.shape[-1] == 1:
            plt.imshow(wi, cmap="gray")
        else:
            plt.imshow(wi)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("W", fig, epoch)
    plt.close()
    return writer


def log_mlp_encoder(writer, net, epoch, reshape=None):
    # plot dict
    a = np.int(np.ceil(np.sqrt(net.hid_dim)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(net.hid_dim):
        ax1 = plt.subplot(gs1[col])
        wi = net.E.weight[col, :].clone().detach().cpu().numpy()
        if reshape:
            wi = np.reshape(wi, reshape)
        plt.imshow(wi, cmap="gray")
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("MLPencoder", fig, epoch)
    plt.close()
    return writer


def log_img(writer, x, xhat, epoch, reshape=None):
    # plot img
    fig = plt.figure()
    ax = plt.subplot(121)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    i = np.random.randint(x.shape[0])
    img = x[i].clone().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if reshape:
        img = np.reshape(img, reshape)
    plt.imshow(img, cmap="gray")
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    ax = plt.subplot(122)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    img_hat = xhat[i].clone().detach().cpu().numpy()
    img_hat = np.transpose(img_hat, (1, 2, 0))
    if reshape:
        img_hat = np.reshape(img_hat, reshape)
    plt.imshow(img_hat, cmap="gray")
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("x", fig, epoch)
    plt.close()
    return writer
