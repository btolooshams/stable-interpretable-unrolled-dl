"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import scipy as sp
import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec


def visualize_dense_dictionary(D, save_path, reshape=(28, 28), cmap="gray"):
    p = D.shape[-1]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = D[:, col].clone().detach().cpu().numpy()
        if reshape:
            wi = np.reshape(wi, reshape)
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_dictionary(D, save_path, cmap="gray"):
    p = D.shape[0]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    W = D.clone().detach().cpu().numpy()
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = W[col]
        wi = np.transpose(wi, (1, 2, 0))
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_feature_maps(Z, save_path, cmap="afmhot"):
    p = Z.shape[0]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = Z[col].clone().detach().cpu().numpy()
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_bias(b, save_path):

    ################################
    axes_fontsize = 30
    legend_fontsize = 30
    tick_fontsize = 30
    title_fontsize = 30

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    eps = 1e-6
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.subplot(111)
    hist, bins = np.histogram(b, bins=50)
    logbins = np.logspace(np.log10(bins[0] + eps), np.log10(bins[-1] + eps), len(bins))
    plt.hist(b, bins=logbins)
    plt.title("Histogram of bias")

    fig.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_image(x, xhat, save_path, cmap="gray"):

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(x, (1, 2, 0)), cmap=cmap)
    plt.title("img")

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(xhat, (1, 2, 0)), cmap=cmap)
    plt.title("est")

    fig.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_singular_values_of_Z(Z, save_path):

    Z_normalized = torch.nn.functional.normalize(Z, dim=0)
    Z_normalized -= torch.mean(Z_normalized, dim=1, keepdims=True)
    s = sp.linalg.svdvals(Z_normalized)

    # print(s[0], s[-1])
    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # x-axis: index, y-axis: singular values

    axn.flat[0].set_xscale("log")
    axn.flat[0].set_yscale("log")
    plt.subplot(1, 2, 1)
    plt.plot(s)
    plt.xlabel("$\mathrm{Singular\ index}$")
    plt.ylabel("$\mathrm{Singular\ values}$")
    plt.title(
        "{} sv greater than 0.1 of largest sv".format(np.where(s > s[0] * 0.1)[0][-1])
    )

    # x-axis: singular values, y-axis: frequency of values
    plt.subplot(1, 2, 2)
    plt.hist(s, bins=20)
    plt.xlabel("$\mathrm{Frequency}$")
    plt.ylabel("$\mathrm{Singular\ values}$")
    plt.title("$\mathrm{Z\ histogram}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_singular_values_of_X(X, save_path):

    X_normalized = torch.nn.functional.normalize(X, dim=0)
    X_normalized -= torch.mean(X_normalized, dim=1, keepdims=True)
    s = sp.linalg.svdvals(X_normalized)

    # print(s[0], s[-1])
    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # x-axis: index, y-axis: singular values

    axn.flat[0].set_xscale("log")
    axn.flat[0].set_yscale("log")
    plt.subplot(1, 2, 1)
    plt.plot(s)
    plt.xlabel("$\mathrm{Singular\ index}$")
    plt.ylabel("$\mathrm{Singular\ values}$")
    plt.title(
        "{} sv greater than 0.1 of largest sv".format(np.where(s > s[0] * 0.1)[0][-1])
    )

    # x-axis: singular values, y-axis: frequency of values
    plt.subplot(1, 2, 2)
    plt.hist(s, bins=100)
    plt.xlabel("$\mathrm{Frequency}$")
    plt.ylabel("$\mathrm{Singular\ values}$")
    plt.title("$\mathrm{X\ histogram}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_singular_values_of_Z_as_a_function_of_beta(Z_all, beta_all, save_path):

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for exp_ctr in range(len(Z_all)):
        Z = Z_all[exp_ctr]
        beta = beta_all[exp_ctr]

        Z_normalized = torch.nn.functional.normalize(Z, dim=0)
        Z_normalized -= torch.mean(Z_normalized, dim=1, keepdims=True)
        s = sp.linalg.svdvals(Z_normalized)

        # x-axis: index, y-axis: singular values
        axn.flat[0].set_xscale("log")
        axn.flat[0].set_yscale("log")
        plt.subplot(1, 2, 1)
        plt.plot(s, label="beta {}".format(beta))
        plt.xlabel("$\mathrm{Singular\ index}$")
        plt.ylabel("$\mathrm{Singular\ values}$")
        plt.title("(1-beta) Lasso + beta Classification")
        plt.legend()

        # x-axis: singular values, y-axis: frequency of values
        eps = 1e-32
        axn.flat[1].set_xscale("log")
        axn.flat[1].set_yscale("log")
        plt.subplot(1, 2, 2)
        hist, bins = np.histogram(s, bins=50)
        logbins = np.logspace(
            np.log10(bins[0] + eps), np.log10(bins[-1] + eps), len(bins)
        )
        plt.hist(s, bins=logbins)
        plt.xlabel("$\mathrm{Singular\ values}$")
        plt.ylabel("$\mathrm{Frequency}$")
        plt.title("$\mathrm{Z\ histogram}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_singular_values_of_Z_as_a_function_of_sparsity(
    Z_all, save_path, thres=0.01
):

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for exp_ctr in range(len(Z_all)):
        Z = Z_all[exp_ctr]

        sparsity = Z.clone()
        sparsity[torch.abs(sparsity) > thres] = 1.0
        sparsity[torch.abs(sparsity) <= thres] = 0.0

        sparsity = torch.mean(sparsity) * 100

        Z_normalized = torch.nn.functional.normalize(Z, dim=0)
        Z_normalized -= torch.mean(Z_normalized, dim=1, keepdims=True)
        s = sp.linalg.svdvals(Z_normalized)

        # x-axis: index, y-axis: singular values
        axn.flat[0].set_xscale("log")
        axn.flat[0].set_yscale("log")
        plt.subplot(1, 2, 1)
        plt.plot(s, label="{:.1f} \% sparse".format(sparsity))
        plt.xlabel("$\mathrm{Singular\ index}$")
        plt.ylabel("$\mathrm{Singular\ values}$")
        plt.title("Rank of Z as a function of lam (sparsity)")
        plt.legend()

        # x-axis: singular values, y-axis: frequency of values
        eps = 1e-32
        axn.flat[1].set_xscale("log")
        axn.flat[1].set_yscale("log")
        plt.subplot(1, 2, 2)
        hist, bins = np.histogram(s, bins=50)
        logbins = np.logspace(
            np.log10(bins[0] + eps), np.log10(bins[-1] + eps), len(bins)
        )
        plt.hist(s, bins=logbins)
        plt.xlabel("$\mathrm{Singular\ values}$")
        plt.ylabel("$\mathrm{Frequency}$")
        plt.title("$\mathrm{Z\ histogram}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_eigenvalues_of_G(G, save_path):

    eigv, _ = np.linalg.eig(G)

    # print(s[0], s[-1])
    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # x-axis: index, y-axis: singular values
    plt.subplot(1, 2, 1)
    plt.plot(eigv)
    plt.xlabel("$\mathrm{Eigen\ index}$")
    plt.ylabel("$\mathrm{Eigenvalues}$")
    plt.title(
        "{} eigv greater than 0.01 of largest eigv".format(
            np.where(eigv > eigv[0] * 0.01)[0][-1]
        )
    )

    # x-axis: singular values, y-axis: frequency of values
    plt.subplot(1, 2, 2)
    plt.hist(eigv, bins=100)
    plt.xlabel("$\mathrm{Frequency}$")
    plt.ylabel("$\mathrm{Eigenvalues}$")
    plt.title("$\mathrm{Histogram}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_sorted_atoms(Z, D, save_path, reshape=(28, 28)):

    eps = 1e-11
    Z_energy = torch.mean(Z.pow(2), dim=-1).detach().cpu().numpy()
    Z_energy_normalized = Z_energy / (np.linalg.norm(Z_energy) + eps)
    Z_energy_normalized = np.nan_to_num(Z_energy_normalized)

    sorted_atoms = np.flip(np.argsort(Z_energy_normalized))
    D_sorted = D[:, sorted_atoms.copy()]

    D_sorted = D_sorted[:, :196]

    visualize_dense_dictionary(D_sorted, save_path=save_path, reshape=reshape)


def visualize_most_used_atoms_energy(Z, D, save_path, reshape=(28, 28)):

    eps = 1e-11
    Z_energy = torch.mean(Z.pow(2), dim=-1).detach().cpu().numpy()
    Z_energy_normalized = Z_energy / (np.linalg.norm(Z_energy) + eps)
    Z_energy_normalized = np.nan_to_num(Z_energy_normalized)

    sorted_atoms = np.argsort(Z_energy_normalized)

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(2, 3)

    ctr = 0
    for ax in axn.flat:
        if ctr <= 1:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr > 1:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    # x-axis: atoms, y-axis: energy
    plt.subplot(2, 3, 1)
    plt.plot(Z_energy_normalized)
    plt.xlabel("$\mathrm{Atom\ index}$")
    plt.ylabel("$\mathrm{Normalized\ energy}$")
    plt.title("$\mathrm{Code\ energy}$")

    # x-axis: energy, y-axis: number of atoms in density
    plt.subplot(2, 3, 2)
    plt.hist(Z_energy_normalized)
    plt.xlabel("$\mathrm{Code\ energy}$")
    plt.ylabel("$\mathrm{Atoms\ density}$")
    plt.title("$\mathrm{Atom\ histogram}$")

    for i in range(4):
        plt.subplot(2, 3, i + 3)
        plt.imshow(np.reshape(D[:, sorted_atoms[-1 - i]], reshape), cmap="gray")
        plt.title("{:.2f}".format(Z_energy_normalized[sorted_atoms[-1 - i]]))

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_most_used_atoms_active(Z, D, save_path, eps=1e-2, reshape=(28, 28)):

    Z_ind = Z.clone()
    Z_ind[torch.abs(Z_ind) > eps] = 1
    Z_ind[torch.abs(Z_ind) <= eps] = 0
    freq_of_an_atom_used = torch.mean(Z_ind, dim=1).detach().cpu().numpy()

    freq_of_an_atom_used = np.nan_to_num(freq_of_an_atom_used)

    sorted_atoms = np.argsort(freq_of_an_atom_used)

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(2, 3)

    ctr = 0
    for ax in axn.flat:
        if ctr <= 1:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr > 1:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    # x-axis: atoms, y-axis: energy
    plt.subplot(2, 3, 1)
    plt.plot(freq_of_an_atom_used)
    plt.xlabel("$\mathrm{Atom\ index}$")
    plt.ylabel("$\mathrm{Normalized\ frequency}$")
    plt.title("$\mathrm{Code\ active}$")

    # x-axis: active, y-axis: number of atoms in density
    plt.subplot(2, 3, 2)
    plt.hist(freq_of_an_atom_used)
    plt.xlabel("$\mathrm{Code\ active}$")
    plt.ylabel("$\mathrm{Atoms\ density}$")
    plt.title("$\mathrm{Atom\ histogram}$")

    for i in range(4):
        plt.subplot(2, 3, i + 3)
        plt.imshow(np.reshape(D[:, sorted_atoms[-1 - i]], reshape), cmap="gray")
        plt.title("{:.2f}".format(freq_of_an_atom_used[sorted_atoms[-1 - i]]))

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_most_similar_training_examples_based_on_GandZ(
    Ginv,
    Z,
    z_new,
    X,
    Y,
    x_new,
    xhat_new,
    params,
    save_path,
    used_num_images=200,
    reshape=(28, 28),
):

    Ginv_normalized = torch.nn.functional.normalize(Ginv, dim=0).clone()
    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()

    Ginv_ZT = torch.matmul(Ginv_normalized, Z_normalized.T)
    beta = torch.matmul(Ginv_ZT, z_new_normalized).detach().cpu().numpy()

    sorted_index = np.argsort(np.abs(beta))

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(4, 5, sharex="row", sharey="row")

    ctr = 0
    for ax in axn.flat:
        if ctr < 5:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr >= 5:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    for c in range(len(params["class_list"])):
        plt.subplot(4, 5, c + 1)
        c_indices = Y == params["class_list"][c]
        beta_c = beta[c_indices]
        plt.hist(beta_c)
        plt.title("Class {} code hist".format(params["class_list"][c]))
        plt.xlabel("$\mathrm{Beta}$")
        if c == 0:
            plt.ylabel("$\mathrm{Frequency}$")

    plt.subplot(4, 5, 6)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("Image")

    plt.subplot(4, 5, 7)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("Rec")

    plt.subplot(4, 5, 8)
    x_new_estimate = np.matmul(
        X[:, sorted_index[-used_num_images:]], beta[sorted_index[-used_num_images:]]
    )
    plt.imshow(np.reshape(x_new_estimate, reshape), cmap="gray")
    plt.title("Estimate")

    # most similar
    for i in range(7):
        plt.subplot(4, 5, i + 9)
        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(beta[sorted_index[-1 - i]]),
            color="green",
        )

    # least similar
    for i in range(5):
        plt.subplot(4, 5, i + 16)
        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title("{:.5f}".format(beta[sorted_index[i]]), color="red")

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_most_similar_training_examples_based_on_GandZ_nohist(
    Ginv,
    Z,
    z_new,
    X,
    Y,
    x_new,
    xhat_new,
    params,
    save_path,
    used_num_images=200,
    reshape=(28, 28),
):

    Ginv_normalized = torch.nn.functional.normalize(Ginv, dim=0).clone()
    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()

    Ginv_ZT = torch.matmul(Ginv_normalized, Z_normalized.T)
    beta = torch.matmul(Ginv_ZT, z_new_normalized).detach().cpu().numpy()

    eps = 1e-14
    beta /= np.linalg.norm(beta) + eps
    beta = np.nan_to_num(beta)

    sorted_index = np.argsort(np.abs(beta))

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 28
    title_fontsize = 28

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(8, 6.5))

    for ax in axn.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(3, 5, 1)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("$\mathrm{Image}$")

    plt.subplot(3, 5, 6)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("$\mathrm{Rec}$")

    plt.subplot(3, 5, 11)
    x_new_estimate = np.matmul(
        X[:, sorted_index[-used_num_images:]], beta[sorted_index[-used_num_images:]]
    )
    plt.imshow(np.reshape(x_new_estimate, reshape), cmap="gray")
    plt.title("$\mathrm{Estimate}$")

    # most contribution
    fig_place = [2, 3, 7, 8, 12, 13]
    for i in range(6):
        plt.subplot(3, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(np.abs(beta[sorted_index[-1 - i]])),
            color="green",
        )

    # least contribution
    fig_place = [4, 5, 9, 10, 14, 15]
    for i in range(6):
        plt.subplot(3, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title("{:.5f}".format(np.abs(beta[sorted_index[i]])), color="red")

    fig.tight_layout(pad=0.00, w_pad=0.2, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def visualize_most_similar_training_examples_based_on_GandZ_nohist_donotnormalize(
    Ginv,
    Z,
    z_new,
    X,
    Y,
    x_new,
    xhat_new,
    params,
    save_path,
    used_num_images=200,
    reshape=(28, 28),
):

    Ginv_normalized = torch.nn.functional.normalize(Ginv, dim=0).clone()
    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()

    Ginv_ZT = torch.matmul(Ginv_normalized, Z_normalized.T)
    beta = torch.matmul(Ginv_ZT, z_new_normalized).detach().cpu().numpy()

    eps = 1e-14
    print("beta norm", np.linalg.norm(beta))
    beta = np.nan_to_num(beta)

    sorted_index = np.argsort(np.abs(beta))

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 28
    title_fontsize = 28

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(8, 6.5))

    for ax in axn.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(3, 5, 1)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("$\mathrm{Image}$")

    plt.subplot(3, 5, 6)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("$\mathrm{Rec}$")

    plt.subplot(3, 5, 11)
    x_new_estimate = np.matmul(
        X[:, sorted_index[-used_num_images:]], beta[sorted_index[-used_num_images:]]
    )
    plt.imshow(np.reshape(x_new_estimate, reshape), cmap="gray")
    plt.title("$\mathrm{Estimate}$")

    # most contribution
    fig_place = [2, 3, 7, 8, 12, 13]
    for i in range(6):
        plt.subplot(3, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(np.abs(beta[sorted_index[-1 - i]])), color="green",
        )

    # least contribution
    fig_place = [4, 5, 9, 10, 14, 15]
    for i in range(6):
        plt.subplot(3, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title("{:.5f}".format(np.abs(beta[sorted_index[i]])), color="red")

    fig.tight_layout(pad=0.00, w_pad=0.2, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def visualize_dense_most_similar_training_examples_based_on_code_similarity(
    Z, z_new, X, Y, x_new, xhat_new, params, save_path, reshape=(28, 28)
):

    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()
    code_similarity = (
        torch.matmul(Z_normalized.T, z_new_normalized).detach().cpu().numpy()
    )

    sorted_index = np.argsort(code_similarity)

    ################################
    axes_fontsize = 24
    legend_fontsize = 24
    tick_fontsize = 24
    title_fontsize = 28

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(8, 3))

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for c in range(len(params["class_list"])):
        plt.subplot(1, 5, c + 1)
        c_indices = Y == params["class_list"][c]
        code_similarity_c = code_similarity[c_indices]
        plt.hist(code_similarity_c, bins=20)
        plt.title("{}-digit".format(params["class_list"][c]))
        plt.xlabel("$\mathrm{Similarity}$")
        plt.xlim([0, 1])
        if c == 0:
            plt.ylabel("$\mathrm{Frequency}$")

    fig.tight_layout(pad=0.00, w_pad=1, h_pad=0.1)

    plt.savefig(
        "{}{}".format(save_path, "hist.pdf"), bbox_inches="tight", pad_inches=0.05
    )
    plt.close()

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 28
    title_fontsize = 28

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(8, 4.5))

    ctr = 0
    for ax in axn.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(2, 5, 1)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("$\mathrm{Image}$")

    plt.subplot(2, 5, 6)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("$\mathrm{Rec}$")

    # most similar
    fig_place = [2, 3, 7, 8]
    for i in range(4):
        if i >= len(sorted_index):
            continue
        plt.subplot(2, 5, fig_place[i])
        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(code_similarity[sorted_index[-1 - i]]),
            color="green",
        )

    # least similar
    fig_place = [4, 5, 9, 10]
    for i in range(4):
        if i >= len(sorted_index):
            continue
        plt.subplot(2, 5, fig_place[i])
        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title("{:.5f}".format(code_similarity[sorted_index[i]]), color="red")

    fig.tight_layout(pad=0.00, w_pad=0.2, h_pad=0.01)

    plt.savefig("{}{}".format(save_path, ".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()


def visualize_conv_most_similar_training_examples_based_on_code_similarity(
    Z,
    z_new,
    y_new,
    X,
    Y,
    x_new,
    xhat_new,
    params,
    save_path,
    lim_x=True,
):

    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()
    code_similarity = (
        torch.matmul(Z_normalized.T, z_new_normalized).detach().cpu().numpy()
    )

    sorted_index = np.argsort(code_similarity)

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(3, 5, sharex="row", sharey="row")

    ctr = 0
    for ax in axn.flat:
        if ctr < 5:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr >= 5:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    for c in range(len(params["class_list"])):
        plt.subplot(4, 5, c + 1)
        c_indices = Y == params["class_list"][c]
        code_similarity_c = code_similarity[c_indices]
        plt.hist(code_similarity_c, bins=20)
        plt.title("Class {} code hist".format(params["class_list"][c]))
        plt.xlabel("$\mathrm{Similarity}$")
        plt.xlim([0, 1])
        if c == 0:
            plt.ylabel("$\mathrm{Frequency}$")

    plt.subplot(4, 5, 6)
    plt.imshow(np.transpose(x_new, (1, 2, 0)), cmap="gray")
    plt.title("Image {:.0f}".format(y_new))

    plt.subplot(4, 5, 7)
    plt.imshow(np.transpose(xhat_new, (1, 2, 0)), cmap="gray")
    plt.title("Rec")

    # most similar
    for i in range(8):
        if i >= len(sorted_index):
            continue
        plt.subplot(4, 5, i + 8)
        plt.imshow(np.transpose(X[sorted_index[-1 - i]], (1, 2, 0)), cmap="gray")
        plt.title(
            "{:.5f}".format(code_similarity[sorted_index[-1 - i]]),
            color="green",
        )

    # least similar
    for i in range(5):
        if i >= len(sorted_index):
            continue
        plt.subplot(4, 5, i + 16)
        plt.imshow(np.transpose(X[sorted_index[i]], (1, 2, 0)), cmap="gray")
        plt.title("{:.5f}".format(code_similarity[sorted_index[i]]), color="red")

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_most_similar_training_examples_based_on_code_similarity_and_feature_maps(
    Z_raw,
    z_new_raw,
    Z,
    z_new,
    y_new,
    X,
    Y,
    x_new,
    xhat_new,
    params,
    save_path,
    lim_x=True,
):

    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()
    code_similarity = (
        torch.matmul(Z_normalized.T, z_new_normalized).detach().cpu().numpy()
    )

    sorted_index = np.argsort(code_similarity)

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(4, 5, sharex="row", sharey="row")

    ctr = 0
    for ax in axn.flat:
        if ctr < 5:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr >= 5:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    for c in range(len(params["class_list"])):
        plt.subplot(4, 5, c + 1)
        c_indices = Y == params["class_list"][c]
        code_similarity_c = code_similarity[c_indices]
        plt.hist(code_similarity_c, bins=20)
        plt.title("Class {} code hist".format(params["class_list"][c]))
        plt.xlabel("$\mathrm{Similarity}$")
        plt.xlim([0, 1])
        if c == 0:
            plt.ylabel("$\mathrm{Frequency}$")

    plt.subplot(4, 5, 6)
    plt.imshow(np.transpose(x_new, (1, 2, 0)), cmap="gray")
    plt.title("Image {:.0f}".format(y_new))

    plt.subplot(4, 5, 7)
    plt.imshow(np.transpose(xhat_new, (1, 2, 0)), cmap="gray")
    plt.title("Rec")

    # most similar
    for i in range(8):
        if i >= len(sorted_index):
            continue
        plt.subplot(4, 5, i + 8)
        plt.imshow(np.transpose(X[sorted_index[-1 - i]], (1, 2, 0)), cmap="gray")
        plt.title(
            "{:.5f}".format(code_similarity[sorted_index[-1 - i]]),
            color="green",
        )

    # least similar
    for i in range(5):
        if i >= len(sorted_index):
            continue
        plt.subplot(4, 5, i + 16)
        plt.imshow(np.transpose(X[sorted_index[i]], (1, 2, 0)), cmap="gray")
        plt.title("{:.5f}".format(code_similarity[sorted_index[i]]), color="red")

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    save_path_image = os.path.join(save_path.split(".png")[0] + "_image.png")
    visualize_conv_feature_maps(z_new_raw, save_path_image, cmap="afmhot")

    for i in range(8):
        if i >= len(sorted_index):
            continue
        save_path_i = os.path.join(
            save_path.split(".png")[0] + "_similar_{}.png".format(i)
        )
        visualize_conv_feature_maps(
            Z_raw[sorted_index[-1 - i]], save_path_i, cmap="afmhot"
        )

    for i in range(5):
        if i >= len(sorted_index):
            continue
        save_path_i = os.path.join(save_path.split(".png")[0] + "_dis_{}.png".format(i))
        visualize_conv_feature_maps(Z_raw[sorted_index[i]], save_path_i, cmap="afmhot")


def visualize_most_similar_training_XG_col_based_on_code_similarity(
    Z, z_new, X, x_new, xhat_new, G, save_path, reshape=(28, 28)
):

    Z_normalized = torch.nn.functional.normalize(Z, dim=0).clone()
    z_new_normalized = torch.nn.functional.normalize(z_new, dim=0).clone()
    code_similarity = (
        torch.matmul(Z_normalized.T, z_new_normalized).detach().cpu().numpy()
    )

    sorted_index = np.argsort(code_similarity)

    XG = torch.matmul(X, G).detach()

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(4, 4)

    ctr = 0
    for ax in axn.flat:
        if ctr == 0:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    plt.subplot(4, 4, 1)
    plt.hist(code_similarity, bins=100)
    plt.xlabel("$\mathrm{Similarity}$")
    plt.ylabel("$\mathrm{Frequency}$")
    plt.title("$\mathrm{Code}$")

    plt.subplot(4, 4, 2)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("$\mathrm{Test\ image}$")

    plt.subplot(4, 4, 3)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("$\mathrm{Test\ rec}$")

    # most similar
    for i in range(6):
        plt.subplot(4, 4, i + 4)
        plt.imshow(np.reshape(XG[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(code_similarity[sorted_index[-1 - i]]),
            color="green",
        )
        if i == 1:
            plt.ylabel("$\mathrm{Most}$")

    # least similar
    for i in range(4):
        plt.subplot(4, 4, i + 9)
        plt.imshow(np.reshape(XG[:, sorted_index[i]], reshape), cmap="gray")
        plt.title("{:.5f}".format(code_similarity[sorted_index[i]]), color="red")
        if i == 0:
            plt.ylabel("$\mathrm{Least}$")

    # in between
    XG_in_between = X.clone()
    code_similarity_in_between = code_similarity.copy()

    indices = np.where(code_similarity_in_between < 0.51)
    code_similarity_in_between = code_similarity_in_between[indices]
    XG_in_between = torch.squeeze(XG_in_between[:, indices])

    indices = np.where(code_similarity_in_between > 0.49)
    code_similarity_in_between = code_similarity_in_between[indices]
    XG_in_between = torch.squeeze(XG_in_between[:, indices])

    for i in range(4):
        plt.subplot(4, 4, i + 13)
        plt.imshow(np.reshape(XG_in_between[:, i], reshape), cmap="gray")
        plt.title("{:.5f}".format(code_similarity_in_between[i]), color="gray")
        if i == 0:
            plt.ylabel("$\mathrm{0.5 sim}$")

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_code_histogram(Z, params, save_path):

    Z = Z.clone().detach().cpu().numpy()

    p = Z.shape[0]
    a = np.int(np.ceil(np.sqrt(p)))

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(a, a, sharex=True, sharey=True)

    ctr = 0
    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # if ctr >= 1:
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.spines["left"].set_visible(False)
        #     ax.spines["bottom"].set_visible(False)
        ctr += 1

    # most similar
    for conv in range(p):
        plt.subplot(a, a, conv + 1)
        plt.hist(Z[conv])

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_code_matrix(Z, save_path, sorted_atom_index=[]):
    # p x n
    # row is for each atom j, and col for each data k
    code_matrix = torch.abs(Z.clone().detach())

    # if sorted_atom_index is provided, plot the sorted. if not, return sorted.
    if len(sorted_atom_index) == 0:
        sorted_atom_index = np.argsort(
            torch.mean(code_matrix[:, -np.int(code_matrix.shape[0] / 2) :], dim=-1)
        )
    else:
        code_matrix = code_matrix[sorted_atom_index]

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "binary"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        code_matrix,
        ax=ax,
        # linewidth=2,
        vmin=0,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Dict\ [j]}$")
    ax.set_title("$\mathrm{(Z)\ presence\ of\ each\ atom\ j\ in\ data\ k}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    return sorted_atom_index


def visualize_ZTZ_matrix(Z, save_path):
    # n x n
    # code similarity matrix
    Z_normalized = torch.nn.functional.normalize(Z, dim=0)
    ZTZ_matrix = torch.matmul(Z_normalized.T, Z_normalized).clone().detach()

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "binary"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        ZTZ_matrix,
        ax=ax,
        # linewidth=2,
        vmin=0,
        vmax=1,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Data\ [k]}$")
    ax.set_title("$\mathrm{ZTZ\ matrix}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_XTX_matrix(X, save_path):
    # n x n
    # code similarity matrix
    X_normalized = torch.nn.functional.normalize(X, dim=0)
    XTX_matrix = torch.matmul(X_normalized.T, X_normalized).clone().detach()

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "binary"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        XTX_matrix,
        ax=ax,
        # linewidth=2,
        vmin=0,
        vmax=1,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Data\ [k]}$")
    ax.set_title("$\mathrm{XTX\ matrix}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_Ginverse_matrix(Ginv, save_path):
    # p x n
    # row is for each atom j, and col for each data k
    Ginv_matrix = Ginv.clone().detach()

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "seismic"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        Ginv_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Data\ [k]}$")
    ax.set_title("$\mathrm{G\ inverse\ matrix}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_XGinverse_matrix(Ginv, X, save_path):
    # p x n
    # row is for each atom j, and col for each data k
    XGinv_matrix = torch.matmul(X, Ginv).detach()

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "seismic"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        XGinv_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Image\ pixel\ [m]}$")
    ax.set_title(
        "$\mathrm{XGinv\ contribution\ of\ each\ data\ k\ into\ estimate\ new\ image\ based\ on\ code\ similarity}$"
    )

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_Ginversew_matrix(Ginv, Z, save_path):
    # p x n
    # row is for each atom j, and col for each data k
    Ginvw_matrix = torch.matmul(Ginv, Z.T).T.detach()

    ################################
    axes_fontsize = 15
    legend_fontsize = 15
    tick_fontsize = 10
    title_fontsize = 15

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([1.0, 0.3, 0.03, 0.4])

    cmap = "seismic"
    cbar = True
    cbar_kws = {
        "shrink": 1,
    }
    sns.heatmap(
        Ginvw_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{Data\ [k]}$")
    ax.set_ylabel("$\mathrm{Dict\ [j]}$")
    ax.set_title(
        "$\mathrm{(Ginvw)\ contribution\ of\ each\ data\ k\ into\ estimate\ of\ atom\ j}$"
    )

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_contribution_of_images_for_dict_j_using_Ginversew(
    D, j, Ginv, Z, X, Y, params, save_path, reshape=(28, 28), D_reshape=(28, 28)
):

    # p x n
    # row is for each atom j, and col for each data k
    eps = 1e-14
    Ginvw_matrix = torch.matmul(Ginv, Z.T).T.detach()
    img_contribution_for_dict_j = Ginvw_matrix[j, :].detach().cpu().numpy()
    img_contribution_for_dict_j /= np.linalg.norm(img_contribution_for_dict_j) + eps
    img_contribution_for_dict_j = np.nan_to_num(img_contribution_for_dict_j)

    sorted_index = np.argsort(np.abs(img_contribution_for_dict_j))

    Dj = D[:, j].clone()

    ################################
    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(4, 5, sharex="row", sharey="row")

    ctr = 0
    for ax in axn.flat:
        if ctr < 5:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr >= 5:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    for c in range(len(params["class_list"])):
        plt.subplot(4, 5, c + 1)
        c_indices = Y == params["class_list"][c]
        img_contribution_for_dict_j_c = img_contribution_for_dict_j[c_indices]
        plt.hist(
            img_contribution_for_dict_j_c,
            bins=100,
            range=(
                np.min(img_contribution_for_dict_j),
                np.max(img_contribution_for_dict_j),
            ),
        )
        plt.title("Class {} hist".format(params["class_list"][c]))
        plt.xlabel("$\mathrm{Contribution}$")
        if c == 0:
            plt.ylabel("$\mathrm{Frequency}$")

    plt.subplot(4, 5, 6)
    plt.imshow(np.reshape(Dj, D_reshape), cmap="gray")
    plt.title("Atom {}".format(j))

    plt.subplot(4, 5, 7)
    Dj_est = np.matmul(X, img_contribution_for_dict_j)
    plt.imshow(np.reshape(Dj_est, reshape), cmap="gray")
    plt.title("Atom estimate")

    # most contribution
    for i in range(8):
        plt.subplot(4, 5, i + 8)

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(img_contribution_for_dict_j[sorted_index[-1 - i]]),
            color="green",
        )

    # least contribution
    for i in range(5):
        plt.subplot(4, 5, i + 16)

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(img_contribution_for_dict_j[sorted_index[i]]), color="red"
        )

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_contribution_of_images_for_dict_j_using_Ginversew_nohist(
    D, j, Ginv, Z, X, Y, params, save_path, reshape=(28, 28), D_reshape=(28, 28)
):

    # p x n
    # row is for each atom j, and col for each data k
    eps = 1e-14
    Ginvw_matrix = torch.matmul(Ginv, Z.T).T.detach()
    img_contribution_for_dict_j = Ginvw_matrix[j, :].detach().cpu().numpy()
    img_contribution_for_dict_j /= np.linalg.norm(img_contribution_for_dict_j) + eps
    img_contribution_for_dict_j = np.nan_to_num(img_contribution_for_dict_j)

    sorted_index = np.argsort(np.abs(img_contribution_for_dict_j))

    Dj = D[:, j].clone()

    ################################
    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 28
    title_fontsize = 28

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(2, 5, sharex="row", sharey="row", figsize=(8, 4.5))

    for ax in axn.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(2, 5, 1)
    plt.imshow(np.reshape(Dj, D_reshape), cmap="gray")
    plt.title("$\mathrm{Learned}$")

    plt.subplot(2, 5, 6)
    Dj_est = np.matmul(X, img_contribution_for_dict_j)
    plt.imshow(np.reshape(Dj_est, reshape), cmap="gray")
    plt.title("$\mathrm{Estimate}$")

    # most contribution
    fig_place = [2, 3, 7, 8]
    for i in range(4):
        plt.subplot(2, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(np.abs(img_contribution_for_dict_j[sorted_index[-1 - i]])),
            color="green",
        )

    # least contribution
    fig_place = [4, 5, 9, 10]
    for i in range(4):
        plt.subplot(2, 5, fig_place[i])

        if i >= len(sorted_index):
            continue

        plt.imshow(np.reshape(X[:, sorted_index[i]], reshape), cmap="gray")
        plt.title(
            "{:.5f}".format(np.abs(img_contribution_for_dict_j[sorted_index[i]])),
            color="red",
        )

    fig.tight_layout(pad=0.00, w_pad=0.2, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def visualize_atoms_for_fixedpoint(x_list, loops, save_path, reshape=(28, 28)):

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(3, 3, sharex="row", sharey="row")

    ctr = 0
    for ax in axn.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ctr += 1

    for itr in range(len(x_list)):
        plt.subplot(3, 3, itr + 1)
        plt.imshow(np.reshape(x_list[itr], reshape), cmap="gray")
        if itr == 0:
            plt.title("atom as input")
        else:
            plt.title("itr {}".format(loops[itr - 1]))

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_contraction(
    z_new, x_new, xhat_new, D, params, save_path, reshape=(28, 28)
):

    D_normalized = torch.nn.functional.normalize(D, dim=0).clone()
    xhat_new_normalized = torch.nn.functional.normalize(xhat_new, dim=0).clone()
    # dictionary_similarity_to_test_image = (
    #     torch.matmul(D_normalized.T, xhat_new_normalized).detach().cpu().numpy()
    # )

    code = torch.nn.functional.normalize(z_new, dim=0).clone().cpu().numpy()
    sorted_index = np.argsort(code)

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(4, 4)

    ctr = 0
    for ax in axn.flat:
        if ctr == 0:
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ctr > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ctr += 1

    plt.subplot(4, 4, 1)
    plt.hist(code, bins=20)
    plt.title("Histogram")
    plt.xlabel("$\mathrm{Code}$")
    plt.ylabel("$\mathrm{Frequency}$")

    plt.subplot(4, 4, 2)
    plt.imshow(np.reshape(x_new, reshape), cmap="gray")
    plt.title("Test image")

    plt.subplot(4, 4, 3)
    plt.imshow(np.reshape(xhat_new, reshape), cmap="gray")
    plt.title("Test rec")

    # most similar dictionary to the image
    for i in range(13):
        plt.subplot(4, 4, i + 4)
        plt.imshow(np.reshape(D[:, sorted_index[-1 - i]], reshape), cmap="gray")
        plt.title(
            "code {:.2f}".format(code[sorted_index[-1 - i]]),
            color="green",
        )
        if i == 0:
            plt.ylabel("dictionary")

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_saliency_map_of_code(
    x, xhat, saliency, z, save_path, s_name="s", cmap="afmhot",
):

    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    ncols = len(x)
    nrows = len(saliency.keys()) + 2 + 1

    fig, axn = plt.subplots(nrows, ncols, sharex="row", sharey="row")

    for ax in axn.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    for c in range(ncols):
        plt.subplot(nrows, ncols, c + 1)
        plt.imshow(x[c], cmap="gray", vmin=0, vmax=1)
        if c == 0:
            plt.ylabel("image")

    for c in range(ncols):
        plt.subplot(nrows, ncols, c + ncols + 1)
        plt.imshow(xhat[c], cmap="gray", vmin=0, vmax=1)
        if c == 0:
            plt.ylabel("rec")

        for index in range(nrows - 3):
            saliency["{}".format(index + 1)][c]

    for c in range(ncols):
        avg_saliency = np.zeros(xhat[0].shape)
        s_vmax = 0
        for index in range(nrows - 3):
            s_vmax = np.maximum(np.max(saliency["{}".format(index + 1)][c]), s_vmax)

        for index in range(nrows - 3):
            plt.subplot(nrows, ncols, c + ncols * (2 + index) + 1)
            plt.imshow(saliency["{}".format(index + 1)][c], cmap=cmap)
            if c == 0:
                plt.ylabel("{} {}".format(s_name, index + 1))

    fig.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()
