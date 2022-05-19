"""
Copyright (c) 2021 Bahareh Tolooshams

representer utils

:author: Bahareh Tolooshams
"""

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def get_X(dataset):
    # m x n
    return torch.squeeze(dataset.x, dim=-1).T


def get_Z(dataset):
    # p x n
    return torch.squeeze(dataset.z, dim=-1).T


def get_ZT(dataset):
    # n x p
    return torch.squeeze(dataset.z, dim=-1)


def get_ZTZ(dataset):
    # n x n
    return torch.matmul(get_ZT(dataset), get_Z(dataset))


def get_ZTZplusrhoI(dataset, rho=1e-7):
    # n x n
    return get_ZTZ(dataset) + rho * torch.eye(dataset.n)


def get_G(dataset, rho=1e-7):
    # n x n
    # (ZTZ + rho I)_inverse
    return torch.inverse(get_ZTZplusrhoI(dataset, rho))


def get_wj(dataset, j):
    # n x 1
    # row j of Z
    return torch.unsqueeze(get_Z(dataset)[j], dim=-1)


def get_Gwj(dataset, j, rho=1e-7):
    # n
    G = get_G(dataset, rho)
    wj = get_wj(dataset, j)
    return torch.matmul(G, wj)


def get_beta(dataset, z_test, rho=1e-7):
    # n x 1
    G = get_G(dataset, rho)
    ZT = get_ZT(dataset)
    GZT = torch.matmul(G, ZT)
    beta = torch.matmul(GZT, z_test)

    ZTz_test = torch.matmul(ZT, z_test)

    supp = z_test.clone()
    supp[supp != 0] = 0.3

    plt.figure()
    plt.plot(beta, ".")
    plt.plot(ZTz_test / 100, ".")
    plt.savefig("../results/beta.png")
    plt.close()
    return torch.matmul(GZT, z_test)


def visualize_G_matrix(dataset):
    # p x n
    # row is for each atom j, and col for each data k
    G_matrix = get_G(dataset, rho=1e-7)

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
            "text.latex.preamble": [r"\usepackage{bm}"],
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
        G_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{data\ [k]}$")
    ax.set_ylabel("$\mathrm{data\ [k]}$")
    ax.set_title("$\mathrm{(G)\ matrix}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(
        "../results/G_matrix.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def visualize_code_matrix(dataset):
    # p x n
    # row is for each atom j, and col for each data k
    Z = get_Z(dataset)
    code_matrix = Z.clone()

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
            "text.latex.preamble": [r"\usepackage{bm}"],
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
        code_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{data\ [k]}$")
    ax.set_ylabel("$\mathrm{dict\ [j]}$")
    ax.set_title("$\mathrm{(Z)\ presence\ of\ each\ atom\ j\ in\ data\ k}$")

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(
        "../results/code_matrix.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def visualize_Gw_matrix(dataset):
    # p x n
    # row is for each atom j, and col for each data k
    Gw_matrix = torch.matmul(get_G(dataset), get_ZT(dataset)).T

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
            "text.latex.preamble": [r"\usepackage{bm}"],
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
        Gw_matrix,
        ax=ax,
        # linewidth=2,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        cmap=cmap,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize - 2, right=False, direction="in")

    ax.set_xlabel("$\mathrm{data\ [k]}$")
    ax.set_ylabel("$\mathrm{dict\ [j]}$")
    ax.set_title(
        "$\mathrm{(Gw)\ contribution\ of\ each\ data\ k\ into\ estimate\ of\ atom\ j}$"
    )

    fig.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(
        "../results/Gw_matrix.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()
