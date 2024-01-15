import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def calc_percentiles(dic_x, dic_y, fe_x, fe_y):
    data_dic_x_percentile = []
    data_dic_y_percentile = []
    data_fe_x_percentile = []
    data_fe_y_percentile = []
    for p in tqdm(range(101)):
        data_dic_x_percentile.append([p, np.percentile(np.abs(dic_x), p)])
        data_dic_y_percentile.append([p, np.percentile(np.abs(dic_y), p)])
        data_fe_x_percentile.append([p, np.percentile(np.abs(fe_x), p)])
        data_fe_y_percentile.append([p, np.percentile(np.abs(fe_y), p)])
        if p == 99:
            for pp in tqdm(range(1, 10)):
                data_dic_x_percentile.append(
                    [p + 0.1 * pp, np.percentile(np.abs(dic_x), p + 0.1 * pp)]
                )
                data_dic_y_percentile.append(
                    [p + 0.1 * pp, np.percentile(np.abs(dic_y), p + 0.1 * pp)]
                )
                data_fe_x_percentile.append(
                    [p + 0.1 * pp, np.percentile(np.abs(fe_x), p + 0.1 * pp)]
                )
                data_fe_y_percentile.append(
                    [p + 0.1 * pp, np.percentile(np.abs(fe_y), p + 0.1 * pp)]
                )
    data_dic_x_percentile = np.array(data_dic_x_percentile)
    data_dic_y_percentile = np.array(data_dic_y_percentile)
    data_fe_x_percentile = np.array(data_fe_x_percentile)
    data_fe_y_percentile = np.array(data_fe_y_percentile)

    return (
        data_dic_x_percentile,
        data_dic_y_percentile,
        data_fe_x_percentile,
        data_fe_y_percentile,
    )


def plot_percentiles(
    data_dic_x_percentile,
    data_dic_y_percentile,
    data_fe_x_percentile,
    data_fe_y_percentile,
    titles=["DIC Percentiles", "FE Percentiles"],
    save=False,
):
    fig, ax = plt.subplots(figsize=(4, 7), nrows=2, ncols=1, constrained_layout=True)
    fig.suptitle(titles[0])
    ax[0].plot(data_dic_x_percentile[:, 0], data_dic_x_percentile[:, 1], ".:")
    # ax[0].set_yscale('log')
    ax[0].set_ylim([0.001, 0.5])
    ax[0].grid()
    ax[0].axhline(
        y=data_dic_x_percentile[-2, 1],
        color="r",
        linestyle="--",
        label="99.9-th percentile",
    )
    ax[0].text(
        x=50, y=data_dic_x_percentile[-2, 1], s=f"{data_dic_x_percentile[-2, 1]:.3f}"
    )
    ax[1].plot(data_dic_y_percentile[:, 0], data_dic_y_percentile[:, 1], ".:")
    # ax[1].set_yscale('log')
    ax[1].set_ylim([0.001, 0.5])
    ax[1].grid()
    ax[1].axhline(
        y=data_dic_y_percentile[-2, 1],
        color="r",
        linestyle="--",
        label="99.9-th percentile",
    )
    ax[1].text(
        x=50, y=data_dic_y_percentile[-2, 1], s=f"{data_dic_y_percentile[-2, 1]:.3f}"
    )
    ax[0].set_title("x-percentiles")
    ax[0].legend()
    ax[1].set_title("y-percentiles")
    ax[1].legend()
    if save:
        plt.savefig(
            os.path.join(
                "..",
                "..",
                "..",
                "..",
                "Masterarbeit",
                "Ergebnisse",
                f"{titles[0]}_percentiles.pdf",
            ),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            os.path.join(
                "..",
                "..",
                "..",
                "..",
                "Masterarbeit",
                "Ergebnisse",
                f"{titles[0]}_percentiles.png",
            ),
            bbox_inches="tight",
            transparent=False,
        )

    fig, ax = plt.subplots(figsize=(4, 7), nrows=2, ncols=1, constrained_layout=True)
    fig.suptitle(titles[1])
    ax[0].plot(data_fe_x_percentile[:, 0], data_fe_x_percentile[:, 1], ".:")
    # ax[0].set_yscale('log')
    ax[0].set_ylim([0.001, 0.5])
    ax[0].grid()
    ax[0].axhline(
        y=data_fe_x_percentile[-2, 1],
        color="r",
        linestyle="--",
        label="99.9-th percentile",
    )
    ax[0].text(
        x=50, y=data_fe_x_percentile[-2, 1], s=f"{data_fe_x_percentile[-2, 1]:.3f}"
    )
    ax[1].plot(data_fe_y_percentile[:, 0], data_fe_y_percentile[:, 1], ".:")
    # ax[1].set_yscale('log')
    ax[1].set_ylim([0.001, 0.5])
    ax[1].grid()
    ax[1].axhline(
        y=data_fe_y_percentile[-2, 1],
        color="r",
        linestyle="--",
        label="99.9-th percentile",
    )
    ax[1].text(
        x=50, y=data_fe_y_percentile[-2, 1], s=f"{data_fe_y_percentile[-2, 1]:.3f}"
    )
    ax[0].set_title("x-percentiles")
    ax[0].legend()
    ax[1].set_title("y-percentiles")
    ax[1].legend()
    if save:
        plt.savefig(
            os.path.join(
                "..",
                "..",
                "..",
                "..",
                "Masterarbeit",
                "Ergebnisse",
                f"{titles[1]}_percentiles.pdf",
            ),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            os.path.join(
                "..",
                "..",
                "..",
                "..",
                "Masterarbeit",
                "Ergebnisse",
                f"{titles[1]}_percentiles.png",
            ),
            bbox_inches="tight",
            transparent=False,
        )
