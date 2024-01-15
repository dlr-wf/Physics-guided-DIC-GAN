import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
import seaborn as sns
import torch


def plot_statistics(
    stats,
    labels,
    xlabel="Iterations",
    ylabel="Loss",
    save_folder=None,
    filename="losses",
    figsize=(10, 5),
    ylims=None,
    yscale="linear",
):
    """
    Plots data points in a line plot.

    :param stats: Data to plot. Needs to be a nested list type
           e.g.: [[stat1_1, stat1_2], [stat2_1]]
           Data in one group (e.g. stat1_1 and stat1_2) is plotted in one subplot
           Different groups (e.g. stat1_x, stat2_x) are plotted in different subplots
    :param labels: String legend labels for the different plot. Needs to be the same format
                   as stats.
    :param xlabel: x-axis label shared across all subplots
    :param ylabel: y-axis label shared across all subplots
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension, defaults to 'losses'
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :param ylims: Bottom and top limits passed to axes.set_ylim for each group
           e.g.: [[0,1],[-1,10],[0,None]]
    :param yscale: scale argument as in pyplot.yscale: "linear", "log", "symlog", "logit", ...
    :return: Figure and Axes object of the plot: fig, ax
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=1,
        nrows=len(stats),
        constrained_layout=False,
        sharex=True,
    )
    if len(stats) == 1:
        for index_group, loss_group in enumerate(stats):
            for index_stat, stat in enumerate(loss_group):
                ax.plot(stat, label=labels[index_group][index_stat])
                ax.legend()
                ax.set_yscale(yscale)
                ax.set_ylabel(ylabel)
            ylim_bot = None if ylims is None else ylims[index_group][0]
            ylim_top = None if ylims is None else ylims[index_group][1]
            ax.set_ylim(bottom=ylim_bot, top=ylim_top)
    else:
        for index_group, loss_group in enumerate(stats):
            for index_stat, stat in enumerate(loss_group):
                ax[index_group].plot(stat, label=labels[index_group][index_stat])
                ax[index_group].legend()
                ax[index_group].set_ylabel(ylabel)
                ax[index_group].set_yscale(yscale)
            ylim_bot = None if ylims is None else ylims[index_group][0]
            ylim_top = None if ylims is None else ylims[index_group][1]
            ax[index_group].set_ylim(bottom=ylim_bot, top=ylim_top)
    plt.xlabel(xlabel)
    plt.subplots_adjust(hspace=0.0)

    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
        )
    return fig, ax


def plot_ssim():
    pass


def plot_ms_ssim():
    pass


def plot_geom_scores(
    csv_file: str,
    gan_names={"GAN": "$\\alpha$GAN", "real": "Real Data"},
    save_folder: str = "plots",
    save_filename: str = "gs",
    figsize: Tuple = (5, 3),
):
    """
    Plots geometry scores mrlts from csv file created from metrics.geometry_score.save_geom_scores function.
    For each epoch present in the csv file a plot is created and saved.

    :param csv_file: Path to the csv file contatining the geometry scores
    :param gan_names: Names (values) used in the plot for specfied GAN types (keys). Has to include 'real' as key.
    :param save_folder: Folder in which the plots are saved
    :param save_filename: Filename of the plots without extension. Filename will be extended by the number of epochs.
    :param figsize: Figsize argument passed to the matplotlib subplots function
    """
    assert (
        gan_names.get("real") is not None
    ), "gan_names has to include 'real' as a key!"

    # Load csv file
    df = pd.read_csv(csv_file, index_col=0)
    epoch_list = df["n_epochs"].unique()

    for i, number_of_epochs in enumerate(epoch_list):
        # Filter number of epochs
        df = df[df["n_epochs"] == number_of_epochs]

        # Rename models
        for gan_type, gan_name in gan_names.items():
            df.loc[df.gan_type == gan_type, "model_name"] = gan_name
        df.loc[df.type == "real", "model_name"] = gan_names["real"]

        # df.loc[df.gan_type == 'uGAN', 'model_name'] = '$u$GAN'
        # df.loc[df.gan_type == 'eGAN', 'model_name'] = '$\\varepsilon$GAN'
        # df.loc[df.type == 'real',     'model_name'] = 'real'

        # Plot
        pal = sns.color_palette("husl", 8)
        pal_short = [pal[i] for i in [5, 1, 3]]
        palette = {"real": pal[1], "$u$GAN": pal[5], "$\\varepsilon$GAN": pal[3]}
        _fig, ax = plt.subplots(figsize=figsize)
        plot = sns.lineplot(
            x="i_in_imax",
            y="mrlt",
            hue="model_name",
            hue_order=gan_names.values(),
            palette=pal_short,
            errorbar=("sd", 1),
            data=df,
        )
        plot.set(xlabel="$i$", ylabel="$\\mathrm{MRLT}(i)$")
        plt.xlim([0, 40])
        plt.ylim([0, 0.2])
        ax.legend(title=None, loc="upper right")
        plt.minorticks_on()
        plt.grid(visible=True)

        plt.savefig(
            os.path.join(save_folder, f"{save_filename}_e{number_of_epochs}.png"),
            dpi=512,
            bbox_inches="tight",
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.savefig(
            os.path.join(save_folder, f"{save_filename}_e{number_of_epochs}.pdf"),
            dpi=512,
            bbox_inches="tight",
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.close()


def plot_swds(
    csv_file: str,
    save_folder: str = "plots",
    save_filename: str = "swd",
    legend_positions: List[str] = None,
    figsize: Tuple = (5, 3),
):
    """
    Plots sliced Wasserstein distances from csv file created from metrics.swd.save_swds function.
    For each epoch present in the csv file a plot is created and saved.

    :param csv_file: Path to the csv file contatining the sliced Wasserstein distances
    :param save_folder: Folder in which the plots are saved
    :param save_filename: Filename of the plots without extension. Filename will be extended by the number of epochs.
    :param legend_positions: List of strings used as loc argument in pyplot.legend function.
                             With this the location of the legend can be specified for each epoch present in the csv_file.
    :param figsize: Figsize argument passed to the matplotlib subplots function
    """
    # Load csv file
    df = pd.read_csv(csv_file, index_col=0)
    epoch_list = df["n_epochs"].unique()

    if legend_positions is None:
        legend_positions = [None for i in range(len(epoch_list))]
    assert len(legend_positions) == len(
        epoch_list
    ), f"Length of legend_positions ({len(legend_positions)}) has to equal number of unique epochs ({len(epoch_list)}) in csv file!"

    for i, number_of_epochs in enumerate(epoch_list):
        # Filter number of epochs
        df = df[df["n_epochs"] == number_of_epochs]

        # Filter out avg swd
        df = df[df["pyramid_size"] != "avg"]

        # Rename models
        df.loc[df.gan_type == "uGAN", "model_name"] = f"$u$GAN ({number_of_epochs})"
        df.loc[
            df.gan_type == "eGAN", "model_name"
        ] = f"$\\varepsilon$GAN ({number_of_epochs})"

        # Plot
        pal = sns.color_palette("husl", 8)
        palette = {
            f"$u$GAN ({number_of_epochs})": pal[5],
            f"$\\varepsilon$GAN ({number_of_epochs})": pal[3],
        }
        _fig, ax = plt.subplots(figsize=figsize)
        plot = sns.boxplot(
            x="pyramid_size",
            y="SWD",
            hue="model_name",
            hue_order=list(palette.keys()),
            whis=[0, 100],
            width=0.5,
            palette=palette,
            data=df,
            order=["256", "128", "64", "32", "16"],
        )
        plot.set(
            xlabel="Laplacian Pyramid Level",
            ylabel="Sliced Wasserstein Distance $\\times10^3$",
        )
        ax.set_xticklabels(
            [
                "$256\\times256$",
                "$128\\times128$",
                "$64\\times64$",
                "$32\\times32$",
                "$16\\times16$",
            ]
        )
        plt.ylim([0, 250])
        plt.legend(title=None, loc=legend_positions[i])
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.grid(True, which="major")
        ax.set_axisbelow(True)

        plt.savefig(
            os.path.join(save_folder, f"{save_filename}_e{number_of_epochs}.png"),
            dpi=512,
            bbox_inches="tight",
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.savefig(
            os.path.join(save_folder, f"{save_filename}_e{number_of_epochs}.pdf"),
            dpi=512,
            bbox_inches="tight",
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.close()


def plot_swd_bar(
    swd_values, names, save_folder=None, filename="swd_bar", figsize=(3.75, 2.25)
):
    """
    Plots sliced Wasserstein distances in bar plot.

    :param swd_values: Sliced Wasserstein distances as returned from
                       metrics.swd.calc_swd or .load_swd functions
    :param names: x-axis labels. Preferably as returned from metrics.swd.calc_swd or .load_swd functions
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension, defaults to 'swd_bar'
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :return: Figure and Axes object of the plot: fig, ax
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    swd_values_tensor = torch.stack([torch.tensor(y) for y in swd_values])
    swd_means = swd_values_tensor.mean(dim=0).tolist()
    swd_stds = swd_values_tensor.std(dim=0).tolist()

    ylim = 250
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    x_labels = [label.split("_")[-1] for label in names]
    ax.bar(
        x_labels,
        swd_means,
        yerr=swd_stds,
        ecolor="red",
        capsize=10,
        label="Sliced Wasserstein Distance",
    )
    ax.set_title("Sliced Wasserstein Distance per pyramid size")
    ax.set_xlabel("Pyramid size")
    ax.set_ylabel(r"Sliced Wasserstein Distance $\times 10^3$")
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 20.0))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for index, value in enumerate(swd_means):
        if value + swd_stds[index] <= ylim - ylim // 10.0:
            ax.text(
                index,
                value + swd_stds[index] + 1.0,
                f"{value:.2f}\n$\pm${swd_stds[index]:.2f}",
                size=8,
                ha="center",
                horizontalalignment="center",
                rotation="horizontal",
                rotation_mode="default",
            )
        else:
            ax.text(
                index - 0.5,
                ylim - 100,
                f"{value:.2f}$\pm${swd_stds[index]:.2f}",
                size=8,
                ha="center",
                rotation="vertical",
                rotation_mode="default",
            )

    plt.show()

    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
        )

    return fig, ax


def plot_swd_line(
    swd_values,
    names,
    save_folder=None,
    filename="swd_line",
    figsize=(3.75, 2.25),
    title="Sliced Wasserstein Distance per pyramid size",
):
    """
    Plots sliced Wasserstein distances in line plot.

    :param swd_values: Sliced Wasserstein distances as returned from
                       metrics.swd.calc_swd or .load_swd functions
    :param names: x-axis labels. Preferably as returned from metrics.swd.calc_swd or .load_swd functions
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension, defaults to 'swd_line'
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :param title: Title of the plot. Defaults to "Sliced Wasserstein Distance per pyramid size"
    :return: Figure and Axes object of the plot: fig, ax
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    swd_values_tensor = torch.stack([torch.tensor(y) for y in swd_values])
    swd_means = swd_values_tensor.mean(dim=0).tolist()
    swd_stds = swd_values_tensor.std(dim=0).tolist()

    ylim = 250
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    x_labels = [label.split("_")[-1] for label in names]
    ax.errorbar(
        x_labels,
        swd_means,
        fmt="--x",
        yerr=swd_stds,
        capsize=5,
        label="Sliced Wasserstein Distance",
    )
    ax.set_title(title)
    ax.set_xlabel("Pyramid size")
    ax.set_ylabel(r"Sliced Wasserstein Distance $\times 10^3$")
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 20.0))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.grid()
    plt.show()

    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
            format="pdf",
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
            format="png",
        )

    return fig, ax
