import os
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from gan.gans import calc_and_concat_strains


def make_plot_data(batch, ncols, minmax=(-1, 1)):
    """
    Reorders a batch of data into a 2D grid for plotting.

    :param batch: The batch of data of shape NxCxHxW. If it is only displacement data,
                  then von-Mises strain data will be added.
    :param ncols: Columns of the 2D grid
    :param minmax: Minimum and maximum of the dataset from which the batch is drawn
    :return: grid of x-displacements, grid of y-displacements, grid of von-Mises strains
    """
    normalize = False
    scale_each = False
    padding = 6

    if batch.shape[1] == 2:
        batch = calc_and_concat_strains(batch, strains="vonMises")
    assert (
        batch.shape[1] == 3
    ), "Expected 3 channels: x-, y-displacements, von-Mises strain"

    plot_x = vutils.make_grid(
        batch[:, 0].unsqueeze(1),
        nrow=ncols,
        padding=padding,
        pad_value=-10,
        normalize=normalize,
        scale_each=scale_each,
        value_range=minmax,
    ).cpu()[0]
    plot_y = vutils.make_grid(
        batch[:, 1].unsqueeze(1),
        nrow=ncols,
        padding=padding,
        pad_value=-10,
        normalize=normalize,
        scale_each=scale_each,
        value_range=minmax,
    ).cpu()[0]
    plot_eps = vutils.make_grid(
        batch[:, 2].unsqueeze(1),
        nrow=ncols,
        padding=padding,
        pad_value=-10,
        normalize=False,
    ).cpu()[0]

    return plot_x, plot_y, plot_eps


def plot_sample(
    sample: torch.Tensor,
    minmax=(-1, 1),
    title=None,
    save_folder=None,
    filename=None,
    figsize=None,
    show_axis=True,
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    cmap = "coolwarm"

    if figsize is None:
        figsize = (4, 4)
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, constrained_layout=True)

    _im = ax.imshow(
        sample, vmin=minmax[0], vmax=minmax[1], cmap=cmap, interpolation=None
    )

    if show_axis is False:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    if save_folder is not None:
        dpi = 512
        filename = "sample" if filename is None else filename
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
    return fig, ax


def plot_batch(
    batch: torch.Tensor,
    minmax=((-1, 1), (-1, 1)),
    minmax_strain=(0, 0.002),
    title=None,
    ax_titles=("$u_x$", "$u_y$", "$\\varepsilon_\\mathrm{vm}$"),
    save_folder=None,
    filename=None,
    figsize=None,
    order=("x", "y", "eps"),
    colorbar_padding=None,
    ncols=None,
    cmap="coolwarm",
):
    """
    Plots and saves a batch of data.

    :param batch: The batch to be plotted. If it is only displacement data, then von-Mises strain data will be added.
    :param minmax: Minimum and maximum of the plotted displacement values ((x_min, x_max), (y_min, y_max))
    :param minmax_strain: Minimum and maximum of the plotted von-Mises strain values
    :param title: Title of the plot
    :param ax_titles: Titles of the subplots
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension, defaults to f"batch_{batch.shape[0]}"
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :param order: Defines the order of the plots from left to right
    :param colorbar_padding: Padding between colorbar and plots
    :param ncols: Number of columns in the plotted data
    :return: Figure and Axes object of the plot: fig, ax
    """
    assert (
        len(order) == 3 and "x" in order and "y" in order and "eps" in order
    ), f"order needs to have entries 'x', 'y', 'eps'. Nothing else! But oder is {order}"

    if ncols is None:
        ncols = int(np.sqrt(batch.shape[0]))

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    plot_x, plot_y, plot_eps = make_plot_data(batch, ncols, minmax)

    scale = plot_eps.shape[0] / plot_eps.shape[1]
    if figsize is None:
        figsize = (8, 4 * scale)
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=3, constrained_layout=True)

    im_eps = ax[order.index("eps")].imshow(
        plot_eps,
        vmin=minmax_strain[0],
        vmax=minmax_strain[1],
        cmap=cmap,
        interpolation=None,
    )
    im_x = ax[order.index("x")].imshow(
        plot_x, vmin=minmax[0][0], vmax=minmax[0][1], cmap=cmap, interpolation=None
    )
    im_y = ax[order.index("y")].imshow(
        plot_y, vmin=minmax[1][0], vmax=minmax[1][1], cmap=cmap, interpolation=None
    )

    ax[0].set_title(ax_titles[0])
    ax[0].axis("off")
    ax[1].set_title(ax_titles[1])
    ax[1].axis("off")
    ax[2].set_title(ax_titles[2])
    ax[2].axis("off")

    if colorbar_padding is None:
        pad = 0.05 if batch.shape[0] > 64 else 0.15
    else:
        pad = colorbar_padding

    extend_u = "both"
    _c_b = fig.colorbar(
        im_eps,
        ax=ax[order.index("eps") : order.index("eps") + 1],
        extend="max",
        location="top",
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_x,
        ax=ax[order.index("x") : order.index("x") + 1],
        extend=extend_u,
        location="top",
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_y,
        ax=ax[order.index("y") : order.index("y") + 1],
        extend=extend_u,
        location="top",
        pad=pad,
    )

    if title is not None:
        fig.suptitle(title)

    if save_folder is not None:
        dpi = (
            4096
            if batch.shape[0] > 1000
            else 2048
            if batch.shape[0] > 500
            else 1024
            if batch.shape[0] > 16
            else 512
        )
        print(f"Using dpi={dpi}.")
        filename = f"batch_{batch.shape[0]}" if filename is None else filename
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
    return fig, ax


def plot_real_vs_fake(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    minmax=(-1, 1),
    minmax_strain=(0, 0.002),
    title=None,
    ax_titles=(
        ("Real $\\varepsilon_\\mathrm{vm}$", "Fake $\\varepsilon_\\mathrm{vm}$"),
        ("Real $u_x$", "Fake $u_x$"),
        ("Real $u_y$", "Fake $u_y$"),
    ),
    save_folder=None,
    filename=None,
    figsize=None,
    order=("x", "y", "eps"),
    colorbar_padding=None,
    ncols=None,
):
    """
    Plots real and fake data side by side.

    :param real_data: Batch of real data. If it is only displacement data, then von-Mises strain data will be added.
    :param fake_data: Batch of fake data. If it is only displacement data, then von-Mises strain data will be added.
    :param minmax: Minimum and maximum of the plotted displacement values ((x_min, x_max), (y_min, y_max))
    :param minmax_strain: Minimum and maximum of the plotted von-Mises strain values
    :param title: Title of the plot
    :param ax_titles: Titles of the subplots
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension, defaults to f"real_vs_fake_{real_data.shape[0]}"
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :param order: Defines the order of the plots from left to right
    :param colorbar_padding: Padding between colorbar and plots
    :param ncols: Number of columns in the plotted data
    :return: Figure and Axes object of the plot: fig, ax
    """
    assert (
        real_data.shape[0] == fake_data.shape[0]
    ), "Real and fake data batches need to have the same batch size"
    assert (
        len(order) == 3 and "x" in order and "y" in order and "eps" in order
    ), f"order needs to have entries 'x', 'y', 'eps'. Nothing else! But oder is {order}"

    if ncols is None:
        ncols = int(np.sqrt(real_data.shape[0]))

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    cmap = "coolwarm"

    real_plot_x, real_plot_y, real_plot_eps = make_plot_data(real_data, ncols, minmax)
    fake_plot_x, fake_plot_y, fake_plot_eps = make_plot_data(fake_data, ncols, minmax)

    scale = fake_plot_eps.shape[0] / fake_plot_eps.shape[1]
    if figsize is None:
        figsize = (3, 4 * scale)
    fig, ax = plt.subplots(figsize=figsize, nrows=3, ncols=2, constrained_layout=True)

    _im_reps = ax[order.index("eps"), 0].imshow(
        real_plot_eps,
        vmin=minmax_strain[0],
        vmax=minmax_strain[1],
        cmap=cmap,
        interpolation=None,
    )
    im_feps = ax[order.index("eps"), 1].imshow(
        fake_plot_eps,
        vmin=minmax_strain[0],
        vmax=minmax_strain[1],
        cmap=cmap,
        interpolation=None,
    )
    _im_rx = ax[order.index("x"), 0].imshow(
        real_plot_x, vmin=minmax[0][0], vmax=minmax[0][1], cmap=cmap, interpolation=None
    )
    im_fx = ax[order.index("x"), 1].imshow(
        fake_plot_x, vmin=minmax[0][0], vmax=minmax[0][1], cmap=cmap, interpolation=None
    )
    _im_ry = ax[order.index("y"), 0].imshow(
        real_plot_y, vmin=minmax[1][0], vmax=minmax[1][1], cmap=cmap, interpolation=None
    )
    im_fy = ax[order.index("y"), 1].imshow(
        fake_plot_y, vmin=minmax[1][0], vmax=minmax[1][1], cmap=cmap, interpolation=None
    )

    ax[0, 0].set_title(ax_titles[0][0])
    ax[0, 0].axis("off")
    ax[0, 1].set_title(ax_titles[0][1])
    ax[0, 1].axis("off")
    ax[1, 0].set_title(ax_titles[1][0])
    ax[1, 0].axis("off")
    ax[1, 1].set_title(ax_titles[1][1])
    ax[1, 1].axis("off")
    ax[2, 0].set_title(ax_titles[2][0])
    ax[2, 0].axis("off")
    ax[2, 1].set_title(ax_titles[2][1])
    ax[2, 1].axis("off")

    if colorbar_padding is None:
        pad = 0.05 if real_data.shape[0] > 64 or fake_data.shape[0] > 64 else 0.15
    else:
        pad = colorbar_padding

    extend_u = "both"
    _c_b = fig.colorbar(
        im_feps,
        ax=ax[order.index("eps") : order.index("eps") + 1, 1],
        extend="max",
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_fx,
        ax=ax[order.index("x") : order.index("x") + 1, 1],
        extend=extend_u,
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_fy,
        ax=ax[order.index("y") : order.index("y") + 1, 1],
        extend=extend_u,
        pad=pad,
    )

    if title is not None:
        fig.suptitle(title)

    if save_folder is not None:
        dpi = 1024 if real_data.shape[0] > 16 else 512
        filename = (
            f"real_vs_fake_{real_data.shape[0]}" if filename is None else filename
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".pdf"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
        plt.savefig(
            os.path.join(save_folder, filename + ".png"),
            bbox_inches="tight",
            transparent=False,
            dpi=dpi,
            facecolor=(1.0, 1.0, 1.0, 1.0),
        )
    return fig, ax


def plot_batch_animation(
    batch: torch.Tensor,
    minmax=(-1, 1),
    minmax_strain=(0, 0.002),
    title="Epoch",
    ax_titles=("$u_x$", "$u_y$", "$\\varepsilon_\\mathrm{vm}$"),
    save_folder=None,
    filename="ani",
    figsize=None,
    order=("x", "y", "eps"),
    colorbar_padding=None,
    ncols=None,
):
    """
    Plots each datapoint in batch to create an animation along the first dimension.

    :param batch: The batch to be plotted. If it is only displacement data, then von-Mises strain data will be added.
    :param minmax: Minimum and maximum of the plotted displacement values ((x_min, x_max), (y_min, y_max))
    :param minmax_strain: Minimum and maximum of the plotted von-Mises strain values
    :param title: Start of the title of the animation.
                  The current frame number c and total number of frames t are added to the title: title c/t
    :param ax_titles: Titles of the subplots
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension
    :param figsize: Figsize argument passed to the matplotlib subplots function
    :param order: Defines the order of the plots from left to right
    :param colorbar_padding: Padding between colorbar and plots
    :param ncols: Number of columns in the plotted data
    :return: Figure, axes and animation object of the plot: fig, ax, ani
    """
    assert (
        len(order) == 3 and "x" in order and "y" in order and "eps" in order
    ), f"order needs to have entries 'x', 'y', 'eps'. Nothing else! But oder is {order}"

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("pgf", texsystem="pdflatex")

    def animation_function(frame):
        print(f"Drawing frame number {frame+1}/{len(batch)}\r", end="")
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[0].axis("off")
        ax[0].set_title(ax_titles[0])
        ax[1].axis("off")
        ax[1].set_title(ax_titles[1])
        ax[2].axis("off")
        ax[2].set_title(ax_titles[2])
        if title is not None:
            fig.suptitle(f"{title} {frame + 1}/{len(batch)}")
        x = batch[frame][0]
        y = batch[frame][1]
        eps = batch[frame][2]

        _im_eps = ax[order.index("eps")].imshow(
            eps, vmin=minmax_strain[0], vmax=minmax_strain[1], cmap="coolwarm"
        )
        _im_x = ax[order.index("x")].imshow(
            x, vmin=minmax[0][0], vmax=minmax[0][1], cmap="coolwarm"
        )
        _im_y = ax[order.index("y")].imshow(
            y, vmin=minmax[1][0], vmax=minmax[1][1], cmap="coolwarm"
        )

    if batch.dim() == 5:
        ncols = int(np.sqrt(batch.shape[1]))
        batch = torch.stack(
            [
                torch.stack(make_plot_data(img_batch, ncols, minmax))
                for img_batch in batch
            ]
        )
    assert (
        batch.dim() == 4
    ), f"Expected the batch to have 4 or 5 dimensions. Got {batch.dim()}."

    if batch.shape[1] == 2:
        batch = calc_and_concat_strains(batch, strains="vonMises")
    assert (
        batch.shape[1] == 3
    ), f"Expected a datapoint in the batch to have 2 or 3 channels. Got {batch.shape[1]}."

    if figsize is None:
        figsize_y = 4 * (batch.shape[3] // batch.shape[2])
        figsize = ((figsize_y) * 2, figsize_y)
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=3, constrained_layout=True)
    x = batch[0][0]
    y = batch[0][1]
    eps = batch[0][2]

    im_eps = ax[order.index("eps")].imshow(
        eps, vmin=minmax_strain[0], vmax=minmax_strain[1], cmap="coolwarm"
    )
    im_x = ax[order.index("x")].imshow(
        x, vmin=minmax[0][0], vmax=minmax[0][1], cmap="coolwarm"
    )
    im_y = ax[order.index("y")].imshow(
        y, vmin=minmax[1][0], vmax=minmax[1][1], cmap="coolwarm"
    )

    if colorbar_padding is None:
        pad = 0.05 if batch.shape[0] > 64 or batch.shape[0] > 64 else 0.15
    else:
        pad = colorbar_padding
    extend_u = "both"
    _c_b = fig.colorbar(
        im_eps,
        ax=ax[order.index("eps") : order.index("eps") + 1],
        extend="max",
        location="top",
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_x,
        ax=ax[order.index("x") : order.index("x") + 1],
        extend=extend_u,
        location="top",
        pad=pad,
    )
    _c_b = fig.colorbar(
        im_y,
        ax=ax[order.index("y") : order.index("y") + 1],
        extend=extend_u,
        location="top",
        pad=pad,
    )

    ani = animation.FuncAnimation(
        fig, animation_function, frames=len(batch), interval=1000, repeat_delay=1000
    )
    if save_folder is not None:
        ani.save(os.path.join(save_folder, filename + ".gif"))
    return fig, ax, ani
