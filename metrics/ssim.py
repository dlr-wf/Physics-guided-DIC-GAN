import os
from pytorch_msssim import ssim as ssim_api
import torch
from tqdm import tqdm


# Source: "Image Quality Assessment: From Error Visibility to Structural Similarity", Wang et al. (2004)
def ssim(real_data, fake_data):
    assert (
        real_data.shape[1:] == fake_data.shape[1:]
    ), "real_data and fake_data must have same shape!"

    ssim_values = (
        torch.ones((real_data.shape[0], fake_data.shape[0]), dtype=torch.float32)
        * -10.0
    )
    for i in tqdm(range(real_data.shape[0])):
        for j in range(fake_data.shape[0]):
            ssim_values[i, j] = ssim_api(
                real_data[i].unsqueeze(0),
                fake_data[j].unsqueeze(0),
                data_range=1.0,
                size_average=False,
            )

    return ssim_values


def calc_ssim(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    minmax=(-1, 1),
    save_folder=None,
    filename: str = "ssim",
):
    """
    Calculates the Structural-Similarity-Index-Measure (SSIM) between two batches.

    :param real_data: First batch of data
    :param fake_data: Second batch of data
    :param minmax: Minimum and maximum of the plotted displacement values
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension
    :return: SSIM values as 2D tensor. First index is index in real_data, second index is index in fake_data.
    """
    real_data = (real_data - minmax[0]) / (minmax[1] - minmax[0])
    fake_data = (fake_data - minmax[0]) / (minmax[1] - minmax[0])

    ssim_values = ssim(real_data, fake_data)
    ssim_max_values, ssim_max_indices = torch.max(ssim_values, 0)
    if save_folder is not None:
        torch.save(ssim_values, os.path.join(save_folder, filename + ".pt"))
    return ssim_values
