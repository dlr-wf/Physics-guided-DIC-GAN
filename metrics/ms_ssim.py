import os
from pytorch_msssim import ms_ssim as ms_ssim_api
import torch
from tqdm import tqdm


# Source: "Multiscale structural similarity for image quality assessment", Wang et al. (2003)
def ms_ssim(data1, data2, data_range):
    assert data1.shape[1:] == data2.shape[1:], "data1 and data2 must have same shape!"

    ms_ssim_values = (
        torch.ones((data1.shape[0], data2.shape[0]), dtype=torch.float32) * -10.0
    )
    for i in tqdm(range(data1.shape[0])):
        for j in range(data2.shape[0]):
            ms_ssim_values[i, j] = ms_ssim_api(
                data1[i].unsqueeze(0),
                data2[j].unsqueeze(0),
                data_range=data_range,
                size_average=False,
            )

    return ms_ssim_values


def calc_ms_ssim(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    minmax=(-1, 1),
    save_folder=None,
    filename: str = "ms_ssim",
):
    """
    Calculates the Multiscale-Structural-Similarity-Index-Measure (MS-SSIM) between two batches.

    :param real_data: First batch of data
    :param fake_data: Second batch of data
    :param minmax: Minimum and maximum of the plotted displacement values
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension
    :return: MS-SSIM values as 2D tensor. First index is index in real_data, second index is index in fake_data.
    """
    real_data = (real_data - minmax[0]) / (minmax[1] - minmax[0])
    fake_data = (fake_data - minmax[0]) / (minmax[1] - minmax[0])

    ms_ssim_values = ms_ssim(real_data, fake_data, data_range=1.0)
    ms_ssim_max_values, ms_ssim_max_indices = torch.max(ms_ssim_values, 0)
    if save_folder is not None:
        torch.save(ms_ssim_values, os.path.join(save_folder, filename + ".pt"))
    return ms_ssim_values
