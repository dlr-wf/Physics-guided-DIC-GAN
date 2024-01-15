import csv
from typing import List, Tuple
from math import ceil
import os
import pandas as pd
import torch
from tqdm import tqdm
from metrics.swd_torch import API as swd_torch_api
from metrics.swd_pggan import API as swd_pggan_api


# Adapted from: "Progressive growing of gans for improved quality, stability, and variation", Karras et al. (2017, arXiv:1710.10196)
def swd_torch(
    data1, data2, device=torch.device("cpu"), variable_n_patches=False, n_patches=0
):
    # data1 = data1.cpu()
    # data2 = data2.cpu()
    swd_obj = swd_torch_api(
        data1.shape[0],
        list(data1.shape[1:]),
        torch.float32,
        data1.shape[0],
        device=device,
        variable_n_patches=variable_n_patches,
        n_patches=n_patches,
    )
    # print(f"-----------------------REALS-----------------------")
    swd_obj.begin(mode="reals")
    swd_obj.feed(mode="reals", minibatch=data1)
    dist = swd_obj.end(mode="reals")
    # print(f"-----------------------FAKES-----------------------")
    swd_obj.begin(mode="fakes")
    swd_obj.feed(mode="fakes", minibatch=data2)
    dist = swd_obj.end(mode="fakes")
    return dist, swd_obj.get_metric_names(), swd_obj


# Source: "Progressive growing of gans for improved quality, stability, and variation", Karras et al. (2017, arXiv:1710.10196)
def swd_pggan(
    data1, data2, device=torch.device("cpu"), variable_n_patches=False, n_patches=0
):
    data1 = data1.cpu()
    data2 = data2.cpu()
    swd_obj = swd_pggan_api(
        data1.shape[0], list(data1.shape[1:]), torch.float32, data1.shape[0]
    )
    # print(f"-----------------------REALS-----------------------")
    swd_obj.begin(mode="reals")
    swd_obj.feed(mode="reals", minibatch=data1)
    dist = swd_obj.end(mode="reals")
    # print(f"-----------------------FAKES-----------------------")
    swd_obj.begin(mode="fakes")
    swd_obj.feed(mode="fakes", minibatch=data2)
    dist = swd_obj.end(mode="fakes")
    return dist, swd_obj.get_metric_names(), swd_obj


def calc_swd(
    real_data: torch.Tensor,
    generator: torch.nn.Module = None,
    fake_data: torch.Tensor = None,
    save_folder=None,
    filename=None,
    n_times: int = 10,
    device: torch.device = torch.device("cpu"),
    n_patches=None,
    minmax=(-1, 1),
    torch_version=True,
):
    """
    Calculates the sliced Wasserstein distance (SWD) between a batch of real data and
    fake data which will be generated with a generator.

    :param real_data: Batch of real data. Normalized as in training.
    :param generator: Generator to generate fake data from noise. If None, fake_data is used as fake data.
    :param fake_data: Batch of fake data. If None, fake data is generated with the specified generator.
    :param save_folder: Folder in which the plot is saved. Using the folder of the used GAN is recommended.
                        If None, nothing will be saved
    :param filename: Filename of the save file without extension.
                     Defaults to 'swd_pggan' or 'swd_pytorch' based on whether torch_version is false or true.
    :param n_times: How often to calculate the SWD between real_data and a newly sampled batch of fake data
    :param device: On which device to do the calculations: CPU | GPU
    :param n_patches: Number of patches per pyramid level as a list
    :param minmax: Minimum and maximum of the real_data values.
                   Needs to be equal to the min and max of the generated fake data.
    :param torch_version: Whether to use the pytorch implementation when calculating the SWDs.
    :return: SWDs and corresponding names: values, names
    """
    if generator is None:
        assert (
            fake_data is not None
        ), "Fake data must be specified if generator is not specified!"
        assert fake_data.shape[2] == fake_data.shape[3], (
            "Data points need to have equal height and width "
            f"but have shape "
            f"{fake_data.shape[2]}x{fake_data.shape[3]}!"
        )
    else:
        generator.eval()
        assert (
            fake_data is None
        ), "Fake data cannot be specified if generator is specified!"

    assert real_data.shape[2] == real_data.shape[3], (
        "Data points need to have equal height and width "
        f"but have shape {real_data.shape[2]}x{real_data.shape[3]}!"
    )

    n_patches_tmp = []
    tmp = int(real_data.shape[2])
    counter = 0
    while tmp >= 16:
        n_patches_tmp.append(2 ** (7 + counter))
        counter += 1
        tmp = tmp // 2

    if n_patches is None:
        n_patches = n_patches_tmp
    assert len(n_patches) == len(n_patches_tmp), (
        f"Length of n_patches needs to be {len(n_patches_tmp)} "
        f"but is {len(n_patches)}!"
    )

    print(f"Using n_patches: {n_patches}")

    real_data = (real_data - minmax[0]) / (minmax[1] - minmax[0])
    swd_func = swd_torch if torch_version else swd_pggan

    num_samples = len(real_data)
    batch_size = 8
    dist = []
    names = []

    if fake_data is not None:
        fake_data = (fake_data - minmax[0]) / (minmax[1] - minmax[0])
        for i in tqdm(range(n_times)):
            dist_tmp, names_tmp, swd_obj = swd_func(
                real_data,
                fake_data,
                device=device,
                variable_n_patches=True,
                n_patches=n_patches,
            )
            print(f"dist_tmp:\n{dist_tmp}")
            dist.append(dist_tmp)
            names = names_tmp
        n_times = 0  # For loop below is not executed, since necessary calculations are already done,
        # but data is not yet saved.
    else:
        for i in tqdm(range(n_times)):
            fake_data_all = torch.tensor([])
            for ii in range(ceil(num_samples / batch_size)):
                noise = torch.randn((batch_size, generator.noise_dim)).to(device)

                with torch.no_grad():
                    fake_data = (generator(noise) - minmax[0]) / (minmax[1] - minmax[0])
                fake_data_all = torch.cat((fake_data_all, fake_data.cpu()), dim=0)
            if len(fake_data_all) > num_samples:
                diff = len(fake_data_all) - num_samples
                fake_data_all = fake_data_all[:-diff]
            dist_tmp, names_tmp, swd_obj = swd_func(
                real_data,
                fake_data_all,
                device=device,
                variable_n_patches=True,
                n_patches=n_patches,
            )
            dist.append(dist_tmp)
            names = names_tmp

    if filename is None:
        filename = "swd"
        filename += "_pytorch" if torch_version else "_pggan"
    file = f"{filename}.csv"
    if save_folder is not None:
        with open(os.path.join(save_folder, file), "w", encoding="utf-8") as f:
            for i, name in enumerate(names):
                f.write(f"{name},")
                for x in dist:
                    f.write(f"{x[i]},")
                f.write("\n")

    return dist, names


def calc_swd_low_variation(
    real_data: torch.Tensor,
    save_folder=None,
    filename=None,
    n_times: int = 10,
    samples=None,
    device: torch.device = torch.device("cpu"),
    n_patches=None,
    minmax=(-1, 1),
    torch_version=True,
):
    """
    Calculates the sliced Wasserstein distance (SWD) between a batch of real data and
    and altered version with artificially decreased variation.

    :param real_data: Batch of real data
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension.
                     Defaults to 'swd_low_variation_pggan' or 'swd_low_variation_pytorch'
                     based on whether torch_version is false or true.
    :param n_times: How often to calculate the SWD between real_data and a newly sampled altered real_data
    :param samples: List of samples to use from real_data to create versions with artificially decreased variation.
                    Defaults to [1, 5, 10, 20, 50, 100, 200, 500, len(real_data)].
    :param device: On which device to do the calculations: CPU | GPU
    :param n_patches: Number of patches per pyramid level as a list
    :param minmax: Minimum and maximum of the real_data values.
                   Needs to be equal to the min and max of the generated fake data.
    :param torch_version: Whether to use the pytorch implementation when calculating the SWDs.
    :return: SWDs and corresponding names: values, names
    """
    assert real_data.shape[2] == real_data.shape[3], (
        "Data points need to have equal height and width "
        f"but have shape {real_data.shape[2]}x{real_data.shape[3]}!"
    )

    n_patches_tmp = []
    tmp = int(real_data.shape[2])
    counter = 0
    while tmp >= 16:
        n_patches_tmp.append(2 ** (7 + counter))
        counter += 1
        tmp = tmp // 2

    if n_patches is None:
        n_patches = n_patches_tmp
    assert len(n_patches) == len(n_patches_tmp), (
        f"Length of n_patches needs to be {len(n_patches_tmp)} "
        f"but is {len(n_patches)}!"
    )

    real_data = (real_data - minmax[0]) / (minmax[1] - minmax[0])
    swd_func = swd_torch if torch_version else swd_pggan
    print("Calculating Sliced Wasserstein Distance of low variation real data...")

    num_samples = len(real_data)
    dists = {}
    if samples is None:
        samples = [1, 5, 10, 20, 50, 100, 200, 500, num_samples]
    for x in samples:
        print(f"Computing SWD with {x} samples.")
        dist = []
        names = []
        for i in tqdm(range(n_times)):
            real_data_low_var = real_data[
                torch.randint(low=0, high=num_samples, size=(x,))
            ].repeat(num_samples // x + 10, 1, 1, 1)[:num_samples]
            dist_tmp, names_tmp, swd_obj = swd_func(
                real_data_low_var,
                real_data,
                device=device,
                variable_n_patches=True,
                n_patches=n_patches,
            )

            dist.append(dist_tmp)
            names = names_tmp
        if filename is None:
            filename = "swd_low_variation"
            filename += "_pytorch" if torch_version else "_pggan"
        file = f"{filename}.csv"
        if save_folder is not None:
            with open(os.path.join(save_folder, file), "w", encoding="utf-8") as f:
                for i in range(len(names)):
                    f.write(f"{names[i]},")
                    for x in dist:
                        f.write(f"{x[i]},")
                    f.write("\n")
        print(f"dist: {dist}, names: {names}\n file: {file} in\n{save_folder}")
        dists[x] = dist
    return dists, names


def calc_swd_low_quality(
    real_data: torch.Tensor,
    save_folder=None,
    filename=None,
    noise_std=(0.0001, 0.0002, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    n_times: int = 10,
    device: torch.device = torch.device("cpu"),
    n_patches=None,
    minmax=(-1, 1),
    torch_version=True,
):
    """
    Calculates the sliced Wasserstein distance (SWD) between a batch of real data and
    and altered version with artificially decreased quality by adding noise.

    :param real_data: Batch of real data
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension.
                     Defaults to 'swd_low_quality_pggan' or 'swd_low_quality_pytorch'
                     based on whether torch_version is false or true.
    :param n_times: How often to calculate the SWD between real_data and a newly sampled altered real_data
    :param noise_std: List of standard deviations used to generate noise, which is added to the real_data
                      to create versions with artificially decreased quality.
    :param device: On which device to do the calculations: CPU | GPU
    :param n_patches: Number of patches per pyramid level as a list
    :param minmax: Minimum and maximum of the real_data values.
                   Needs to be equal to the min and max of the generated fake data.
    :param torch_version: Whether to use the pytorch implementation when calculating the SWDs.
    :return: SWDs and corresponding names: values, names
    """
    assert real_data.shape[2] == real_data.shape[3], (
        "Data points need to have equal height and width "
        f"but have shape {real_data.shape[2]}x{real_data.shape[3]}!"
    )

    n_patches_tmp = []
    tmp = int(real_data.shape[2])
    counter = 0
    while tmp >= 16:
        n_patches_tmp.append(2 ** (7 + counter))
        counter += 1
        tmp = tmp // 2

    if n_patches is None:
        n_patches = n_patches_tmp
    assert len(n_patches) == len(n_patches_tmp), (
        f"Length of n_patches needs to be {len(n_patches_tmp)} "
        f"but is {len(n_patches)}!"
    )

    real_data = (real_data - minmax[0]) / (minmax[1] - minmax[0])
    swd_func = swd_torch if torch_version else swd_pggan
    print("Calculating Sliced Wasserstein Distance of low quality real data...")

    dists = {}
    names = None
    for x in noise_std:
        print(f"Computing SWD with {x} standard deviation.")
        dist = []
        names = []
        for i in tqdm(range(n_times)):
            real_data_noise = x * torch.randn(size=real_data.size())
            real_data_noisy = real_data.cpu() + real_data_noise
            real_data_noisy_clamp = torch.clamp(real_data_noisy, min=0, max=1)
            dist_tmp, names_tmp, swd_obj = swd_func(
                real_data_noisy_clamp,
                real_data,
                device=device,
                variable_n_patches=True,
                n_patches=n_patches,
            )

            dist.append(dist_tmp)
            names = names_tmp
        if filename is None:
            filename = "swd_low_quality"
            filename += "_pytorch" if torch_version else "_pggan"
        file = f"{filename}.csv"
        if save_folder is not None:
            with open(os.path.join(save_folder, file), "w", encoding="utf-8") as f:
                for i in range(len(names)):
                    f.write(f"{names[i]},")
                    for x in dist:
                        f.write(f"{x[i]},")
                    f.write("\n")
        print(f"dist: {dist}, names: {names}\n FILE: {file} in\n{save_folder}")
        dists[x] = dist
    return dists, names


def save_swds(
    csv_files: List[str],
    save_folder: str,
    save_filename: str,
    pyramid_sizes: Tuple = (256, 128, 64, 32, 16, "avg"),
    additional_columns: dict = {},
):
    """
    Saves multiple csv files created from calc_swd function in one csv file.

    :param csv_files: List of csv files to read.
    :param save_folder: Folder in which output csv file will be written
    :param save_filename: Filename of output csv file without extension
    :param pyramid_sizes: Pyramid sizes to read from all csv files
    :param additional_columns: Specify additional columns saved to output csv file.
                               Each key will be a column name. Values need to be lists with same length as csv_files.
                               Each list entry corresponds to the file in csv_files at equal index.
                               Default columns are: # of SWD run (depends on n_times parameter in metrics.swd.calc_swd), pyramid size and SWD value
    """
    _additional_columns_old_keys = ["model_number", "gan_type", "n_epochs"]
    for key, value in additional_columns.items():
        assert isinstance(
            value, list
        ), f"All values in additional_columns need to be lists, but values of key {key} are of type {type(value)}!"
        assert len(value) == len(csv_files), (
            f"All values in additional_columns need to be lists with same length as csv_files ({len(csv_files)}), "
            f"but list of key {key} has length {len(value)}!"
        )

    df_all = pd.DataFrame(
        columns=list(additional_columns.keys()) + ["SWD_run", "pyramid_size", "SWD"]
    )

    for file_count, file in tqdm(enumerate(csv_files)):
        # model_number = int(file.split('_')[0])
        # gan_type = folder.split('_')[0]
        # n_epochs = int(folder.split('_')[1].rstrip('Epochs'))

        df_tmp = pd.read_csv(file, header=None).T.dropna()
        df_tmp = df_tmp.rename(columns=df_tmp.iloc[0]).drop(
            df_tmp.index[0]
        )  # Use first row as header

        file_additional_columns = {}
        for key in additional_columns.keys():
            file_additional_columns[key] = additional_columns[key][file_count]

        for index, row in df_tmp.iterrows():
            for pyr_size in pyramid_sizes:
                swd = row[f"SWDx1e3_{pyr_size}"]

                new_entry = {
                    **file_additional_columns,
                    **{"SWD_run": index - 1, "pyramid_size": pyr_size, "SWD": swd},
                }
                df_all = pd.concat(
                    [df_all, pd.DataFrame(new_entry, index=[0])], ignore_index=True
                )

    df_all.to_csv(os.path.join(save_folder, f"{save_filename}.csv"))
    df_all.to_excel(os.path.join(save_folder, f"{save_filename}.xlsx"))

    return df_all


def load_swd(path_to_file):
    """
    Loads SWD values from csv file created from a calc_swd function.

    :param path_to_file: Path to the csv file
    :return: SWDs and corresponding names as returned from a calc_swd funtion: values, names
    """
    reader = csv.reader(open(path_to_file, "r", encoding="utf-8"))
    swd = {}
    for row in reader:
        entries = row
        key = entries[0]
        values = [float(x) for x in entries[1:-1]]
        swd[key] = values

    swd_values = list(swd.values())
    swd_values = torch.stack([torch.tensor(y) for y in swd_values]).t().tolist()
    swd_names = list(swd.keys())

    return swd_values, swd_names
