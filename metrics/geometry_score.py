from math import ceil
from typing import List
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import metrics.gs as gs


def calc_geom_score(
    real_data: torch.Tensor,
    generator: torch.nn.Module = None,
    fake_data: torch.Tensor = None,
    save_folder=None,
    filename="geometry_score",
    n_times: int = 1000,
    i_max=100,
    only_fake_mrlts=False,
    device: torch.device = torch.device("cpu"),
):
    """
    Calculates the geometry score between a batch of real data and
    fake data which will be generated with a generator.
    Source: "Geometry Score: A Method For Comparing Generative Adversarial Networks", Khrulkov et al. (2018, arXiv:1802.02664).

    :param real_data: Batch of real data. Normalized as in training.
    :param generator: Generator to generate fake data from noise. If None, fake_data is used as fake data.
    :param fake_data: Batch of fake data. If None, fake data is generated with the specified generator.
    :param save_folder: Folder in which the plot is saved. If None, nothing will be saved
    :param filename: Filename of the save file without extension.
    :param n_times: How often to calculate the rlts values
    :param i_max: i_max parameter passed to metrics.gs.rlts function
    :param only_fake_mrlts: Calculate only fake mrlts. Geometry score and real mrlts are set to zero.
                            real_data arguments has to still be a batch of real data.
    :param device: On which device to do the calculations: CPU | GPU
    :return: geometry score
    """
    if generator is None:
        assert (
            fake_data is not None
        ), "Fake data must be specified if generator is not specified!"
    else:
        assert (
            fake_data is None
        ), "Fake data cannot be specified if generator is specified!"

    num_samples = len(real_data)
    batch_size = 8

    if fake_data is None:
        generator.eval()
        print("Sampling fake data with generator...")
        fake_data = torch.tensor([])
        for i in range(ceil(num_samples / batch_size)):
            noise = torch.randn((batch_size, generator.noise_dim)).to(device)

            with torch.no_grad():
                fake_data_tmp = generator(noise)
            fake_data = torch.cat((fake_data, fake_data_tmp.cpu()), dim=0)
        if len(fake_data) > num_samples:
            diff = len(fake_data) - num_samples
            fake_data = fake_data[:-diff]

    assert fake_data.shape[0] == real_data.shape[0]

    # Reshape into 2d array
    real_data_2d = real_data.reshape(real_data.shape[0], -1)
    fake_data_2d = fake_data.reshape(fake_data.shape[0], -1)

    print("Calculating rlts for fake data...")
    rlts_fake = gs.rlts(fake_data_2d.clone().cpu().numpy(), n=n_times, i_max=i_max)
    mrlt_fake = np.mean(rlts_fake, axis=0)
    df = pd.DataFrame(
        {
            "geometry_score": [0.0] * len(mrlt_fake),
            "mrlt_real": [0.0] * len(mrlt_fake),
            "mrlt_fake": mrlt_fake,
        }
    )

    if only_fake_mrlts is False:
        print("Calculating rlts for real data...")
        rlts_real = gs.rlts(real_data_2d.clone().cpu().numpy(), n=n_times, i_max=i_max)
        mrlt_real = np.mean(rlts_real, axis=0)

        score = gs.geom_score(rlts1=rlts_real, rlts2=rlts_fake)

        df = pd.DataFrame(
            {"geometry_score": score, "mrlt_real": mrlt_real, "mrlt_fake": mrlt_fake}
        )

    file = f"{filename}.csv"
    if save_folder is not None:
        df.to_csv(os.path.join(save_folder, file))

    torch.save(fake_data, "geometry_score_fake_data.pt")
    torch.save(real_data, "geometry_score_real_data.pt")

    if only_fake_mrlts is False:
        return score, mrlt_real, mrlt_fake
    return 0.0, 0.0, mrlt_fake


def save_geom_scores(
    csv_files: List[str],
    save_folder: str,
    save_filename: str,
    additional_columns: dict = {},
):
    """
    Saves multiple csv files created from calc_geom_score function in one csv file.

    :param csv_files: List of csv files to read.
    :param save_folder: Folder in which output csv file will be written
    :param save_filename: Filename of output csv file without extension
    :param additional_columns: Specify additional columns saved to output csv file.
                               Each key will be a column name. Values need to be lists with same length as csv_files.
                               Each list entry corresponds to the file in csv_files at equal index.
                               Default columns are: type (real or fake), i_in_imax, geometry_score, mrlt
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
        columns=list(additional_columns.keys())
        + ["type", "i_in_imax", "geometry_score", "mrlt"]
    )

    for file_count, file in tqdm(enumerate(csv_files)):
        # model_number = int(file.split('_')[0])
        # gan_type = folder.split('_')[0]
        # n_epochs = int(folder.split('_')[1].rstrip('Epochs'))

        # Geometry Score
        df_tmp = pd.read_csv(file).dropna().drop(["Unnamed: 0"], axis=1)

        file_additional_columns = {}
        for key in additional_columns.keys():
            file_additional_columns[key] = additional_columns[key][file_count]

        for index, row in df_tmp.iterrows():
            # Geometry Score
            new_entry_fake = {
                **file_additional_columns,
                **{
                    "type": "fake",
                    "i_in_imax": index,
                    "geometry_score": row["geometry_score"],
                    "mrlt": row["mrlt_fake"],
                },
            }
            new_entry_real = {
                **file_additional_columns,
                **{
                    "type": "real",
                    "i_in_imax": index,
                    "geometry_score": 0.0,
                    "mrlt": row["mrlt_real"],
                },
            }

            df_all = pd.concat(
                [df_all, pd.DataFrame(new_entry_fake, index=[0])], ignore_index=True
            )
            df_all = pd.concat(
                [df_all, pd.DataFrame(new_entry_real, index=[0])], ignore_index=True
            )

    df_all.to_csv(os.path.join(save_folder, f"{save_filename}.csv"))
    df_all.to_excel(os.path.join(save_folder, f"{save_filename}.xlsx"))

    return df_all
