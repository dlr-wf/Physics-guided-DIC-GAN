import os
from typing import List
import torch
from torch.utils.data import Dataset
import torchvision.transforms as v_transforms
import datasets.import_nodemaps as data_processing
import datasets.interpolation as interpolation
from datasets.import_nodemaps import NodemapStructure


def numpy_to_tensor(numpy_dict):
    """
    Convert a dict of numpy arrays into a dict of 'unsqueezed' tensors.

    :param numpy_dict: Dictionary to convert
    :return: Dictionary with torch tensors as values instead of numpy arrays
    """
    return {
        key: torch.from_numpy(value).unsqueeze(0).to(torch.float32)
        for key, value in numpy_dict.items()
    }


def dict_to_list(dictionary):
    """
    Converts a dictionary into a list by loosing the keys.

    :param dictionary: Dictionary to convert
    :return: List of dictionary values
    """

    return [value for key, value in dictionary.items()]


class DisplacementDataset(Dataset):
    """
    Create dataset of displacement data from a tensor file.

    :param data_paths: List of tensor files
    :param scaling: Type of scaling used to preprocess data: minmax | no_scaling
    :param norm_data_paths: List of tensor files used to normalize/scale the data from data_paths.
                            This data is only used for normalization.
    """

    def __init__(self, data_paths: List[str], scaling="minmax", norm_data_paths=None):
        self.supported_scalings = ["minmax", "no_scaling"]
        if scaling not in self.supported_scalings:
            raise NotImplementedError(
                f"Scaling {scaling} not implemented. Use one of {self.supported_scalings}"
            )
        self.scaling = scaling

        self.path = data_paths
        self.data = torch.cat([torch.load(data) for data in self.path], dim=0)
        if norm_data_paths is None:
            self.x_min = self.data[:, 0].min()
            self.x_max = self.data[:, 0].max()
            self.y_min = self.data[:, 1].min()
            self.y_max = self.data[:, 1].max()
        else:
            print(f"Using statistics from {norm_data_paths}")
            norm_data = torch.cat([torch.load(data) for data in norm_data_paths], dim=0)
            self.x_min = norm_data[:, 0].min()
            self.x_max = norm_data[:, 0].max()
            self.y_min = norm_data[:, 1].min()
            self.y_max = norm_data[:, 1].max()

        print(
            f"Data min/max: (x: {self.x_min}/{self.x_max}, y: {self.y_min}/{self.y_max})"
        )

        self.minmax_transforms = v_transforms.Compose(
            [
                # Min-Max scaling with range [0,1]
                v_transforms.Normalize(
                    mean=(self.x_min, self.y_min),
                    std=(self.x_max - self.x_min, self.y_max - self.y_min),
                ),
                v_transforms.Lambda(lambda x: (x * 2.0) - 1.0)  # Shift to range [-1,1]
                # Shifting data to [-1,1] is important because of the tanh activation in the last generator layer!
            ]
        )
        self.noscaling_transforms = v_transforms.Compose([])

        if self.scaling == "minmax":
            print("Using min-max scaling.")
            self.transforms = self.minmax_transforms
        elif self.scaling == "no_scaling":
            print("Using no scaling.")
            self.transforms = self.noscaling_transforms

    def remove(self, idx):
        """
        Remove the data point with index idx from the dataset.

        :param idx: Index of the data point you want to remove
        """
        self.data = torch.cat((self.data[:idx], self.data[idx + 1 :]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.data)

    def normalize_data(self, data, scaling=None):
        """
        Apply the same normalization used for the data of this dataset to specified data.

        :param data: Data on which to apply the normalization
        """

        if scaling is None:
            return self.transforms(data)
        elif scaling not in self.supported_scalings:
            raise NotImplementedError(
                f"Scaling {scaling} not implemented. Use one of {self.supported_scalings}"
            )

        if scaling == "minmax":
            return self.minmax_transforms(data)
        elif scaling == "no_scaling":
            return self.noscaling_transforms(data)
        else:
            raise NotImplementedError(
                f"Scaling {scaling} not implemented for unnormalization yet!"
            )

    def unnormalize_data(self, data, scaling=None):
        """
        Reverse the normalization used for the data of this dataset on specified data.

        :param data: Data on which to reverse the normalization
        """

        if scaling is None:
            scaling = self.scaling
        elif scaling not in self.supported_scalings:
            raise NotImplementedError(
                f"Scaling {scaling} not implemented. Use one of {self.supported_scalings}"
            )

        if scaling == "minmax":
            data_renorm = (data + 1.0) / 2.0

            data_renorm[:, 0] = (
                data_renorm[:, 0] * (self.x_max - self.x_min)
            ) + self.x_min
            data_renorm[:, 1] = (
                data_renorm[:, 1] * (self.y_max - self.y_min)
            ) + self.y_min
        elif scaling == "no_scaling":
            data_renorm = data
        else:
            raise NotImplementedError(
                f"Scaling {scaling} not implemented for unnormalization yet!"
            )

        return data_renorm
