# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# Costum PyTorch implementation of the sliced wasserstein distance. Source code adapted from:
# "Progressive growing of gans for improved quality, stability, and variation", Karras et al. (2017, arXiv:1710.10196)

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    minibatch_np = minibatch
    S = minibatch_np.shape  # (minibatch, channel, height, width)
    assert len(S) == 4
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0 : S[1], -H : H + 1, -H : H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch_np.flat[idx]


# ----------------------------------------------------------------------------


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


# ----------------------------------------------------------------------------


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat, device=torch.device("cpu")):
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    A = torch.tensor(A).to(device)
    B = torch.tensor(B).to(device)
    results = []
    for repeat in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = torch.randn(A.shape[1], dirs_per_repeat).to(device)
        # normalize descriptor components for each direction
        dirs /= torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True))

        # (neighborhood, direction)
        projA = torch.matmul(A, dirs)
        projB = torch.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA_s, _ = torch.sort(projA, dim=0)
        projB_s, _ = torch.sort(projB, dim=0)
        dists = torch.abs(projA_s - projB_s)  # pointwise wasserstein distances
        dists = dists.cpu().numpy()
        results.append(np.mean(dists))  # average over neighborhoods and directions
    return np.mean(results)


# ----------------------------------------------------------------------------


def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (
            t[:, :, 0::2, 0::2]
            + t[:, :, 0::2, 1::2]
            + t[:, :, 1::2, 0::2]
            + t[:, :, 1::2, 1::2]
        ) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------


gaussian_filter = (
    np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=np.float32,
    )
    / 256.0
)


def pyr_down(minibatch, device=torch.device("cpu")):  # matches cv2.pyrDown()
    assert minibatch.ndim == 4

    gaussian_k = torch.tensor(gaussian_filter[np.newaxis, np.newaxis, :, :]).to(device)

    minibatch = F.pad(torch.tensor(minibatch, device=device), (2, 2, 2, 2), "reflect")
    multiband = [
        F.conv2d(minibatch[:, i : i + 1, :, :], gaussian_k, padding=0)
        for i in range(minibatch.shape[1])
    ]
    down_image = torch.cat(multiband, dim=1)[:, :, ::2, ::2]
    return down_image.cpu().numpy()


def pyr_up(minibatch, device=torch.device("cpu")):  # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2))
    res[:, :, ::2, ::2] = minibatch
    res = torch.tensor(res).to(torch.float).to(device)

    gaussian_k = torch.tensor(gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0).to(
        device
    )

    res = F.pad(res, (2, 2, 2, 2), "reflect")
    multiband = [
        F.conv2d(res[:, i : i + 1, :, :], gaussian_k, padding=0)
        for i in range(res.shape[1])
    ]
    up_image = torch.cat(multiband, dim=1)
    return up_image.cpu().numpy()


def generate_laplacian_pyramid(minibatch, num_levels, device=torch.device("cpu")):
    pyramid = [minibatch.clone().cpu().numpy()]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1], device=device))
        pyramid[-2] -= pyr_up(pyramid[-1], device=device)
    return pyramid


def reconstruct_laplacian_pyramid(pyramid, device=torch.device("cpu")):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


# ----------------------------------------------------------------------------


class API:
    def __init__(
        self,
        num_images,
        image_shape,
        image_dtype,
        minibatch_size,
        device=torch.device("cpu"),
        variable_n_patches=False,
        n_patches=0,
    ):
        self.resolutions = []
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2
        self.nhood_size = 7
        self.nhoods_per_image = [128] * len(self.resolutions)
        if variable_n_patches:
            # Increasing powers of 2, takes ~27s for about 800 images per dataset, e.g. 128, 256, 512, 1024, 2048
            self.nhoods_per_image = []
            for i in range(len(self.resolutions)):
                self.nhoods_per_image.append(2 ** (7 + i))

            # Undersample patches
            # self.nhoods_per_image = [128, 512, 2048, 8192, 32768]

            # Number of distinct patches, takes ca. 13 min for about 800 images per dataset
            # self.nhoods_per_image = [100, 676, 3364, 14885, 62500]

            # Oversample patches
            # self.nhoods_per_image = [128, 1024, 4096, 16384, 65536]
            if n_patches != 0:
                assert isinstance(
                    n_patches, list
                ), f"n_patches needs to be list of 0 but is {n_patches}"
                resolution = (
                    image_shape[1]
                    if image_shape[1] < image_shape[2]
                    else image_shape[2]
                )
                expected_size = int(np.floor(np.log2(resolution) - np.log2(16) + 1))
                assert (
                    len(n_patches) == expected_size
                ), f"Expected list of size {expected_size} for n_patches, but got list of size {len(n_patches)}: {n_patches}"
                self.nhoods_per_image = n_patches

        # print(f"Using n_patches = {self.nhoods_per_image}")

        self.dir_repeats = 4
        self.dirs_per_repeat = 128
        self.device = device
        self.pyramid_real = []
        self.pyramid_fake = []

    def get_metric_names(self):
        return ["SWDx1e3_%d" % res for res in self.resolutions] + ["SWDx1e3_avg"]

    def get_metric_formatting(self):
        return ["%-13.4f"] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ["warmup", "reals", "fakes"]
        self.descriptors = [[] for res in self.resolutions]

    def feed(self, mode, minibatch):
        for lod, level in enumerate(
            generate_laplacian_pyramid(
                minibatch, len(self.resolutions), device=self.device
            )
        ):
            if mode == "reals":
                self.pyramid_real.append(level)
            if mode == "fakes":
                self.pyramid_fake.append(level)
            desc = get_descriptors_for_minibatch(
                level, self.nhood_size, self.nhoods_per_image[lod]
            )
            self.descriptors[lod].append(desc)
            del desc

    def end(self, mode):
        desc = [finalize_descriptors(d) for d in self.descriptors]
        if mode in ["warmup", "reals"]:
            self.desc_real = desc
            return 0
        dist = [
            sliced_wasserstein(
                dreal, dfake, self.dir_repeats, self.dirs_per_repeat, device=self.device
            )
            for dreal, dfake in zip(self.desc_real, desc)
        ]
        del desc
        dist = [d * 1e3 for d in dist]  # multiply by 10^3
        return dist + [np.mean(dist)]


# ----------------------------------------------------------------------------
