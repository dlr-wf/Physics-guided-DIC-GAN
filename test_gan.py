import imageio.v3 as iio
import os
import random
import numpy as np
import torch
import configs.gan as conf
from datasets.displacement_datasets import DisplacementDataset
from data_processing.strain_calc import calc_and_concat_strains
from gan import gans
from gan.noise_vec_interpolation import interpolate_noise_vectors
from metrics import swd, ssim, ms_ssim, geometry_score
import metrics.gs as gs
import plot.data as plt_d
import plot.metrics as plt_m

##########################
# Define global variables:
##########################
if conf.seed is not None:
    print(f"Using manual seed ({conf.seed}).")
    torch.manual_seed(conf.seed)
    random.seed(conf.seed)
    np.random.seed(conf.seed)

print("Defining global variables...")
DATA_PATH = os.path.join("data", "training_data.pt")
MODEL_PATH = os.path.join("models", "GANs", "Test")
MODEL_FOLDER = "EPOCHS2_BATCHS8_DISCDCGANDiscriminator_GENUpsampleGenerator_ACTrelu_UPMODEnearest_ND5_S1LAST_LOSSGANLoss_TYPEvanilla_LLOSS1.0_OPTDISCAdam_OPTGENAdam_0"
MODEL_FOLDER = os.path.join(MODEL_PATH, MODEL_FOLDER)
SAVE_FOLDER = MODEL_FOLDER

SCALING = "minmax"

####################
# Define dataset:
####################
print("Defining dataloader...")
dataset = DisplacementDataset([DATA_PATH], scaling=SCALING)
real_data = dataset.data
real_data_normalized = dataset.transforms(dataset.data)

###########
# Load GAN:
###########
print("Loading GAN...")
# checkpoint_path = os.path.join(MODEL_FOLDER, 'checkpoints', 'Epoch10')
checkpoint_path = MODEL_FOLDER
gan_state_dict = torch.load(
    os.path.join(checkpoint_path, "gan_state_dict.pt"), map_location=conf.device
)
my_gan = gans.load_gan_model_by_folder(
    MODEL_FOLDER, gan_state_dict=gan_state_dict, device=conf.device
)

##################
# Load statistics:
##################
print("Loading losses...")
losses_disc = torch.load(
    os.path.join(checkpoint_path, "losses_disc.pt")
)  # Dict['epoch'][epoch_number] is list
mean_losses_disc = torch.stack(
    [torch.stack(epoch_list).mean() for epoch_list in losses_disc["epoch"].values()]
)
losses_gen = torch.load(os.path.join(checkpoint_path, "losses_gen.pt"))
mean_losses_gen = torch.stack(
    [torch.stack(epoch_list).mean() for epoch_list in losses_gen["epoch"].values()]
)
print("Loading discriminator outputs...")
outputs_disc = torch.load(os.path.join(checkpoint_path, "outputs_disc.pt"))
mean_outputs_disc_real = torch.stack(
    [
        torch.stack(epoch_list["real"]).mean()
        for epoch_list in outputs_disc["epoch"].values()
    ]
)
mean_outputs_disc_fake = torch.stack(
    [
        torch.stack(epoch_list["fake"]).mean()
        for epoch_list in outputs_disc["epoch"].values()
    ]
)
mean_outputs_disc_fake_2 = torch.stack(
    [
        torch.stack(epoch_list["fake_2"]).mean()
        for epoch_list in outputs_disc["epoch"].values()
    ]
)

print("Loading fake data list...")
fake_data_list = torch.load(os.path.join(checkpoint_path, "fake_data_list.pt"))
fake_data_list[:, :, 2] = (
    fake_data_list[:, :, 2] * 100
)  # Convert to vonMises strains to %

print("Plotting fake data as animation and saving as gif...")
plt_d.plot_batch_animation(
    fake_data_list,
    save_folder=MODEL_FOLDER,
    filename="fake_data_list",
    minmax=((-0.1, 0.1), (-0.2, 0.2)),
    minmax_strain=(0.0, 0.2),
    ax_titles=("$u_x$\,[mm]", "$u_y$\,[mm]", "$\\varepsilon_\\mathrm{vm}$\,[\%]"),
    figsize=(6, 3),
    colorbar_padding=0.02,
    ncols=4,
)
print("Saving animation as mp4...")
gif_org = iio.imread(os.path.join(MODEL_FOLDER, "fake_data_list.gif"), index=None)
iio.imwrite(os.path.join(MODEL_FOLDER, "fake_data_list.mp4"), gif_org, fps=10)
print("Saved animation as mp4.")

#####################
# Generate fake data:
#####################
print("Generating fake data...")
noise_vec = torch.randn(9, my_gan.generator.noise_dim, device=conf.device)
with torch.no_grad():
    fake_data = my_gan.generator(noise_vec).cpu()

fake_data_w_eps = calc_and_concat_strains(
    dataset.unnormalize_data(fake_data), "vonMises"
)
fake_data_w_eps[:, 2] = fake_data_w_eps[:, 2] * 100  # Convert von-Mises strains to %

real_data_w_eps = calc_and_concat_strains(real_data, "vonMises")
real_data_w_eps[:, 2] = real_data_w_eps[:, 2] * 100  # Convert von-Mises strains to %

plt_d.plot_batch(
    fake_data_w_eps,
    save_folder=MODEL_FOLDER,
    filename=f"fake_batch_seed_{conf.seed}",
    minmax=((-0.1, 0.1), (-0.2, 0.2)),
    minmax_strain=(0.0, 0.2),
    ax_titles=("$u_x$\,[mm]", "$u_y$\,[mm]", "$\\varepsilon_\\mathrm{vm}$\,[\%]"),
    figsize=(6, 8),
    colorbar_padding=0.02,
    ncols=3,
    cmap="coolwarm",
)

##########################
# Plot real vs. fake data:
##########################
idc = torch.randint(low=0, high=real_data_w_eps.shape[0], size=(9,))
# plt_d.plot_real_vs_fake(real_data_w_eps[idc], fake_data_w_eps,
#                         save_folder=MODEL_FOLDER,
#                         filename='real_batch',
#                         minmax=((-0.1, 0.1), (-0.2, 0.2)), minmax_strain=(0.0, 0.2),
#                         ax_titles=(('Real $\\varepsilon_\\mathrm{vm}$\,[\%]', 'Fake $\\varepsilon_\\mathrm{vm}$\,[\%]'),
#                                    ('Real $u_x$\,[mm]', 'Fake $u_x$\,[mm]'),
#                                    ('Real $u_y$\,[mm]', 'Fake $u_y$\,[mm]')),
#                         figsize=(6, 8), colorbar_padding=0.02, ncols=4)

# Plot real data
plt_d.plot_batch(
    real_data_w_eps[idc],
    save_folder=MODEL_FOLDER,
    filename=f"real_batch_seed_{conf.seed}",
    minmax=((-0.1, 0.1), (-0.2, 0.2)),
    minmax_strain=(0.0, 0.2),
    ax_titles=("$u_x$\,[mm]", "$u_y$\,[mm]", "$\\varepsilon_\\mathrm{vm}$\,[\%]"),
    figsize=(6, 8),
    colorbar_padding=0.02,
    ncols=3,
    cmap="coolwarm",
)

################################################
# Interpolate between two noise vectors metrics:
################################################
# print("Interpolating between noise vectors...")
# n1 = torch.randn(1, my_gan.generator.noise_dim, device=conf.device)
# n2 = torch.randn(my_gan.generator.noise_dim, device=conf.device)
# vecs, gen_data = interpolate_noise_vectors(n1, n2, num=100, generator=my_gan.generator, device=conf.device)

# gen_data_w_eps = calc_and_concat_strains(gen_data, 'vonMises')
# real_data_w_eps[:, 2] = real_data_w_eps[:, 2]*100  # Convert to vonMises strains to %

# ani = plt_d.plot_batch_animation(gen_data_w_eps, save_folder=MODEL_FOLDER, filename='interp',
#                                  title='Interpolation',
#                                  minmax=((-0.1, 0.1), (-0.2, 0.2)), minmax_strain=(0.0, 0.2),
#                                  ax_titles=('$u_x$\,[mm]', '$u_y$\,[mm]', '$\\varepsilon_\\mathrm{vm}$\,[\%]'),
#                                  figsize=(6, 3), colorbar_padding=0.02, ncols=4)
# gif_org = iio.mimread(os.path.join(MODEL_FOLDER, "interp.gif"), memtest=False)
# iio.mimsave(os.path.join(MODEL_FOLDER, f"interp.mp4"), gif_org, fps=30)

####################
# Calculate metrics:
####################
print("Calculating metrics...")
# --- Sliced Wasserstein Distance ---
print("Calculating Sliced Wasserstein Distance...")
swd_values, swd_names = swd.calc_swd(
    generator=my_gan.generator,
    real_data=real_data_normalized,
    save_folder=MODEL_FOLDER,
    device=conf.device,
    n_times=10,
    torch_version=True,
)
swd_values, swd_names = swd.load_swd(os.path.join(MODEL_FOLDER, "swd_pytorch.csv"))

# --- Structural Similarity Index Measure ---
# print("Calculating Structural Similarity Index Measure...")
# ssim_values = ssim.calc_ssim(real_data=real_data, fake_data=fake_data, save_folder=MODEL_FOLDER)
# ssim_values = torch.load(os.path.join(MODEL_FOLDER, 'ssim.pt'))

# --- Multi-Scale Structural Similarity Index Measure ---
# print("Calculating Multi-Scale Structural Similarity Index Measure...")
# ms_ssim_values = ms_ssim.calc_ms_ssim(real_data=real_data, fake_data=fake_data, save_folder=MODEL_FOLDER)


# --- Geometry Score ---
print("Calculating Geometry Score...")
score, mrlt_real, mrlt_fake = geometry_score.calc_geom_score(
    generator=my_gan.generator,
    real_data=real_data_normalized,
    save_folder=MODEL_FOLDER,
    n_times=2,
    i_max=50,
    only_fake_mrlts=False,
    device=conf.device,
)

##################################
# Combining SWDs of multiple GANs:
##################################
# base_folder_name = "EPOCHS100_BATCHS8_DISCDCGANDiscriminator_GENUpsampleGenerator_ACTrelu_UPMODEnearest_ND5_S1LAST_LOSSGANLoss_TYPEvanilla_LLOSS1.0_OPTDISCAdam_OPTGENAdam_"
# model_folders = [base_folder_name + str(i) for i in range(10)]
# model_folders = model_folders + [base_folder_name+ "EPSvonMises_" + str(i) for i in range(10)]
# model_folders = [os.path.join(MODEL_PATH, folder) for folder in model_folders]
# csv_files = [os.path.join(folder, 'swd_pytorch.csv') for folder in model_folders]
# model_numbers = [folder.split('_')[-1] for folder in model_folders]
# n_epochs = [os.path.split(folder)[-1].split('_')[0][len('EPOCHS'):] for folder in model_folders]
# additional_columns={'gan_type': ['uGAN']*10 + ['eGAN']*10,
#                     'model_number': model_numbers, 'n_epochs': n_epochs}
# df = swd.save_swds(csv_files=csv_files, save_folder=MODEL_PATH, save_filename='SWDs_100',
#                    additional_columns=additional_columns)

#################################
# Combining GSs of multiple GANs:
#################################
# base_folder_name = "EPOCHS100_BATCHS8_DISCDCGANDiscriminator_GENUpsampleGenerator_ACTrelu_UPMODEnearest_ND5_S1LAST_LOSSGANLoss_TYPEvanilla_LLOSS1.0_OPTDISCAdam_OPTGENAdam_"
# model_folders = [base_folder_name + str(i) for i in range(10)]
# model_folders = model_folders + [base_folder_name+ "EPSvonMises_" + str(i) for i in range(11)]
# model_folders = [os.path.join(MODEL_PATH, folder) for folder in model_folders]
# csv_files = [os.path.join(folder, 'geometry_score.csv') for folder in model_folders]
# model_numbers = [folder.split('_')[-1] for folder in model_folders]
# n_epochs = [os.path.split(folder)[-1].split('_')[0][len('EPOCHS'):] for folder in model_folders]
# additional_columns={'gan_type': ['uGAN']*10 + ['eGAN']*10, 'model_number': model_numbers, 'n_epochs': n_epochs}
# df = geometry_score.save_geom_scores(csv_files=csv_files, save_folder=MODEL_PATH,
#                                      save_filename='GSs_100',
#                                      additional_columns=additional_columns)

base_folder_name = "_".join(MODEL_FOLDER.split("_")[:-1]) + "_"
model_folders = [base_folder_name + str(i) for i in range(1)]
model_numbers = [folder.split("_")[-1] for folder in model_folders]
n_epochs = [
    os.path.split(folder)[-1].split("_")[0][len("EPOCHS") :] for folder in model_folders
]
additional_columns = {
    "gan_type": ["uGAN"] * 1,
    "model_number": model_numbers,
    "n_epochs": n_epochs,
}
df = geometry_score.save_geom_scores(
    csv_files=[os.path.join(MODEL_FOLDER, "geometry_score.csv")],
    save_folder=MODEL_PATH,
    save_filename="GSs",
    additional_columns=additional_columns,
)

###############
# Plot metrics:
###############
print("Plotting training statistics...")
plt_m.plot_statistics(
    [[mean_outputs_disc_real, mean_outputs_disc_fake], [mean_outputs_disc_fake_2]],
    [["real", "fake"], ["fake_2"]],
    save_folder=MODEL_FOLDER,
    filename="outputs",
)

plt_m.plot_statistics(
    [
        [mean_losses_disc, mean_losses_gen],
    ],
    [["mean disc loss", "mean gen loss"]],
    save_folder=MODEL_FOLDER,
    filename="losses",
)

print("Plotting Sliced Wasserstein Distance...")
swd_filename = "swd_pytorch"

plt_m.plot_swd_bar(
    swd_values=swd_values,
    names=swd_names,
    save_folder=MODEL_FOLDER,
    filename=swd_filename + "_Nodemaps_1",
)


print("Plotting Geometry Score...")
gan_names = {"uGAN": "$u$GAN", "eGAN": "$\epsilon$GAN", "real": "Real Data"}
gs_filename = "GSs"
plt_m.plot_geom_scores(
    os.path.join(MODEL_PATH, gs_filename + ".csv"),
    gan_names=gan_names,
    save_filename=gs_filename + "_Nodemaps_1",
)

# print("Plotting Combined Sliced Wasserstein Distance...")
# swd_filename = "SWDs_100"
# plt_m.plot_swds(
#     os.path.join(MODEL_FOLDER, swd_filename + ".csv"),
#     save_filename=swd_filename + "_Nodemaps_1",
# )

# print("Plotting Combined Geometry Score...")
# gan_names = {"uGAN": "$u$GAN", "eGAN": "$\epsilon$GAN", "real": "Real Data"}
# gs_filename = "GSs_100_w_mrlt_real_no_eGAN10"
# plt_m.plot_geom_scores(
#     os.path.join(MODEL_FOLDER, gs_filename + ".csv"),
#     gan_names=gan_names,
#     save_filename=gs_filename + "_Nodemaps_1",
# )
