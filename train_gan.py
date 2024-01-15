import os
import time
from pathlib import Path
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
import configs.gan as conf
from datasets.displacement_datasets import DisplacementDataset
from data_processing.strain_calc import calc_and_concat_strains
from gan import gans, generators, discriminators

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
SAVE_FOLDER = os.path.join("models", "GANs")

Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

####################
# Define dataloader:
####################
print("Defining dataloader...")
dataset = DisplacementDataset([DATA_PATH], scaling=conf.scaling)
dataloader = DataLoader(
    dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.workers
)

###################
# Define generator:
###################
print("Defining generator...")
generator = generators.UpsampleGenerator(
    channels=conf.channels_gen,
    filters=conf.filters_gen,
    noise_dim=conf.noise_dim,
    kernel_size=conf.kernel_size_gen,
    upsample_mode=conf.upsample_mode,
    activation=conf.activation,
    layernorm=conf.layernorm_gen,
    stride_1_as_last_layer=conf.stride_1_as_last_layer,
    more_conv2d_in_last_layer=conf.more_conv2d_in_last_layer,
    no_linear_layer=conf.no_linear_layer,
)

#######################
# Define discriminator:
#######################
print("Defining discriminator...")
discriminator = discriminators.DCGANDiscriminator(
    channels=conf.channels_disc,
    filters=conf.filters_disc,
    kernel_size=conf.kernel_size_disc,
    layernorm=conf.layernorm_disc,
    stride_1_as_first_layer=conf.stride_1_as_first_layer,
    dropout=conf.dropout,
)

##############
# Plot models:
##############
fixed_noise = torch.randn(16, conf.noise_dim, device=conf.device)
real_batch = next(iter(dataloader))
real_batch = calc_and_concat_strains(real_batch, conf.strains)
assert real_batch.shape[1] == conf.channels_disc, (
    f"Number of input channels ({conf.channels_disc}) has to "
    "match specified strain type ({conf.strains}), "
    f"but batch of real data after concataneting specified strains has {real_batch.shape[1]} channels!"
)
print("--------------GENERATOR--------------")
summary(
    model=generator,
    input_size=fixed_noise.size(),
    depth=5,
    verbose=1,
    device=conf.device.type,
)
print("\n\n\n--------------DISCRIMINATOR--------------")
summary(
    model=discriminator,
    input_size=real_batch.size(),
    depth=5,
    verbose=1,
    device=conf.device.type,
)

#############
# Define GAN:
#############
gan_loss = gans.GANLoss(
    gan_type=conf.gan_type,
    reg_type=conf.reg_type,
    lambda_loss=conf.lambda_loss,
    lambda_reg=conf.lambda_reg,
    real_label=conf.real_label,
    fake_label=conf.fake_label,
)
optimizer_gen = optim.Adam(
    generator.parameters(), lr=conf.lr_gen, betas=(conf.beta1, conf.beta2)
)
optimizer_disc = optim.Adam(
    discriminator.parameters(), lr=conf.lr_disc, betas=(conf.beta1, conf.beta2)
)
gan = gans.GAN(
    discriminator,
    generator,
    gan_loss,
    optimizer_gen,
    optimizer_disc,
    conf.device,
    conf.amp,
    conf.strains,
)
gans.init_weights(generator, conf.init_type, conf.init_gain)
gans.init_weights(discriminator, conf.init_type, conf.init_gain)

############
# Train GAN:
############
print("Training GAN...")
start = time.time()
gan.train(
    dataloader,
    conf.epochs,
    checkpoint_rate=conf.checkpoint_rate,
    save_folder=SAVE_FOLDER,
)
print(f"Finished. Training took {(time.time()-start)/60} minutes")
