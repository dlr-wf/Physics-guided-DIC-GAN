import torch

seed = 48  # Seed used for random number generators

# ----------------- Training and dataset parameters -----------------
epochs = 2  # Number of training epochs
checkpoint_rate = 1  # Every checkpoint_rate number of training epochs a model checkpoint is saved. If 0 no checkpoints are saved.
batch_size = 8  # Batch size used in the data loader
lambda_loss = 1.0  # Weighting factor the adversarial loss is mutliplied before an optimization step
lambda_reg = 0.0  # 0.2  # Weighting factor the regularization loss is mutliplied before an optimization step
amp = False  # Whether to use PyTorch automatic mixed precission
workers = 2  # Number of PyTorch dataloader workers
scaling = "minmax"  # Scaling type used to scale training data
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # Device used for training

# ----------------- Generator parameters -----------------
channels_gen = 2  # Number of output channels
filters_gen = 64  # Number of filters in the Conv2D layers
noise_dim = 5  # Dimension of the input noise
kernel_size_gen = 5  # Kernel size used in the Conv2D layers
upsample_mode = "nearest"  # Upsampling mode used in the upsampling layers
activation = "relu"  # Used activation function
layernorm_gen = False  # Whether to use layer normalization
stride_1_as_last_layer = (
    True  # Whether to use a Conv2D layer with stride set to 1 as the last layer
)
more_conv2d_in_last_layer = (
    False  # Whether to use extra Conv2D layers as the last layers
)
no_linear_layer = False  # Whether to use no linear layers for mapping the input noise to a higher dimension

# ----------------- Discriminator parameters -----------------
channels_disc = 2  # Number of input channels
filters_disc = 64  # Number of filters in the Conv2D layers
kernel_size_disc = 4  # Kernel size used in the Conv2D layers
layernorm_disc = False  # Whether to use layer normalization
stride_1_as_first_layer = (
    False  # Whether to use a Conv2D layer with stride set to 1 as the first layer
)
dropout = False  # Whether to use dropout layers

# ----------------- GAN parameters -----------------
gan_type = "vanilla"  # Used GAN type. Defines the used loss function
reg_type = None  # 'gp-real'  # Used regularization type.
real_label = 1.0  # Value used as label for real data
fake_label = 0.0  # Value used as label for fake data
strains = None  # Strains calculated in the training process. Has to agree with the number of input channels in the discriminator.

# ----------------- Optimizer parameters -----------------
lr_gen = 0.0002  # Learning rate used for training the generator
lr_disc = 0.0002  # Learning rate used for training the discriminator
beta1 = 0.5  # Beta1 used for the Adam optimizer
beta2 = 0.999  # Beta2 used for the Adam optimizer

# ----------------- Weight initialization parameters -----------------
# Parameters used as in "Unsupervised representation learning with deep convolutional generative adversarial networks", Radford et al. (2015, arXiv:1511.06434)
init_type = "normal"  # Used method for initializing network parameters
init_gain = 0.02  # Used gain for initializing network parameters
