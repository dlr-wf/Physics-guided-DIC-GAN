import torch
from torch import nn
from gan.layers import Reshape


##############################################################################
# GAN Generators
##############################################################################
class DCGANGenerator(nn.Module):
    """
        Generator following the guidelines of
        "Unsupervised representation learning with deep convolutional generative adversarial networks", Radford et al. (2015, arXiv:1511.06434).

        :param channels: number of output channels.
        :param filters: number of filters used in convolutional layers.
                        Number of filters start from channels to filters
                        to increasing multiples of filters in every convolutional layer.
        :param noise_dim: Input size for the Generator.
        :param kernel_size: filter size in convolutional layers.
        :param layernorm: If True Layer Normalization is applied instead of Batch Normalization.
        :param stride_1_as_last_layer: If True the last layer is a convolution with stride 1.
    """

    def __init__(self, channels, filters=64, noise_dim=100, kernel_size=4,
                 layernorm=False, stride_1_as_last_layer=False):
        super().__init__()

        self.register_buffer('channels', torch.tensor([channels]))
        self.register_buffer('filters', torch.tensor([filters]))
        self.register_buffer('noise_dim', torch.tensor([noise_dim]))
        self.register_buffer('kernel_size', torch.tensor([kernel_size]))

        self.name = f"LNORM{layernorm}_S1LAST{stride_1_as_last_layer}"

        # To guarantee that ConvTranspose2d doubles input size
        # kernel_size MUST be multiple of 2 and
        # padding = (kernel_size - stride) / 2   must be an integer
        # see pytorch documentation of ConvTranspose2d
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(in_features=noise_dim, out_features=filters * 8 * 8 * 8, bias=False),
            nn.LayerNorm(filters * 8 * 8 * 8) if layernorm else nn.BatchNorm1d(filters * 8 * 8 * 8),
            nn.ReLU(inplace=True),
            # state size. filters*8 * 8 * 8
            Reshape(dim=(-1, filters * 8, 8, 8)),
            # state size. (filters*8) x 8 x 8

            nn.ConvTranspose2d(in_channels=filters * 8, out_channels=filters * 4, kernel_size=5,
                               stride=1, padding=2, bias=False),
            nn.LayerNorm([filters * 4, 8, 8]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True),
            # state size. (filters*4) x 8 x 8

            nn.ConvTranspose2d(in_channels=filters * 4, out_channels=filters * 4,
                               kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 2) // 2, bias=False),
            nn.LayerNorm([filters * 4, 16, 16]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True),
            # state size. (filters*4) x 16 x 16

            nn.ConvTranspose2d(in_channels=filters * 4, out_channels=filters * 4,
                               kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 2) // 2, bias=False),
            nn.LayerNorm([filters * 4, 32, 32]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True),
            # state size. (filters*4) x 32 x 32

            nn.ConvTranspose2d(in_channels=filters * 4, out_channels=filters * 2,
                               kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 2) // 2, bias=False),
            nn.LayerNorm([filters * 2, 64, 64]) if layernorm else nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True),
            # state size. (filters*2) x 64 x 64

            nn.ConvTranspose2d(in_channels=filters * 2, out_channels=filters * 2,
                               kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 2) // 2, bias=False),
            nn.LayerNorm([filters * 2, 128, 128]) if layernorm else nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True),
            # state size. (filters*2) x 128 x 128

            nn.ConvTranspose2d(in_channels=filters * 2, out_channels=channels,
                               kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 2) // 2, bias=False),
            # state size. (CHANNELS) x 256 x 256

            nn.Sequential(nn.LayerNorm([channels, 256, 256]) if layernorm
                          else nn.BatchNorm2d(channels),
                          nn.ReLU(inplace=True),
                          # padding = (kernel_size - 1) / 2   to get get input size as output size
                          nn.ConvTranspose2d(in_channels=channels, out_channels=channels,
                                             kernel_size=5,
                                             stride=1, padding=2, bias=False),
                          nn.Tanh()) if stride_1_as_last_layer else (nn.Tanh())
        )

    def forward(self, x):
        return self.main(x)

    def get_name(self):
        """
            Returns name of the generator object. This name is used to save the generator.

            :return: Name of the generator
        """
        return self.name

    @classmethod
    def from_name_and_state_dict(cls, name, state_dict):
        """
            Creates generator object from save folder name and state_dict.

            :param name: Save folder name
            :param state_dict: state_dict of the saved generator
            :return: Generator object
        """
        layernorm = False
        stride_1_as_last_layer = False

        parts = name.split('_')
        for part in parts:
            if part.startswith('LNORM'):
                layernorm = True
            elif part.startswith('S1LAST'):
                stride_1_as_last_layer = True

        channels = state_dict['channels'].item()
        filters = state_dict['filters'].item()
        noise_dim = state_dict['noise_dim'].item()
        kernel_size = state_dict['kernel_size'].item()

        gen = cls(channels=channels, filters=filters, noise_dim=noise_dim, kernel_size=kernel_size,
                  layernorm=layernorm, stride_1_as_last_layer=stride_1_as_last_layer)
        gen.load_state_dict(state_dict)
        gen.eval()
        return gen


class UpsampleGenerator(nn.Module):
    """
        DCGAN Generator with Upsampling and Convolutional layers instead of
        Transposed Convolutional layers
        This Generator created the most stable results and was used most of the time
        for generating DIC-data.

        :param channels: number of channels in the input
        :param filters: number of filters used in convolutional layers.
                        Number of filters start from channels to filters
                        to increasing multiples of filters in every convolutional layer.
        :param noise_dim: Input size for the Generator.
        :param kernel_size: filter size in convolutional layers.
        :param activation: Activation function used.
                        Can be one of ['relu', 'prelu', 'swish', 'silu', 'elu', 'LeakyReLU'].
                        ELU activation seems to reduce mode collapse im comparison to relu,
                        which might be caused by "dead ReLUs".
        :param upsample_mode: Used method of upsampling. Can be one of ['nearest', 'linear', bilinear'].
        :param layernorm: If True Layer Normalization is applied instead of Batch Normalization.
        :param stride_1_as_last_layer: If True the last layer is a convolution with stride 1.
        :param more_conv2d_in_last_layer: If True the more convolutional layers with
                                        stride 1 are added in the last layers.
    """

    def __init__(self, channels, filters=64, noise_dim=100, kernel_size=5,
                 activation='relu',
                 upsample_mode='nearest', layernorm=False, stride_1_as_last_layer=True,
                 more_conv2d_in_last_layer=False,
                 no_linear_layer=False):
        super().__init__()

        self.register_buffer('channels', torch.tensor([channels]))
        self.register_buffer('filters', torch.tensor([filters]))
        self.register_buffer('noise_dim', torch.tensor([noise_dim]))
        self.register_buffer('kernel_size', torch.tensor([kernel_size]))

        supported_activations = ['relu', 'prelu', 'swish', 'silu', 'elu', 'leakyrelu']
        if activation not in supported_activations:
            raise ValueError(f"Activation has to be in {supported_activations}"
                             f"but is {activation}.")
        if activation == 'swish':
            activation = 'silu'

        self.no_linear_layer = no_linear_layer

        names = [f"ACT{activation}", f"UPMODE{upsample_mode}", f"ND{noise_dim}"]
        if layernorm:
            names.append("LNORM")
        if stride_1_as_last_layer:
            names.append("S1LAST")
        if more_conv2d_in_last_layer:
            names.append("_MORECONV")
        if no_linear_layer:
            names.append("NOLIN")
        self.name = '_'.join(names)

        # Conv2d output size same as input size:
        # padding = (kernel_size - stride) / 2 works only for stride 1
        # see pytorch documentation of ConvTranspose2d
        self.main = nn.Sequential(
            nn.Sequential(
                # input is Z, going into a convolution
                nn.Linear(in_features=noise_dim, out_features=filters * 8 * 8 * 8, bias=False),
                nn.LayerNorm(filters * 8 * 8 * 8) if layernorm else nn.BatchNorm1d(filters * 8 * 8 * 8),
                nn.ReLU(inplace=True) if activation == 'relu'
                else nn.PReLU() if activation == 'prelu'
                else nn.ELU() if activation == 'elu'
                else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
                else nn.SiLU(inplace=True),
                # state size. filters*8 * 8 * 8
                Reshape(dim=(-1, filters * 8, 8, 8))
                # state size. (filters*8) x 8 x 8
            ) if not no_linear_layer else
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=noise_dim, out_channels=filters * 8, kernel_size=4,
                                   stride=1, padding=0, bias=False),
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                nn.LayerNorm([filters * 8, 16, 16]) if layernorm else nn.BatchNorm2d(filters * 8),
                nn.ReLU(inplace=True) if activation == 'relu'
                else nn.PReLU() if activation == 'prelu'
                else nn.ELU() if activation == 'elu'
                else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
                else nn.SiLU(inplace=True),
                # state size. (filters*8) x 8 x 8
            ),
            # state size. (filters*8) x 8 x 8

            nn.Conv2d(in_channels=filters * 8, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.LayerNorm([filters * 4, 8, 8]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.ELU() if activation == 'elu'
            else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*4) x 8 x 8

            nn.Conv2d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),

            nn.LayerNorm([filters * 4, 16, 16]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.ELU() if activation == 'elu'
            else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*4) x 16 x 16

            nn.Conv2d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.LayerNorm([filters * 4, 32, 32]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.ELU() if activation == 'elu'
            else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*4) x 32 x 32

            nn.Conv2d(in_channels=filters * 4, out_channels=filters * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.LayerNorm([filters * 2, 64, 64]) if layernorm else nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.ELU() if activation == 'elu'
            else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*2) x 64 x 64

            nn.Conv2d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.LayerNorm([filters * 2, 128, 128]) if layernorm else nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.ELU() if activation == 'elu'
            else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*2) x 128 x 128

            nn.Sequential(
                nn.Conv2d(in_channels=filters * 2, out_channels=filters,
                          kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                # state size. (CHANNELS) x 256 x 256

                nn.LayerNorm([filters, 256, 256]) if layernorm
                else nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True) if activation == 'relu'
                else nn.PReLU() if activation == 'prelu'
                else nn.ELU() if activation == 'elu'
                else nn.LeakyReLU(0.01) if activation == 'leakyrelu'
                else nn.SiLU(inplace=True),
                # padding = (kernel_size - 1) / 2   to get get input size as output size
                nn.Conv2d(in_channels=filters, out_channels=channels,
                          kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2, bias=False),

                nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels,
                                        kernel_size=kernel_size,
                                        stride=1, padding=(kernel_size - 1) // 2, bias=False),
                              nn.Tanh()) if more_conv2d_in_last_layer
                else (nn.Tanh())) if stride_1_as_last_layer else
            nn.Sequential(
                nn.Conv2d(in_channels=filters * 2, out_channels=channels, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                # state size. (CHANNELS) x 256 x 256
                nn.Tanh())
        )

    def forward(self, x):
        # if self.no_linear_layer:
        #     x = x.unsqueeze(2).unsqueeze(3)
        return self.main(x)

    def get_name(self):
        """
            Returns name of the generator object. This name is used to save the generator.

            :return: Name of the generator
        """
        return self.name

    @classmethod
    def from_name_and_state_dict(cls, name, state_dict):
        """
            Creates generator object from save folder name and state_dict.

            :param name: Save folder name
            :param state_dict: state_dict of the saved generator
            :return: Generator object
        """
        activation = None
        upsample_mode = None
        layernorm = False
        stride_1_as_last_layer = False
        more_conv2d_in_last_layer = False
        no_linear_layer = False

        parts = name.split('_')
        for part in parts:
            if part.startswith('ACT'):
                activation = part[3:]
            elif part.startswith('UPMODE'):
                upsample_mode = part[6:]
            elif part.startswith('LNORM'):
                layernorm = True
            elif part.startswith('S1LAST'):
                stride_1_as_last_layer = True
            elif part.startswith('MORECONV'):
                more_conv2d_in_last_layer = True
            elif part.startswith('NOLIN'):
                no_linear_layer = True
        assert activation is not None, "Activation not specified!"
        assert upsample_mode is not None, "Upsampling mode not specified!"

        channels = state_dict['channels'].item()
        filters = state_dict['filters'].item()
        noise_dim = state_dict['noise_dim'].item()
        kernel_size = state_dict['kernel_size'].item()

        gen = cls(channels=channels, filters=filters, noise_dim=noise_dim, kernel_size=kernel_size,
                  activation=activation, upsample_mode=upsample_mode, layernorm=layernorm,
                  stride_1_as_last_layer=stride_1_as_last_layer, more_conv2d_in_last_layer=more_conv2d_in_last_layer,
                  no_linear_layer=no_linear_layer)
        gen.load_state_dict(state_dict)
        gen.eval()
        return gen


class MoreConvUpsampleGenerator(nn.Module):
    """
        DCGAN Generator with Upsampling and Convolutional layers instead of
        Transposed Convolutional layers. In addition to the Upsample Generator,
        this generator uses convolutional layers before AND after each upsampling layer.

        This generator showed very good quality samples with no gargabe samples, but with low variation.
        This generator was not exensively tested.

        :param channels: number of channels in the input
        :param filters: number of filters used in convolutional layers.
                        Number of filters start from channels to filters
                        to increasing multiples of filters in every convolutional layer.
        :param noise_dim: Input size for the Generator.
        :param kernel_size: filter size in convolutional layers.
        :param activation: Activation function used.
                        Can be one of ['relu', 'prelu', 'swish', 'silu', 'elu', 'LeakyReLU'].
                        ELU activation seems to reduce mode collapse im comparison to relu,
                        which might be caused by "dead ReLUs".
        :param upsample_mode: Used method of upsampling. Can be one of ['nearest', 'linear', bilinear'].
        :param layernorm: If True Layer Normalization is applied instead of Batch Normalization.
    """

    def __init__(self, channels, filters=64, noise_dim=100, kernel_size=5,
                 activation='relu',
                 upsample_mode='nearest', layernorm=False):
        super().__init__()

        self.register_buffer('channels', torch.tensor([channels]))
        self.register_buffer('filters', torch.tensor([filters]))
        self.register_buffer('noise_dim', torch.tensor([noise_dim]))
        self.register_buffer('kernel_size', torch.tensor([kernel_size]))

        if activation not in ['relu', 'prelu', 'swish', 'silu']:
            raise ValueError(f"Activation has to be relu, prelu or swish/silu but is {activation}.")
        if activation == 'swish':
            activation = 'silu'

        self.name = f"ACT{activation}_UPMODE{upsample_mode}_LNORM{layernorm}"

        # Conv2d output size same as input size:
        # padding = (kernel_size - stride) / 2 works only for stride 1
        # see pytorch documentation of ConvTranspose2d
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(in_features=noise_dim, out_features=filters * 16 * 8 * 8, bias=False),
            nn.LayerNorm(filters * 16 * 8 * 8) if layernorm else nn.BatchNorm1d(filters * 16 * 8 * 8),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.SiLU(inplace=True),
            # state size. filters*16 * 8 * 8
            Reshape(dim=(-1, filters * 16, 8, 8)),
            # state size. (filters*16) x 8 x 8

            nn.Conv2d(in_channels=filters * 16, out_channels=filters * 16, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(in_channels=filters * 16, out_channels=filters * 8, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),

            nn.LayerNorm([filters * 8, 8, 8]) if layernorm else nn.BatchNorm2d(filters * 8),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*8) x 16 x 16

            nn.Conv2d(in_channels=filters * 8, out_channels=filters * 8, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(in_channels=filters * 8, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),

            nn.LayerNorm([filters * 4, 16, 16]) if layernorm else nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*4) x 32 x 32

            nn.Conv2d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(in_channels=filters * 4, out_channels=filters * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),

            nn.LayerNorm([filters * 2, 32, 32]) if layernorm else nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*2) x 64 x 64

            nn.Conv2d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(in_channels=filters * 2, out_channels=filters * 1, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),

            nn.LayerNorm([filters * 1, 64, 64]) if layernorm else nn.BatchNorm2d(filters * 1),
            nn.ReLU(inplace=True) if activation == 'relu'
            else nn.PReLU() if activation == 'prelu'
            else nn.SiLU(inplace=True),
            # state size. (filters*1) x 128 x 128

            nn.Conv2d(in_channels=filters * 1, out_channels=filters * 1, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            # nn.Conv2d(in_channels=filters * 2, out_channels=filters, kernel_size=kernel_size,
            #           stride=1, padding=(kernel_size-1)//2, bias=False),
            #
            #
            # nn.LayerNorm([filters, 256, 256]) if layernorm else nn.BatchNorm2d(filters),
            # nn.ReLU(inplace=True) if activation == 'relu'
            # else nn.PReLU() if activation == 'prelu'
            # else nn.SiLU(inplace=True),
            # # state size. (filters) x 256 x 256

            nn.Conv2d(in_channels=filters, out_channels=channels, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            # state size. (channels) x 256 x 256
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def get_name(self):
        """
            Returns name of the generator object. This name is used to save the generator.

            :return: Name of the generator
        """
        return self.name

    @classmethod
    def from_name_and_state_dict(cls, name, state_dict):
        """
            Creates generator object from save folder name and state_dict.

            :param name: Save folder name
            :param state_dict: state_dict of the saved generator
            :return: Generator object
        """
        activation = None
        upsample_mode = None
        layernorm = False

        parts = name.split('_')
        for part in parts:
            if part.startswith('ACT'):
                activation = part[3:]
            elif part.startswith('UPMODE'):
                upsample_mode = part[6:]
            elif part.startswith('LNORM'):
                layernorm = True
        assert activation is not None, "Activation not specified!"
        assert upsample_mode is not None, "Upsampling mode not specified!"

        channels = state_dict['channels'].item()
        filters = state_dict['filters'].item()
        noise_dim = state_dict['noise_dim'].item()
        kernel_size = state_dict['kernel_size'].item()

        gen = cls(channels=channels, filters=filters, noise_dim=noise_dim, kernel_size=kernel_size,
                  activation=activation, upsample_mode=upsample_mode, layernorm=layernorm)
        gen.load_state_dict(state_dict)
        gen.eval()
        return gen
