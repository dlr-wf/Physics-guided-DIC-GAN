from warnings import warn
import torch
from torch import nn

##############################################################################
# GAN Discriminators
##############################################################################
class DCGANDiscriminator(nn.Module):
    """
        Discriminator following the guidelines of
        "Unsupervised representation learning with deep convolutional generative adversarial networks", Radford et al. (2015, arXiv:1511.06434).

        :param channels: number of input channels.
        :param filters: number of filters used in convolutional layers.
                        Number of filters start from channels to filters
                        to increasing multiples of filters in every convolutional layer.
        :param kernel_size: filter size in convolutional layers.
        :param layernorm: If True Layer Normalization is applied instead of Batch Normalization.
        :param stride_1_as_first_layer: If True the first layer is a convolution with stride 1.
        :param dropout: If True Dropout is applied instead of Batch Normalization.
    """
    def __init__(self, channels, filters=64, kernel_size=4,
                 layernorm=False, stride_1_as_first_layer=False, dropout=False):
        super().__init__()

        self.register_buffer('channels', torch.tensor([channels]))
        self.register_buffer('filters', torch.tensor([filters]))
        self.register_buffer('kernel_size', torch.tensor([kernel_size]))

        dropout_p = 0.1

        names = []
        if layernorm:
            names.append("LNORM")
        if stride_1_as_first_layer:
            names.append("S1FIRST")
        if dropout:
            names.append("DROP")
        self.name = '_'.join(names)

        self.phi = 0.0

        self.main = nn.Sequential(
            # Conv2d output:
            # padding = (kernel_size - stride) / 2 works only for stride 2

            # Conv2d layer with stride 1 in first layer, reduces quality,
            # maybe because discriminator is to strong then
            nn.Sequential(nn.LayerNorm([channels, 256, 256]) if layernorm
                          else nn.Dropout(p=dropout_p, inplace=True) if dropout
                          else nn.BatchNorm2d(channels),
                          nn.ReLU(inplace=True),
                          # padding = (kernel_size - 1) / 2   to get get input size as output size
                          nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5,
                                    stride=1, padding=2, bias=False),

                          # input is (channels) x 256 x 256
                          nn.Conv2d(in_channels=channels, out_channels=filters,
                                    kernel_size=kernel_size,
                                    stride=2, padding=(kernel_size - 2) // 2, dilation=1,
                                    bias=False)) if stride_1_as_first_layer
            else (# input is (channels) x 256 x 256
                nn.Conv2d(in_channels=channels,
                          out_channels=filters,
                          kernel_size=kernel_size,
                          stride=2,
                          padding=(kernel_size-2)//2,
                          dilation=1, bias=False))

            , nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters) x 128 x 128

            nn.Conv2d(in_channels=filters, out_channels=filters * 2, kernel_size=kernel_size,
                      stride=2, padding=(kernel_size-2)//2, dilation=1, bias=False),
            nn.LayerNorm([filters * 2, 64, 64]) if layernorm
            else nn.Dropout(p=dropout_p, inplace=True) if dropout else nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters*2) x 64 x 64

            nn.Conv2d(in_channels=filters * 2, out_channels=filters * 4, kernel_size=kernel_size,
                      stride=2, padding=(kernel_size-2)//2, dilation=1, bias=False),
            nn.LayerNorm([filters * 4, 32, 32]) if layernorm
            else nn.Dropout(p=dropout_p, inplace=True) if dropout else nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters*4) x 32 x 32

            nn.Conv2d(in_channels=filters * 4, out_channels=1, kernel_size=5,
                      stride=1, padding=2, dilation=1, bias=False),
            # nn.BatchNorm2d(1), # This BatchNorm layer delays learning for FashionMNIST
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 32 x 32

            nn.Flatten()
        )

        # For BCE Loss
        # self.last = nn.Sequential(nn.Linear(in_features=32 * 32, out_features=1, bias=True),
        #                           nn.Sigmoid())

        # For BCEWithLogitsLoss
        self.last = nn.Sequential(nn.Linear(in_features=32 * 32, out_features=1, bias=True))

    def forward(self, x):
        self.phi = self.main(x)
        return self.last(self.phi)

    def get_name(self):
        """
            Returns name of the discriminator object. This name is used to save the discriminator.

            :return: Name of the discriminator
        """
        return self.name

    @classmethod
    def from_name_and_state_dict(cls, name, state_dict):
        """
            Creates discriminator object from save folder name and state_dict.

            :param name: Save folder name
            :param state_dict: state_dict of the saved discriminator
            :return: Discriminator object
        """
        layernorm = False
        stride_1_as_first_layer = False
        dropout = False

        parts = name.split('_')
        for part in parts:
            if part.startswith('LNORM'):
                layernorm = True
            elif part.startswith('S1FIRST'):
                stride_1_as_first_layer = True
            elif part.startswith('DROP'):
                dropout = True

        try:
            channels = state_dict['channels'].item()
        except KeyError:
            channels = 2
            if name.find('VonMises') != -1:
                channels += 1
            if name.find('WithStrains') != -1:
                channels += 3
            state_dict['channels'] = torch.tensor([channels])
            warn(f"Channels not found in state_dict! Setting value to {channels}.")
            channels = state_dict['channels'].item()
        try:
            filters = state_dict['filters'].item()
        except KeyError:
            warn("filters not found in state_dict! Setting value to 64.")
            state_dict['filters'] = torch.tensor([64])
            filters = state_dict['filters'].item()
        try:
            kernel_size = state_dict['kernel_size'].item()
        except KeyError:
            warn("kernel_size not found in state_dict! Setting value to 4.")
            state_dict['kernel_size'] = torch.tensor([4])
            kernel_size = state_dict['kernel_size'].item()

        disc = cls(channels=channels, filters=filters, kernel_size=kernel_size,
                   layernorm=layernorm, stride_1_as_first_layer=stride_1_as_first_layer,
                   dropout=dropout)
        disc.load_state_dict(state_dict)
        return disc
