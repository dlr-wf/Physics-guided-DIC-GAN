from collections import OrderedDict
import contextlib
import os
from pathlib import Path
import sys
from warnings import warn
import torch
from torch import nn, autograd
from data_processing.strain_calc import calc_and_concat_strains
from gan import discriminators, generators


def compute_grad2(d_out, x_in):
    """
    Calculates sum of squared gradients with respect to an input.
    Source: "Which training methods for GANs do actually converge?", Mescheder et al. (2018, arXiv:1801.04406).

    :param d_out: Output of the discriminator
    :param x_in: Batch of input data of the discriminator
    :return: Sum of squared gradients
    """
    batch_size = x_in.shape[0]
    grad_dout = autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    # grad_dout2 = (grad_dout.view(grad_dout.shape[0], -1).norm(2, dim=1) ** 2).mean()
    assert grad_dout2.size() == x_in.size()
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def reg_wgan_gp(disc, x_real, x_fake, center):
    """
    Calculates and returns regularization term of the Wasserstein-GAN gradient penalty.
    Source: "Which training methods for GANs do actually converge?", Mescheder et al. (2018, arXiv:1801.04406).

    :param disc: Discriminator model
    :param x_real: Batch of real data on which the discriminator is trained
    :param x_fake: Batch of fake data on which the discriminator is trained
    :param center: Center of the gradient penalty.
                   "Improved Training of Wasserstein GANs", Gulrajani et al. (2017, arXiv:1704.00028v3) uses 1,
                   "Which training methods for GANs do actually converge?", Mescheder et al. (2018, arXiv:1801.04406) suggests 0.
    :return: Regularization term
    """
    assert x_real.shape[0] == x_fake.shape[0]
    batch_size = x_fake.shape[0]
    eps = torch.rand(batch_size, device=x_fake.device).view(batch_size, 1, 1, 1)
    x_interp = (1 - eps) * x_real + eps * x_fake
    x_interp = x_interp.detach()
    x_interp.requires_grad_()
    d_out = disc(x_interp)

    reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

    return reg


def init_weights(model, init_type: str = "normal", init_gain: float = 0.02):
    """
    Initializes network weights.
    Default parameters are adapted from "Unsupervised representation learning with deep convolutional generative adversarial networks", Radford et al. (2015, arXiv:1511.06434).
    Source: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", Zhu et al. (2017, arXiv:1703.10593).

    :param model: The network which weights are to be initialized
    :param init_type: Initialization method: normal | xavier | kaiming | orthogonal
    :param init_gain: Scaling factor for normal, xavier and orthogonal
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method {init_type} is not implemented!"
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find("BatchNorm2d") != -1 or classname.find("BatchNorm1d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.zeros_(m.bias.data)

    print(f"Initializing model with {init_type}!")
    model.apply(init_func)


def load_gan_model_by_folder(
    folder_path, gan_state_dict=None, device=torch.device("cpu"), old_format=False
):
    # TODO: Remove old_format parameter if not needed anymore
    """
    Creates and returns GAN object from saved GAN.

    :param folder_path: Path to the saved GAN
    :param gan_state_dict: The saved state_dict of the GAN. If None, os.path.join(folder_path, 'gan_state_dict.pt') is used.
    :param device: Device on which the GAN will be saved
    """
    folder = folder_path.split(os.sep)[-1]

    if gan_state_dict is None and old_format is False:
        gan_state_dict = torch.load(os.path.join(folder_path, "gan_state_dict.pt"))
    # if old_format:
    #     assert gan_state_dict is None, "GAN state_dict cannot be specified if old_format is True!"

    n_epochs = None
    optimizer_disc_name = None
    optimizer_gen_name = None
    disc_name = None
    gen_name = None
    loss_name = None

    # -------------------------- Parsing Old Format --------------------------
    if old_format:
        generator_state_dict_tmp = torch.load(
            os.path.join(folder_path, "generator_state_dict.pt")
        )
        discriminator_state_dict_tmp = torch.load(
            os.path.join(folder_path, "discriminator_state_dict.pt")
        )
        generator_state_dict = OrderedDict()
        discriminator_state_dict = OrderedDict()
        reformat_gen_state_dict = False
        for k, v in generator_state_dict_tmp.items():
            name = k
            if k.startswith("module."):
                name = k[7:]  # remove 'module.'
            if name == "main.0.weight":
                reformat_gen_state_dict = True
            generator_state_dict[name] = v
        if reformat_gen_state_dict:
            generator_state_dict_tmp = generator_state_dict
            generator_state_dict = {}
            generator_state_dict["main.0.0.weight"] = generator_state_dict_tmp[
                "main.0.weight"
            ]
            generator_state_dict["main.0.1.weight"] = generator_state_dict_tmp[
                "main.1.weight"
            ]
            generator_state_dict["main.0.1.bias"] = generator_state_dict_tmp[
                "main.1.bias"
            ]
            generator_state_dict["main.0.1.running_mean"] = generator_state_dict_tmp[
                "main.1.running_mean"
            ]
            generator_state_dict["main.0.1.running_var"] = generator_state_dict_tmp[
                "main.1.running_var"
            ]
            generator_state_dict[
                "main.0.1.num_batches_tracked"
            ] = generator_state_dict_tmp["main.1.num_batches_tracked"]
            for k, v in generator_state_dict_tmp.items():
                layer = int(k.split(".")[1])
                if layer >= 4:
                    name = k.split(".")
                    name[1] = str(layer - 3)
                    name = ".".join(name)
                    generator_state_dict[name] = generator_state_dict_tmp[k]

        for k, v in discriminator_state_dict_tmp.items():
            name = k
            if k.startswith("module."):
                name = k[7:]  # remove 'module.'
            discriminator_state_dict[name] = v

        loss_name = "GANLoss"
        disc_name = "DCGANDiscriminator"
        channels_disc = 2
        noise_dim = None
        lr = None
        beta1 = None
        beta2 = None
        activation = None
        up_mode = None
        kernel_size_disc = None
        kernel_size_gen = None
        filters_disc = None
        filters_gen = None
        stride_1_as_last_layer = False
        withStrains = False
        vonMises = False

        for part in folder.split("_"):
            if part.startswith("E"):
                n_epochs = int(part[1:])
            elif part.startswith("LR"):
                lr = float(part[2:])
            elif part.startswith("B1"):
                beta1 = float(part[2:])
            elif part.startswith("B2"):
                beta2 = float(part[2:])
            elif part.startswith("UPSAMPLEGen"):
                gen_name = "UpsampleGenerator"
                if part.find("relu") != -1:
                    activation = "relu"
                if part.find("nearest") != -1:
                    up_mode = "nearest"
            elif part.startswith("ND"):
                noise_dim = int(part[2:])
            elif part.startswith("KSD"):
                kernel_size_disc = int(part[3:])
            elif part.startswith("KSG"):
                kernel_size_gen = int(part[3:])
            elif part.startswith("s1LastLayer"):
                stride_1_as_last_layer = True
            elif part.startswith("FD"):
                filters_disc = int(part[2:])
            elif part.startswith("FG"):
                filters_gen = int(part[2:])
            elif part.startswith("WithStrains"):
                withStrains = True
            elif part.startswith("VonMises"):
                vonMises = True
        assert n_epochs is not None, "Number of epochs not specified!"
        assert lr is not None, "Learning rate not specified!"
        assert beta1 is not None, "Beta 1 not specified!"
        assert beta2 is not None, "Beta 2 not specified!"
        assert activation is not None, "Activation not specified!"
        assert up_mode is not None, "Upsampling mode not specified!"
        assert (
            gen_name is not None
        ), "Generator class not specified! Only UpsampleGenerator supported!"
        assert noise_dim is not None, "Noise dimension not specified!"
        assert kernel_size_disc is not None, "Discriminator kernel size not specified!"
        assert kernel_size_gen is not None, "Generator kernel size not specified!"
        assert (
            filters_disc is not None
        ), "Discriminator number of filters not specified!"
        assert filters_gen is not None, "Generator number of filters not specified!"

        strains = None
        if withStrains and vonMises:
            strains = "strains_vonMises"
            channels_disc = 6
        elif withStrains:
            strains = "strains"
            channels_disc = 5
        elif vonMises:
            strains = "vonMises"
            channels_disc = 3

        discriminator = discriminators.DCGANDiscriminator(
            channels=channels_disc, filters=filters_disc, kernel_size=kernel_size_disc
        )
        discriminator_state_dict["channels"] = torch.tensor([channels_disc])
        discriminator_state_dict["filters"] = torch.tensor([filters_disc])
        discriminator_state_dict["kernel_size"] = torch.tensor([kernel_size_disc])
        discriminator.load_state_dict(discriminator_state_dict)
        discriminator.eval()
        generator = generators.UpsampleGenerator(
            channels=2,
            filters=filters_gen,
            noise_dim=noise_dim,
            kernel_size=kernel_size_gen,
            activation=activation,
            upsample_mode="nearest",
            stride_1_as_last_layer=stride_1_as_last_layer,
        )
        generator_state_dict["channels"] = torch.tensor([2])
        generator_state_dict["filters"] = torch.tensor([filters_gen])
        generator_state_dict["noise_dim"] = torch.tensor([noise_dim])
        generator_state_dict["kernel_size"] = torch.tensor([kernel_size_gen])
        generator.load_state_dict(generator_state_dict)
        generator.eval()

        optimizer_disc = torch.optim.Adam(
            discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        optimizer_gen = torch.optim.Adam(
            generator.parameters(), lr=lr, betas=(beta1, beta2)
        )

        loss = GANLoss(gan_type="vanilla", reg_type=None)

        new_gan = GAN(
            discriminator=discriminator,
            generator=generator,
            gan_loss=loss,
            optimizer_disc=optimizer_disc,
            optimizer_gen=optimizer_gen,
            device=device,
            amp=False,
            strains=strains,
        )
        new_gan.n_epochs = n_epochs
        return new_gan

    # -------------------------- Parsing New Format --------------------------
    for part in folder.split("_"):
        if part.startswith("EPOCHS"):
            n_epochs = int(part[6:])
        elif part.startswith("OPTDISC"):
            optimizer_disc_name = part[7:]
        elif part.startswith("OPTGEN"):
            optimizer_gen_name = part[6:]
        elif part.startswith("DISC"):
            disc_name = part[4:]
        elif part.startswith("GEN"):
            gen_name = part[3:]
        elif part.startswith("LOSS"):
            loss_name = part[4:]
    if n_epochs is None:
        warn("Number of epochs not specified! Setting value to 0.")
        n_epochs = 0
    if optimizer_disc_name is None:
        warn("Discriminator optimizer not specified! Using Adam optimizer.")
        optimizer_disc_name = "Adam"
    if optimizer_gen_name is None:
        warn("Generator optimizer not specified! Using Adam optimizer.")
        optimizer_gen_name = "Adam"
    if disc_name is None:
        warn("No discriminator name found! Using 'DCGANDiscriminator'.")
        disc_name = "DCGANDiscriminator"
    if gen_name is None:
        if folder.find("UPSAMPLEGen") != -1:
            warn("Found old keyword 'UPSAMPLEGen'! Using 'UpsampleGenerator'.")
            gen_name = "UpsampleGenerator"
    if loss_name is None:
        warn("Loss class not specified! Using GANLoss.")
        loss_name = "GANLoss"
    # assert n_epochs is not None, "Number of epochs not specified!"
    # assert optimizer_disc_name is not None, "Discriminator optimizer not specified!"
    # assert optimizer_gen_name is not None, "Generator optimizer not specified!"
    assert disc_name is not None, "Discriminator class not specified!"
    assert gen_name is not None, "Generator class not specified!"
    # assert loss_name is not None, "Loss class not specified!"

    disc_class = getattr(discriminators, disc_name)
    if not ("from_name_and_state_dict" in dir(disc_class)):
        raise NotImplementedError(
            f"Cannot create discriminator! "
            f"from_name_and_state_dict function is not implemented in {disc_class}!"
        )
    discriminator = disc_class.from_name_and_state_dict(
        folder, gan_state_dict["discriminator.state_dict"]
    )

    gen_class = getattr(generators, gen_name)
    if not ("from_name_and_state_dict" in dir(gen_class)):
        raise NotImplementedError(
            f"Cannot create generator! "
            f"from_name_and_state_dict function is not implemented in {gen_class}!"
        )
    generator = gen_class.from_name_and_state_dict(
        folder, gan_state_dict["generator.state_dict"]
    )

    loss_class = globals()[loss_name]
    if not ("from_name_and_state_dict" in dir(loss_class)):
        raise NotImplementedError(
            f"Cannot create loss! "
            f"from_name_and_state_dict function is not implemented in {loss_class}!"
        )
    loss = loss_class.from_name_and_state_dict(
        folder, gan_state_dict["gan_loss.state_dict"]
    )

    optimizer_disc_class = getattr(torch.optim, optimizer_disc_name)
    optimizer_disc = optimizer_disc_class(discriminator.parameters())
    optimizer_disc.load_state_dict(gan_state_dict["optimizer_disc.state_dict"])
    optimizer_gen_class = getattr(torch.optim, optimizer_gen_name)
    optimizer_gen = optimizer_gen_class(generator.parameters())
    optimizer_gen.load_state_dict(gan_state_dict["optimizer_gen.state_dict"])

    amp = False
    if folder.find("AMP") != -1:
        amp = True
    strains = None
    if folder.find("EPSstrains") != -1:
        strains = "strains"
    elif folder.find("EPSvonMises") != -1:
        strains = "vonMises"
    elif folder.find("EPSstrains_vonMises") != -1:
        strains = "strains_vonMises"
    new_gan = GAN(
        discriminator=discriminator,
        generator=generator,
        gan_loss=loss,
        optimizer_disc=optimizer_disc,
        optimizer_gen=optimizer_gen,
        device=device,
        amp=amp,
        strains=strains,
    )
    new_gan.n_epochs = n_epochs
    new_gan.model_number = int(folder.split("_")[-1])

    return new_gan


class GANLoss(nn.Module):
    """
    Defines different GAN objectives.
    Use an instance of this class as loss for a GAN. This class automatically creates a target label tensor
    that has the same size as the input. Call calc_regularization to get the specified regularization term
    which also has to be backpropagated.
    Source: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", Zhu et al. (2017, arXiv:1703.10593).

    :param gan_type: Type of GAN objective: vanilla | lsgan | wgan
    :param reg_type: Type of regularization:
           wgangp: Original Wasserstein-GAN gradient penalty (arXiv:1704.00028v3)
           wgangp0: Zero centered gradient penalty (arXiv:1801.04406)
           gp-real: L2 gradient penalty on real samples (arXiv:1801.04406)
           gp-fake: L2 gradient penalty on fake samples (arXiv:1801.04406)
           gp-real-fake: L2 gradient penalty on real and fake samples (arXiv:1801.04406)
    :param lambda_loss: The gan loss is multiplied with this factor
    :param lambda_reg: The regularization loss is multiplied with this factor
                       (arXiv:1801.04406 and arXiv:1704.00028v3 use 10.0)
    :param real_label: Label used for real data
    :param fake_label: Label used for fake data
    """

    def __init__(
        self,
        gan_type,
        reg_type,
        lambda_loss=1.0,
        lambda_reg=0.2,
        real_label=1.0,
        fake_label=0.0,
        wgan_n_critic=5,
    ):
        super().__init__()

        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        self.register_buffer("lambda_loss", torch.tensor(lambda_loss))
        self.register_buffer("lambda_reg", torch.tensor(lambda_reg))

        self.gan_type = gan_type
        n_critic = 1
        if gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type == "wgan":
            self.loss = None
            n_critic = wgan_n_critic
        else:
            raise NotImplementedError(f"Gan type {gan_type} not implemented!")

        self.reg_type = reg_type
        supp_reg_types = ["wgangp", "wgangp0", "gp-real", "gp-fake", "gp-real-fake"]
        if reg_type not in supp_reg_types and reg_type is not None:
            raise NotImplementedError(
                f"Regularization type {reg_type} not implemented! Use one of {supp_reg_types}!"
            )

        self.register_buffer("n_critic", torch.tensor(n_critic))

    @staticmethod
    def forward(inp):
        return inp

    def get_name(self):
        """
        Returns name of the GANLoss object. This name is used to save the GANLoss.

        :return: Name of the GANLoss
        """
        name = f"TYPE{self.gan_type}_LLOSS{self.lambda_loss}"
        if self.reg_type is not None:
            name += f"_REG{self.reg_type}_LREG{self.lambda_reg:.03f}"
        return name

    @classmethod
    def from_name_and_state_dict(cls, name, state_dict):
        """
        Creates GANLoss object from save folder name and state_dict.

        :param name: Save folder name
        :param state_dict: state_dict of the saved GANLoss
        :return: GANLoss object
        """
        gan_type = None
        reg_type = None

        parts = name.split("_")
        for part in parts:
            if part.startswith("TYPE"):
                gan_type = part[4:]
            elif part.startswith("REG"):
                reg_type = None if part[3:] == "None" else part[3:]
        if gan_type is None:
            warn("GAN type not specified! Using 'vanilla'.")
            gan_type = "vanilla"
        # assert gan_type is not None, "GAN type not specified!"

        try:
            real_label = state_dict["real_label"].item()
        except KeyError:
            warn("real_label not found in state_dict! Setting value to 1.")
            state_dict["real_label"] = torch.tensor([1])
            real_label = state_dict["real_label"].item()
        try:
            fake_label = state_dict["fake_label"].item()
        except KeyError:
            warn("fake_label not found in state_dict! Setting value to 0.")
            state_dict["fake_label"] = torch.tensor([0])
            fake_label = state_dict["fake_label"].item()
        try:
            lambda_loss = state_dict["lambda_loss"].item()
        except KeyError:
            warn("lambda_loss not found in state_dict! Setting value to 1.")
            state_dict["lambda_loss"] = torch.tensor([1])
            lambda_loss = state_dict["lambda_loss"].item()
        try:
            lambda_reg = state_dict["lambda_reg"].item()
        except KeyError:
            warn("lambda_reg not found in state_dict! Setting value to 0.2.")
            state_dict["lambda_reg"] = torch.tensor([0.2])
            lambda_reg = state_dict["lambda_reg"].item()
        try:
            n_critic = state_dict["n_critic"].item()
        except KeyError:
            warn("n_critic not found in state_dict! Setting value to 1.")
            state_dict["n_critic"] = torch.tensor([0.2])
            n_critic = state_dict["n_critic"].item()

        return cls(
            gan_type=gan_type,
            reg_type=reg_type,
            real_label=real_label,
            fake_label=fake_label,
            lambda_loss=lambda_loss,
            lambda_reg=lambda_reg,
            wgan_n_critic=n_critic,
        )

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensor with the same size as the input.

        :param prediction: Output of the discriminator
        :param target_is_real: If the ground truth label is for real images or fake images
        :return: A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = torch.full_like(prediction, fill_value=self.real_label)
        else:
            target_tensor = torch.full_like(prediction, fill_value=self.fake_label)
        return target_tensor

    def calc_regularization(
        self, disc=None, x_real=None, x_fake=None, d_real=None, d_fake=None
    ):
        """
        Calculate specified regularization.

        :param disc: Discriminator. Necessary for: wgangp, wgangp0
        :param x_real: Batch of real data. Necessary for: wgangp, wgangp0, gp-real, gp-real-fake
        :param x_fake: Batch of fake data. Necessary for: wgangp, wgangp0, gp-fake, gp-real-fake
        :param d_real: Output of the discriminator for x_real. Necessary for: gp-real, gp-real-fake
        :param d_fake: Output of the discriminator for x_fake. Necessary for: gp-fake, gp-real-fake
        :return: The calculated regularization
        """
        reg = None

        if (
            disc is not None
            and x_real is not None
            and x_fake is not None
            and d_real is None
            and d_fake is None
        ):
            if self.reg_type == "wgangp":
                reg = reg_wgan_gp(disc, x_real, x_fake, center=1)
            elif self.reg_type == "wgangp0":
                reg = reg_wgan_gp(disc, x_real, x_fake, center=0)
        elif (
            disc is not None
            and x_real is not None
            and x_fake is None
            and d_real is not None
            and d_fake is None
        ):
            if self.reg_type in ["gp-real", "gp-real-fake"]:
                reg = compute_grad2(d_real, x_real).mean()
        elif (
            disc is not None
            and x_real is None
            and x_fake is not None
            and d_real is None
            and d_fake is not None
        ):
            if self.reg_type in ["gp-fake", "gp-real-fake"]:
                reg = compute_grad2(d_fake, x_fake).mean()
        else:
            raise RuntimeError(
                f"Given parameters are not compatible with regularization {self.reg_type}"
            )
        return None if reg is None else 0.5 * self.lambda_reg * reg

    def __call__(self, prediction, target_is_real):
        """
        Calculate loss given Discriminator's output and ground truth labels.

        :param prediction: Output of the discriminator
        :param target_is_real: If the ground truth label is for real images or fake images
        :return: The calculated loss
        """
        loss = None
        if self.gan_type in ["vanilla", "lsgan"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_type == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return self.lambda_loss * loss


class GAN:
    """
    Defines a GAN.

    :param discriminator: Discriminator model
    :param generator: Generator model
    :param gan_loss: Instance of GANLoss class characterizing the GAN objective
    :param optimizer_gen: Optimizer used to update the weights of the discriminator
    :param optimizer_disc: Optimizer used to update the weights of the generator
    :param device: CPU / GPU
    :param amp: If automatic mixed precision should be used
    :param strains: Type of strains used as additional input for the discriminator:
           None: No strains are used
           strains: Strains ε_x, ε_y, ε_xy are used as additional input for the discriminator
           vonMises: Von-Mises equivalent strain ε_vm is used as additional input for the discriminator
           strains_vonMises: Both of the above are used as additional input for the discriminator
    """

    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        gan_loss: GANLoss,
        optimizer_gen: torch.optim.Optimizer,
        optimizer_disc: torch.optim.Optimizer,
        device=torch.device("cpu"),
        amp=False,
        strains=None,
    ):
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)
        self.gan_loss = gan_loss
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.device = device
        self.amp = amp
        self.scaler_gen = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scaler_disc = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.losses_gen = {"epoch": {}}
        self.losses_disc = {"epoch": {}}
        self.outputs_disc = {"epoch": {}}
        self.fake_data_list = []

        self.strains = strains
        supp_strain_types = ["strains", "vonMises", "strains_vonMises"]
        if strains is not None and strains not in supp_strain_types:
            raise NotImplementedError(
                f"Strain type {strains} not implemented! Use one of {supp_strain_types}!"
            )

        self.n_epochs = 0
        self.checkpoint_rate = 0
        self.batch_size = -1

        self.model_number = 0

    def get_name(self):
        """
        Returns name of the GAN object. This name is used to save the GAN.

        :return: Name of the GAN
        """
        name = ""
        name_disc = f"DISC{self.discriminator.__class__.__name__}"
        if len(self.discriminator.get_name()) != 0:
            name_disc += f"_{self.discriminator.get_name()}"

        name_gen = f"GEN{self.generator.__class__.__name__}"
        if len(self.generator.get_name()) != 0:
            name_gen += f"_{self.generator.get_name()}"

        name_loss = f"LOSS{self.gan_loss.__class__.__name__}"
        if len(self.gan_loss.get_name()) != 0:
            name_loss += f"_{self.gan_loss.get_name()}"

        name_optim_gen = f"OPTGEN{self.optimizer_gen.__class__.__name__}"
        name_optim_disc = f"OPTDISC{self.optimizer_disc.__class__.__name__}"
        name += f"EPOCHS{self.n_epochs}_"
        if self.batch_size != -1:
            name += f"BATCHS{self.batch_size}_"
        name += (
            f"{name_disc}_{name_gen}_" f"{name_loss}_{name_optim_disc}_{name_optim_gen}"
        )
        if self.amp:
            name += f"_AMP{self.amp}"
        if self.strains is not None:
            name += f"_EPS{self.strains}"
        return f"{name}_{self.model_number}"

    def save_model_checkpoint(self, epoch, save_folder):
        """
        Saves GAN checkpoint. Saves every state_dict and training statistics.

        :param epoch: The data is saved to separate folder depending on the epoch
        :param save_folder: Folder in which the GAN is saved. In this folder a 'checkpoints' folder is created.
                            In the checkpoints folder an folder is created for each checkpoint, named after the epoch.
        """
        save_name = self.get_name()

        save_folder_epoch = os.path.join(
            save_folder, save_name, "checkpoints", f"Epoch{str(epoch)}"
        )

        Path(save_folder_epoch).mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint in {save_folder_epoch}")
        # Save GAN state dict
        torch.save(
            {
                "generator.state_dict": self.generator.state_dict(),
                "discriminator.state_dict": self.discriminator.state_dict(),
                "gan_loss.state_dict": self.gan_loss.state_dict(),
                "optimizer_gen.state_dict": self.optimizer_gen.state_dict(),
                "optimizer_disc.state_dict": self.optimizer_disc.state_dict(),
            },
            os.path.join(save_folder_epoch, "gan_state_dict.pt"),
        )
        # Save statistics
        torch.save(self.losses_gen, os.path.join(save_folder_epoch, "losses_gen.pt"))
        torch.save(self.losses_disc, os.path.join(save_folder_epoch, "losses_disc.pt"))
        torch.save(
            self.outputs_disc, os.path.join(save_folder_epoch, "outputs_disc.pt")
        )
        fake_data_list = torch.stack(self.fake_data_list)
        torch.save(fake_data_list, os.path.join(save_folder_epoch, "fake_data_list.pt"))

        # Delete previous fake_data_list.pt
        checkpoints = [
            name
            for name in os.listdir(os.path.join(save_folder, save_name, "checkpoints"))
            if os.path.isdir(os.path.join(save_folder, save_name, "checkpoints", name))
        ]
        if len(checkpoints) > 1:
            os.remove(
                os.path.join(
                    save_folder,
                    save_name,
                    "checkpoints",
                    checkpoints[-2],
                    "fake_data_list.pt",
                )
            )

    def save_gan(self, save_folder):
        """
        Saves GAN. Saves every state_dict and training statistics.

        :param save_folder: Folder in which the model is saved
        """
        save_name = self.get_name()

        save_folder_gan = os.path.join(save_folder, save_name)
        Path(save_folder_gan).mkdir(parents=True, exist_ok=True)

        print(f"Saving GAN in {save_folder_gan}")
        # Save GAN state dict
        torch.save(
            {
                "generator.state_dict": self.generator.state_dict(),
                "discriminator.state_dict": self.discriminator.state_dict(),
                "gan_loss.state_dict": self.gan_loss.state_dict(),
                "optimizer_gen.state_dict": self.optimizer_gen.state_dict(),
                "optimizer_disc.state_dict": self.optimizer_disc.state_dict(),
            },
            os.path.join(save_folder_gan, "gan_state_dict.pt"),
        )
        # Save statistics
        torch.save(self.losses_gen, os.path.join(save_folder_gan, "losses_gen.pt"))
        torch.save(self.losses_disc, os.path.join(save_folder_gan, "losses_disc.pt"))
        torch.save(self.outputs_disc, os.path.join(save_folder_gan, "outputs_disc.pt"))
        fake_data_list = torch.stack(self.fake_data_list)
        torch.save(fake_data_list, os.path.join(save_folder_gan, "fake_data_list.pt"))

    def train_step_disc(self, real_data, noise_vec):
        """
        Performs one training iteration on the discriminator.

        :param real_data: Batch of real data
        :param noise_vec: Batch of noise vectors used as input for the generator
        """
        context = torch.cuda.amp.autocast() if self.amp else contextlib.suppress()
        self.discriminator.zero_grad()
        real_data.requires_grad_()
        # --------------------------------- Train with real data ---------------------------------
        with context:
            real_data = calc_and_concat_strains(real_data, strains=self.strains)
            disc_real = self.discriminator(real_data)
            disc_loss_real = self.gan_loss(prediction=disc_real, target_is_real=True)
            disc_reg = self.gan_loss.calc_regularization(
                disc=self.discriminator,  # Regularization on real data
                x_real=real_data,
                d_real=disc_real,
            )
        self.scaler_disc.scale(disc_loss_real).backward(retain_graph=True)
        if disc_reg is not None:
            self.scaler_disc.scale(disc_reg).backward()

        # --------------------------------- Train with fake data ---------------------------------
        with context:
            fake_data = self.generator(noise_vec)
            fake_data = calc_and_concat_strains(fake_data, strains=self.strains)
            disc_fake = self.discriminator(fake_data.detach())
            disc_loss_fake = self.gan_loss(prediction=disc_fake, target_is_real=False)
            disc_reg = self.gan_loss.calc_regularization(
                disc=self.discriminator,  # Regularization on fake data
                x_fake=fake_data,
                d_fake=disc_fake,
            )
        self.scaler_disc.scale(disc_loss_fake).backward()
        if disc_reg is not None:
            self.scaler_disc.scale(disc_reg).backward()

        disc_reg = self.gan_loss.calc_regularization(
            disc=self.discriminator,  # WGAN-GP Regularization
            x_real=real_data,
            x_fake=fake_data,
        )
        if disc_reg is not None:
            self.scaler_disc.scale(disc_reg).backward()

        self.scaler_disc.step(self.optimizer_disc)
        self.scaler_disc.update()

        ##################################
        # Calculate and return statistics:
        ##################################
        disc_loss = disc_loss_real + disc_loss_fake
        stats = {
            "disc_out_real": disc_real.detach().mean().cpu(),
            "disc_out_fake": disc_fake.detach().mean().cpu(),
            "disc_loss": disc_loss.detach().cpu(),
        }

        return stats

    def train_step_gen(self, noise_vec):
        """
        Performs one training iteration on the generator.

        :param noise_vec: Batch of noise vectors used as input for the generator
        """
        context = torch.cuda.amp.autocast() if self.amp else contextlib.suppress()
        self.generator.zero_grad()

        with context:
            fake_data = self.generator(noise_vec)
            fake_data = calc_and_concat_strains(fake_data, strains=self.strains)
            disc_fake_2 = self.discriminator(fake_data)
            gen_loss = self.gan_loss(prediction=disc_fake_2, target_is_real=True)
        self.scaler_gen.scale(gen_loss).backward()
        self.scaler_gen.step(self.optimizer_gen)
        self.scaler_gen.update()

        ##################################
        # Calculate and return statistics:
        ##################################
        stats = {
            "disc_out_fake_2": disc_fake_2.detach().mean().cpu(),
            "gen_loss": gen_loss.detach().cpu(),
        }

        return stats

    def train(
        self,
        dataloader,
        n_epochs,
        fixed_noise=None,
        checkpoint_rate=0,
        save_folder=os.getcwd(),
    ):
        """
        Trains the GAN.

        :param dataloader: Dataloader object holding the training data
        :param n_epochs: Number of epochs
        :param fixed_noise: Fixed batch of noise vectors to create and save fake data throughout the training.
                            If not specified one will be created.
        :param checkpoint_rate: Frequency of saving model checkpoints at the end of epochs. If 0 no checkpoints are saved.
        :param save_folder: Folder in which the models are saved at every checkpoint and at the end of training
        """
        batch_size = next(iter(dataloader)).shape[0]
        self.batch_size = batch_size

        self.n_epochs += n_epochs

        # If folder with same model name already exists, create new folder with additional index
        # model_name -> model_name_0 or mode_name_1 if model_name_0 already exists, and so on
        save_name = self.get_name()
        folders = os.listdir(save_folder)
        same_names = []
        for folder in folders:
            if os.path.isdir(os.path.join(save_folder, folder)) and "_".join(
                folder.split("_")[:-1]
            ) == "_".join(save_name.split("_")[:-1]):
                same_names.append(folder)
        model_numbers = sorted([int(x.split("_")[-1]) for x in same_names])
        n_model = 0 if len(model_numbers) == 0 else model_numbers[-1] + 1
        self.model_number = n_model
        print(
            f"Model and checkpoints will be saved to\n{os.path.join(save_folder, self.get_name())}"
        )

        if os.path.isdir(os.path.join(save_folder, self.get_name())):
            print(f"A model with name {self.get_name()} already exists.")
            abort_or_continue = None
            while abort_or_continue not in ["y", "n"]:
                print("Do you want to continue and overwrite existing model (y/n)?.")
                abort_or_continue = input()
            if abort_or_continue == "n":
                sys.exit()

        Path(os.path.join(save_folder, self.get_name())).mkdir(
            parents=True, exist_ok=True
        )

        if fixed_noise is None:
            fixed_noise = torch.randn(16, self.generator.noise_dim, device=self.device)

        assert checkpoint_rate >= 0, "Checkpoint rate cannot be negative!"
        self.checkpoint_rate = checkpoint_rate

        self.generator.train()
        self.discriminator.train()

        print_rate = 10  # Iterations after training statistics are printed

        for epoch in range(n_epochs):
            self.losses_gen["epoch"][epoch + 1] = []
            self.losses_disc["epoch"][epoch + 1] = []
            self.outputs_disc["epoch"][epoch + 1] = {
                "real": [],
                "fake": [],
                "fake_2": [],
            }

            for i, data in enumerate(dataloader, start=0):
                real_batch = data.to(self.device)
                noise_vec = torch.randn(
                    batch_size, self.generator.noise_dim, device=self.device
                )

                stats_disc = self.train_step_disc(real_batch, noise_vec)
                stats_gen = None

                if (
                    i % self.gan_loss.n_critic.item() == 0
                ):  # The generator model is updated after n_critic iterations, default 1
                    stats_gen = self.train_step_gen(noise_vec)
                    if isinstance(
                        self.gan_loss.loss, nn.BCEWithLogitsLoss
                    ):  # Apply Sigmoid function if outputs are logits
                        stats_gen["disc_out_fake_2"] = stats_gen[
                            "disc_out_fake_2"
                        ].sigmoid()
                    self.outputs_disc["epoch"][epoch + 1]["fake_2"].append(
                        stats_gen["disc_out_fake_2"]
                    )
                    self.losses_gen["epoch"][epoch + 1].append(stats_gen["gen_loss"])

                if isinstance(
                    self.gan_loss.loss, nn.BCEWithLogitsLoss
                ):  # Apply Sigmoid function if outputs are logits
                    stats_disc["disc_out_real"] = stats_disc["disc_out_real"].sigmoid()
                    stats_disc["disc_out_fake"] = stats_disc["disc_out_fake"].sigmoid()

                self.losses_disc["epoch"][epoch + 1].append(stats_disc["disc_loss"])
                self.outputs_disc["epoch"][epoch + 1]["real"].append(
                    stats_disc["disc_out_real"]
                )
                self.outputs_disc["epoch"][epoch + 1]["fake"].append(
                    stats_disc["disc_out_fake"]
                )

                if i % print_rate == 0:
                    print(
                        f"[{epoch+1:02}/{n_epochs:02}][{i:03}/{len(dataloader):03}]",
                        f"\tLoss_D: {stats_disc['disc_loss'].item():+.4f}",
                        end="",
                    )
                    if i % self.gan_loss.n_critic.item() == 0:
                        print(f"\tLoss_G: {stats_gen['gen_loss'].item():+.4f}", end="")
                    print(
                        f"\tD(x): {stats_disc['disc_out_real']:+.4f}\t",
                        f"D(G(z)): {stats_disc['disc_out_fake']:+.4f}",
                        end="",
                    )
                    if i % self.gan_loss.n_critic.item() == 0:
                        print(f" / {stats_gen['disc_out_fake_2']:+.4f}", end="\n")
                    else:
                        print()

                if i == len(dataloader) - 1:
                    with torch.no_grad():
                        context = (
                            torch.cuda.amp.autocast()
                            if self.amp
                            else contextlib.suppress()
                        )
                        with context:
                            fake_data = self.generator(fixed_noise).detach().cpu()
                    fake_data = calc_and_concat_strains(fake_data, strains="vonMises")
                    self.fake_data_list.append(fake_data)

            if checkpoint_rate > 0 and (epoch + 1) % self.checkpoint_rate == 0:
                self.save_model_checkpoint(epoch + 1, save_folder)
        self.save_gan(save_folder)
        torch.cuda.empty_cache()
