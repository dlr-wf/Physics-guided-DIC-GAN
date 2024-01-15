import numpy as np
import torch


def calc_strains_from_interpolated_data(interp_data, nu=0.5, min_order=6, max_order=6):
    """
    Calculating strains from interpolated displacement data.
    Derivatives are taken between pixels which are separated by min_order number of pixels.
    This separation increases
    to max_order number of pixels and all calculated derivatives are then averaged.
    This implies that the resolution of the strains is reduced by max_order-1 pixels.
    Since the formula for calculating the von Mises equivalent strain contains a square-root,
    which is not
    differentiable at zero, a smoothing value of 1e-40 is added inside the square-root.

    Args:
        interp_data: Interpolated displacement data
        min_order: Minimum distance used for calculating derivative
        max_order: Maximum distance used for calculating derivative

    Returns:
        eps_x, eps_y, eps_xy, eps_vm_squared, eps_vm_adjusted, eps_vm:  2-dimensional strains,
         squared von Mises strains, adjusted von Mises strains and von Mises equivalent strain
    """

    # eps_x = du_x/dx
    # eps_y = du_y/dy
    # eps_xy = (du_y/dx + du_x/dy)/2
    # eps_vm = 2 / 3 * np.sqrt(3 / 2 * (eps_x ** 2 + eps_y ** 2) + 3 * eps_xy ** 2)

    # ---------------------------------------------------------------------------------------------
    # calculate derivative with f'(x) = (f(x+h)-f(x))/h (error of order h+h^2+h^3+...),
    # called forward difference
    # NOTE: If we want to calculate the derivative wrt. x of a 2-dim function f(x,y)
    #  we want (f(x+h,y)-f(x,y))/h but in a numpy array or torch tensor x and y are
    #  swapped for readibility.
    #  simply plot np.random.rand(2,2) or torch.randn((2,2)) to familiarize yourself with it
    #  because indices in array refer to row, column (which is y,x) not x,y
    #
    # One could also use f'(x) = (f(x+h)-f(x-h))/2h (error of order h^2+h^3+...),
    # called central difference.
    # Implementation wise this just changes h to 2*h.
    # ---------------------------------------------------------------------------------------------

    u_x = interp_data[:, 0]  # x-displacements with size batch_size x 256 x 256
    u_y = interp_data[:, 1]  # y-displacements with size batch_size x 256 x 256

    h = min_order
    h2 = h // 2
    start, end = h2, (-h2 if h % 2 == 0 else -h2 - 1)
    eps_x = (u_x[:, start:end, h:] - u_x[:, start:end, :-h]) / h
    eps_y = (u_y[:, h:, start:end] - u_y[:, :-h, start:end]) / h
    u_xy = (u_x[:, h:, start:end] - u_x[:, :-h, start:end]) / h
    u_yx = (u_y[:, start:end, h:] - u_y[:, start:end, :-h]) / h

    # Loop over all h from min_order+1 to max_order. h=min_order was already calculated above
    for h in np.arange(min_order + 1, max_order + 1, step=1):
        # First calculate difference with order h and save it in tmp variable,
        #  then take mean with current difference and save it in corresponding strain variable
        h2 = h // 2
        start, end = h2, (-h2 if h % 2 == 0 else -h2 - 1)
        eps_x_tmp = (u_x[:, start:end, h:] - u_x[:, start:end, :-h]) / h
        eps_x = (eps_x[:, 1:, 1:] + eps_x_tmp) / 2
        eps_y_tmp = (u_y[:, h:, start:end] - u_y[:, :-h, start:end]) / h
        eps_y = (eps_y[:, 1:, 1:] + eps_y_tmp) / 2
        u_xy_tmp = (u_x[:, h:, start:end] - u_x[:, :-h, start:end]) / h
        u_xy = (u_xy[:, 1:, 1:] + u_xy_tmp) / 2
        u_yx_tmp = (u_y[:, start:end, h:] - u_y[:, start:end, :-h]) / h
        u_yx = (u_yx[:, 1:, 1:] + u_yx_tmp) / 2

    # Wikipedia (https://en.wikipedia.org/wiki/Infinitesimal_strain_theory) and
    # DianaFEA User Manual (https://manuals.dianafea.com/d944/Analys/node405.html)
    # yield the same result under plane stress assumption (e_z = -nu/(1-nu) * (eps_x+eps_y)):
    # eps_vm = 2/3 * sqrt( eps_x^2 + eps_y^2 + eps_z^2 - eps_x*eps_y - eps_x*eps_z - eps_y*eps_z + 3*eps_xy^2 )
    # eps_vm = 2/3 * sqrt( eps_x^2 + eps_y^2 + nu^2/(1-nu)^2(eps_x+eps_y)^2 - eps_x*eps_y + nu/(1-nu)(eps_x+eps_y)^2 + 3*eps_xy^2 )
    # eps_vm = 2/3 * sqrt( eps_x^2 + eps_y^2 + ( nu^2/(1-nu)^2 + nu/(1-nu) ) (eps_x+eps_y)^2 - eps_x*eps_y + 3*eps_xy^2 )
    # With eps_z = - v/(1-v) * (eps_x + eps_y)

    epsilon = 1e-40  # Smoothing value to keep the square root tractable
    eps_xy = (u_xy + u_yx) / 2

    # von-Mises equivalent strain with plane stress
    # eps_vm = 2 / 3 * torch.sqrt( torch.square(eps_x) + torch.square(eps_y) + 3*torch.square(eps_xy) +
    #                              nu**2/(1-nu)**2 * (eps_x+eps_y)**2 - eps_x*eps_y + nu/(1-nu)*(eps_x+eps_y)**2 )
    eps_vm_adjusted = (
        2
        / 3
        * torch.sqrt(
            torch.square(eps_x)
            + torch.square(eps_y)
            + 3 * torch.square(eps_xy)
            + nu**2 / (1 - nu) ** 2 * (eps_x + eps_y) ** 2
            - eps_x * eps_y
            + nu / (1 - nu) * (eps_x + eps_y) ** 2
            + epsilon
        )
    )
    # eps_vm_squared = (2 / 3)**2 * ( torch.square(eps_x) + torch.square(eps_y) + 3*torch.square(eps_xy) +
    #                                 nu**2/(1-nu)**2 * (eps_x+eps_y)**2 - eps_x*eps_y + nu/(1-nu)*(eps_x+eps_y)**2 )

    return eps_x, eps_y, eps_xy, eps_vm_adjusted


def calc_and_concat_strains(
    batch,
    strains=None,
    nu=0.5,
    strain_min_order=6,
    strain_max_order=6,
):
    """
    Calculate strains and add as additional channels to input batch.

    :param batch: Input batch
    :param strains: Strains to be concatenated: strains | vonMises | vonMises_strains
    :param strain_min_order: Minimum order of strain calculation.
                             See data_processing.strain_calc.calc_strains_from_interpolated_data
    :param strain_max_order: Maximum order of strain calculation.
                             See data_processing.strain_calc.calc_strains_from_interpolated_data
    :return: Input batch with specified strains as additional channels
    """
    if strains is None:
        return batch

    pad = strain_max_order // 2

    eps_x, eps_y, eps_xy, eps_vm_adjusted = calc_strains_from_interpolated_data(
        batch,
        nu=nu,
        min_order=strain_min_order,
        max_order=strain_max_order,
    )
    eps = []
    if strains == "strains" or strains == "vonMises_strains":
        eps.append(eps_x)
        eps.append(eps_y)
        eps.append(eps_xy)
    if strains == "vonMises" or strains == "vonMises_strains":
        eps.append(eps_vm_adjusted)

    eps = torch.stack(list(eps), dim=1)
    eps_pad = torch.nn.functional.pad(eps, (pad, pad, pad, pad), value=0.0)
    batch = torch.cat((batch, eps_pad), dim=1)

    return batch
