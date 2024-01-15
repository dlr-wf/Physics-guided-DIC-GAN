from typing import Tuple
import torch


def interpolate_noise_vectors(n_1: torch.Tensor, n_2: torch.Tensor, num: int,
                              generator: torch.nn.Module,
                              device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Interpolates between two noise vectors and generated fake data with the interpolated noise vectors.

        :param n_1: (torch.Tensor): Start noise vector
        :param n_2: (torch.Tensor): End noise vector
        :param num: (int): Number of support vectors between n_1 and n_2
        :param generator: (torch.nn.Module): Generator network
        :param device: (torch.device): Device to use, defaults to 'cpu'

        :return: Tuple of (Interpolated noise vectors, corresponding generated data)
    """
    if n_1.dim() == 2:
        n_1 = n_1.squeeze(dim=0)
    if n_2.dim() == 2:
        n_2 = n_2.squeeze(dim=0)

    assert n_1.dim() == 1
    assert n_2.dim() == 1
    n_1 = n_1.cpu()
    n_2 = n_2.cpu()

    generator.to(device)
    generator.eval()

    diff = n_2 - n_1
    diff_step = diff / (num+1)
    vecs = [n_1]
    for i in range(1, num+1):  # 1, 2, ..., num
        vecs.append(n_1 + diff_step * i)
    vecs.append(n_2)
    assert len(vecs) == num + 2

    vecs = torch.stack(vecs)

    gen_data = []
    for vec in vecs:
        vec = vec.unsqueeze(dim=0)
        with torch.no_grad():
            gen_data.append(generator(vec.to(device)).cpu())
    assert len(gen_data) == num + 2
    gen_data = torch.cat(gen_data)

    return vecs, gen_data
