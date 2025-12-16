import numpy as np
import matplotlib.pyplot as plt
import torch


def absId(kappa, w_over_l, vgs: torch.Tensor, vth, vds: torch.Tensor):
    """
        MOSFET drain current
    """
    sat = 0.5 * kappa * w_over_l * (vgs - vth) ** 2
    lin = 0.5 * kappa * w_over_l * (2 * (vgs - vth) - vds) * vds
    Ids = torch.where(vds >= vgs - vth, sat, lin) * torch.where(vgs > vth, 1, 0)

    return Ids

