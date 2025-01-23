import numpy as np


def get_rapidity(num_sites, lsp, mass, wavenumber=None, with_wn=False, npmod=np):
    if wavenumber is None:
        half_lat = num_sites // 2
        wavenumber = npmod.arange(-half_lat // 2, half_lat // 2)
    momentum = npmod.sin(2 * npmod.pi / num_sites * wavenumber) / lsp
    energy = npmod.sqrt(npmod.square(mass) + npmod.square(momentum))
    rapidity = npmod.log((momentum + energy) / mass)

    if with_wn:
        return rapidity, wavenumber
    return rapidity
