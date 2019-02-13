import os
import numpy as np
import h5py
from ElegantWrapper.watcher import Watcher

def h5_out(h5_file, dict_, overwrite=False):
    if not overwrite and os.path.isfile(h5_file):
        raise OSError('File %s exists')

    with h5py.File(h5_file, 'w') as f:
        for key, val in dict_.items():
            f.create_dataset(key, data=val)

def match_dist(h5_in, bxm, axm, bym, aym, n_slices=None, proj=True, center=True):
    beta_match = {'x': 1, 'y': 1}
    alpha_match = {'x': 1, 'y': 1}
    particle_match = {}

    dist = Watcher(h5_in)

    slices = dist.slice_beam(n_slices)
    slice_to_match = slices[n_slices//2]

    for dim in ('x', 'y'):
        beta0 = slice_to_match.get_beta_from_beam(dim)
        alpha0 = slice_to_match.get_alpha_from_beam(dim)
        betam = beta_match[dim]
        alpham = alpha_match[dim]

        r11 = np.sqrt(betam/beta0)
        r12 = 0.
        r21 = (alpha0-alpham)/np.sqrt(betam*beta0)
        r22 = 1./r11

        particle_match[dim] = dist[dim]*r11 + dist[dim+'p']*r12
        particle_match[dim+'p'] = dist[dim]*r21 + dist[dim+'p']*r22

        if center:
            particle_match[dim] -= slice_to_match[dim].mean()
            particle_match[dim+'p'] -= slice_to_match[dim+'p'].mean()

    return particle_match

