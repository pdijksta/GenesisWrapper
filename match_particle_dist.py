import os
import numpy as np
import h5py
from ElegantWrapper.watcher import Watcher, Watcher2

def h5_out(h5_file, dict_, overwrite=False):
    if not overwrite and os.path.isfile(h5_file):
        raise OSError('File %s exists')

    with h5py.File(h5_file, 'w') as f:
        for key, val in dict_.items():
            f.create_dataset(key, data=val)

def match_dist(h5_in, h5_out_filename, bxm, axm, bym, aym, n_slices=None, n_slice_to_match=None, proj=False, center=True, overwrite=False):
    beta_match = {'x': bxm, 'y': bym}
    alpha_match = {'x': axm, 'y': aym}
    particle_match = {}

    dist = Watcher(h5_in)

    if proj:
        slice_to_match = dist
    else:
        slices = dist.slice_beam(n_slices)
        if n_slice_to_match is None:
            n_slice_to_match = n_slices // 2
        slice_to_match = slices[n_slice_to_match]

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

        particle_match['p'] = dist['p']
        particle_match['t'] = dist['t']

        if center:
            if proj:
                particle_match[dim] -= particle_match[dim].mean()
                particle_match[dim+'p'] -= particle_match[dim+'p'].mean()
            else:
                new_dist = Watcher2({}, particle_match)
                slice_to_match2 = new_dist.slice_beam(n_slices)[n_slice_to_match]

                particle_match[dim] = particle_match[dim] - slice_to_match2[dim].mean()
                particle_match[dim+'p'] = particle_match[dim+'p'] - slice_to_match2[dim+'p'].mean()

    #new_dist2 = Watcher2({}, particle_match)
    #slice_to_match3 = new_dist2.slice_beam(n_slices)[n_slice_to_match]
    #import pdb; pdb.set_trace()

    h5_out(h5_out_filename, particle_match, overwrite=overwrite)

