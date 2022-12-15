import os
from scipy.constants import c
import numpy as np
import h5py
from ElegantWrapper.watcher import Watcher, Watcher2, SliceCollection
from copy import deepcopy

def h5_out(h5_file, dict_, overwrite=False):
    if not overwrite and os.path.isfile(h5_file):
        raise OSError('File %s exists')

    with h5py.File(h5_file, 'w') as f:
        for key, val in dict_.items():
            f.create_dataset(key, data=val)

def h5_in_genesis(h5_file):
    out = {}
    with h5py.File(h5_file, 'r') as f:
        for key, val in f.items():
            out[key] = np.array(val)
    return out

def h5_in_elegant(h5_file):
    out = {}
    with h5py.File(h5_file, 'r') as f:
        for key, val in f['page1/columns'].items():
            out[key] = np.array(val)
    return out

def get_watcher_genesis(h5_file):
    h5_dist = h5_in_genesis(h5_file)
    return Watcher2({}, h5_dist)

def match_dist2(dist, bxm, axm, bym, aym, n_slices, n_slice_to_match):
    beta_match = {'x': bxm, 'y': bym}
    alpha_match = {'x': axm, 'y': aym}
    particle_match = {}

    watch = Watcher2({}, dist)

    slices = watch.slice_beam(n_slices)
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
    return particle_match

def match_dist_proj(dist, bxm, axm, bym, aym):
    beta_match = {'x': bxm, 'y': bym}
    alpha_match = {'x': axm, 'y': aym}
    particle_match = {}

    watch = Watcher2({}, dist)

    for dim in ('x', 'y'):
        beta0 = watch.get_beta_from_beam(dim)
        alpha0 = watch.get_alpha_from_beam(dim)
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
    return particle_match

def center_dist(dist, n_slices, n_slice_to_center):
    particle_center = {}

    watch = Watcher2({}, dist)
    slices = watch.slice_beam(n_slices)
    slice_to_center = slices[n_slice_to_center]

    for dim in ('x', 'y'):
        particle_center[dim] = dist[dim] - slice_to_center[dim].mean()
        particle_center[dim+'p'] = dist[dim+'p'] - slice_to_center[dim+'p'].mean()

    for dim in ('t', 'p'):
        particle_center[dim] = dist[dim]

    return particle_center

def center_dist_proj(dist):
    particle_center = deepcopy(dist)
    for dim in ('x', 'y', 'xp', 'yp'):
        particle_center[dim] -= particle_center[dim].mean()

    return particle_center

def match_dist(h5_in, h5_out_filename, bxm, axm, bym, aym, n_slices=None, t_to_match=None, n_slice_to_match=None, proj=False, center=True, overwrite=False):
    beta_match = {'x': bxm, 'y': bym}
    alpha_match = {'x': axm, 'y': aym}
    particle_match = {}

    if type(h5_in) is str and os.path.isfile(h5_in):
        dist = Watcher(h5_in)
    else:
        dist = Watcher2({}, h5_in)

    if proj:
        slice_to_match = dist
    else:
        slices = dist.slice_beam(n_slices)
        if t_to_match is not None:
            slice_coll = SliceCollection(slices, dist)
            t_arr = -slice_coll.s_arr/c
            n_slice_to_match = int(np.argmin((t_arr - t_to_match)**2))
        elif n_slice_to_match is None:
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

def add_chirp(dist, chirp_peak_to_peak):

    dist_out = deepcopy(dist)
    t_min = dist['t'].min()
    t_max = dist['t'].max()
    del_t = t_max - t_min
    p_central = dist['p'].mean()

    p_add = chirp_peak_to_peak*p_central * (dist['t'] - dist['t'].mean())/del_t

    dist_out['p'] = dist['p'] + p_add
    return dist_out

def change_energy(dist, gamma_factor):
    dist_out = deepcopy(dist)
    dist_out['p'] *= gamma_factor

    for dim in ('x', 'xp', 'y', 'yp'):
        dist_out[dim] *= np.sqrt(1/gamma_factor)

    return dist_out

