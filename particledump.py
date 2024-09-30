import numpy as np
from scipy.constants import c

from .view import ViewBase
from ElegantWrapper import watcher

class Particledump(ViewBase):
    def __init__(self, outfile):
        ViewBase.__init__(self, outfile)
        self.npslice = len(self['slice000001/x'])
        self.n_particles = self['slicecount'] * self.npslice
        current_arr = []
        for n_slice in range(self['slicecount']):
            current_arr.append(self['slice%06i/current' % (n_slice+1)])
        self.current_arr = np.array(current_arr)

    def to_watcher(self, n_particles, slice_indices=None, particle_indices=None):
        out_arr = np.zeros(n_particles)
        out = {
                'x': out_arr,
                'xp': out_arr.copy(),
                'y': out_arr.copy(),
                'yp': out_arr.copy(),
                'p': out_arr.copy(),
                't': out_arr.copy(),
                }
        if slice_indices is None:
            rand1 = np.random.rand(n_particles)
            #rand2 = np.random.rand(n_particles)
            rand3 = (np.random.rand(n_particles)*self.npslice).astype(int)
            cdf = np.cumsum(self.current_arr)
            cdf /= cdf[-1]
            slice_indices0 = np.arange(len(self.current_arr))
            slice_nums = np.interp(rand1, cdf, slice_indices0)
            self.slice_indices, counts = np.unique(slice_nums.astype(int), return_counts=True)
            self.particle_indices = []
        else:
            self.slice_indices = slice_indices
            counts = [None]*len(slice_indices)
            self.particle_indices = particle_indices

        ctr = 0
        for n_slice, (slice_index, count) in enumerate(zip(self.slice_indices, counts)):
            if particle_indices is None:
                particle_index = rand3[ctr:ctr+count]
                self.particle_indices.append(particle_index)
            else:
                particle_index = particle_indices[n_slice]
                count = len(particle_index)
            slice_t = -self['slicespacing']*slice_index/c

            gamma = np.take(self['slice%06i/gamma' % (slice_index+1)], particle_index)
            theta = np.take(self['slice%06i/theta' % (slice_index+1)], particle_index)
            out['x'][ctr:ctr+count] = np.take(self['slice%06i/x' % (slice_index+1)], particle_index)
            out['y'][ctr:ctr+count] = np.take(self['slice%06i/y' % (slice_index+1)], particle_index)
            out['xp'][ctr:ctr+count] = np.take(self['slice%06i/px' % (slice_index+1)], particle_index)/gamma
            out['yp'][ctr:ctr+count] = np.take(self['slice%06i/py' % (slice_index+1)], particle_index)/gamma
            out['p'][ctr:ctr+count] = gamma
            out['t'][ctr:ctr+count] = slice_t + theta/(2*np.pi)*self['slicespacing']/c
            ctr += count
        assert ctr == n_particles
        out['t'] -= out['t'].min()
        return watcher.Watcher2({}, out)

    def to_sdds(self, filename, n_particles):
        watch = self.to_watcher(n_particles)
        watch.toSDDS(filename)

