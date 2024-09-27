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

    def to_watcher(self, n_particles):
        out_arr = np.zeros(n_particles)
        out = {
                'x': out_arr,
                'xp': out_arr.copy(),
                'y': out_arr.copy(),
                'yp': out_arr.copy(),
                'p': out_arr.copy(),
                't': out_arr.copy(),
                }
        rand1 = np.random.rand(n_particles)
        rand2 = np.random.rand(n_particles)
        rand3 = (np.random.rand(n_particles)*self.npslice).astype(int)
        cdf = np.cumsum(self.current_arr)
        cdf /= cdf[-1]
        indices = np.arange(len(self.current_arr))
        slice_nums = np.interp(rand1, cdf, indices)
        indices, count = np.unique(slice_nums.astype(int), return_counts=True)

        ctr = 0
        for slice_index, count in zip(indices, count):
            particle_index = rand3[ctr:ctr+count]
            slice_t = -self['slicespacing']*slice_index/c

            gamma = np.take(self['slice%06i/gamma' % (slice_index+1)], particle_index)
            #theta = np.take(self['slice%06i/theta' % (slice_index+1)], particle_index)
            out['x'][ctr:ctr+count] = np.take(self['slice%06i/x' % (slice_index+1)], particle_index)
            out['y'][ctr:ctr+count] = np.take(self['slice%06i/y' % (slice_index+1)], particle_index)
            out['xp'][ctr:ctr+count] = np.take(self['slice%06i/px' % (slice_index+1)], particle_index)/gamma
            out['yp'][ctr:ctr+count] = np.take(self['slice%06i/py' % (slice_index+1)], particle_index)/gamma
            out['p'][ctr:ctr+count] = gamma
            out['t'][ctr:ctr+count] = slice_t + rand2[ctr:ctr+count]*self['slicespacing']/c
            ctr += count
        assert ctr == n_particles
        return watcher.Watcher2({}, out)

    def to_sdds(self, filename, n_particles):
        watch = self.to_watcher(n_particles)
        watch.toSDDS(filename)

