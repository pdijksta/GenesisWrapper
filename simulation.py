import os
import h5py
import numpy as np
from scipy.constants import c

from . import averagePower
from . import parser
from .gaussfit import GaussFit


class GenesisSimulation:

    comment_chars = ('!',)
    default_dict = {
            'npart': 8192.,
            'sample': 2.,
            }
    warn_geo = True

    def __init__(self, infile, _file_=None):

        if _file_ is None:
            self.infile = infile
        else:
            self.infile = os.path.join(os.path.dirname(_file_), infile)

        self.input = parser.GenesisInputParser(self.infile)# , self.comment_chars, self.default_dict)
        dirname = os.path.dirname(self.infile)
        self.outfile = os.path.join(dirname, self.input['setup']['rootname']+'.out.h5')

        self._dict = {}
        try:
            self.zplot = self['Global/zplot']
        except KeyError:
            print('Old version of genesis. No zplot available.')
            if (self['Field/power'].shape[0] == self['Lattice/z'].shape[0]+1):
                self.zplot = np.append(self['Lattice/z'], self['Lattice/z'][-1]+self['Lattice/dz'][-1])
            else:
                sample = int(self['Global/sample'])
                self.zplot = self['Lattice/z'][::sample]
        if self['Field/power'].shape[0] != self.zplot.shape[0]:
            print('error', self['Field/power'].shape[0], self.zplot.shape[0])
            import pdb; pdb.set_trace()

        time = self['Global/time']

        if time.shape == ():
            time = np.arange(0, self['Beam/emitx'].shape[1], dtype=float)*self['Global/sample']*self['Global/lambdaref']/c
        self.time = time

        if GenesisSimulation.warn_geo:
            print('Warning, adjusting for geometric emittance in buggy genesis version')
            GenesisSimulation.warn_geo = False


        xy = ('x', 'y')
        self.geom_emittance = {x: self.get_geometric_emittance(x) for x in xy}

        self.beta_twiss = {x: self['Beam/%ssize' % x][0,:]**2 / self.geom_emittance[x] for x in xy}
        self.alpha_twiss = {x: self['Beam/alpha%s' % x][0,:] for x in xy}
        self.gamma_twiss = {x: (1.+self.alpha_twiss[x]**2)/self.beta_twiss[x] for x in xy}

        self.powerfit = GaussFit(self.time, self['Field/power'][-1,:])


    def __getitem__(self, key):
        if key not in self._dict:
            with h5py.File(self.outfile, 'r') as ff:
                try:
                    raise_ = False
                    val = np.array(ff[key])
                except KeyError:
                    raise_ = True
                # Reduce verbosity
                if raise_:
                    raise KeyError('Key %s not found in %s' % (key, self.outfile))
                if len(val.shape) == 1:
                    val = np.squeeze(val)
                val.setflags(write=False) # Immutable array
                self._dict[key] = val
        return self._dict[key]

    def keys(self):
        with h5py.File(self.outfile, 'r') as ff:
            out = list(ff.keys())
        return out

    def print_tree(self):
        def name_and_size(key, ff):
            try:
                print((key, ff[key].shape, ff[key].dtype))
            except:
                print(key)

        with h5py.File(self.outfile, 'r') as ff:
            ff.visit(lambda x: name_and_size(x, ff))

    def get_rms_pulse_length(self):
        time = self.time
        power = self['Field/power'][-1,:].copy()
        return averagePower.get_rms_pulse_length(time, power)

    def get_total_pulse_energy(self):
        return averagePower.get_total_pulse_energy(self.time, self['Field/power'][-1,:])

    def get_m1(self, dimension, mu, mup):
        assert dimension in ('x', 'y')

        beta = self.beta_twiss[dimension][0]
        alpha = self.alpha_twiss[dimension][0]
        gamma = self.gamma_twiss[dimension][0]
        geom_emittance = self.geom_emittance[dimension]

        rms_bunch_length = self.get_rms_bunch_length()
        m1 = 1./geom_emittance * (beta*mup**2 + gamma*mu**2 + 2*alpha*mu*mup) * rms_bunch_length**2
        return m1

    def get_rms_bunch_length(self):
        zz = self.time*c
        curr = self['Beam/current']

        int_zz_sq = np.sum(zz**2*curr)/np.sum(curr)
        int_zz = np.sum(zz*curr)/np.sum(curr)

        return np.sqrt(int_zz_sq - int_zz**2)

    def get_geometric_emittance(self, dimension):
        assert dimension in ('x', 'y')
        geom_emittance = self['Beam/emit'+dimension][0,0]/self['Global/gamma0']

        if abs(geom_emittance - self.input['beam']['ex'])/geom_emittance < 1e-4:
            if self.warn_geo:
                print('Warning! Wrong emittance in output!')
                self.warn_geo = False
            geom_emittance /= self['Global/gamma0']

        return geom_emittance

    def get_average_beta(self, dimension):
        assert dimension in ('x', 'y')

        return np.nanmean(self.get_beta_func(dimension))

    def get_beta_func(self, dimension):
        assert dimension in ('x', 'y')

        xsize = self['Beam/%ssize' % dimension][:,0]

        # assert that beam is uniform along bunch.
        # Otherwise this method is wrong!
        s = self['Beam/%ssize' % dimension][0,:]
        assert np.nanmax(np.abs(s - np.nanmean(s))/np.nanmean(s)) < 1e-4

        em = self.get_geometric_emittance(dimension)
        return xsize**2/em


# Obsolete
class InputParser(dict):

    def __init__(self, infile, comment_chars, default_dict):
        super().__init__(self)
        self.infile = infile
        self.comment_chars = comment_chars
        self.default_dict = default_dict

        with open(self.infile, 'r') as f:
            lines = f.readlines()

        self.update(default_dict)
        for line in lines:
            line = line.strip().replace(',','').replace(';','')
            if line and line[0] in self.comment_chars:
                pass
            elif '=' in line:
                attr = line.split('=')[0].strip()
                value = line.split('=')[-1].strip()
                try:
                    value = float(value)
                except:
                    pass
                self[attr] = value


    def estimate_memory(self):
        # From Sven
        n_slices = self['slen'] / self['lambda0'] / self['sample']
        memory_field = self['ngrid']**2*n_slices*16
        memory_beam = self['npart']*n_slices*6*8
        safety_factor = 1.5 # more accurate than factor of 2?

        return (memory_field+memory_beam)*safety_factor

