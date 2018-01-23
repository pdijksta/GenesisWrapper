import os
import h5py
import numpy as np
from scipy.constants import c


class GenesisSimulation:

    comment_chars = ('!',)
    default_dict = {
            'npart': 8192.,
            'sample': 2.,
            }

    def __init__(self, infile):
        self.infile = infile
        self.input = InputParser(self.infile, self.comment_chars, self.default_dict)
        dirname = os.path.dirname(self.infile)
        self.outfile = os.path.join(dirname, self.input['rootname']+'.out.h5')

        self._dict = {}
        try:
            self.zplot = self['Global/zplot']
        except KeyError:
            print('Old version of genesis. No zplot available.')
            sample = int(self['Global/sample'])
            self.zplot = np.append(self['Lattice/z'][::sample],
                    self['Lattice/z'][-1]+self['Lattice/dz'][-1])
        time = self['Global/time']
        if time.shape == ():
            time = np.arange(0, self['Beam/emitx'].shape[1], dtype=float)*self['Global/sample']*self['Global/lambdaref']/c
        self.time = time

    def __getitem__(self, key):
        if key not in self._dict:
            with h5py.File(self.outfile, 'r') as ff:
                val = np.array(ff[key])
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
        power = self['Field/power'][-1,:]

        int_power = np.trapz(power, time)
        int_time_sq = np.trapz(time**2*power, time) / int_power
        int_time = np.trapz(time*power, time) / int_power

        rms_time = np.sqrt(int_time_sq - int_time**2)
        return rms_time

    def get_total_pulse_power(self):
        return np.trapz(self['Field/power'][-1,:], self.time)

    def get_m1(self, dimension, mu, mup):
        assert dimension in ('x', 'y')

        beta = self['Beam/beta'+dimension]
        if np.all(beta < 0.01):
            print('Correcting for wrong beta in Genesis')
            beta = beta*self['Global/gamma0']
        assert not np.any(np.diff(beta) > 1e-5)

        beta = beta[0,0]
        alpha = self['Beam/alpha'+dimension][0,0]
        gamma = (1. + alpha**2)/beta
        emittance = self['Beam/emit'+dimension][0,0]

        rms_bunch_length = self.get_rms_bunch_length()
        m1 = 1./emittance * (beta*mup**2 + gamma*mu**2 + 2*alpha*mu*mup) * rms_bunch_length
        return m1

    def get_rms_bunch_length(self):
        zz = self.time*c
        curr = self['Beam/current']

        int_zz_sq = np.sum(zz**2*curr)/np.sum(curr)
        int_zz = np.sum(zz*curr)/np.sum(curr)

        return np.sqrt(int_zz_sq - int_zz**2)


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

