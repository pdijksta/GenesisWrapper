import os
import numpy as np
import h5py

class GenesisSimulation:
    def __init__(self, infile):
        self.infile = infile
        self.input = InputParser(self.infile)
        dirname = os.path.dirname(self.infile)
        self.outfile = os.path.join(dirname, self.input['rootname']+'.out.h5')

        self._dict = {}
        self.zplot = self['Global/zplot']
        self.time = self['Global/time']

    def __getitem__(self, key):
        if key not in self._dict:
            with h5py.File(self.outfile, 'r') as ff:
                self._dict[key] = np.array(ff[key])
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

class InputParser(dict):


    def __init__(self, infile):
        dict.__init__(self)
        self.infile = infile

        with open(self.infile, 'r') as f:
            lines = f.readlines()

        self.update({
            'npart': 8192.,
            'sample': 2.,
            })
        for line in lines:
            line = line.strip()
            if '=' in line:
                attr = line.split('=')[0].strip()
                value = line.split('=')[-1].strip()
                try:
                    value = float(value)
                except:
                    pass
                self[attr] = value


    def estimate_memory(self):
        n_slices = self['slen'] / self['lambda0'] / self['sample']
        memory_field = self['ngrid']**2*n_slices*16
        memory_beam = self['npart']*n_slices*6*8
        safety_factor = 1.5

        return (memory_field+memory_beam)*safety_factor
