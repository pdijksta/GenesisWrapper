import numpy as np
import h5py

class GenesisSimulation:
    def __init__(self, filepath_in, filepath_out):
        self.infile = filepath_in
        self.outfile = filepath_out

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
                print((key, ff[key].shape))
            except:
                print(key)

        with h5py.File(self.outfile, 'r') as ff:
            ff.visit(lambda x: name_and_size(x, ff))

