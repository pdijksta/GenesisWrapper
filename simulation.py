import os
import numpy as np
import h5py

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
            self.zplot = None
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
        safety_factor = 1.2 # more accurate than factor of 2?

        return (memory_field+memory_beam)*safety_factor

