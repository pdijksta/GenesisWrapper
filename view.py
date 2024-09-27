import h5py
import numpy as np

class ViewBase:
    def __init__(self, outfile, getter_factor=None, croptime=None):
        self.outfile = outfile
        self.getter_factor = getter_factor
        self.croptime = croptime
        self._dict = {}

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

    def __getitem__(self, key):
        if key not in self._dict:
            with h5py.File(self.outfile, 'r') as ff:
                try:
                    raise_ = False
                    shape = ff[key].shape
                    if len(shape) == 1 or self.getter_factor is None:
                        val = np.array(ff[key])
                    else:
                        val = np.array(ff[key][:,::self.getter_factor])
                except KeyError:
                    raise_ = True
                # Reduce verbosity
                if raise_:
                    raise KeyError('Key %s not found in %s' % (key, self.outfile))
                if len(val.shape) == 1:
                    val = np.squeeze(val)
                if val.ndim > 1 and self.croptime is not None:
                    val = val[:,:self.croptime]

                val.setflags(write=False) # Immutable array
                self._dict[key] = val
        return self._dict[key]

