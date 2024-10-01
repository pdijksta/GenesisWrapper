import os
import re
import glob
import h5py
import numpy as np
import numpy.fft as fft
from scipy.constants import c

from . import averagePower
from .view import ViewBase
from . import parser
from .gaussfit import GaussFit
from .gainlengthfit import GainLengthFit

_xy = ('x', 'y',)

class GenesisWrapperError(Exception):
    pass

class GenesisSimulation(ViewBase):

    comment_chars = ('!',)
    default_dict = {
            'npart': 8192.,
            'sample': 2.,
            }
    warn_geo = True

    def __init__(self, infile, _file_=None, max_time_len=None, croptime=None, getter_factor=None):

        if _file_ is None:
            self.infile = infile
        else:
            self.infile = os.path.join(os.path.dirname(_file_), infile)

        self.input = parser.GenesisInputParser(self.infile)
        dirname = os.path.dirname(self.infile)
        rootname = self.input['setup']['rootname']
        outfile = os.path.join(dirname, rootname+'.out.h5')
        ViewBase.__init__(self, outfile, getter_factor, croptime)
        self._init()

        particle_dump_files = glob.glob(os.path.join(dirname, rootname)+'.*.par.h5')
        particle_dump_z = []
        regex = re.compile('%s.(\d+).par.h5' % rootname)
        for pd in particle_dump_files:
            index = min(int(regex.match(os.path.basename(pd)).group(1)), len(self['Lattice/z'])-1)
            particle_dump_z.append(self['Lattice/z'][index])
        particle_dump_z, self.particle_dump_files = zip(*sorted(zip(particle_dump_z, particle_dump_files)))
        self.particle_dump_z = np.array(particle_dump_z)

    def _init(self):
        self._dict = {}
        zshape, tshape = self['Field/power'].shape
        try:
            self.zplot = self['Global/zplot']
        except KeyError:
            #print('Old version of genesis. No zplot available.')

            if zshape == self['Lattice/z'].shape[0]+1:
                print('Lattice/z shape is weird')
                self.zplot = np.append(self['Lattice/z'], self['Lattice/z'][-1]+self['Lattice/dz'][-1])
            else:
                output_step = int(self.input['track']['output_step'])
                self.zplot = self['Lattice/z'][::output_step]
        if zshape != self.zplot.shape[0]:
            raise ValueError('error', zshape, self.zplot.shape[0])
            #import pdb; pdb.set_trace()

        #time = self['Global/time']

        #if time.shape == ():
        #    time = (np.arange(0, tshape, dtype=float)*self['Global/sample']*self['Global/lambdaref']/c)[::-1]
        time = -self['Global/s']/c
        time -= time.min()
        self.time = time

        #if GenesisSimulation.warn_geo:
        #    print('Warning, adjusting for geometric emittance in buggy genesis version')
        #    GenesisSimulation.warn_geo = False

        # Moved to properties
        self._powerfit = None
        self._gaussian_pulselength = None

        self._beta_twiss, self._alpha_twiss, self._gamma_twiss = None, None, None
        self._geom_emittance = None

    @property
    def gaussian_pulselength(self):
        return np.abs(self.powerfit.sigma)

    @property
    def powerfit(self):
        if self._powerfit is None:
            self._powerfit = GaussFit(self.time, self['Field/power'][-1,:])
        return self._powerfit

    def do_powerfit(self, z_index=-1, **kwargs):
        return GaussFit(self.time, self['Field/power'][z_index,:], **kwargs)

    def pulselength_evolution(self, method, lims=None, maxrange=None):
        sigmas = []
        means = []

        for n_z, z in enumerate(self.zplot):
            xx = self.time
            yy = self['Field/power'][n_z]
            mask = None
            if lims is not None:
                mask = np.logical_and(xx >= lims[0], xx <= lims[1])
            elif maxrange is not None:
                where_max = xx[int(np.argmax(yy).squeeze())]
                mask = np.logical_and(xx >= where_max - maxrange/2, xx <= where_max + maxrange/2)
            if mask is not None:
                xx = xx[mask]
                yy = yy[mask]

            if method == 'gauss':
                gf = GaussFit(xx, yy, fit_const=False)
                sigmas.append(gf.sigma)
                means.append(gf.mean)
            elif method == 'rms':
                if np.sum(yy) == 0:
                    mean, rms = 0, 0
                else:
                    mean = np.sum(xx*yy)/np.sum(yy)
                    rms = np.sqrt(np.sum((xx-mean)**2*yy)/np.sum(yy))
                sigmas.append(rms)
                means.append(mean)
        return np.array(means), np.array(sigmas)

    @property
    def beta_twiss(self):
        if self._beta_twiss is None:
            self._beta_twiss = {x: self['Beam/%ssize' % x][0,:]**2 / self.geom_emittance[x] for x in _xy}
        return self._beta_twiss

    @property
    def alpha_twiss(self):
        if self._alpha_twiss is None:
            self._alpha_twiss = {x: self['Beam/alpha%s' % x][0,:] for x in _xy}
        return self._alpha_twiss

    @property
    def gamma_twiss(self):
        if self._gamma_twiss is None:
            self._gamma_twiss = {x: (1.+self.alpha_twiss[x]**2)/self.beta_twiss[x] for x in _xy}
        return self._gamma_twiss

    @property
    def geom_emittance(self):
        if self._geom_emittance is None:
            self._geom_emittance = {x: self.get_geometric_emittance(x) for x in _xy}
        return self._geom_emittance

    def get_rms_pulse_length(self, treshold=None):
        """
        Treshold: fraction of max value that is set to 0
        """
        time = self.time
        power = self['Field/power'][-1,:].copy()
        if treshold is not None:
            assert 0 <= treshold < 1
            power[power < (np.max(power)*treshold)] = 0
        return averagePower.get_rms_pulse_length(time, power)

    def get_total_pulse_energy(self, z_index=-1):
        return -averagePower.get_total_pulse_energy(self.time, self['Field/power'][z_index,:])

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

    def get_wavelength_spectrum(self, *args, **kwargs):
        raise ValueError('Use get_frequency_spectrum instead')

    def get_frequency_spectrum(self, z_index=-1, multiply_length=None, mask_time=None, multiply_arr=None):
        time = self.time
        if mask_time is None:
            field_abs = self['Field/intensity-farfield'][z_index,:]
            field_phase = self['Field/phase-farfield'][z_index,:]
        else:
            field_abs = self['Field/intensity-farfield'][z_index,:].copy()
            field_phase = self['Field/phase-farfield'][z_index,:].copy()
            field_abs[~mask_time] = 0
            field_phase[~mask_time] = 0

        if multiply_arr is not None:
            field_abs = field_abs*multiply_arr

        if multiply_length is not None:
            assert multiply_length % 2 == 1
            field_abs2 = np.zeros(len(field_abs)*multiply_length)
            field_phase2 = field_abs2.copy()
            index1 = (len(field_abs2)-len(field_abs))//2
            index2 = (len(field_abs2)+len(field_abs))//2
            field_abs2[index1:index2] = field_abs
            field_phase2[index1:index2] = field_phase

            field_abs = field_abs2
            field_phase = field_phase2
            t1 = time.min()
            t2 = t1 + (time.max() - t1)*multiply_length
            time = np.linspace(t1, t2, len(time)*multiply_length)
        return self._get_frequency_spectrum(time, field_abs, field_phase, self['Global/lambdaref'])

    @staticmethod
    def _get_frequency_spectrum(time, field_abs, field_phase, lambda_ref):
        signal0 = np.sqrt(field_abs)*np.exp(1j*field_phase)
        f0 = c/lambda_ref

        signal_fft = fft.fft(signal0)
        signal_fft_shift = fft.fftshift(signal_fft)

        dt = abs(np.diff(time)[0]) # "Sample" already included
        nq = 1/(2*dt)
        xx = np.linspace(f0-nq, f0+nq, signal_fft.size)

        return xx, np.abs(signal_fft_shift)

    def z_index(self, z, warn=True):
        index = int(np.squeeze(np.argmin(np.abs(self.zplot-z))))
        if warn and index in (0, len(self.zplot)-1):
            print('Warning: Index is at limit!')
        return index

    def t_index(self, t):
        index = int(np.squeeze(np.argmin(np.abs(self.time-t))))
        if index in (0, len(self.time)-1):
            print('Warning: Index is at limit!')
        return index

    def _get_vertical_size(self, dimension, index):
        assert dimension in ('x', 'y')
        tt = self['Field/%ssize' % dimension][index,:]
        power = self['Field/power'][index,:]
        return np.sum(tt*power)/np.sum(power)

    def xsize(self, index=-1):
        return self._get_vertical_size('x', index)

    def ysize(self, index=-1):
        return self._get_vertical_size('y', index)

    def convertZ(self, array):
        output_step = int(self.input['track']['output_step'])
        return np.copy(array[::output_step])

    def maskCutDrifts(self):
        # Filter out drifts
        # To be checked if this works for tapered sections
        diff = np.diff(self.zplot)
        mask_diff = np.concatenate([[True], (1-diff/diff.min())**2 < 1.01])
        return mask_diff

    def zplotCutDrifts(self):
        mask_diff = self.maskCutDrifts()
        diff_arr = np.concatenate([[0], np.diff(self.zplot)])
        zplot_cut = np.cumsum(diff_arr[mask_diff])
        return zplot_cut

    def fit_gainLength(self, limits, energy=None):
        if energy is None:
            energy = np.trapz(self['Field/power'], -self.time, axis=-1)

        mask_diff = self.maskCutDrifts()
        mask_cut = np.logical_and(self.zplot > limits[0], self.zplot < limits[1])
        mask = np.logical_and(mask_cut, mask_diff)

        energy_cut = energy[mask]
        zplot_fit = self.zplot[mask]
        return GainLengthFit(zplot_fit, energy_cut)

    def get_input_watcher(self):
        if 'importdistribution' not in self.input:
            raise GenesisWrapperError('Needs importdistribution')
        indistribution = os.path.join(os.path.dirname(self.infile), self.input['importdistribution']['file'])
        from ElegantWrapper.watcher import Watcher
        return Watcher(indistribution, no_page1=True)

    def getSliceSPEmittance(self, dimension, ref='proj'):
        assert dimension in ('x', 'y')

        if 'importdistribution' not in self.input:
            raise GenesisWrapperError('Needs importdistribution')
        indistribution = os.path.join(os.path.dirname(self.infile), self.input['importdistribution']['file'])
        #print(indistribution)
        with h5py.File(indistribution, 'r') as f:
            x = np.array(f[dimension])
            xp = np.array(f[dimension+'p'])
            #p = np.array(f['p'])
        x0 = x - x.mean()
        xp0 = xp - xp.mean()

        if ref == 'proj':
            emit_ref = np.sqrt(np.mean(x0**2)*np.mean(xp0**2) - np.mean(x0*xp0)**2)
            beta_ref = np.mean(x0**2)/emit_ref
            alpha_ref = - np.mean(xp0*x0)/emit_ref
        else:
            emit_ref = self['Beam/emit%s' % dimension][0,ref]/self['Beam/energy'][0,ref]
            beta_ref = self['Beam/beta%s' % dimension][0,ref]
            # correct a bug in Genesis in the calculation of alpha:
            alpha_ref = self['Beam/alpha%s' % dimension][0,ref]/self['Beam/energy'][0,ref]

        gamma_ref = (1+alpha_ref**2)/beta_ref

        xpos = self['Beam/%sposition' % dimension][0,:]
        xppos = self['Beam/p%sposition' % dimension][0,:]/self['Beam/energy'][0,:]
        slice_invariant = gamma_ref*xpos**2 + 2*alpha_ref*xpos*xppos + beta_ref*xppos**2

        return slice_invariant/emit_ref


class MultiGenesisSimulation(GenesisSimulation):
    def __init__(self, infiles, _file_=None, max_time_len=None, croptime=None, getter_factor=None):
        self.croptime = croptime # Wierd bug
        self.getter_factor = getter_factor

        if _file_ is None:
            self.infiles = infiles
        else:
            self.infiles = [os.path.join(os.path.dirname(_file_), x) for x in infiles]
        self.infile = self.infiles[0]

        self.input = parser.GenesisInputParser(self.infile)
        self.outfiles = []
        for infile in self.infiles:
            dirname = os.path.dirname(infile)
            self.outfiles.append(os.path.join(dirname, self.input['setup']['rootname']+'.out.h5'))

        self._init()


    def __getitem__(self, key):
        if key not in self._dict:
            outp = 0
            for outfile in self.outfiles:
                if not os.path.isfile(outfile):
                    raise FileNotFoundError(outfile)
                with h5py.File(outfile, 'r') as ff:
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
                        raise KeyError('Key %s not found in %s' % (key, outfile))
                    if len(val.shape) == 1:
                        val = np.squeeze(val)
                    if val.ndim > 1 and self.croptime is not None:
                        val = val[:,:self.croptime]

                    val.setflags(write=False) # Immutable array
                    outp += val
            self._dict[key] = outp/len(self.outfiles)

        return self._dict[key]

    def get_frequency_spectrum(self, z_index=-1):
        out_xx = 0
        out_yy = 0

        for outfile in self.outfiles:
            with h5py.File(outfile, 'r') as ff:
                field_abs = np.array(ff['Field/intensity-farfield'])[z_index,:]
                field_phase = np.array(ff['Field/phase-farfield'])[z_index,:]
            xx, yy = self._get_frequency_spectrum(self.time, field_abs, field_phase, self['Global/lambdaref'])
            out_xx += xx
            out_yy += yy

        return out_xx/len(self.outfiles), out_yy/len(self.outfiles)

    @staticmethod
    def get_infiles(glob_, basename):
        """
        Returns list of files
        """
        dirs = glob.glob(glob_)
        return [os.path.join(d, basename) for d in dirs]

def get_simulation_from_glob(glob_, *args, **kwargs):
    files = glob.glob(glob_)
    return MultiGenesisSimulation(files, *args, **kwargs)





# Obsolete, but used by ElegantWrapper
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
                except ValueError:
                    pass
                self[attr] = value


    def estimate_memory(self):
        # From Sven
        n_slices = self['slen'] / self['lambda0'] / self['sample']
        memory_field = self['ngrid']**2*n_slices*16
        memory_beam = self['npart']*n_slices*6*8
        safety_factor = 1.5 # more accurate than factor of 2?

        return (memory_field+memory_beam)*safety_factor

