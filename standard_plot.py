import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from . import myplotstyle as ms
from .gaussfit import GaussFit

class StandardPlot:
    def __init__(self, sim):
        self.sim = sim
        self.mask_current = sim['Beam/current'].squeeze() != 0
        self.time = sim.time[self.mask_current]

def plot(sim, title=None):
    mask_current = sim['Beam/current'].squeeze() != 0
    time = sim.time[mask_current]
    #z_plot = time*c

    if title is None:
        title = 'Standard plot for %s' % sim.infile
    fig = ms.figure(title)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    _subplot = ms.subplot_factory(3,4)
    subplot_list = []

    def subplot(*args, **kwargs):
        output = _subplot(*args, **kwargs)
        subplot_list.append(output)
        return output
    sp_ctr = 1
    sp = None

    #sp = subplot(sp_ctr, title='Beam centroid coordinates', xlabel='z [m]', ylabel='x [m]', sciy=True, scix=True)
    #sp2 = sp.twinx()
    #ms.sciy()
    #sp2.set_ylabel('x\'')
    #sp_ctr += 1
    #sp.plot(z_plot, sim['Beam/xposition'][0,:][mask_current], label='xpos begin')
    #sp2.plot(z_plot, sim['Beam/pxposition'][0,:][mask_current], label='pxpos begin',color='orange')
    #sp.plot(z_plot, sim['Beam/xposition'][-1,:][mask_current], label='xpos end')
    #sp2.plot(z_plot, sim['Beam/pxposition'][-1,:][mask_current], label='pxpos end',color='red')
    #ms.comb_legend(sp, sp2)

    sp = subplot(sp_ctr, title='Projected optics', xlabel='s [m]', ylabel=r'$\beta$ [m]')
    sp_ctr += 1

    #x0 = np.sqrt(sim['Beam/xposition']**2+sim['Beam/xsize']**2)
    for xy in 'x','y':
        x0 = sim['Beam/%ssize' % xy]**2/sim['Beam/%ssize' % xy][0]**2*sim['Beam/beta%s' % xy]
        x = np.nansum(x0*sim['Beam/current'].squeeze(), axis=1)/np.sum(sim['Beam/current'].squeeze())
        sp.plot(sim.zplot, x, label=xy)

    sp.legend()

    sp = sp_slice = subplot(sp_ctr, title='Slice invariant', xlabel='t [s]', ylabel='$\epsilon$ (single-particle)/$\epsilon_0$', scix=True)
    sp_ctr += 1
    ref = int(np.argmin((time-4e-14)**2).squeeze())
    sp.plot(time, sim.getSliceSPEmittance('x', ref=ref)[mask_current], label='x')
    sp.plot(time, sim.getSliceSPEmittance('y', ref=ref)[mask_current], label='y')
    sp.axvline(time[len(time)//2], color='black', ls='--')
    sp.legend()

    sp = subplot(sp_ctr, title='Mismatch', xlabel='t [s]', ylabel='M', scix=True, sharex=sp_slice)
    sp_ctr += 1

    mean_slice = len(time)//2
    isnan = np.isnan(sim['Beam/energy'][0])
    energy = sim['Beam/energy'][0][~isnan]
    current = sim['Beam/current'][0][~isnan]
    mean_energy = np.sum(energy*current)/np.sum(current)
    for xy in 'x','y':
        beta = sim['Beam/beta%s' % xy].squeeze()[mask_current]
        alpha = sim['Beam/alpha%s' % xy].squeeze()[mask_current]/mean_energy
        gamma = (1+alpha**2)/beta

        mismatch = (beta*gamma[mean_slice] - 2*alpha*alpha[mean_slice] + gamma*beta[mean_slice])/2.
        sp.plot(time, mismatch, label=xy)
    sp.axvline(time[mean_slice], ls='--', color='black')
    sp.legend()

    sp = subplot(sp_ctr, title='Beam current', xlabel='t [s]', ylabel='I [A]', sciy=True, scix=True, sharex=sp_slice)
    sp_ctr += 1
    sp.plot(time, sim['Beam/current'].squeeze()[mask_current])

    sp = subplot(sp_ctr, title='Emittance', xlabel='t [s]', ylabel='$\epsilon$', sciy=True, scix=True, sharex=sp_slice)
    sp_ctr += 1
    sp.plot(time, sim['Beam/emitx'][0,:][mask_current], label='$\epsilon_x$')
    sp.plot(time, sim['Beam/emity'][0,:][mask_current], label='$\epsilon_y$')
    sp.legend()


    sp = subplot(sp_ctr, title='Initial Energy', xlabel='t [s]', ylabel='$\gamma$', sciy=True, scix=True, sharex=sp_slice)
    sp_ctr += 1
    sp.plot(time, sim['Beam/energy'][0,:][mask_current])

    sp = subplot(sp_ctr, title='Pulse energy', xlabel='s [m]', ylabel='Energy [J]', sciy=True)
    sp_ctr += 1
    sp.semilogy(sim.zplot, np.trapz(sim['Field/power'], sim.time, axis=1))

    sp = subplot(sp_ctr, title='Final Pulse', xlabel='t [s]', ylabel='Power [W]', sciy=True, sharex=sp_slice)
    sp_ctr += 1
    sp.plot(time, sim['Field/power'][-1,mask_current])

    sp = subplot(sp_ctr, title='Spectrum', xlabel='$\lambda$ [m]', ylabel='Power')
    sp_ctr += 1
    xx, spectrum = sim.get_wavelength_spectrum()
    sp.semilogy(c/xx, spectrum)

    gf = GaussFit(c/xx, spectrum, sigma_00=1e-13)
    sp.plot(gf.xx, gf.yy, label='%e m' % gf.sigma)
    sp.legend()

    for dim in ('x', 'y'):
        sp = subplot(sp_ctr, title='Centroid movement %s' % dim, xlabel='s [m]', ylabel='Displacement [m]', sciy=True)
        sp_ctr += 1
        len_ = np.sum(mask_current)
        n_slices = 10
        for n_index, index in enumerate(np.linspace(0, len_-1, n_slices)):
            if n_index not in (0, n_slices-1):
                index = int(index)
                sp.plot(sim.zplot, sim['Beam/%sposition' % dim][:,mask_current][:,index], label=n_index-1)
        sp.legend(title='Slice count')

    sp = subplot(sp_ctr, title='Initial optics', xlabel='t [s]', ylabel=r'$\beta$', scix=True, sharex=sp_slice)
    sp_ctr += 1
    for dim in ('x', 'y'):
        sp.plot(time, sim['Beam/beta%s' % dim][0][mask_current], label=dim)
    sp.legend()

    return fig, subplot_list

