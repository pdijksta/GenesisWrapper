import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from . import myplotstyle as ms

def plot(sim, title=None):
    mask_current = sim['Beam/current'].squeeze() != 0
    time = sim.time[mask_current]
    z_plot = time*c

    if title is None:
        title = 'Standard plot for %s' % sim.infile
    fig = ms.figure(title)
    plt.subplots_adjust(wspace=0.3)

    subplot = ms.subplot_factory(2,4)
    sp_ctr = 1

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

    sp = subplot(sp_ctr, title='Projected optics', xlabel='s [m]', ylabel='Beam size [m]', sciy=True)
    sp_ctr += 1

    x0 = np.sqrt(sim['Beam/xposition']**2+sim['Beam/xsize']**2)
    x = np.nansum(x0*sim['Beam/current'].squeeze(), axis=1)/np.sum(sim['Beam/current'].squeeze())
    y0 = np.sqrt(sim['Beam/yposition']**2+sim['Beam/ysize']**2)
    y = np.nansum(y0*sim['Beam/current'].squeeze(), axis=1)/np.sum(sim['Beam/current'].squeeze())
    sp.plot(sim.zplot, x, label='x')
    sp.plot(sim.zplot, y, label='y')

    sp.legend()

    sp = subplot(sp_ctr, title='Slice invariant', xlabel='z [m]', ylabel='$\epsilon$ (single-particle)/$\epsilon_0$', sciy=True, scix=True)
    sp_ctr += 1
    sp.plot(z_plot, sim.getSliceSPEmittance('x')[mask_current], label='x')
    sp.plot(z_plot, sim.getSliceSPEmittance('y')[mask_current], label='y')
    sp.legend()


    sp = subplot(sp_ctr, title='Beam current', xlabel='z [m]', ylabel='I [A]', sciy=True, scix=True)
    sp_ctr += 1
    sp.plot(z_plot, sim['Beam/current'].squeeze()[mask_current])

    sp = subplot(sp_ctr, title='Emittance', xlabel='z [m]', ylabel='$\epsilon$', sciy=True, scix=True)
    sp_ctr += 1
    sp.plot(z_plot, sim['Beam/emitx'][0,:][mask_current], label='$\epsilon_x$')
    sp.plot(z_plot, sim['Beam/emity'][0,:][mask_current], label='$\epsilon_y$')
    sp.legend()

    sp = subplot(sp_ctr, title='Pulse energy', xlabel='s [m]', ylabel='Energy [J]')
    sp_ctr += 1
    sp.plot(sim.zplot, np.trapz(sim['Field/power'], sim.time, axis=1))

    sp = subplot(sp_ctr, title='Final Pulse', xlabel='t [s]', ylabel='Power [W]', sciy=True)
    sp_ctr += 1
    sp.plot(time, sim['Field/power'][-1][mask_current])

    sp = subplot(sp_ctr, title='Spectrum', xlabel='$\lambda$ [m]', ylabel='Power')
    sp_ctr += 1
    xx, spectrum = sim.get_wavelength_spectrum()
    sp.plot(c/xx, spectrum)

    return fig

