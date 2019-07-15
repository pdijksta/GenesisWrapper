import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from . import myplotstyle as ms
#from .gaussfit import GaussFit

class StandardPlot:
    def __init__(self, sim):
        self.sim = sim
        self.mask_current = sim['Beam/current'].squeeze() != 0
        self.time = sim.time[self.mask_current]

def plot(sim, title=None, s_final_pulse=None, n_slices=10):
    """
    Output: fig, subplots_list
    """
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
        beta = np.nansum(x0*sim['Beam/current'].squeeze(), axis=1)/np.sum(sim['Beam/current'].squeeze())
        sp.plot(sim.zplot, beta, label=xy)
        #print(title, xy, np.mean(beta))

    sp.legend()

    sp_inv = subplot(sp_ctr, title='Slice invariant w.r.t. proj.', xlabel='t [s]', ylabel='$\epsilon$ (single-particle)/$\epsilon_0$', scix=True)
    sp_ctr += 1
    #ref = int(np.argmin((time-4e-14)**2).squeeze())
    ref = 'proj'
    try:
        sp_inv.plot(time, sim.getSliceSPEmittance('x', ref=ref)[mask_current], label='x')
        sp_inv.plot(time, sim.getSliceSPEmittance('y', ref=ref)[mask_current], label='y')
    except:
        pass
    #sp.axvline(time[len(time)//2], color='black', ls='--')
    sp_inv.legend()

    sp_mm = subplot(sp_ctr, title='Mismatch w.r.t. projected', xlabel='t [s]', ylabel='M', scix=True, sharex=sp_inv)
    sp_ctr += 1

    #mean_slice = len(time)//2
    isnan = np.isnan(sim['Beam/energy'][0])
    energy = sim['Beam/energy'][0][~isnan]
    current = sim['Beam/current'][0][~isnan]
    mean_energy = np.sum(energy*current)/np.sum(current)


    for xy in 'x','y':
        beta_ref = np.sum(sim['Beam/beta%s' % xy][0][mask_current]*sim['Beam/current'][0][mask_current])/np.sum(sim['Beam/current'][0][mask_current])
        alpha_ref = np.sum(sim['Beam/alpha%s' % xy][0][mask_current]*sim['Beam/current'][0][mask_current])/np.sum(sim['Beam/current'][0][mask_current])/mean_energy

        gamma_ref = (1+alpha_ref**2)/beta_ref
        beta = sim['Beam/beta%s' % xy].squeeze()[mask_current]
        alpha = sim['Beam/alpha%s' % xy].squeeze()[mask_current]/mean_energy
        gamma = (1+alpha**2)/beta

        mismatch = (beta*gamma_ref - 2*alpha*alpha_ref + gamma*beta_ref)/2.
        sp_mm.plot(time, mismatch, label=xy)

    sp_mm.set_ylim(1, None)

    #sp.axvline(time[mean_slice], ls='--', color='black')
    sp.legend()

    sp = subplot(sp_ctr, title='Beam current', xlabel='t [s]', ylabel='I [A]', sciy=True, scix=True, sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time, sim['Beam/current'].squeeze()[mask_current])

    sp = subplot(sp_ctr, title='Emittance', xlabel='t [s]', ylabel='$\epsilon_n$', sciy=True, scix=True, sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time, sim['Beam/emitx'][0,:][mask_current], label='$\epsilon_x$')
    sp.plot(time, sim['Beam/emity'][0,:][mask_current], label='$\epsilon_y$')
    sp.legend()


    sp = subplot(sp_ctr, title='Initial Energy', xlabel='t [s]', ylabel='$\gamma$', sciy=True, scix=True, sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time, sim['Beam/energy'][0,:][mask_current])

    sp = subplot(sp_ctr, title='Pulse energy', xlabel='s [m]', ylabel='Energy [J]', sciy=True)
    sp_ctr += 1
    sp.semilogy(sim.zplot, np.trapz(sim['Field/power'], -sim.time, axis=1))

    if s_final_pulse is None:
        index_final_pulse = -1
        z_final_pulse = sim.zplot[-1]
    else:
        index_final_pulse = np.argmin((sim.zplot-s_final_pulse)**2).squeeze()
        z_final_pulse = sim.zplot[index_final_pulse]
    sp = subplot(sp_ctr, title='Final Pulse at %i m' % round(z_final_pulse), xlabel='t [s]', ylabel='Power [W]', sciy=True, sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time, sim['Field/power'][index_final_pulse,mask_current])

    sp = subplot(sp_ctr, title='Spectrum at %i m' % round(z_final_pulse), xlabel='$\lambda$ [m]', ylabel='Power')
    sp_ctr += 1
    xx, spectrum = sim.get_frequency_spectrum(z_index=index_final_pulse)
    sp.semilogy(c/xx, spectrum)

    #gf = GaussFit(c/xx, spectrum, sigma_00=1e-13)
    #sp.plot(gf.xx, gf.yy, label='%e m' % gf.sigma)
    #sp.legend()

    for dim in ('x', 'y'):
        sp = subplot(sp_ctr, title='Centroid movement %s' % dim, xlabel='s [m]', ylabel='Displacement [m]', sciy=True)
        sp_ctr += 1
        len_full_position = int(np.sum(mask_current))
        remainder = len_full_position % n_slices
        len_full_position -= remainder
        len_first = sim['Beam/%sposition' % dim].shape[0]
        position0 = sim['Beam/%sposition' % dim][:, mask_current]
        reshaped_position = position0[:,remainder:].reshape((len_first, n_slices, len_full_position//n_slices))
        reshaped_t = time[remainder:].reshape((n_slices, len_full_position//n_slices))

        averaged_position = np.mean(reshaped_position, axis=-1)
        averaged_t = np.mean(reshaped_t, axis=-1)

        for n_index in range(n_slices):
            color = ms.colorprog(n_index, n_slices)
            sp.plot(sim.zplot, averaged_position[:, n_index], label=n_index, color=color)
            sp_inv.axvline(averaged_t[n_index], ls='--', color=color)
        #sp.legend(title='Slice count')

    sp = subplot(sp_ctr, title='Initial slice optics', xlabel='t [s]', ylabel=r'$\beta$ [m]', scix=True, sharex=sp_inv)
    sp_ctr += 1
    for dim in ('x', 'y'):
        sp.plot(time, sim['Beam/beta%s' % dim][0][mask_current], label=dim)
    sp.legend()

    return fig, subplot_list

