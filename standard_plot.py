import numpy as np
from scipy.constants import c, m_e, e, h
import matplotlib.pyplot as plt

from PassiveWFMeasurement import myplotstyle as ms
from PassiveWFMeasurement import h5_storage

from . import simulation
from .gaussfit import GaussFit

m_e_eV = m_e*c**2/e

def plot(sim, title=None, s_final_pulse=None, n_slices=10, fit_pulse_length=None, centroid_dim='x', figsize=(12,10), cut_spectrum=False):
    """
    fit_pulse_length may be 'gauss'
    Output: {} with keys fig, subplot_list, gf
    """
    mask_current = sim['Beam/current'].squeeze() != 0
    time = sim.time[mask_current]
    #z_plot = time*c

    if title is None:
        title = 'Standard plot for %s' % sim.infile
    fig = ms.figure(title, figsize=figsize)
    plt.subplots_adjust(wspace=0.35, hspace=0.3)

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

    sp = subplot(sp_ctr, title='Projected optics', xlabel='s (m)', ylabel=r'$\beta$ (m)')
    sp_ctr += 1

    #x0 = np.sqrt(sim['Beam/xposition']**2+sim['Beam/xsize']**2)
    for xy in 'x','y':
        size0 = sim['Beam/%ssize' % xy][0]
        x0 = np.zeros_like(sim['Beam/%ssize' % xy])
        mx0 = size0 != 0
        x0[:,mx0] = sim['Beam/%ssize' % xy][:,mx0]**2/size0[mx0]**2*sim['Beam/beta%s' % xy][:,mx0]
        beta = np.nansum(x0*sim['Beam/current'].squeeze(), axis=1)/np.sum(sim['Beam/current'].squeeze())
        sp.plot(sim.zplot, beta, label=xy)
        #print(title, xy, np.mean(beta))

    sp.legend()

    sp_inv = subplot(sp_ctr, title='Slice invariant w.r.t. proj.', xlabel='t (fs)', ylabel='$\epsilon$ (single-particle)/$\epsilon_0$')
    sp_ctr += 1
    #ref = int(np.argmin((time-4e-14)**2).squeeze())
    ref = 'proj'
    try:
        sp_inv.plot(time*1e15, sim.getSliceSPEmittance('x', ref=ref)[mask_current], label='x')
        sp_inv.plot(time*1e15, sim.getSliceSPEmittance('y', ref=ref)[mask_current], label='y')
        sp_inv.legend()
    except:
        pass
    #sp.axvline(time[len(time)//2], color='black', ls='--')

    sp_mm = subplot(sp_ctr, title='Mismatch w.r.t. projected', xlabel='t (fs)', ylabel='M', sharex=sp_inv)
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
        sp_mm.plot(time*1e15, mismatch, label=xy)

    sp_mm.set_ylim(1, None)

    #sp.axvline(time[mean_slice], ls='--', color='black')
    sp.legend()

    sp = subplot(sp_ctr, title='Beam current', xlabel='t (fs)', ylabel='I (kA)', sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time*1e15, sim['Beam/current'].squeeze()[mask_current]/1e3, label='Current')

    sp2 = sp.twinx()
    sp2.set_ylabel('Init. energy spread (MeV)')
    espread = sim['Beam/energyspread'][0,mask_current]*m_e_eV
    sp2.plot(time*1e15, espread/1e6, color='tab:orange', label='Espread')
    ms.comb_legend(sp, sp2)

    sp = subplot(sp_ctr, title='Emittance', xlabel='t (fs)', ylabel='$\epsilon_n$ (nm)', sciy=False, sharex=sp_inv)
    sp_ctr += 1
    sp.plot(time*1e15, sim['Beam/emitx'][0,:][mask_current]*1e9, label='$\epsilon_x$')
    sp.plot(time*1e15, sim['Beam/emity'][0,:][mask_current]*1e9, label='$\epsilon_y$')
    sp.legend()


    sp_slice_ene = subplot(sp_ctr, title='Slice energy', xlabel='t (fs)', ylabel='E (GeV)', sharex=sp_inv)
    sp_ctr += 1

    sp = subplot(sp_ctr, title='Pulse energy', xlabel='s (m)', ylabel='Energy (J)', sciy=True)
    sp_ctr += 1
    sp.semilogy(sim.zplot, np.trapz(sim['Field/power'], -sim.time, axis=1), label='Pulse energy')
    sp2 = sp.twinx()
    sp2.set_ylabel('Rms duration (fs)')
    _, pl = sim.pulselength_evolution(method='rms')
    sp2.plot(sim.zplot, pl*1e15, color='tab:orange', label='Pulse duration')
    ms.comb_legend(sp, sp2)

    if s_final_pulse is None:
        index_final_pulse = -1
        z_final_pulse = sim.zplot[-1]
    else:
        index_final_pulse = np.argmin((sim.zplot-s_final_pulse)**2).squeeze()
        z_final_pulse = sim.zplot[index_final_pulse]
    sp = subplot(sp_ctr, title='Final Pulse at %i m' % round(z_final_pulse), xlabel='t (fs)', ylabel='Power (W)', sciy=True, sharex=sp_inv)
    sp_ctr += 1

    yy_final_pulse = sim['Field/power'][index_final_pulse]
    sp.plot(sim.time*1e15, yy_final_pulse)
    gf = None
    if fit_pulse_length == 'gauss':
        gf = GaussFit(sim.time, yy_final_pulse)
        sp.plot(gf.xx*1e15, gf.yy, label='$\sigma$=%.3e' % gf.sigma)
        sp.legend()
    elif fit_pulse_length is None:
        pass
    else:
        raise ValueError('fit_pulse_length', fit_pulse_length)


    sp = subplot(sp_ctr, title='Spectrum at %i m' % round(z_final_pulse), xlabel='$E$ (eV)', ylabel='Power (arb. units)')
    sp_ctr += 1
    xx, spectrum = sim.get_frequency_spectrum(z_index=index_final_pulse)
    phot_energy = xx*h/e
    if cut_spectrum:
        mean_E = np.sum(phot_energy * spectrum) / np.sum(spectrum)
        rms = np.sqrt(np.sum(phot_energy**2 * spectrum) / np.sum(spectrum) - (mean_E)**2)
        mask = np.logical_and(phot_energy > mean_E - 2*rms, phot_energy < mean_E + 2*rms)
        sp.plot(phot_energy[mask], spectrum[mask])
    else:
        sp.semilogy(phot_energy, spectrum)

    #gf = GaussFit(c/xx, spectrum, sigma_00=1e-13)
    #sp.plot(gf.xx, gf.yy, label='%e m' % gf.sigma)
    #sp.legend()

    sp = subplot(sp_ctr, title='Centroid movement %s' % centroid_dim, xlabel='s (m)', ylabel='Displacement ($\mu$m)', sciy=True)
    sp_ctr += 1
    len_full_position = int(np.sum(mask_current))
    remainder = len_full_position % n_slices
    len_full_position -= remainder
    len_first = sim['Beam/%sposition' % centroid_dim].shape[0]
    position0 = sim['Beam/%sposition' % centroid_dim][:, mask_current]
    reshaped_position = position0[:,remainder:].reshape((len_first, n_slices, len_full_position//n_slices))
    reshaped_t = time[remainder:].reshape((n_slices, len_full_position//n_slices))

    averaged_position = np.mean(reshaped_position, axis=-1)
    averaged_t = np.mean(reshaped_t, axis=-1)

    for n_index in range(n_slices):
        color = ms.colorprog(n_index, n_slices)
        sp.plot(sim.zplot, averaged_position[:, n_index]*1e6, label=n_index, color=color)
        sp_inv.axvline(averaged_t[n_index]*1e15, ls='--', color=color)
    #sp.legend(title='Slice count')

    sp = subplot(sp_ctr, title='Initial slice optics', xlabel='t (fs)', ylabel=r'$\beta$ (m)', sharex=sp_inv)
    sp_ctr += 1
    for dim in ('x', 'y'):
        sp.plot(time*1e15, sim['Beam/beta%s' % dim][0][mask_current], label=dim)
    sp.legend()

    sp = subplot(sp_ctr, title='Pulse evolution', xlabel='t (fs)', ylabel='Power (W)', sciy=True)
    sp_ctr += 1
    for z_pos in np.arange(10, sim.zplot.max(), 10):
        index = np.argmin((sim.zplot-z_pos)**2).squeeze()
        z_final_pulse = sim.zplot[index]
        yy = sim['Field/power'][index,mask_current]
        label = '%i' % int(z_pos)
        sp.semilogy(time[::100]*1e15, yy[::100], label=label)
        sp_slice_ene.plot(time*1e15, sim['Beam/energy'][index,mask_current]*m_e_eV/1e9, label=label)
    sp.set_ylim(1e5,None)
    sp.legend(title='s (m)')
    sp_slice_ene.legend(title='s (m)')

    outp_dict = {
            'fig': fig,
            'subplot_list': subplot_list,
            'gf': gf
            }

    return outp_dict

def self_seeding_plot(sim2, seed_file, sim1=None, standard_plot=True, sase_spectrum_plot_range=30, figtitle=None, s_stage2=None, seed_spectrum_plot_range=4):
    if standard_plot:
        if sim1:
            plot(sim1)
        plot(sim2)


    figtitle = figtitle or 'Self-seeding simulation %s' % sim2.infile
    fig = ms.figure(figtitle)
    fig.set_constrained_layout(True)
    subplots = []
    _subplot = ms.subplot_factory(2, 3)
    sp_ctr = 1

    def subplot(*args, **kwargs):
        sp = _subplot(*args, **kwargs)
        subplots.append(sp)
        return sp

    sp_pulse_energy_evo = subplot(sp_ctr, title='Pulse energy evolution', xlabel='$s$ (m)', ylabel='$E$ (J)')
    sp_ctr += 1
    sp_pulse_energy_evo.set_yscale('log')

    seed_data = h5_storage.loadH5Recursive(seed_file)

    s_stage2 = s_stage2_plot = s_stage2 or sim2.zplot[-1]

    if sim1:
        z_index = seed_data['z_index']
        zplot1 = sim1.zplot[:z_index]
        zplot = np.concatenate([zplot1, zplot1[-1]+sim2.zplot])
        pulse_energy = np.concatenate([sim1.get_all_pulse_energy()[:z_index], sim2.get_all_pulse_energy()])
        sp_pulse_energy_evo.axvline(zplot1[-1], ls='--', color='black')
        s_stage2_plot += zplot1[-1]
    else:
        zplot = sim2.zplot
        pulse_energy = sim2.get_all_pulse_energy()

    sp_pulse_energy_evo.axvline(s_stage2_plot, color='tab:red', ls='--')

    sp_pulse_energy_evo.plot(zplot, pulse_energy)

    sp_sase_spectrum = subplot(sp_ctr, title='SASE spectrum', xlabel='$E$ (eV)', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    xx = seed_data['mult_outp']['photon_energy_sim']
    xx_ref = seed_data['mult_outp']['photon_energy_sim_ref']
    mask = np.abs(xx - xx_ref) < sase_spectrum_plot_range/2
    sp_sase_spectrum.plot(xx[mask], np.abs(seed_data['mult_outp']['spectrum_sim'][mask])**2)
    sp_sase_spectrum.axvline(xx_ref, color='black', ls='--')

    sp_crystal_wake = subplot(sp_ctr, title='Crystal wake', xlabel='$t$ (fs)', ylabel=r'$|\tilde{G}^2_{00}|$ (fs$^{-2}$)')
    sp_ctr +=1
    sp_crystal_wake.set_yscale('log')

    sp_crystal_trans = subplot(sp_ctr, title='Crystal and seed spectrum', xlabel='$E$ (eV)', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    xx = seed_data['mult_outp']['mult_photon_energy'] + xx_ref
    yy = yy0 = seed_data['mult_outp']['mult_spectrum']
    yy = np.abs(yy - yy[-1])**2
    mask = np.abs(xx - xx_ref) < seed_spectrum_plot_range/2

    f_diff = (xx[1]-xx[0])*e/h
    wake_time = np.fft.fftshift(np.fft.fftfreq(len(xx),f_diff))
    wake = np.fft.fftshift(np.fft.fft(np.fft.fftshift(yy0-yy0[-1]))*(f_diff))
    mask_t = wake_time>0
    sp_crystal_wake.plot(wake_time[mask_t]*1e15, np.abs(wake[mask_t])**2/1e30)

    yy2 = np.abs(seed_data['mult_outp']['spectrum_wake'])**2
    yy2 = yy2/yy2.max()*yy.max()

    time = seed_data['s']/c
    field = np.sqrt(seed_data['power'])
    phase = seed_data['phase']

    time2, field2, phase2 = simulation.prepare_arrays(time, field, phase, multiply_length=5)
    xx3, yy3 = simulation.get_frequency_spectrum2(time2, field2, phase2, sim2['Global/lambdaref'])
    xx3 = xx3*h/e
    mask3 = np.abs(xx3 - xx_ref) < seed_spectrum_plot_range/2
    yy3 = yy3/yy3[mask3].max()*yy.max()

    #sp_crystal_trans.plot(xx[mask], yy2[mask], label='Entire wake')
    sp_crystal_trans.plot(xx3[mask3], yy3[mask3], label='Wake seen by beam')
    sp_crystal_trans.plot(xx[mask], yy[mask], label='Crystal trans.')

    mask_t = seed_data['mult_outp']['time'] > 0

    sp_all_seed_power = subplot(sp_ctr, title='Transmitted pulse amp', xlabel='$t$ (fs)', ylabel='$P$ (W)')
    sp_ctr += 1
    sp_all_seed_power.set_yscale('log')
    sp_all_seed_power.plot(seed_data['mult_outp']['time'][mask_t]*1e15, seed_data['mult_outp']['power'][mask_t])

    sp_all_seed_phase = sp_all_seed_power.twinx()
    sp_all_seed_phase.set_ylabel('Phase (rad)', color='tab:orange')
    mask_phase = seed_data['mult_outp']['time'] > seed_data['t_min']
    sp_all_seed_phase.plot(seed_data['mult_outp']['time'][mask_phase]*1e15, seed_data['mult_outp']['phase'][mask_phase], color='tab:orange')

    sp_all_seed_power.axvline(seed_data['t_min']*1e15, color='black', ls='--')
    sp_all_seed_power.axvline(seed_data['t_max']*1e15, color='black', ls='--')

    z_index = sim2.z_index(s_stage2) if s_stage2 else -1

    xx4, yy4 = sim2.get_frequency_spectrum(z_index, multiply_length=5)
    xx4 = xx4*h/e
    mask = np.abs(xx4 - xx_ref) < seed_spectrum_plot_range/2
    yy4 = yy4/yy4.max()*yy.max()

    sp_crystal_trans.plot(xx4[mask], yy4[mask], label='FEL spectrum')
    sp_crystal_trans.legend()

    sp_wake_actual = subplot(sp_ctr, title='Wake input file', xlabel='$s$ ($\mu$m)', ylabel='$P$ (W)')
    sp_ctr += 1
    sp_wake_actual.set_yscale('log')
    sp_wake_phase = sp_wake_actual.twinx()
    sp_wake_phase.set_ylabel('Phase (rad)', color='tab:orange')

    sp_wake_actual.plot(seed_data['s']*1e6, seed_data['power'])
    sp_wake_phase.plot(seed_data['s']*1e6, seed_data['phase'], color='tab:orange')

    current = sim2['Beam/current'][0]
    mask_current = current != 0
    current_s = sim2['Global/s'][mask_current]

    sp_wake_actual.axvline(current_s[0]*1e6, color='black', ls='--')
    sp_wake_actual.axvline(current_s[-1]*1e6, color='black', ls='--')


    return fig, subplots

