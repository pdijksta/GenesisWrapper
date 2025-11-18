import itertools
import numpy as np
from scipy.special import jv
from scipy.constants import hbar, h, c, e, epsilon_0

from PassiveWFMeasurement import h5_storage
from PassiveWFMeasurement import gaussfit
from PassiveWFMeasurement import beam_profile

try:
    from . import bragg_data
except ImportError:
    import bragg_data

sqrt = np.emath.sqrt

crystal_table = {}
crystal_table['diamond'] = { # Shvyd'Ko and Lindberg 2012, Appendix
        'A': 3.5668e-10, # https://x-server.gmca.aps.anl.gov
        (3, 3, 3): {
            #'E_H': 9.03035e3,
            'Lambda_bar_s_H': 7.83e-6,
            'w_s_H': 0.89e-5,
            #'Delta_E_H': 27.3e-3,
            },
        (0, 0, 4): {
            #'E_H': 6.95161e3,
            'Lambda_bar_s_H': 3.63e-6,
            'w_s_H': 1.51e-5,
            #'Delta_E_H': 60.6e-3,
            },
        (2, 2, 0): {
            #'E_H': 4.91561e3,
            'Lambda_bar_s_H': 1.98e-6,
            'w_s_H': 3.04e-5,
            #'Delta_E_H': 106.0e-3,
            },
        (5, 1, 1): {
            #'E_H': 4.91561e3,
            'Lambda_bar_s_H': 7.83e-6,
            'w_s_H': 30e-6,
            #'Delta_E_H': 106.0e-3,
            },
        }

#chi_0_dict = {
#        # xr0, xi0 from https://x-server.gmca.aps.anl.gov/
#        ('diamond', 0, 0, 4, 9.83e3): -0.15124E-4+ 1j*0.13222e-7,
#        ('diamond', 0, 0, 4, 9.70e3): -0.15532E-4+ 1j*0.13981e-7,
#        }
#
#chi_h_dict = {
#        # xrh, xih from https://x-server.gmca.aps.anl.gov/
#        # Somehow need to flip sign of real part
#        ('diamond', 0, 0, 4, 9.83e3, 'sigma'): -0.37824e-5 + 1j*0.12060e-7,
#        ('diamond', 0, 0, 4, 9.70e3, 'sigma'): -0.38851e-5 + 1j*0.12768e-7,
#        }


def plane_norm(plane_points):
    vec_1 = plane_points[0]-plane_points[1]
    vec_2 = plane_points[0]-plane_points[2]
    result = np.cross(vec_1,vec_2)
    result = result/np.linalg.norm(result)
    # turn x to +
    if result[0]<0:
        result = -result
    return result

def build_crystal_plane(a1,a2,a3,hkl_vec):
    plane_coordinate=[]
    plane_points=[]
    # 1. first check how many zeros
    zero_vec=[0, 0, 0]
    zero_vec=np.array(zero_vec)
    for i in range(3):
        if abs(hkl_vec[i])<0.5:
            zero_vec[i]=1
    zero_sum=np.sum(zero_vec)

    # 2. based on number of zeros, get three points on the plane
    a=[a1, a2, a3]
    a=np.array(a)
    if zero_sum==0:
        plane_points=[a[0]/hkl_vec[0], a[1]/hkl_vec[1], a[2]/hkl_vec[2]]
    elif zero_sum==1:
        for i in range(3):
            if zero_vec[i]==0:
                plane_points.append(a[i]/hkl_vec[i])
            else:
                vec_temp=a[i]
        plane_points.append(plane_points[0]+vec_temp)
    elif zero_sum==2:
        vec_temp=[]
        for i in range(3):
            if zero_vec[i]==0:
                plane_points.append(a[i]/hkl_vec[i])
            else:
                vec_temp.append(a[i])
        plane_points.append(plane_points[0]+vec_temp[0])
        plane_points.append(plane_points[0]+vec_temp[1])
    else:
        print('warning: invalid hkl / a input!')
        return

    # 3. use these three points get a, b, c, d
    vec_norm=plane_norm(plane_points)
    d=plane_points[0][0]*vec_norm[0]+plane_points[0][1]*vec_norm[1]+plane_points[0][2]*vec_norm[2]
    d=-d

    plane_coordinate=[vec_norm[0], vec_norm[1], vec_norm[2], d]
    return plane_coordinate

def crystal_plane_distance(a1,a2,a3,hkl_vec):
    hkl_vec=np.array(hkl_vec)
    plane_coordinate_1=build_crystal_plane(a1,a2,a3,hkl_vec)
    plane_coordinate_2=build_crystal_plane(a1,a2,a3,hkl_vec/2)
    return plane_distance(plane_coordinate_1,plane_coordinate_2)

def plane_distance(plane_coordinates_1,plane_coordinates_2):
    # 1. check if parallel or not
    norm1=plane_coordinates_1[0:3]
    norm2=plane_coordinates_2[0:3]
    cross_product=np.cross(norm1,norm2)
    if np.linalg.norm(cross_product)/np.linalg.norm(norm1)>1e-4:
        print('warning: unparallel planes')
        return 0
    else:
        k=np.array(norm1)/np.array(norm2)
        k=np.nanmean(k)
        return abs(plane_coordinates_2[-1]*k-plane_coordinates_1[-1])/np.linalg.norm(norm1)

def plane_angle(plane_coordinates_1,plane_coordinates_2):
    norm1=plane_coordinates_1[0:3]
    norm2=plane_coordinates_2[0:3]
    return np.arccos((norm1[0]*norm2[0]+norm1[1]*norm2[1]+norm1[2]*norm2[2])/np.sqrt(norm1[0]**2+norm1[1]**2+norm1[2]**2)/np.sqrt(norm2[0]**2+norm2[1]**2+norm2[2]**2))

class SimpleCrystal:
    def __init__(self, material, cut, hkl, thickness, photon_energy, polarization='sigma', force_table=True, allow_laue=False):
        self.material = material
        self.hkl = hkl
        self.cut = cut
        self.photon_energy = photon_energy
        self.polarization = polarization
        if polarization != 'sigma':
            raise ValueError('Pi polarization not implemented!')

        h_, k_, l_ = hkl
        for permutation in itertools.permutations([abs(h_), abs(k_), abs(l_)]):
            if permutation in crystal_table[self.material]:
                self.material_properties = crystal_table[self.material][permutation].copy()
                break
        else:
            if force_table:
                self.material_properties = {}
            else:
                raise ValueError(hkl)
        self.material_properties['A'] = A = crystal_table[self.material]['A']
        self.material_properties['d'] = thickness

        self.lambda0 = h*c/(self.photon_energy*e)
        self.K0 = 2*np.pi/self.lambda0
        self.omega_0 = self.K0*c
        self.P = 1 if self.polarization == 'sigma' else np.cos(2*self.theta)

        a1=[A, 0, 0]
        a2=[0, A, 0]
        a3=[0, 0, A]
        self.d_H = crystal_plane_distance(a1,a2,a3,np.array(self.hkl)) # distance along the plane used for diffraction
        self.H = 2*np.pi/self.d_H # reciprocal lattice vector
        theta_arg = self.H/(2*self.K0)
        if abs(theta_arg) > 1:
            raise bragg_data.PhotonEnergyException('Angle impossible')
        self.theta = np.arcsin(theta_arg) # Bragg's law
        self.eta = plane_angle(cut, hkl) # angle between H and surface

        if -self.theta < self.eta < self.theta: # Caption of Fig. 1 from Shvydko & Lindberg 2012
            self.type = 'Bragg'
        elif - self.theta < self.eta < np.pi - self.theta: # Caption of Fig. 1 from Shvydko & Lindberg 2012
            self.type = 'Laue'
            if not allow_laue:
                raise bragg_data.PhotonEnergyException('Laue geometry')
        else:
            raise bragg_data.PhotonEnergyException('Angles wrong')

        self.psi_0 = self.theta + self.eta - np.pi/2 # Caption of Fig. 1 from Shvydko & Lindberg 2012
        self.psi_H = np.pi/2 + self.theta - self.eta # Caption of Fig. 1 from Shvydko & Lindberg 2012
        self.gamma_0 = np.cos(self.psi_0) # Eq. 15 from Shvydko & Lindberg 2012
        self.gamma_H = np.cos(self.psi_H) # Eq. 15 from Shvydko & Lindberg 2012
        self.b = self.gamma_0/self.gamma_H # Eq. 15 from Shvydko & Lindberg 2012

        #self.material_properties['chi_0'] = -self.material_properties['w_s_H']*2*np.sin(self.theta)**2 # Eq. 44 from Shvydko & Lindberg 2012
        #print('before', self.material_properties['chi_0'], self.material_properties['Lambda_bar_s_H'], self.material_properties['w_s_H'])
        try:
            table_data = bragg_data.get_x0h_data(self.material, *self.hkl, photon_energy)
            self.material_properties['chi_0'] = table_data['xr0'] + 1j*table_data['xi0']
            self.material_properties['chi_H'] = table_data['xrh'] - 1j*table_data['xih'] # somehow need to flip the sign of one of the parts
            self.material_properties['Lambda_bar_s_H'] = np.sin(self.theta)/(self.K0*np.abs(self.P)*self.material_properties['chi_H']) # Eq. 40 from Shvydko & Lindberg 2012
            self.material_properties['w_s_H'] = -self.material_properties['chi_0']/(2*np.sin(self.theta)) # Eq. 44 from Shvydko & Lindberg 2012
        except:
            if force_table:
                raise
            self.material_properties['chi_0'] = -self.material_properties['w_s_H']*2*np.sin(self.theta)**2 # Eq. 44 from Shvydko & Lindberg 2012
        #print('after', self.material_properties['chi_0'], self.material_properties['Lambda_bar_s_H'], self.material_properties['w_s_H'])


        self.Tau_0 = (2*self.material_properties['Lambda_bar_s_H']**2) / (c*self.material_properties['d']/self.gamma_0) # Eq. 8 from Yang & Shvydko 2013
        self.Tau_d = (2*self.material_properties['d']*np.sin(self.theta)**2)/(c*np.abs(self.gamma_H)) # Eq. 8 from Yang & Shvydko 2013

        self.C = np.exp(1j*self.material_properties['chi_0'] * (self.K0*self.material_properties['d'])/(2*np.cos(self.psi_0))) # Eq. 45 from Shvydko & Lindberg 2012

    def calc_G_tilde_00_simple(self, xi_0):
        """
        Eq. 6 of Yang & ShvydKo (2013)
        """
        mask = xi_0 != 0
        arg = np.sqrt(xi_0[mask]/self.Tau_0 * (1+xi_0[mask]/self.Tau_d))
        G_tilde_00 = np.empty_like(xi_0, dtype=complex)
        G_tilde_00[~mask] = -1/(4*self.Tau_0)
        G_tilde_00[mask] = -1/(2*self.Tau_0) * jv(1, arg)/arg
        return TransferFunctionSimple(self.C, xi_0, G_tilde_00, self.omega_0)

    def calc_G_tilde_00_simple2(self, xi_0):
        """
        Eq. 9 of Yang & ShvydKo (2013)
        """
        mask = xi_0 != 0
        arg = np.sqrt(xi_0[mask]/self.Tau_0)
        G_tilde_00 = np.empty_like(xi_0, dtype=complex)
        G_tilde_00[~mask] = 1/(4*self.Tau_0) * np.sign(self.b)
        G_tilde_00[mask] = 1/(2*self.Tau_0) * jv(1, arg)/arg * np.sign(self.b)
        return TransferFunctionSimple(self.C, xi_0, G_tilde_00, self.omega_0)


class Crystal(SimpleCrystal):
    def __init__(self, *args, **kwargs):
        SimpleCrystal.__init__(self, *args, **kwargs)
        self.Lambda_bar_H = np.sqrt(self.gamma_0*np.abs(self.gamma_H))/np.sin(self.theta)*self.material_properties['Lambda_bar_s_H'] # Eq. 40 from Shvydko & Lindberg 2012
        self.A = self.material_properties['d'] / self.Lambda_bar_H
        self.w_H = self.material_properties['w_s_H'] * (self.b-1)/(2*self.b) # Eq. 44 from Shvydko & Lindberg 2012
        Tau_s_Lambda = 2*self.material_properties['Lambda_bar_s_H']/c # Eq. 43 from Shvydko & Lindberg 2012
        self.Tau_Lambda = Tau_s_Lambda*np.sqrt(np.abs(self.b))*np.sin(self.theta) # Eq. 42 from Shvydko & Lindberg 2012

    def calc_y(self, Omega):
        """
        Eq. 41 from Shvydko & Lindberg 2012
        """
        y = (Omega - self.w_H*(self.omega_0-Omega)) * self.Tau_Lambda / (-np.sign(self.b))
        return y

    def calc_R_00(self, Omega, outp='tf'):
        """
        Eq. 38 from Shvydko & Lindberg 2012
        """
        G = 1 # Assumption for symmetric crystals (?). Anyway it cancels out for R_00.
        y = self.calc_y(Omega)
        Y_1 = -y + sqrt(y**2 + self.b/np.abs(self.b))
        Y_2 = -y - sqrt(y**2 + self.b/np.abs(self.b))
        R_1 = G*Y_1
        R_2 = G*Y_2
        kappa_1d = self.material_properties['chi_0'] * (self.K0*self.material_properties['d'])/(2*self.gamma_0) + self.A/2*Y_1
        kappa_2d = self.material_properties['chi_0'] * (self.K0*self.material_properties['d'])/(2*self.gamma_0) + self.A/2*Y_2

        R_00 = np.exp(1j*kappa_1d) * (R_2 - R_1) / (R_2 - R_1*np.exp(1j*(kappa_1d - kappa_2d)))
        if outp == 'tf':
            return TransferFunction(Omega, R_00, self.C, self.omega_0)
        elif outp == 'mult':
            return Multiplication(Omega, R_00, self.C, self.omega_0)

    def get_mult(self, Omega):
        return self.calc_R_00(Omega, outp='mult')

    def calc_R_00_v2(self, Omega):
        """
        Eq. 49 from Shvydko & Lindberg 2012
        Analytical approximation
        """
        y = self.calc_y(Omega)
        outp = np.zeros_like(y, dtype=complex)
        mask = np.abs(y) < 1
        y1 = y[mask]
        y2 = y[~mask]
        outp[mask] = self.C*np.exp(-self.A/2*(1j*y1+np.sqrt(1-y1**2)))
        outp[~mask] = self.C*np.exp(1j*self.A/2*(-y2+np.sign(y2)*np.sqrt(y2**2-1)))
        return TransferFunction(Omega, outp, self.C, self.omega_0)

class Multiplication:
    def __init__(self, Omega, R_00, C, omega_ref):
        self.Omega = Omega
        self.R_00 = R_00
        self.C = self.R_00[-1] if C is None else C
        self.omega_ref = omega_ref

    def multiplication(self, photon_energy_sim, spectrum_sim, photon_energy_ref_sim):
        outp = {
                'photon_energy_sim': photon_energy_sim,
                'spectrum_sim': spectrum_sim,
                'photon_energy_sim_ref': photon_energy_ref_sim,
                }
        outp['mult_photon_energy'] = mult_photon_energy = photon_energy_sim - self.omega_ref*hbar/e
        outp['mult_spectrum'] = mult_spectrum = np.interp(mult_photon_energy, self.Omega*hbar/e, self.R_00)
        outp['spectrum'] = spectrum_sim*mult_spectrum
        outp['spectrum_wake'] = spectrum_sim*(mult_spectrum-self.C)
        f_diff = (photon_energy_sim[1]-photon_energy_sim[0])*e/h
        outp['time'] = np.fft.fftshift(np.fft.fftfreq(len(photon_energy_sim), f_diff))
        outp['field'] = field = np.fft.fft(np.fft.fftshift(outp['spectrum']))*f_diff
        outp['power'] = np.abs(field)**2*epsilon_0
        outp['phase'] = np.angle(field)
        outp['seed_power'] = SeedPower(outp['time'], outp['power'], outp['phase'])
        return outp


class TransferFunctionSimple:
    def __init__(self, C, xi_0, G_tilde_00, omega_ref):
        self.C = C
        self.G_tilde_00 = G_tilde_00
        self.omega_ref = omega_ref
        self.xi_0 = xi_0

    def convolute_power_profile(self, time, power_amplitude, phase, lambda_ref, max_time=None):
        if max_time is None:
            max_time = self.xi_0.max()
        assert max_time <= self.xi_0.max()
        diff_time = np.diff(time)[0]
        add_time = max_time - (time[-1] - time[0])
        if add_time > 0:
            print('Extend array')
            n_add = int(add_time // diff_time)
            time = np.arange(time[0], time[-1]+n_add*diff_time, diff_time)
            zero_arr = np.zeros(n_add, dtype=power_amplitude.dtype)
            power_amplitude = np.concatenate([power_amplitude, zero_arr])
            phase = np.concatenate([phase, zero_arr])

        photon_energy_crystal = self.omega_ref*hbar/e
        photon_energy_power_profile = c/lambda_ref*h/e
        print('Convolution with crystal photon energy %.2f eV and power profile carrier photon energy %.2f eV' % (photon_energy_crystal, photon_energy_power_profile))
        diff_xi = np.diff(self.xi_0)[0]
        assert diff_time > 0
        assert diff_xi > 0
        assert abs((diff_time - diff_xi)) / diff_xi < 1e-4
        assert len(time) <= len(self.xi_0)

        input_field = np.sqrt(power_amplitude)*np.exp(1j*phase)
        self.outp_field0 = self.C * input_field
        self.outp_field1 = self.C * diff_xi * np.convolve(input_field, self.G_tilde_00)[:len(time)] # Eq. 5 from Yang and Shvydko 2015

        self.outp_field = self.outp_field0 + self.outp_field1
        power_amplitude2 = np.abs(self.outp_field)**2
        phase2 = np.angle(self.outp_field)
        return SeedPower(time, power_amplitude2, phase2)


class SeedPower:
    def __init__(self, time, power_amplitude, phase):
        self.time = time
        self.power_amplitude = power_amplitude
        self.phase = phase

    def shift_time(self, input_time, input_power, clip=True):
        prof0 = beam_profile.AnyProfile(input_time, input_power)
        prof0.cutoff(5e-2)
        prof = beam_profile.AnyProfile(self.time, self.power_amplitude)
        prof.cutoff(5e-2)
        time_shift = prof0.mean() - prof.mean()
        self.time += time_shift
        if clip:
            mask = self.time > prof0._xx[prof0._yy != 0].min()
            self.time = self.time[mask]
            self.power_amplitude = self.power_amplitude[mask]
            self.phase = self.phase[mask]
        return time_shift

    def writeH5(self, filename, t_min, t_max, s_0=0):
        mask = np.logical_and(self.time >= t_min, self.time <= t_max)
        s = -c*self.time[mask][::-1]
        s = s - s[0] + s_0
        assert s[1] > s[0]
        outp_dict = {
                's': s,
                'power': self.power_amplitude[mask][::-1],
                'phase': self.phase[mask][::-1],
                }
        if filename is not None:
            h5_storage.saveH5Recursive(filename, outp_dict)
            print('Wrote %s with keys s, power, phase' % filename)
        return outp_dict


class TransferFunction(TransferFunctionSimple):
    def __init__(self, Omega, R_00, C, omega_ref, only_pos_time=True):
        self.Omega = Omega
        self.R_00 = R_00
        self.C = C
        self.omega_ref = omega_ref

        self.R_tilde_00 = self.R_00 - self.C
        self.f_diff = (Omega[1]-Omega[0])/(2*np.pi)
        self.xi_0 = np.fft.fftshift(np.fft.fftfreq(len(Omega), self.f_diff))
        self.G_tilde_00 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.R_tilde_00))*self.f_diff) # Eq. 47 from Shvydko & Lindberg 2012

        #print(np.trapz(np.abs(self.G_tilde_00)**2, self.Omega/2/np.pi))
        #print(np.trapz(np.abs(self.R_tilde_00)**2, self.xi_0))

        if only_pos_time:
            mask_xi = self.xi_0 >= 0
            self.xi_0 = self.xi_0[mask_xi]
            self.G_tilde_00 = self.G_tilde_00[mask_xi]

        #self.G_tilde_00_plus_phase = self.G_tilde_00 * np.exp(1j*self.omega_ref*self.xi_0)
        #self.G_tilde_00_minus_phase = self.G_tilde_00 * np.exp(-1j*self.omega_ref*self.xi_0)
        #factor = np.sqrt(1/np.abs(self.G_tilde_00[0])**2)
        #self.G_tilde_00 *= factor
        #print('Scale G_00 by factor %e' % factor)
        #self.G_00 = np.fft.fftshift(np.fft.fft(self.R_00)) * np.exp(1j*self.omega_ref*self.xi_0)


class SeedGenerator:
    def __init__(self, sim, photon_energy_window=10, pew_size=1e5, z_index=-1, crystal=None):
        self.sim = sim
        self.z_index = z_index
        self.crystal = crystal
        self.Omega_arr = np.linspace(-photon_energy_window/2, photon_energy_window/2, int(pew_size))/hbar*e
        self.freq, self.spectrum = self.sim.get_frequency_spectrum(self.z_index, multiply_length=5, key_amp='Field/power', key_phase='Field/phase-nearfield', type_='field')

    def set_crystal(self, crystal):
        self.crystal = crystal

    def get_spectrum_peak(self):
        self.gf = gaussfit.GaussFit(self.freq*h/e, np.abs(self.spectrum)**2, fit_const=False)
        return self.gf.mean

    def generate_seed(self, filename, delay):
        self.mult = mult = self.crystal.get_mult(self.Omega_arr)
        self.mult_outp = mult_outp = mult.multiplication(self.freq*h/e, self.spectrum, self.sim.photon_energy_ref)
        seed_power = mult_outp['seed_power']
        seed_power.shift_time(self.sim.time, self.sim['Field/power'][self.z_index])

        s_len = self.sim.input['time']['slen']
        current_mask = self.sim['Beam/current'][0] != 0
        blen_arr = self.sim['Global/s'][current_mask]
        bunch_len = abs(blen_arr[-1] - blen_arr[0])

        shift = (s_len-bunch_len)/c
        tmin, tmax = delay+shift, delay+bunch_len/c+shift
        seed_dict = seed_power.writeH5(filename, tmin, tmax, 0)
        return seed_dict

def generate_seed(sim, crystal, z_pos, max_time, *write_args):
    z_index = sim.z_index(z_pos)
    xi_0 = np.arange(0, max_time+5e-15, abs(np.diff(sim.time)[0]))
    tf_simple = crystal.calc_G_tilde_00_simple(xi_0)
    seed_power = tf_simple.convolute_power_profile(sim.time[::-1], sim['Field/power'][z_index][::-1], sim['Field/phase-nearfield'][z_index][::-1], sim['Global/lambdaref'], max_time=max_time)
    seed_power.writeH5(*write_args)


if __name__ == '__main__':
    from PassiveWFMeasurement import myplotstyle as ms
    ms.closeall()
    E_c = 9.83e3
    #E_c = 9.6983884e3

    E_arr = np.linspace(-5, 5, int(1e4))
    E_mask = np.logical_and(E_arr > -0.2, E_arr < 0.4)
    Omega_arr = E_arr*e/hbar

    fig = ms.figure('Fig 4 from Shvyd\'Ko and Lindberg (2012)')
    fig.set_constrained_layout(True)
    subplot = ms.subplot_factory(3, 3)
    sp_ctr = 1

    for thickness in [0.2e-3, 0.1e-3, 0.05e-3]:
        sp_r00 = subplot(sp_ctr, title='Thickness %.2f mm' % (thickness*1e3), xlabel='$E$-$E_c$ (eV)', ylabel='$|R_{00}(E)|^2$')
        sp_ctr += 1
        sp_r00_re_im = sp_r00.twinx()
        sp_r00_re_im.set_ylabel('Components')

        sp_tilde_r00 = subplot(sp_ctr, title='Thickness %.2f mm' % (thickness*1e3), xlabel='$E$-$E_c$ (eV)', ylabel=r'$|\tilde{R}_{00}(E)|^2$')
        sp_ctr += 1

        sp_tilde_g00 = subplot(sp_ctr, title='Thickness %.2f mm' % (thickness*1e3), xlabel=r'$\xi_0$ (fs)', ylabel=r'$|\tilde{G}_{00}(\xi_0)|^2$')
        sp_ctr += 1

        crystal = Crystal('diamond', (0, 0, 1), (0, 0, 4), thickness, E_c, 'sigma')
        for ctr, (transfer_function, label) in enumerate([
                (crystal.calc_R_00(Omega_arr), 'v1'),
                (crystal.calc_R_00_v2(Omega_arr), 'v2'),
                ]):
            R_tilde_00 = transfer_function.R_tilde_00
            R_00 = transfer_function.R_00
            sp_r00.plot(E_arr[E_mask], np.abs(R_00[E_mask])**2)

            if ctr == 0:
                sp_r00_re_im.plot(E_arr[E_mask], np.real(R_00[E_mask]), color='tab:red')
                sp_r00_re_im.plot(E_arr[E_mask], np.imag(R_00[E_mask]), color='tab:green')

            sp_tilde_r00.plot(E_arr[E_mask], np.abs(R_tilde_00[E_mask])**2)
            sp_tilde_g00.semilogy(transfer_function.xi_0*1e15, np.abs(transfer_function.G_tilde_00)**2, label=label)
            #print(transfer_function.C)

        tf_simple = crystal.calc_G_tilde_00_simple(transfer_function.xi_0)
        sp_tilde_g00.semilogy(tf_simple.xi_0*1e15, np.abs(tf_simple.G_tilde_00)**2, label='simple')
        tf_simple2 = crystal.calc_G_tilde_00_simple2(transfer_function.xi_0)
        sp_tilde_g00.semilogy(tf_simple2.xi_0*1e15, np.abs(tf_simple2.G_tilde_00)**2, label='simple2')

        sp_tilde_g00.set_xlim(0, 300)
        sp_tilde_g00.legend()

    ms.show()

