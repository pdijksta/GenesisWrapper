import numpy as np
from scipy.constants import hbar, h, c, e

sqrt = np.emath.sqrt

crystal_table = {}
crystal_table['diamond'] = { # Shvyd'Ko and Lindberg 2012, Appendix
        'd_H': 3.56712e-10,
        (3, 3, 3): {
            'E_H': 9.03035e3,
            'Lambda_bar_s_H': 7.83e-6,
            'w_s_H': 0.89e-5,
            'Delta_E_H': 27.3e-3,
            },
        (0, 0, 4): {
            'E_H': 6.95161e3,
            'Lambda_bar_s_H': 3.63e-6,
            'w_s_H': 1.51e-5,
            'Delta_E_H': 60.6e-3,
            },
        }

chi_0_dict = {
        ('diamond', 0, 0, 4, 9.83e3): -0.15124E-4+ 1j*0.13222e-7,
        }

chi_h_dict = {
        #('diamond', 0, 0, 4, 9.83e3, 'sigma'): -0.37824e-5 + 1j*0.12060e-7,
        }


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


class Crystal:
    def __init__(self, material, cut, hkl, thickness, photon_energy, polarization='sigma'):
        self.material = material
        self.hkl = hkl
        self.cut = cut
        self.polarization = polarization
        self.material_properties = crystal_table[self.material][hkl].copy()
        self.material_properties['d_H'] = d_H = crystal_table[self.material]['d_H']
        self.material_properties['d'] = thickness

        self.photon_energy = photon_energy
        self.lambda0 = h*c/(self.photon_energy*e)
        self.K0 = 2*np.pi/self.lambda0
        self.omega = self.K0*c

        a1=[d_H, 0, 0]
        a2=[0, d_H, 0]
        a3=[0, 0, d_H]
        self.d_H = crystal_plane_distance(a1,a2,a3,np.array(self.hkl))
        self.H = 2*np.pi/self.d_H
        self.theta = np.arcsin(self.H/(2*self.K0))
        chi_h_key = (self.material,)+hkl+(self.photon_energy, self.polarization)
        if chi_h_key in chi_h_dict:
            print('lambda old', self.material_properties['Lambda_bar_s_H'])
            P = 1 if polarization == 'sigma' else np.cos(2*self.theta)
            self.material_properties['Lambda_bar_s_H'] = np.sin(self.theta)/(self.K0*np.abs(P)*np.sqrt(chi_h_dict[chi_h_key]**2))
            print('lambda new', self.material_properties['Lambda_bar_s_H'])
        elif self.polarization == 'pi':
            self.material_properties['Lambda_bar_s_H'] = self.material_properties['Lambda_bar_s_H']/np.cos(2*self.theta)
        if cut != (1, 0, 0):
            raise NotImplementedError
        self.psi_0 = self.theta + 0 - np.pi/2 # Caption of Fig. 1 from Shvydko & Lindberg 2012
        self.gamma_0 = np.cos(self.theta - np.pi/2)
        self.gamma_H = np.cos(self.theta + np.pi/2) # Eq. 15 from Shvydko & Lindberg 2012
        self.b = self.gamma_0/self.gamma_H # Eq. 15 from Shvydko & Lindberg 2012
        self.Lambda_bar_H = np.sqrt(self.gamma_0*np.abs(self.gamma_H))/np.sin(self.theta)*self.material_properties['Lambda_bar_s_H'] # Eq. 40 from Shvydko & Lindberg 2012
        self.A = self.material_properties['d'] / self.Lambda_bar_H
        self.w_H = self.material_properties['w_s_H'] * (self.b-1)/(2*self.b) # Eq. 44 from Shvydko & Lindberg 2012
        Tau_s_Lambda = 2*self.material_properties['Lambda_bar_s_H']/c # Eq. 43 from Shvydko & Lindberg 2012
        self.Tau_Lambda = Tau_s_Lambda*np.sqrt(np.abs(self.b))*np.sin(self.theta) # Eq. 42 from Shvydko & Lindberg 2012

        self.chi_0 = -self.material_properties['w_s_H']*2*np.sin(self.theta)**2 # Eq. 44 from Shvydko & Lindberg 2012
        self.chi_0 -= 0.001*1j*self.chi_0 # trial and error
        hkl_key = (self.material,)+hkl+(photon_energy,)
        if hkl_key in chi_0_dict:
            print('chi_0 before', self.chi_0)
            self.chi_0 = chi_0_dict[hkl_key]
            print('chi_0 after', self.chi_0)
        self.C = np.exp(1j*self.chi_0 * (self.K0*self.material_properties['d'])/(2*np.cos(self.psi_0))) # Eq. 45 from Shvydko & Lindberg 2012

    def calc_y(self, Omega):
        """
        Eq. 41 from Shvydko & Lindberg 2012
        """
        y = (Omega - self.w_H*self.omega) * self.Tau_Lambda / (-np.sign(self.b))
        return y

    def calc_R_00(self, Omega):
        """
        Eq. 38 from Shvydko & Lindberg 2012
        """
        G = 1 # Assumption for symmetric crystals
        y = self.calc_y(Omega)
        Y_1 = -y + sqrt(y**2 + self.b/np.abs(self.b))
        Y_2 = -y - sqrt(y**2 + self.b/np.abs(self.b))
        R_1 = G*Y_1
        R_2 = G*Y_2
        kappa_1d = self.chi_0 * (self.K0*self.material_properties['d'])/(2*self.gamma_0) + self.A/2*Y_1
        kappa_2d = self.chi_0 * (self.K0*self.material_properties['d'])/(2*self.gamma_0) + self.A/2*Y_2

        R_00 = np.exp(1j*kappa_1d) * (R_2 - R_1) / (R_2 - R_1*np.exp(1j*(kappa_1d - kappa_2d)))
        return TransferFunction(Omega, R_00, self.C, self.omega)


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
        return TransferFunction(Omega, outp, self.C, self.omega)

class TransferFunction:
    def __init__(self, Omega, R_00, C, omega_ref):
        self.Omega = Omega
        self.R_00 = R_00
        self.C = C
        self.omega_ref = omega_ref

        self.R_tilde_00 = self.R_00 - self.C
        self.f_diff = (Omega[1]-Omega[0])/(2*np.pi)
        self.xi_0 = np.fft.fftshift(np.fft.fftfreq(len(Omega), self.f_diff))
        self.G_tilde_00 = np.fft.fftshift(np.fft.fft(self.R_tilde_00)) * np.exp(1j*self.omega_ref*self.xi_0) # Eq. 47 from Shvydko & Lindberg 2012

        mask_xi = self.xi_0 > 0
        self.xi_0 = self.xi_0[mask_xi]
        self.G_tilde_00 = self.G_tilde_00[mask_xi]
        #self.G_00 = np.fft.fftshift(np.fft.fft(self.R_00)) * np.exp(1j*self.omega_ref*self.xi_0)

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
        assert abs((diff_time - diff_xi)) / diff_xi < 1e-4
        assert len(time) <= len(self.xi_0)

        input_field = np.sqrt(power_amplitude)*np.exp(1j*phase)
        output_field = self.C * (input_field + np.convolve(input_field, self.G_tilde_00)[:len(time)]) # Eq. 5 from Yang and Shvydko 2015
        power_amplitude2 = np.abs(output_field)**2
        phase2 = np.angle(output_field)
        return time, power_amplitude2, phase2

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

        sp_tilde_r00 = subplot(sp_ctr, title='Thickness %.2f mm' % (thickness*1e3), xlabel='$E$-$E_c$ (eV)', ylabel=r'$|\tilde{R}_{00}(E)|^2$')
        sp_ctr += 1

        sp_tilde_g00 = subplot(sp_ctr, title='Thickness %.2f mm' % (thickness*1e3), xlabel=r'$\xi_0$ (fs)', ylabel=r'$|\tilde{G}_{00}(\xi_0)|^2$')
        sp_ctr += 1

        crystal = Crystal('diamond', (1, 0, 0), (0, 0, 4), thickness, E_c, 'sigma')
        for transfer_function in [
                crystal.calc_R_00(Omega_arr),
                crystal.calc_R_00_v2(Omega_arr),
                ]:
            R_tilde_00 = transfer_function.R_tilde_00
            R_00 = transfer_function.R_00
            sp_r00.plot(E_arr[E_mask], np.abs(R_00[E_mask])**2)

            sp_tilde_r00.plot(E_arr[E_mask], np.abs(R_tilde_00[E_mask])**2)
            sp_tilde_g00.semilogy(transfer_function.xi_0*1e15, np.abs(transfer_function.G_tilde_00)**2)
        sp_tilde_g00.set_xlim(0, 300)

    ms.show()

