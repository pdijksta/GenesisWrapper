import numpy as np
from scipy.special import jv
from scipy.constants import c, m_e, e, epsilon_0

m_e_eV = m_e*c**2/e
I_Alvfen = 4*np.pi*epsilon_0*m_e*c**3/e

def calc_k(gamma, lambdaR, lambda_u):
    return np.sqrt(2*((lambdaR*2*gamma**2)/lambda_u - 1))

def calc_lambdaR(gamma, lambda_u, k):
    return lambda_u/(2*gamma**2) * (1+k**2/2)

def chirp_to_lin_taper(n_wig, lambdaR, gamma0, lambda_u, chirp_eV_s):
    l_s = n_wig*lambdaR # slippage per module
    k0 = calc_k(gamma0, lambdaR, lambda_u) # k in first module
    chirp_gamma_s = chirp_eV_s/m_e_eV
    gamma1 = gamma0 - l_s/c * chirp_gamma_s # neg. sign since tail (large t) seeds head (small t) -> chirp needs to be inverted
    k1 = calc_k(gamma1, lambdaR, lambda_u)
    t1 = (k0-k1)/k0
    # default chirp is negative, i.e. positive t1 decreases k1 w.r.t. k0
    return t1

def chirp_to_lin_taper_saldin(n_wig, lambdaR, gamma0, lambda_u, chirp_eV_s):
    # Agrees perfectly with function above
    k0 = calc_k(gamma0, lambdaR, lambda_u)
    chirp_gamma_s = chirp_eV_s/m_e_eV
    dkdz = -(1+k0**2/2)**2/k0/gamma0**3/c*chirp_gamma_s
    k1 = k0 + dkdz * n_wig*lambda_u
    t1 = (k0-k1)/k0
    return t1

def calc_rho(sim, lambda_u):
    gamma = sim['Beam/energy'][0].mean()
    lambdaR = sim['Global/lambdaref']
    K = calc_k(gamma, lambdaR, lambda_u)
    j_arg = K**2/(4+2*K**2)
    JJ = jv(0, j_arg) - jv(1, j_arg)
    current = np.median(sim['Beam/current'][0])
    sig_sq = sim['Beam/xsize'][0].mean() * sim['Beam/ysize'][0].mean()

    rho = (K**2 * JJ**2 / (8*np.pi) * current/I_Alvfen * lambda_u**2 / (gamma**3*8*np.pi*sig_sq))**(1/3)
    return rho

