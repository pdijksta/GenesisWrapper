import numpy as np
from scipy.constants import c, m_e, e

m_e_eV = m_e*c**2/e
def calc_k(gamma, lambdaR, lambda_u):
    return np.sqrt(2*((lambdaR*2*gamma**2)/lambda_u - 1))

def calc_lambdaR(gamma, lambda_u, k):
    return lambda_u/(2*gamma**2) * (1+k**2/2)

def chirp_to_lin_taper(n_wig, lambdaR, gamma0, lambda_u, chirp_eV_s):
    l_s = n_wig*lambdaR # slippage per module
    k0 = calc_k(gamma0, lambdaR, lambda_u) # k in first module
    chirp_gamma_s = chirp_eV_s/m_e_eV
    gamma1 = gamma0 + l_s/c * chirp_gamma_s
    k1 = calc_k(gamma1, lambdaR, lambda_u)
    t1 = (k1-k0)/k0
    return t1

