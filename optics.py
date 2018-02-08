import numpy as np
import numpy.linalg as linalg

def ml(l):
    return np.array([[1, l], [0,1]], float)

def mf(f):
    return np.array([[1, 0], [1/f,1]], float)

def get_m_tot(*matrices):
    out = np.eye(2)
    for m in matrices[::-1]:
        out = out.dot(m)
    return out


def beam_matrix(beta, alpha):
    gamma = (1+alpha**2)/beta
    return np.array([[beta, -alpha], [-alpha, gamma]], dtype=float)

class FodoCell:
    def __init__(self, L, m):


        assert np.abs(np.trace(m)) <= 2
        assert np.abs(linalg.det(m)-1) < 1e-10

        self.m = m

        [[C, S],[Cp, Sp]] = m

        m33 = self.m33 = np.array([
            [C**2, -2*S*C, S**2],
            [-C*Cp, S*Cp+Sp*C, -S*Sp],
            [Cp**2, -2*Sp*Cp, Sp**2]], dtype=float)

        mu = self.mu = np.arccos(0.5*np.trace(self.m))
        self.beta_max = L*(1+np.sin(mu))/np.sin(mu)
        self.beta_min = L*(1-np.sin(mu))/np.sin(mu)


        eigenvals, eigenvecs = linalg.eig(m33)
        mask_real = np.abs(eigenvals.real-1) < 1e-5
        assert np.sum(mask_real) == 1
        assert 1 - 1e-5 < eigenvals[mask_real][0] < 1+1e-5
        eigenvec = e = np.real(eigenvecs[:,mask_real].squeeze())
        eigenvec /= np.sqrt(e[0]*e[2]-e[1]**2)
        eigenvec *= np.sign(eigenvec[0])
        self.beta, self.alpha, self.gamma = eigenvec

def get_m_aramis(k):
    f = 1/(0.08*k)
    m_aramis = get_m_tot(
            ml(265*0.015),
            ml(0.355),
            mf(f),
            ml(0.34),
            ml(265*0.015),
            ml(0.355),
            mf(-f),
            ml(0.34),
            )
    return m_aramis


def get_fodo_aramis(k):
    L = 9.5
    m = get_m_aramis(k)
    return FodoCell(L, m)


