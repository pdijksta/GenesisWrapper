import numpy as np
from scipy.optimize import curve_fit
#import scipy.stats as stats

class GaussFit:
    """
    p0 for 'gauss': scale, mean, sig, const
    p0 for 'double_gauss': scale1, scale2, mean1, mean2, sig1, sig2, konst
    """
    def __init__(self, xx, yy, print_=False, fit='gauss', p0=None):

        if fit == 'gauss':
            self.fit_func = singleGauss
        elif fit == 'double_gauss':
            self.fit_func = doubleGauss

        #p0 = (1e10, 3.5e-14, 1e-14)
        scale_0 = 0.9*np.max(yy)
        mean_0 = np.squeeze(xx[np.argmax(yy)])

        # Third instead of half for better stability
        mask_above_half = np.squeeze(yy > np.max(yy)/3)

        if np.sum(mask_above_half) != 0:
            sigma_0 = abs(xx[mask_above_half][-1] - mean_0)
        const_0 = yy[0]

        if p0 is None and fit == 'gauss':
            p0 = self.p0 = (scale_0, mean_0, sigma_0, const_0)
        elif p0 is None and fit == 'double_gauss':
            raise ValueError('p0 must be provided for double gauss fit')
        elif p0 is not None:
            self.p0 = p0
        else:
            raise Exception('Error in code here')

        self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, p0=p0)
        self.yy = self.fit_func(xx, *self.popt)
        self.xx = xx

        self.yy_in = self.power = yy

        if fit == 'gauss':
            self.scale, self.mean, self.sigma, self.const = self.popt
        elif fit == 'double_gauss':
            (self.scale1, self.scale2,
                self.mean1, self.mean2,
                self.sigma1, self.sigma2,
                self.const) = self.popt

        if print_:
            print(p0, '\t\t', self.popt)

def singleGauss(xx, scale, mean, sig, const):
    #return scale*stats.norm.pdf(xx, mean, sig)
    return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const

def doubleGauss(xx, scale1, scale2, mean1, mean2, sig1, sig2, const):
    g1 = singleGauss(xx, scale1, mean1, sig1, 0)
    g2 = singleGauss(xx, scale2, mean2, sig2, 0)
    return g1 + g2 + const

