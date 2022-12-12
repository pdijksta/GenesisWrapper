import numpy as np
from scipy.optimize import curve_fit

factor_fwhm = 2*np.sqrt(2*np.log(2))

class GaussFit:
    """
    p0 for 'gauss': scale, mean, sig, const
    p0 for 'double_gauss': scale1, scale2, mean1, mean2, sig1, sig2, const
    """
    def __init__(self, xx, yy, yy_std=None, print_=False, fit='gauss', p0=None, fit_const=True):

        if fit == 'gauss':
            self.fit_func = singleGauss
            n_vars = 4
            jac = self.jacobi_single_gauss
        elif fit == 'double_gauss':
            self.fit_func = doubleGauss
            n_vars = 7
            jac = self.jacobi_double_gauss
        else:
            raise ValueError("Fit must be 'gauss' or 'double_gauss', not %s" % fit)

        if not fit_const:
            n_vars -= 1
        self.jacobi_arr = np.ones((len(xx), n_vars))

        if p0 is None and fit == 'gauss':
            scale_0 = 0.9*np.max(yy)
            mean_0 = np.squeeze(xx[np.argmax(yy)])
            const_0 = yy[0]

            mask_above_half = np.squeeze(yy > np.max(yy)/2)
            if np.sum(mask_above_half) != 0:
                sigma_0 = abs(xx[mask_above_half][-1] - xx[mask_above_half][0]) / factor_fwhm
            else:
                sigma_0 = 1e-15

            p0 = self.p0 = (scale_0, mean_0, sigma_0, const_0)
            if not fit_const:
                p0 = self.p0 = p0[:-1]
        elif p0 is None and fit == 'double_gauss':
            raise ValueError('p0 must be provided for double gauss fit')
        elif p0 is not None:
            self.p0 = p0

        self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, sigma=yy_std, p0=p0, jac=jac)
        if not fit_const:
            self.popt = np.append(self.popt, [0.])
        self.yy = self.fit_func(xx, *self.popt)
        self.xx = xx

        self.yy_in = self.power = yy
        self.yy_std = yy_std

        if fit == 'gauss':
            self.scale, self.mean, self.sigma, self.const = self.popt
        elif fit == 'double_gauss':
            (self.scale1, self.scale2,
                self.mean1, self.mean2,
                self.sigma1, self.sigma2,
                self.const) = self.popt

        if print_:
            print("p0, '\t\t', self.popt")
            print(p0, '\t\t', self.popt)

    def jacobi_single_gauss(self, xx, scale, mean, sig, const=0):
        g_minus_const = singleGauss(xx, scale, mean, sig, 0)
        if scale == 0:
            self.jacobi_arr[:,0] = np.inf
        else:
            self.jacobi_arr[:,0] = g_minus_const/scale
        if sig == 0:
            self.jacobi_arr[:,1] = np.inf
            self.jacobi_arr[:,2] = np.inf
        else:
            self.jacobi_arr[:,1] = g_minus_const * (xx-mean)/sig**2
            self.jacobi_arr[:,2] = g_minus_const * (xx-mean)**2/sig**3
        #self.jacobi_arr[:,3] = 1 # This never changes
        return self.jacobi_arr

    def jacobi_double_gauss(self, xx, scale1, scale2, mean1, mean2, sig1, sig2, const=0):
        g1_minus_const = singleGauss(xx, scale1, mean1, sig1, 0)
        g2_minus_const = singleGauss(xx, scale2, mean2, sig2, 0)
        self.jacobi_arr[:,0] = g1_minus_const/scale1
        self.jacobi_arr[:,2] = g1_minus_const * (xx-mean1)/sig1**2
        self.jacobi_arr[:,4] = g1_minus_const * (xx-mean1)**2/sig1**3

        self.jacobi_arr[:,1] = g2_minus_const/scale2
        self.jacobi_arr[:,3] = g2_minus_const * (xx-mean2)/sig2**2
        self.jacobi_arr[:,5] = g2_minus_const * (xx-mean2)**2/sig2**3
        #self.jacobi_arr[:,3] = 1 # This never changes
        return self.jacobi_arr



def singleGauss(xx, scale, mean, sig, const=0):
    #return scale*stats.norm.pdf(xx, mean, sig)
    return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const

def doubleGauss(xx, scale1, scale2, mean1, mean2, sig1, sig2, const=0):
    g1 = singleGauss(xx, scale1, mean1, sig1, 0)
    g2 = singleGauss(xx, scale2, mean2, sig2, 0)
    return g1 + g2 + const

