import numpy as np
from scipy.optimize import curve_fit
#import scipy.stats as stats

class GaussFit:
    def __init__(self, time, power, print_=False, sigma_00=3e-15):
        #p0 = (1e10, 3.5e-14, 1e-14)
        scale_0 = 0.9*np.max(power)
        mean_0 = np.squeeze(time[np.argmax(power)])

        # Third instead of half for better stability
        mask_above_half = np.squeeze(power > np.max(power)/3)

        if np.sum(mask_above_half) != 0:
            sigma_0 = abs(time[mask_above_half][-1] - mean_0)
        else:
            sigma_0 = sigma_00

        sigma_0 = max(sigma_0, sigma_00)
        const_0 = power[0]
        p0 = self.p0 = (scale_0, mean_0, sigma_0, const_0)

        self.popt, self.pcov = curve_fit(self.fit_func, time, power, p0=p0)
        self.yy = self.fit_func(time, *self.popt)
        self.xx = time

        self.power = power
        self.scale, self.mean, self.sigma, self.const = self.popt

        if print_:
            print(p0, '\t\t', self.popt)

    def fit_func(self, xx, scale, mean, sig, const):
        #return scale*stats.norm.pdf(xx, mean, sig)
        return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const

