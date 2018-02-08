from scipy.optimize import curve_fit
import scipy.stats as stats

class GaussFit:
    def __init__(self, time, power):
        p0 = (1e10, 3.5e-14, 1e-14)

        self.popt, self.pcov = curve_fit(self.fit_func, time, power, p0=p0)
        self.yy = self.fit_func(time, *self.popt)
        self.xx = time

        self.power = power

        self.scale, self.mean, self.sigma = self.popt

    def fit_func(self, xx, scale, mean, sig):
        return scale*stats.norm.pdf(xx, mean, sig)

