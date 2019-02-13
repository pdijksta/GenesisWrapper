import numpy as np
from scipy.optimize import curve_fit

class GainLengthFit:
    def __init__(self, zplot, energy):

        self.popt, self.pcov = curve_fit(self.fit_func, zplot, np.log(energy), p0=(np.log(energy[0]), 1.))
        self.yy = np.exp(self.fit_func(zplot, *self.popt))
        self.xx = zplot

        self.energy = energy
        self.a, self.b = self.popt
        self.gainlength = 1./self.b


    def fit_func(self, xx, a, b):
        return a + b*(xx-xx[0])

