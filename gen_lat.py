import numpy as np
from collections import OrderedDict

# Example lattice file:
"""
QF1: Quadrupole = { l=0.408, k1=0.333700};
QF2: Quadrupole = { l=0.408, k1=-0.333700};

D1:   Drift = { l=0.544 };
D2:   Drift = { l=0.544 };

Un: Undulator = {lambdau=0.068,nwig=74, aw=6.17198};
FODO: LINE = {Un,D1,QF1,D2,Un,D1,QF2,D2};

FEL: LINE = {10*FODO,Un};
"""

def def_drift(ctr, ld):
    key = 'D%i' % ctr
    string = '%s: Drift = { l=%.8f };\n' % (key, ld)
    return key, string

def def_quad(ctr, lq, k1):
    key = 'QF%i' % ctr
    string = '%s: Quadrupole = { l=%.8f, k1=%.8f};\n' % (key, lq, k1)
    return key, string

def def_und(ctr, lambdau, nwig, k):
    aw = k/np.sqrt(2)
    key = 'Un%i' % ctr
    string = '%s: Undulator = {lambdau=%.8f,nwig=%i, aw=%.8f};\n' % (key, lambdau, nwig, aw)
    return key, string

class lat_file:
    def __init__(self, lin_taper=0, quad_taper=0, und_ctr_quad=1):
        self.lin_taper = lin_taper
        self.quad_taper = quad_taper
        self.und_ctr_quad = und_ctr_quad
        self.und_ctr = 1
        self.drift_ctr = 1
        self.quad_ctr = 1
        self.elem_dict = OrderedDict()
        self.elem_list = []
        self.k_list = []

    def add_undulator(self, lambdau, nwig, k):
        k_lin = k*(1+self.lin_taper*self.und_ctr)
        if self.und_ctr >= self.und_ctr_quad:
            k_quad = k_lin*(1+self.quad_taper*(self.und_ctr - self.und_ctr_quad)**2)
        else:
            k_quad = k_lin
        key, string = def_und(self.und_ctr, lambdau, nwig, k_quad)
        self.und_ctr += 1
        self.elem_dict[key] = string
        self.k_list.append(k_quad)
        return key

    def add_drift(self, ld):
        key, string = def_drift(self.drift_ctr, ld)
        self.drift_ctr += 1
        self.elem_dict[key] = string
        return key

    def add_quad(self, lq, k1):
        key, string = def_quad(self.quad_ctr, lq, k1)
        self.quad_ctr += 1
        self.elem_dict[key] = string
        return key

    def add_elem(self, key):
        assert key in self.elem_dict
        self.elem_list.append(key)

    def add_line(self, key, elems):
        elem_str = ''
        for elem in elems:
            assert elem in self.elem_dict.keys()
            elem_str += '%s,' % elem
        string = '%s: LINE = {%s};\n' % (key, elem_str[:-1])
        self.elem_list.append(key)
        self.elem_dict[key] = string

    def add_final_line(self):
        self.add_line('FEL', self.elem_list)

    def write_lat(self, fname):
        content = ''.join(self.elem_dict.values())
        with open(fname, 'w') as f:
            f.write(content)
        return content

def sase3_lat(filename, k_init, lin_taper=0, quad_taper=0, und_ctr_quad=0, k1_foc=0.3337, k1_defoc=-0.3337):
    ld = 0.544
    lq = 0.408
    lambdau = 0.068
    nwig = 74
    n_fodo = 10
    extra_un = True

    lat = lat_file(lin_taper, quad_taper, und_ctr_quad)
    d = lat.add_drift(ld)
    q1 = lat.add_quad(lq, k1_foc)
    q2 = lat.add_quad(lq, k1_defoc)
    for fodo_ctr in range(n_fodo):
        un1 = lat.add_undulator(lambdau, nwig, k_init)
        un2 = lat.add_undulator(lambdau, nwig, k_init)
        lat.add_line('FODO%i' % (fodo_ctr+1), [un1, d, q1, d, un2, d, q2, d])
    if extra_un:
        un = lat.add_undulator(lambdau, nwig, k_init)
        lat.add_elem(un)

    lat.add_final_line()
    content = lat.write_lat(filename)
    return lat, content

