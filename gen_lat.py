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
        self.und_ctr = 0
        self.drift_ctr = 0
        self.quad_ctr = 0
        self.elem_dict = OrderedDict()
        self.elem_list = []
        self.k_list = []

    def add_undulator(self, lambdau, nwig, k):
        if self.und_ctr < self.und_ctr_quad:
            quad_taper = 0
        else:
            quad_taper = self.quad_taper
        k_tapered = k*(1-self.lin_taper*self.und_ctr-quad_taper*(self.und_ctr - self.und_ctr_quad)**2)
        key, string = def_und(self.und_ctr, lambdau, nwig, k_tapered)
        self.und_ctr += 1
        self.elem_dict[key] = string
        self.k_list.append(k_tapered)
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

def gen_fodo_beamline_tapered(ld1, ld2, lq, k1_foc, k1_defoc, lambdau, nwig, k_init, n_fodo, extra_un, lin_taper, quad_taper, und_ctr_quad):
    lat = lat_file(lin_taper, quad_taper, und_ctr_quad)
    d1 = lat.add_drift(ld1)
    d2 = lat.add_drift(ld2)
    q1 = lat.add_quad(lq, k1_foc)
    q2 = lat.add_quad(lq, k1_defoc)
    for fodo_ctr in range(n_fodo):
        un1 = lat.add_undulator(lambdau, nwig, k_init)
        un2 = lat.add_undulator(lambdau, nwig, k_init)
        lat.add_line('FODO%i' % (fodo_ctr+1), [un1, d1, q1, d2, un2, d1, q2, d2])
    if extra_un:
        un = lat.add_undulator(lambdau, nwig, k_init)
        lat.add_elem(un)
    lat.add_final_line()
    return lat

def gen_fodo_beamline_k_arr(ld1, ld2, lq, k1_foc, k1_defoc, lambdau, nwig, k_arr, n_fodo, extra_un):
    if not hasattr(k_arr, '__len__'):
        n_und = int(n_fodo*2 + 1*extra_un)
        k_arr = [k_arr]*n_und
    lat = lat_file(0, 0, 0)
    d1 = lat.add_drift(ld1)
    d2 = lat.add_drift(ld2)
    q1 = lat.add_quad(lq, k1_foc)
    q2 = lat.add_quad(lq, k1_defoc)
    for fodo_ctr in range(n_fodo):
        un1 = lat.add_undulator(lambdau, nwig, k_arr[2*fodo_ctr])
        un2 = lat.add_undulator(lambdau, nwig, k_arr[2*fodo_ctr+1])
        lat.add_line('FODO%i' % (fodo_ctr+1), [un1, d1, q1, d2, un2, d1, q2, d2])
    if extra_un:
        un = lat.add_undulator(lambdau, nwig, k_arr[-1])
        lat.add_elem(un)
    lat.add_final_line()
    return lat



def sase3_lat(filename, k_init, lin_taper=0, quad_taper=0, und_ctr_quad=0, k1_foc=0.554, k1_defoc=-0.554, n_fodo=None):
    ld = 0.47465
    lq = 0.1137
    lambdau = 0.068
    nwig = 74
    if n_fodo is None:
        n_fodo = 10
        extra_un = True
    else:
        n_fodo = n_fodo
        extra_un = False
    lat = gen_fodo_beamline_tapered(ld, ld, lq, k1_foc, k1_defoc, lambdau, nwig, k_init, n_fodo, extra_un, lin_taper, quad_taper, und_ctr_quad)
    content = lat.write_lat(filename)
    return lat, content

def sase1_lat(filename, k_init, lin_taper=0, quad_taper=0, und_ctr_quad=0, k1_foc=0.554, k1_defoc=-0.554, n_fodo=None):
    ld = 0.51315
    lq = 0.1137
    lambdau = 0.04
    nwig = 124
    if n_fodo is None:
        n_fodo = 17
        extra_un = True
    else:
        n_fodo = n_fodo
        extra_un = False
    lat = gen_fodo_beamline_tapered(ld, ld, lq, k1_foc, k1_defoc, lambdau, nwig, k_init, n_fodo, extra_un, lin_taper, quad_taper, und_ctr_quad)
    content = lat.write_lat(filename)
    return lat, content

sase2_lat = sase1_lat

def aramis_lat(filename, k_init, lin_taper=0, quad_taper=0, und_ctr_quad=0, k1_foc=2.5, k1_defoc=-2.5, n_fodo=6):
    ld1 = 0.355
    ld2 = 0.34
    lq = 0.08
    lambdau = 0.015
    nwig = 265
    if hasattr(k_init, '__len__'):
        lat = gen_fodo_beamline_k_arr(ld1, ld2, lq, k1_foc, k1_defoc, lambdau, nwig, k_init, n_fodo, False)
    else:
        lat = gen_fodo_beamline_tapered(ld1, ld2, lq, k1_foc, k1_defoc, lambdau, nwig, k_init, n_fodo, False, lin_taper, quad_taper, und_ctr_quad)
    content = lat.write_lat(filename)
    return lat, content

