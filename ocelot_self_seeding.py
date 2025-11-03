#import numpy as np
from scipy.constants import hbar, c, e

from ocelot.optics.elements import Crystal
from ocelot.optics.bragg import get_crystal_filter, CrystalLattice
from . import self_seeding

material_dict = {
        'diamond': 'C',
        'silicon': 'Si',
        'germanium': 'Ge',
        }

class OcelotCrystal:
    def __init__(self, material, cut, hkl, thickness, photon_energy):
        self.material = material
        self.cut = cut
        self.hkl = hkl
        self.thickness = thickness
        self.photon_energy = photon_energy

        self.ocelot_crystal = Crystal(hkl, cut, [1, 1, thickness])
        self.ocelot_crystal.lattice = CrystalLattice(material_dict[material])

        self.ocelot_crystal.cut = cut
        self.ocelot_crystal.ref_idx = hkl
        self.filt = get_crystal_filter(self.ocelot_crystal, photon_energy)

    def get_transfer_function(self):
        omega = self.filt.k*c
        omega_ref = self.photon_energy*e/hbar
        tr = self.filt.tr
        tf = self_seeding.TransferFunction(omega-omega_ref, tr[::-1], tr[0], omega_ref)
        return tf

