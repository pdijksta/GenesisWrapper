import os
import re
import glob
import itertools
import numpy as np

data = {}
re_dfile = re.compile('x0h_results(\w+)(\d{3}).dat')
dfiles = glob.glob(os.path.join(os.path.dirname(__file__), './data/*.dat'))
dfile_columns = ['photon_energy_keV', 'xr0', 'xi0', 'xrh', 'xih']
material_label_dict = {
        'C': 'diamond',
        }

class PhotonEnergyException(ValueError):
    pass

for dfile in dfiles:
    basename = os.path.basename(dfile)
    groups = re_dfile.match(basename).groups()
    material = material_label_dict[groups[0]]
    hkl = tuple([int(x) for x in groups[1]])
    if material not in data:
        data[material] = {}
    data[material][hkl] = {}
    this_data = np.loadtxt(dfile, delimiter=',', comments='#')
    for ctr, key in enumerate(dfile_columns):
        data[material][hkl][key] = this_data[:,ctr]


def get_x0h_data(material, h_, k_, l_, photon_energy_eV):
    if material not in data:
        raise ValueError('No data for %s' % material)
    for permutation in itertools.permutations([abs(h_), abs(k_), abs(l_)]):
        if tuple(permutation) in data[material]:
            this_data = data[material][permutation]
            break
    else:
        raise ValueError('No data for %s' % str([h_, k_, l_]))

    data_photon_energy = this_data['photon_energy_keV']*1e3
    if photon_energy_eV < data_photon_energy[0] or photon_energy_eV > data_photon_energy[-1]:
        raise PhotonEnergyException('Data only available between %.0f and %.0f eV, not for %.0f eV' % (data_photon_energy[0], data_photon_energy[-1], photon_energy_eV))

    outp ={}
    for key in dfile_columns[1:]:
        outp[key] = np.interp(photon_energy_eV, data_photon_energy, this_data[key])
    return outp


