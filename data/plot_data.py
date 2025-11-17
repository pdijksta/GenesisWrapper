import numpy as np
from PassiveWFMeasurement import myplotstyle as ms

ms.closeall()

fig = ms.figure('Crystal data')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1
dfiles = [
        './x0h_resultsC111.dat', # PAL-XFEL at 3.5 keV (with 110 cut)
        './x0h_resultsC220.dat', # Mentioned in Yang & Shvyd'Ko paper
        './x0h_resultsC331.dat', # Mentioned in Yang & Shvyd'Ko paper
        './x0h_resultsC004.dat', # EuXFEL at 6 keV (with 100 cut) and LCLS at 8.3 keV (also with 100 cut)
        #'./x0h_resultsC511.dat', # Mentioned in Yang & Shvyd'Ko paper
        './x0h_resultsC333.dat', # PAL-XFEL at 9.7 keV (with 100 cut)
        './x0h_resultsC533.dat', # PAL-XFEL at 14.6 keV (with 100 cut)
        ]

for n_key, key in enumerate(['xr0', 'xi0', 'xrh', 'xih']):
    sp = subplot(sp_ctr, title='|%s|' % key, xlabel='Energy (keV)', ylabel=key)
    sp_ctr += 1
    for dfile in dfiles:
        reflection = dfile.replace('./x0h_results', '').replace('.dat', '')
        data = np.loadtxt(dfile, comments='#', delimiter=',')
        energy = data[:,0]
        vals = data[:,1+n_key]
        sp.semilogy(energy, np.abs(vals), label=reflection)
    sp.legend()

ms.show()

