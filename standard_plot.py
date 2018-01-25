import numpy as np
import matplotlib.pyplot as plt

from .simulation import GenesisSimulation
from . import myplotstyle as ms


def run(sim):
    fig = ms.figure('Standard plot for %s' % sim.infile)

    sp = plt.subplot(2,2,1)
    sp.grid(True)
    sp.set_title('Power')
    sp.set_xlabel('s [m]')
    sp.set_ylabel('Power [W]')
    power = np.nan_to_num(sim['Field/power'])
    try:
        sp.plot(sim.zplot, np.trapz(power, sim.time), axis=1)
    except:
        import pdb; pdb.set_trace()


