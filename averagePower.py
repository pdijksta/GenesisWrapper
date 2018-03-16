import numpy as np

def get_average_power(simulations, axis=0, energy=False):

    avg_power = 0
    for sim in simulations:
        if axis == 0:
            avg_power += sim['Field/power'][-1,:]
        elif axis == 1 and energy:
            avg_power += np.trapz(sim['Field/power'], x=sim.time, axis=1)
        elif axis == 1 and not energy:
            avg_power += np.mean(sim['Field/power'], axis=1)

    avg_power /= len(simulations)

    return avg_power


def get_rms_pulse_length(time, power, treshold=None):
    power = np.nan_to_num(power)

    if treshold is not None:
        power[power < power.max()*treshold] = 0

    int_power = np.trapz(power, time)
    if int_power == 0:
        rms_time = 0
    else:
        int_time_sq = np.trapz(time**2*power, time) / int_power
        int_time = np.trapz(time*power, time) / int_power

        rms_time = np.sqrt(int_time_sq - int_time**2)
    return rms_time

def get_total_pulse_energy(time, power):
    return np.trapz(power, time)

