import numpy as np

import py0D.global_data as gd


def generate_Q(alpha, density, energy, Y, velocity_x = 0, velocity_y = 0):
    energy_int = energy
    Q = np.zeros(gd.len_Q_g)
    Q[gd.i_d]  = np.copy(density * alpha * 1)
    Q[gd.i_vx] = np.copy(density * alpha * velocity_x)
    Q[gd.i_vy] = np.copy(density * alpha * velocity_y)
    Q[gd.i_e]  = np.copy(density * alpha * energy_int + 0.5 * (velocity_x**2))
    Q[gd.i_Y:] = np.copy(density * alpha * Y)
    return Q


def generate_dQ_dt(gas, volume, Q_shape):
    dQ_dt_0D = np.zeros(Q_shape)
    dQ_dt_0D[gd.i_d] = np.sum(gas.dm_dt_flux) / volume
    dQ_dt_0D[gd.i_e] = gas.dQ_dt / volume
    dQ_dt_0D[gd.i_Y:] = (gas.dm_dt_flux) / volume
    return dQ_dt_0D

def degenerate_Q(alpha, Q_prev, Q, dQ_dt_0D, volume, dt):
    if len(Q.shape)>=2:
        Q_last = Q[-1]
    else:
        Q_last = Q
    dQi_dt = np.zeros(gd.nb_gas_species)

    # puisqu'on évalue la variation de la cinétique par rapport
    # à la variation 0D, afin d'éviter des approximation pouvant
    # etre a l'origine de Y<0, dQi_dt est légèrement réduit:
    dQi_dt = ((Q_last - Q_prev)/dt - dQ_dt_0D) * (1 - 1e-14)

    dmi_dt = dQi_dt[gd.i_Y:] * volume
    # afin de ne me pas considérer des résidus relatifs à la méthode d'intégration sur
    # des espèces réactives:
    dmi_dt = np.where(np.abs(dmi_dt)/np.mean(np.abs(dmi_dt))<1e-14, 0, dmi_dt)
    return dmi_dt
