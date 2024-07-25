from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

import module_kin as mkin

P_atm = 1e5


def kinetics(gas, volume, dt):
    """
    La cinétique est calculée en considérant un alpha_g constant
    C'est la raison pour laquelle l'état du gas en fin de cinétique
    diffère légèrement de l'état du gaz en fin d'itération
    """

    tau  = 0                                            # temps d'avancement de la sous-itération (s)
    dtau = 0                                            # sous-pas de temps (s)

    phi = gas.dm_dt_flux                                # flux d'espèce provenant des échanges interphase sur une itération # (kg/s)
    dQ_dt   = gas.dQ_dt                                 # flux d'énergie provenant des échanges interphase sur une itération # (J/s)

    # in fine: dmi_dt_tot = dm_cin_tot_dt + phi

    dm_cin_dt = np.zeros(gas.nb_species)                # variation temporelle finale de l'espece "i" sur une itération # (kg/s)
    wdot    = np.zeros(gas.nb_species)                  # taux de réaction sur une sous-itération # (kg/s)
    density = gas.density                               # densité du gaz (kg/m3)
    energy  = gas.energy                                # energie massique du gas (J/kg)
    phi_ct  = np.zeros(gas.nb_species)
    Y_ct    = np.zeros(gas.nb_species)

    # passage de l'indexation maison à celle de cantera pour les flux et Y d'espèces
    for i,x in enumerate(gas.gas_ct.species_names):
        phi_ct[i] = phi[gas.species_name.index(x)]
        Y_ct[i]   = gas.Y[gas.species_name.index(x)]
    T_l = [gas.temperature]
    t_l = [0]
    Y_ct_save_prev_ite = deepcopy(Y_ct)                 # fraction massique des espèces à l'itération précédente (kg/m3)
    density_save_prev_ite = deepcopy(gas.density)       # densité du gaz                à l'itération précédente (kg/m3)
    cpt = 0
    while tau<dt:
        energy_save  = deepcopy(energy)
        density_save = deepcopy(density)
        Y_ct_save    = deepcopy(Y_ct)
        tau_save     = deepcopy(tau)

        # entrée des grandeurs dans cantera
        gas.gas_ct.UVY = energy, 1/density, Y_ct        # (J/kg), (m3/kg), (kg/kg)
        # Calcul des taux de réaction via cantera
        wdot = gas.gas_ct.net_production_rates * gas.gas_ct.molecular_weights * gas.alpha * volume # (kg/s)
        # determination du sous-pas de temps
        # if cpt >= 303:
        #     print("T55")
        # print(cpt)
        dtau = dt_cin(phi_ct, wdot, density, gas.alpha, volume, Y_ct, tau, dt)

        # MAJ des grandeurs d'interet du gaz
        density_prev =  deepcopy(density)
        density = (density_prev * gas.alpha * volume * 1      + (np.sum(phi_ct)) * dtau) / (   1    * gas.alpha * volume) # (kg/m3)
        energy  = (density_prev * gas.alpha * volume * energy + (dQ_dt)          * dtau) / (density * gas.alpha * volume) # (J/kg)
        Y_ct    = (density_prev * gas.alpha * volume * Y_ct   + (phi_ct + wdot)  * dtau) / (density * gas.alpha * volume) # (kg/kg)
        # avancement temporel
        tau = tau + dtau
        cpt = cpt + 1
        T_l.append(gas.gas_ct.T)
        t_l.append(tau)
        # print("T4",cpt, dtau)
        if cpt>1e7:
            print("cpt",cpt, dtau)
            raise ValueError("kinetics did not converge")
    if tau > dt: # si on dépasse le pas de temps
        gas.gas_ct.UVY = energy_save, 1/density_save, Y_ct_save
        wdot    = gas.gas_ct.net_production_rates * gas.gas_ct.molecular_weights * gas.alpha * volume
        dtau    = dt - tau_save

        density_prev =  deepcopy(density)
        density = (density_prev * gas.alpha * volume * 1      + (np.sum(phi_ct)) * dtau) / (   1    * gas.alpha * volume) # (kg/m3)
        energy  = (density_prev * gas.alpha * volume * energy + (dQ_dt)          * dtau) / (density * gas.alpha * volume) # (J/kg)
        Y_ct    = (density_prev * gas.alpha * volume * Y_ct   + (phi_ct + wdot)  * dtau) / (density * gas.alpha * volume) # (kg/kg)
        tau     = tau_save + dtau
        T_l.append(gas.gas_ct.T)
        t_l.append(tau)

    # la variation totale temporelle est définie comme la différence entre l'état
    # final du gaz et l'état initial sur un pas de temps
    dm_dt_tot_ct = (density               * gas.alpha * volume * Y_ct - 
                    density_save_prev_ite * gas.alpha * volume * Y_ct_save_prev_ite) / dt

    # La variation temporelle étant égale à la somme de la variation induite par
    # les échanges interphases et la cinétique gazeuse, on obtient la variation
    # sur un pas de temps relative à la cinétique:
    dm_dt_cin_ct = dm_dt_tot_ct - phi_ct

    # passage de l'indexation cantera à celle maison pour les variation dmi
    for i,x in enumerate(gas.species_name):
        dm_cin_dt[i]=dm_dt_cin_ct[gas.gas_ct.species_index(x)]
    if gas.gas_ct.T>=3000 and len(T_l)>5:
        # plt.plot(T_l)
        print("TETET")
    dmi_dt_kin, int_BDF, t_BDF, Q_BDF = mkin.compute_kin(gas, dt, volume, int_BDF = gas.int_BDF)
    import py0D.kinetics.enthalpy_numba as enth_nb
    plt.plot(t_l, T_l)
    plt.plot(t_BDF, enth_nb.find_g_temperature(np.full(Q_BDF.shape[0] ,2200),Q_BDF), '--')
    plt.show()
    print("relative difference between mass species increment:", (dmi_dt_kin*volume - dm_cin_dt)/dm_cin_dt)
    return dm_cin_dt, int_BDF


def dt_cin(phi, wdot, rho, alpha, volume, Y, tau, dt):
    A=np.where((phi+wdot)<0, (-(rho*alpha*volume*Y)/(phi+wdot)+1e-50) / 1000, dt - tau)
    min_dt = dt / 1000
    # min_dt = 1
    return np.min((np.min(A), min_dt))

