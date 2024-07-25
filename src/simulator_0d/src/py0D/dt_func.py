import numpy as np

import simulator_0d.src.py0D.Balance_eq_source_term as bes
import simulator_0d.src.py0D.enthalpy as enth
import simulator_0d.src.py0D.functions as func
import simulator_0d.src.py0D.global_data as gd
import simulator_0d.src.py0D.kinetics.module_kin as mkin


#---------------------------------------------------------------------------------------------------------------------------------
def dt_determination(system,gas,p_Al,p_MeO, *args):
    '''
    This function is used to find dt the step of time. It takes the minimum
    between different conditions.
    
    Return
    -------
    dt: float
        step of time
    '''
    max_flux=0
    for arg in args:
        dic=arg[1].__dict__
        for key in dic:
            if (type(dic[key]) is not dict):
                if key!="Al_ext_history" and key!="Al_ext_history_wo_mwa" and key!="t_history" and np.max([np.abs(dic[key])])>max_flux:
                    max_flux=max(max_flux,np.max([np.abs(dic[key])]))

    dm_dt_0=np.zeros(gas.nb_species)
    dm_dt_1=np.zeros(gas.nb_species)
    dt_dict = dict()
    flux_pAl = args[0]
    flux_pMeO = args[1]
    source_pAl = args[2]
    source_pMeO = args[3]
    heat_flux_pAl = args[4]
    heat_flux_pMeO = args[5]

    tot_heat_flux_1 = (heat_flux_pAl[1].thermal + heat_flux_pMeO[1].thermal
                      +heat_flux_pAl[1].species_flux_heat + heat_flux_pMeO[1].species_flux_heat)
    tot_heat_flux_0 = (heat_flux_pAl[0].thermal + heat_flux_pMeO[0].thermal
                      +heat_flux_pAl[0].species_flux_heat + heat_flux_pMeO[0].species_flux_heat)

    for i in range (gas.nb_species):
        dm_dt_0[i],dm_dt_1[i]=bes.dm_dt_computation_gas(gas.species_name[i],flux_pMeO,flux_pAl,source_pMeO,source_pAl)

    if gas.bool_kin:
        wdot_estim = estim_first_wdot_for_dt_determin(gas, gas.volume, dm_dt_0, dm_dt_1, tot_heat_flux_0, tot_heat_flux_1, system.dt)
    else :
        wdot_estim = 0

    dt_min_gamma =( gas.density * gas.volume) * 2/np.abs(np.sum(dm_dt_0+dm_dt_1+1e-50))
    if np.sum(dm_dt_0+dm_dt_1+1e-20)>0:
        dt_min_gamma=1
    psi_cond = (gas.density * gas.volume * gas.Y) * 2/np.abs((dm_dt_0+dm_dt_1+wdot_estim)+1e-50)
    cond_dm = np.where((((dm_dt_0+dm_dt_1+wdot_estim)>=0) | (gas.Y==0)),1,psi_cond)
    dt_dict['dt_min_psi'] = 0.95*np.min(cond_dm)

    dt_min_psi_index_species = np.argwhere(cond_dm == np.min(cond_dm))[0][0]
    dt_min_psi_species = gas.species_name[dt_min_psi_index_species]

    dt_dict['dt_min_gamma'] = 0.9*np.where(dt_min_gamma<0,1,dt_min_gamma)

    dT_dt_pAl = dT_dt_computation(gas, p_Al, flux_pAl, heat_flux_pAl, source_pAl)
    dT_dt_pMeO = dT_dt_computation(gas, p_MeO, flux_pMeO, heat_flux_pMeO, source_pMeO)

    dT_max_p = 10
    dt_dict['dt_min_T_pAl'] = dt_min_T(dT_dt_pAl, dT_max_p, p_Al)
    dt_dict['dt_min_T_pMeO'] = dt_min_T(dT_dt_pMeO, dT_max_p, p_MeO)

    dt_dict['dt_min_mass_pAl'] = dt_min_mass(p_Al, flux_pAl, source_pAl)
    dt_dict['dt_min_mass_pMeO'] = dt_min_mass(p_MeO, flux_pMeO, source_pMeO)


    dT_dt_g = func.compute_T_derivative(tot_heat_flux_0, tot_heat_flux_1, dm_dt_0, dm_dt_1, gas)

    if dT_dt_g<0:
        dt_min_T_g = (299.9 - gas.temperature) / dT_dt_g / 1.2
    else:
        dt_min_T_g = 1

    dT_max_g = system.inputs_dic['max_dTg_ite']
    dt_dict['dt_min_T_g'] = min(dt_min_T_g, dT_max_g / np.abs(dT_dt_g))

    dtau_g = compute_tau_th_0D_g(p_Al, p_MeO, gas, system.volume)
    H_cond_pq = heat_flux_pAl[1].add_values["H_cond_pq"]
    dtau_p = compute_tau_th_0D_p(p_Al, p_MeO, gas, H_cond_pq, system.volume)
    dtau_q = compute_tau_th_0D_p(p_MeO, p_Al, gas, H_cond_pq, system.volume)
    dt_dict['dt_min_T_g_n'] = dt_min_T_cond_0D(dtau_g)
    dt_dict['dt_min_T_p_n'] = dt_min_T_cond_0D(dtau_p)
    dt_dict['dt_min_T_q_n'] = dt_min_T_cond_0D(dtau_q)

    num_stab = 100
    dt_dict['dt_num_stab'] = num_stab / max_flux

    dt_dict['max_dt'] = system.max_dt

    dt = min(dt_dict.values())
    if dt == dt_dict['dt_min_psi']:
        dt_species_lim = dt_min_psi_species
    else:
        dt_species_lim = None

    system.dt_restrict = ([k  for (k, val) in dt_dict.items() if val == dt][0], dt_species_lim)
    if dt <= 0:
        print("\n --------------------------ATTENTION: Pas de temps négatif ou nul \n")
        exit()
    # print("T14","%.2e" %dtau_g, "%.2e" %dt)
    # print(system.dt_restrict)
    # print()
    return dt

#---------------------------------------------------------------------------------------------------------------------------------
def dt_min_T_cond_0D(tau):
    dt = tau * 0.9
    return dt


#---------------------------------------------------------------------------------------------------------------------------------
def dT_dt_computation(gas, particle, flux, heat_flux, source):
    '''
    This function calculates the derivative of the temperature used to 
    find the step of time.

    Return
    ------
    dT_dt: float
        derivative of the temperature
    '''
    dUH_dt = (-heat_flux[0].thermal-heat_flux[1].thermal
            -heat_flux[0].species_flux_heat-heat_flux[1].species_flux_heat
            +heat_flux[0].radiative+heat_flux[1].radiative
            +heat_flux[0].p_to_p+heat_flux[1].p_to_p
            +heat_flux[0].source_input+heat_flux[1].source_input)/2

    dm_dt_0=np.zeros(len(particle.species_name))
    dm_dt_1=np.zeros(len(particle.species_name))

    if np.any(np.array(particle.phase)=="PC1") or np.any(np.array(particle.phase)=="PC2"):
        dT_dt = 1

    else:
        phase_index = np.where(np.array(particle.phase)=='s',0,1)

        if gas.transform_type=="P=cste":
            c_vp = np.array([enth.c_p(particle.temperature,particle.species_carac[index_species],phase_index[index_species]) for index_species in range(0,len(particle.species_name))])
            uh_i = np.array([enth.H_tot(particle.temperature,particle.species_carac[index_species],phase_index[index_species]) for index_species in range(0,len(particle.species_name))])

        elif gas.transform_type=="V=cste":
            c_vp = np.array([enth.c_v(particle.temperature,particle.species_carac[index_species],phase_index[index_species]) for index_species in range(0,len(particle.species_name))])
            uh_i = np.array([enth.U_tot(particle.temperature,particle.species_carac[index_species],phase_index[index_species]) for index_species in range(0,len(particle.species_name))])

        species = ["Al_pAl_core","MeO_pMeO_core","Me_pMeO_core","Al2O3_p_shell"]
        for i in range(len(species)):
            dm_dt_0[i], dm_dt_1[i] = bes.dm_dt_computation_solid(species[i], flux,source)

        dT_dt = (dUH_dt - np.sum(uh_i*(dm_dt_1+dm_dt_0)/2))/(np.sum(c_vp*particle.composition))
    return dT_dt


#---------------------------------------------------------------------------------------------------------------------------------
def dt_min_T(dT_dt_p, dT_max, particle):
    '''
    This function is used to find the step of time for the condition of realisability
    on the particules temperature (T(n+1)-T(n)>0).

    Attributes
    ----------
    dT_dt: float
        derivative of the temperature
    
    dT_max: float
        maximum difference between T(n+1) and T(n)

    Return
    ------
    dt_min_T: float
        step of time
    '''
    dt_min_T_p = dT_max/np.abs(dT_dt_p)
    dt_min_T_p_2 = (299.9 - particle.temperature) / dT_dt_p / 1.2
    if dT_dt_p < 0 :
        dt_min_T_p = min (dt_min_T_p_2,dt_min_T_p)
    return dt_min_T_p


def dt_min_mass_species(particule, index_species, flux, source):
    '''
    Determination of the time step for each species according to
    the feasibility condition on the mass of the particles (m>0).

    Return
    ------
    dt_min_mass_species: float
        the time step
    '''
    species_vec = ["Al_pAl_core", "MeO_pMeO_core", "Me_pMeO_core", "Al2O3_p_shell"]
    psi_part0, psi_part1 = bes.dm_dt_computation_solid(species_vec[index_species], flux, source)
    mass_part = particule.composition[index_species]

    if (psi_part1+psi_part0)<0 and mass_part != 0:
        dt_min_mass_species = - 2*(mass_part+1e-100)/(psi_part1+psi_part0-1e-45)*0.9
    else:
        dt_min_mass_species = 1

    return dt_min_mass_species


#---------------------------------------------------------------------------------------------------------------------------------
def dt_min_mass(particule, flux, source):
    '''
    Determination of the general time step according to the
    feasibility condition on the mass of the particles (m>0).

    Return
    ------
    dt_min_mass: float
        the time step
    '''
    #Détermination du pas de temps général en fonction de la condition de réalisabilité sur la masse des particules (m>0)
    dt_min_mass_Al = dt_min_mass_species(particule, 0, flux, source)
    dt_min_mass_MeO = dt_min_mass_species(particule, 1, flux, source)
    dt_min_mass_Me = dt_min_mass_species(particule, 2, flux, source)
    dt_min_mass_Al2O3 = dt_min_mass_species(particule, 3, flux, source)

    dt_min_mass = min([dt_min_mass_Al, dt_min_mass_MeO, dt_min_mass_Me, dt_min_mass_Al2O3])
    return dt_min_mass


#---------------------------------------------------------------------------------------------------------------------------------
def estim_first_wdot_for_dt_determin(gas, volume, dm_dt_0, dm_dt_1, tot_heat_flux_0, tot_heat_flux_1, dt_min):
    """
    dt_min est une estimation du pas de temps afin d'estimer l'incrément de la T°
    """

    # dm_cin_dt = np.zeros(gas.nb_species)                # variation temporelle finale de l'espece "i" sur une itération # (kg/s)
    # Y_ct    = np.zeros(gas.nb_species)

    # # passage de l'indexation maison à celle de cantera pour les flux et Y d'espèces
    # for i,x in enumerate(gas.gas_ct.species_names):
    #     Y_ct[i]   = gas.Y[gas.species_name.index(x)]

    # # entrée des grandeurs dans cantera
    # gas.gas_ct.UVY = gas.energy, 1/gas.density, Y_ct        # (J/kg), (m3/kg), (kg/kg)
    # # Calcul des taux de réaction via cantera
    # wdot = gas.gas_ct.net_production_rates * gas.gas_ct.molecular_weights * gas.volume # (kg/s)

    # for i,x in enumerate(gas.species_name):
    #     dm_cin_dt[i]=wdot[gas.gas_ct.species_index(x)]
    dm_cin_dt = mkin.compute_wdot_nb((gas.temperature), gas.Y, gas.density, gas.alpha) * gas.volume / gas.alpha

    return dm_cin_dt


def compute_tau_th_0D_g(p_Al, p_MeO, gas, volume):
    Re = 0
    Pr = 0
    Nu=(7-10*gas.alpha+5*gas.alpha**2)*(1+0.7*Re**0.2*Pr**(1/3))
    tau_th_1 = (gas.alpha * gas.density * gas.cv) / (gas.lambda_f * np.pi * (p_Al.r_ext + p_MeO.r_ext) * Nu * (p_Al.n + p_MeO.n))

    T_g_eq = compute_T_g_eq_th(p_Al, p_MeO, gas)

    DELTA_T_gp = gas.temperature - p_Al.temperature
    DELTA_T_gq = gas.temperature - p_MeO.temperature
    DELTA_T_eq = T_g_eq - gas.temperature

    C1 = np.abs(DELTA_T_gp)<1e-6
    C2 = np.abs(DELTA_T_gq)<1e-6
    C3 = np.abs(DELTA_T_eq)<1e-3
    C  = C1 and C2 or C3

    rho = gas.density
    Cv = gas.cv
    lambda_g = gas.lambda_f
    p = p_Al
    q = p_MeO
    tau_th_2 = 1
    num = gas.alpha * rho * Cv * (T_g_eq - gas.temperature + 1e-12) * volume
    # num = gas.alpha * rho * Cv * (T_g_eq - gas.temperature)
    denum = lambda_g * np.pi * Nu * (p.r_ext*2 * (p.temperature - gas.temperature + 1e-12) * p.n
                                       + q.r_ext*2 * (q.temperature - gas.temperature + 1e-12) * q.n)
                                    #    + gd.eps)
    tau_th_2 = (num/denum)# * system.volume.volume[gd.N_gc:-gd.N_gc]
    tau_th_2 = np.where(T_g_eq - gas.temperature == 0, 1, tau_th_2)
    # print("T7",min(tau_th_2))
    if C:
        tau_th_2 = np.array(1)
    if (tau_th_2<0).any():
        print("error on characteristic thermal time")
    return tau_th_2



def compute_tau_th_0D_p(p, q, gas, H_cond_pq, volume):
    T_p_eq, H_g, H_q = compute_T_p_eq_th(p, q, gas, H_cond_pq)

    DELTA_T_gp = gas.temperature - p.temperature
    DELTA_T_qp = q.temperature - p.temperature
    DELTA_T_eq = T_p_eq - p.temperature

    C1 = np.abs(DELTA_T_gp)<1e-6
    C2 = np.abs(DELTA_T_qp)<1e-6
    C3 = np.abs(DELTA_T_eq)<1e-3
    C  = C1 & C2 | C3

    DELTA_T_gp = np.where(C1, 0, DELTA_T_gp)
    DELTA_T_qp = np.where(C2, 0, DELTA_T_qp)

    # num = p.alpha * p.density * p.cp * (DELTA_T_eq) * volume
    num = np.sum(p.composition) * p.cp * (DELTA_T_eq)
    denum = (H_g * DELTA_T_gp
            +H_q * DELTA_T_qp + gd.eps)

    tau_th = 1
    tau_th = num / denum # * system.volume.volume[gd.N_gc:-gd.N_gc]
    # tau_th = np.where(T_p_eq - p.temperature == 0, 1, tau_th)
    if C:
        tau_th = 1
    else:
        tau_th = tau_th

    return tau_th


def compute_T_g_eq_th(p_Al, p_MeO, gas):
    p = p_Al
    q = p_MeO
    A = - (q.r_ext/(p.r_ext + gd.eps)
          *q.n/(p.n + gd.eps))
    T_p = p.temperature
    T_q = q.temperature
    T_g_eq = (T_p - A * T_q)/(1 - A)
    return T_g_eq


def compute_T_p_eq_th(p, q, gas, H_cond_pq):
    Re = 0
    Pr = 0
    Nu=(7-10*gas.alpha+5*gas.alpha**2)*(1+0.7*Re**0.2*Pr**(1/3))

    H_g = gas.lambda_f * np.pi * p.r_ext*2 * Nu * p.n
    H_q = H_cond_pq
    norm = np.max([H_g, H_q])
    # T_p_eq = (H_g / norm * gas.temperature + H_q / norm * q.temperature) / (H_g / norm + H_q / norm + gd.eps)
    T_p_eq = (H_g / norm * gas.temperature + H_q / norm * q.temperature) / (H_g / norm + H_q / norm)
    """ reduction erreur numérique en normalisant H par norm"""
    return T_p_eq, H_g, H_q