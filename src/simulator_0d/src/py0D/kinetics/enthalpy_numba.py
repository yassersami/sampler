import time

import numexpr as ne
import numpy as np
from numba import jit, njit
from scipy import optimize

import simulator_0d.src.py0D.global_data as gd

R=8.31446261815324

#---------------------------------------------------------------------------------------------------------------------------------
def find_g_temperature(T_init, Q, bool_check_T_range = True):
    energy_int = energy_int_from_Q(Q)
    Y = np.transpose(np.transpose(Q[:, gd.i_Y:]) / Q[:, gd.i_d])
    T = optimize.newton(f_enthalpy_g, T_init, args = (energy_int, Y,))
    if ((T<250) | (T>10e3) |(np.isnan(T))).any() and bool_check_T_range:
        # raise ValueError("test")
        print(T)
    return T


#---------------------------------------------------------------------------------------------------------------------------------
def f_enthalpy_g(T, energy_int, Y):
    #cette fonction est utilisée dans la recherche de la temperature de la particule p_Al en connaissant l'enthalpie de celle-ci. cf: "find_p_temeprature"
    #En effet, sachant que l'enthalpie totale de la particule s'ecrit "H_tot=H_sensible(T)+H_formation", où H_sensible est fonction de la temperature,
    #il suffit de résoudre l'équation "f=0=H_tot-H_sensible(T)-H_formation" pour trouver la temperature à partir de l'enthalpie totale de la particule.
    uh_i, Cvp = uCv_i_g_vec2(T)
    f = energy_int - np.sum(
        np.transpose(np.transpose(Y))
        * uh_i, axis = 1)
    return f


#---------------------------------------------------------------------------------------------------------------------------------
@njit
def hs_i_g_vec(T_vec):
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]

    if isinstance(T_vec, float) or isinstance(T_vec, int):
        # chron2 = time.time()
        T_vec = np.array([T_vec])

    h = np.zeros((len(T_vec),gd.nb_gas_species))
    s = np.zeros((len(T_vec),gd.nb_gas_species))   
  
    for i in range(len(T_vec)):
        for j in range(gd.nb_gas_species):
            if T_vec[i] < T_mid[j]:
                # A = 3
                h[i,j], s[i,j] = array_poly_2_hs_g(coeffs_0[j], 1, T_vec[i])
            else:
                h[i,j], s[i,j] = array_poly_2_hs_g(coeffs_1[j], 1, T_vec[i])
        # B = array_poly_2_hs_g_multi_species_ne(coeffs_0, 1, np.tile(T,(gd.nb_gas_species, 1)))
        # C = array_poly_2_hs_g_multi_species_ne(coeffs_1, 1, np.tile(T,(gd.nb_gas_species, 1)))
        # h_ct, s_ct = ne.evaluate('where(A,B,C)')
    return h, s


@njit
def uCv_i_g_vec2(T_vec):

    if isinstance(T_vec, float) or isinstance(T_vec, int):
        # chron2 = time.time()
        T_vec = np.array([T_vec])

    u  = np.zeros((len(T_vec),gd.nb_gas_species))
    Cv  = np.zeros((len(T_vec),gd.nb_gas_species))
    # Cv = np.zeros((gd.nb_gas_species,len(T_vec)))
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]
    MW_gas_arr = gd.MW_gas_arr


    for i in range(len(T_vec)):
        for j in range(gd.nb_gas_species):
            if T_vec[i] < T_mid[j]:
                # A = 3
                u[i,j], Cv[i,j] = array_poly_2_Cv_g(coeffs_0[j], MW_gas_arr[j], T_vec[i])
            else:
                u[i,j], Cv[i,j] = array_poly_2_Cv_g(coeffs_1[j], MW_gas_arr[j], T_vec[i])
                # print(gd.MW_gas_vec[j])
        # B = array_poly_2_hs_g_multi_species_ne(coeffs_0, 1, np.tile(T,(gd.nb_gas_species, 1)))
        # C = array_poly_2_hs_g_multi_species_ne(coeffs_1, 1, np.tile(T,(gd.nb_gas_species, 1)))
        # h_ct, s_ct = ne.evaluate('where(A,B,C)')
    return u, Cv
#################################################################
#################################################################
#################################################################
#################################################################

@njit
def g_0_vec(T):
    t = np.transpose
    h, s = hs_i_g_vec(T)
    g_0 = t(t(h) - T * t(s))
    return g_0


# def g_0(T):
#     t = np.transpose
#     h, s = hs_i_g(T)
#     g_0 = h - T * s
#     return g_0


#---------------------------------------------------------------------------------------------------------------------------------
def hs_i_g(T, name_species):
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
    else:
        len_x = len(T)
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]

    i_species = np.where(np.array(gd.name_gas_species) == name_species)[0][0]
    h_i, s_i = np.where(T < T_mid[i_species],
                        array_poly_2_hs_g(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                        array_poly_2_hs_g(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    return h_i, s_i


#---------------------------------------------------------------------------------------------------------------------------------
def uCv_i_g(T, name_species):
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
    else:
        len_x = len(T)
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]

    i_species = np.where(np.array(gd.name_gas_species) == name_species)[0][0]
    u_i, cv_i = np.where(T < T_mid[i_species],
                        array_poly_2_Cv_g(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                        array_poly_2_Cv_g(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    return u_i, cv_i


#---------------------------------------------------------------------------------------------------------------------------------
def uCv_i_p_vec(T, phase_index = None):
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
        cv_ct = np.zeros([gd.nb_p_species, len_x])
        u_ct  = np.zeros([gd.nb_p_species, len_x])
        for i_species in range(gd.nb_p_species):
            u_ct[i_species, :], cv_ct[i_species, :] = np.where(T < T_mid[i_species],
                                                            array_poly_2_Cv_p(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                                                            array_poly_2_Cv_p(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    else:
        len_x = len(T)
        T_mid_vec = np.tile(T_mid,(len_x, 1))
        A = T < np.transpose(T_mid_vec)
        B = array_poly_2_Cv_p_multi_species(coeffs_0, gd.MW_gas_vec, np.tile(T,(gd.nb_p_species, 1)))
        C = array_poly_2_Cv_p_multi_species(coeffs_1, gd.MW_gas_vec, np.tile(T,(gd.nb_p_species, 1)))
        u_ct, cv_ct = ne.evaluate('where(A,B,C)')
    return np.transpose(u_ct), np.transpose(cv_ct)


#---------------------------------------------------------------------------------------------------------------------------------
def uCv_i_p(T,name_species):
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
    else:
        len_x = len(T)
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]

    i_species = np.where(np.array(gd.name_p_species) == name_species)[0][0]
    u_i, cv_i = np.where(T < T_mid[i_species],
                        array_poly_2_Cv_p(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                        array_poly_2_Cv_p(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    return u_i, cv_i


#---------------------------------------------------------------------------------------------------------------------------------
@jit
def array_poly_2_Cv_g(coeffs, MW, T):
    """ from NASA7 coefficients """
    Cv = (coeffs[0] + coeffs[1]*T   + coeffs[2]*T**2   + coeffs[3]*T**3   + coeffs[4]*T**4                 - 1) * R / MW
    u =  (coeffs[0] + coeffs[1]*T/2 + coeffs[2]*T**2/3 + coeffs[3]*T**3/4 + coeffs[4]*T**4/5 + coeffs[5]/T - 1) * R * T / MW
    return u, Cv


#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cv_g_multi_species(coeffs, MW, T):
    """ from NASA7 coefficients """
    T = np.transpose(T)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]

    Cv = np.transpose((A + B*T   + C*T**2   + D*T**3   + E*T**4 - 1) * R / MW)
    u  = np.transpose((A + B*T/2   + C*T**2/3   + D*T**3/4   + E*T**4/5 + F/T - 1) * R * T / MW)

    return u, Cv

#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cv_g_multi_species_ne(coeffs, MW, T):
    """ from NASA7 coefficients """
    T = np.transpose(T)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]

    Cv = np.transpose(ne.evaluate('(A + B*T   + C*T**2   + D*T**3   + E*T**4 - 1) * R / MW'))
    u  = np.transpose(ne.evaluate('(A + B*T/2   + C*T**2/3   + D*T**3/4   + E*T**4/5 + F/T - 1) * R * T / MW'))

    return u, Cv


#---------------------------------------------------------------------------------------------------------------------------------
@jit
def array_poly_2_hs_g(coeffs, MW, T):
    """ from NASA7 coefficients """
    h = (coeffs[0] + coeffs[1]*T/2 + coeffs[2]*T**2/3 + coeffs[3]*T**3/4 + coeffs[4]*T**4/5 + coeffs[5]/T) * R * T / MW
    s = (coeffs[0] * np.log(T) + coeffs[1] * T + coeffs[2]*T**2/2 + coeffs[3]*T**3/3 + coeffs[4]*T**4/4 + coeffs[6]) * R / MW
    return h, s


#----------------------------------------------------------------------------------------------------------------------------------

def array_poly_2_hs_g_multi_species(coeffs, MW, T):
    """ from NASA7 coefficients """
    T = np.transpose(T)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]
    G = coeffs[:, 6]
    h  = np.transpose((A + B*T/2   + C*T**2/3   + D*T**3/4   + E*T**4/5 + F/T ) * R * T / MW)
    s = np.transpose((A * np.log(T) + B * T + C*T**2/2 + D*T**3/3 + E*T**4/4 + G) * R / MW)

    return h, s


#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_hs_g_multi_species_ne(coeffs, MW, T):
    """ from NASA7 coefficients """
    T = np.transpose(T)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]
    G = coeffs[:, 6]
    # chron1 = time.time()
    h  = np.transpose(ne.evaluate('(A + B*T/2   + C*T**2/3   + D*T**3/4   + E*T**4/5 + F/T ) * R * T / MW'))
    s = np.transpose(ne.evaluate('(A * log(T) + B * T + C*T**2/2 + D*T**3/3 + E*T**4/4 + G) * R / MW'))

    return h, s

#---------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cv_p(coeffs, MW, T):
    """ from NIST coefficients : shomate formulation """
    t = T/1000
    Cv = (coeffs[0] + coeffs[1]*t   + coeffs[2]*t**2   + coeffs[3]*t**3   + coeffs[4]/t**2)/1000
    u =  (coeffs[0]*t + coeffs[1]*t**2/2 + coeffs[2]*t**3/3 + coeffs[3]*t**4/4 - coeffs[4]/t + coeffs[5] - coeffs[6])
    return u, Cv


#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cv_p_multi_species(coeffs, MW, T):
    """ from NIST coefficients : shomate formulation """
    t = np.transpose(T/1000)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]
    G = coeffs[:, 6]

    Cv = np.transpose(ne.evaluate('(A + B*t   + C*t**2   + D*t**3   + E/t**2)/1000'))
    u  = np.transpose(ne.evaluate('A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - G'))

    # Cv = np.transpose(coeffs[:,0] + coeffs[:,1]*t   + coeffs[:,2]*t**2   + coeffs[:,3]*t**3   + coeffs[:,4]/t**2)
    # u =  np.transpose(coeffs[:,0]*t + coeffs[:,1]*t**2/2 + coeffs[:,2]*t**3/3 + coeffs[:,3]*t**4/4 - coeffs[:,4]/t + coeffs[:,5] - coeffs[:,6])

    return u, Cv


#---------------------------------------------------------------------------------------------------------------------------------
def hCp_i_g_vec(T):
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]

    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
        cp_ct = np.zeros([gd.nb_gas_species, len_x])
        h_ct  = np.zeros([gd.nb_gas_species, len_x])
        for i_species in range(gd.nb_gas_species):
            h_ct[i_species, :], cp_ct[i_species, :] = np.where(T < T_mid[i_species],
                                                            array_poly_2_Cp_g(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                                                            array_poly_2_Cp_g(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    else:
        len_x = len(T)
        T_mid_vec = np.tile(T_mid,(len_x, 1))
        A = T < np.transpose(T_mid_vec)
        B = array_poly_2_Cp_g_multi_species(coeffs_0, gd.MW_gas_vec, np.tile(T,(gd.nb_gas_species, 1)))
        C = array_poly_2_Cp_g_multi_species(coeffs_1, gd.MW_gas_vec, np.tile(T,(gd.nb_gas_species, 1)))
        h_ct, cp_ct = ne.evaluate('where(A,B,C)')
    return np.transpose(h_ct), np.transpose(cp_ct)


#---------------------------------------------------------------------------------------------------------------------------------
def hCp_i_g(T, name_species):
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
    else:
        len_x = len(T)
    len_poly = 7
    T_mid = gd.thermo_coeff_g[:, 0]
    coeffs_0 = gd.thermo_coeff_g[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_g[:, 1 : 7+1]

    i_species = np.where(np.array(gd.name_gas_species) == name_species)[0][0]
    h_i, cp_i = np.where(T < T_mid[i_species],
                        array_poly_2_Cp_g(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                        array_poly_2_Cp_g(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    return h_i, cp_i


#---------------------------------------------------------------------------------------------------------------------------------
def hCp_i_p_vec(T, phase_index = None):
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
        cp_ct = np.zeros([gd.nb_p_species, len_x])
        h_ct  = np.zeros([gd.nb_p_species, len_x])
        for i_species in range(gd.nb_p_species):
            h_ct[i_species, :], cp_ct[i_species, :] = np.where(T < T_mid[i_species],
                                                            array_poly_2_Cp_p(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                                                            array_poly_2_Cp_p(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    else:
        len_x = len(T)
        T_mid_vec = np.tile(T_mid,(len_x, 1))
        A = T < np.transpose(T_mid_vec)
        B = array_poly_2_Cp_p_multi_species(coeffs_0, gd.MW_gas_vec, np.tile(T,(gd.nb_p_species, 1)))
        C = array_poly_2_Cp_p_multi_species(coeffs_1, gd.MW_gas_vec, np.tile(T,(gd.nb_p_species, 1)))
        h_ct, cp_ct = ne.evaluate('where(A,B,C)')
    return np.transpose(h_ct), np.transpose(cp_ct)


#---------------------------------------------------------------------------------------------------------------------------------
def hCp_i_p(T,name_species):
    if isinstance(T, float) or isinstance(T, int):
        len_x = 1
    else:
        len_x = len(T)
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]

    i_species = np.where(np.array(gd.name_p_species) == name_species)[0][0]
    h_i, cp_i = np.where(T < T_mid[i_species],
                        array_poly_2_Cp_p(coeffs_0[i_species], gd.MW_gas_vec[i_species], T),
                        array_poly_2_Cp_p(coeffs_1[i_species], gd.MW_gas_vec[i_species], T))
    return h_i, cp_i


#---------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cp_g(coeffs, MW, T):
    """ from NASA7 coefficients """
    Cp = (coeffs[0] + coeffs[1]*T   + coeffs[2]*T**2   + coeffs[3]*T**3   + coeffs[4]*T**4                 - 0) * R / MW
    h  = (coeffs[0] + coeffs[1]*T/2 + coeffs[2]*T**2/3 + coeffs[3]*T**3/4 + coeffs[4]*T**4/5 + coeffs[5]/T - 0) * R * T / MW
    return h, Cp


#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cp_g_multi_species(coeffs, MW, T):
    """ from NASA7 coefficients """
    T = np.transpose(T)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]

    Cp = np.transpose(ne.evaluate('(A + B*T   + C*T**2   + D*T**3   + E*T**4 - 0) * R / MW'))
    h  = np.transpose(ne.evaluate('(A + B*T/2   + C*T**2/3   + D*T**3/4   + E*T**4/5 + F/T - 0) * R * T / MW'))

    # Cv = np.transpose((coeffs[:, 0] + coeffs[:, 1]*T   + coeffs[:, 2]*T**2   + coeffs[:, 3]*T**3   + coeffs[:, 4]*T**4                 - 1) * R / MW)
    # u =  np.transpose((coeffs[:, 0] + coeffs[:, 1]*T/2 + coeffs[:, 2]*T**2/3 + coeffs[:, 3]*T**3/4 + coeffs[:, 4]*T**4/5 + coeffs[:, 5]/T - 1) * R * T / MW)
    return h, Cp


#---------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cp_p(coeffs, MW, T):
    """ from NIST coefficients : shomate formulation """
    t = T/1000
    Cp = (coeffs[0] + coeffs[1]*t   + coeffs[2]*t**2   + coeffs[3]*t**3   + coeffs[4]/t**2)/1000
    h  = (coeffs[0]*t + coeffs[1]*t**2/2 + coeffs[2]*t**3/3 + coeffs[3]*t**4/4 - coeffs[4]/t + coeffs[5] - coeffs[6])
    return h, Cp


#----------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cp_p_multi_species(coeffs, MW, T):
    """ from NIST coefficients : shomate formulation """
    t = np.transpose(T/1000)
    A = coeffs[:, 0]
    B = coeffs[:, 1]
    C = coeffs[:, 2]
    D = coeffs[:, 3]
    E = coeffs[:, 4]
    F = coeffs[:, 5]
    G = coeffs[:, 6]

    Cp = np.transpose(ne.evaluate('(A + B*t   + C*t**2   + D*t**3   + E/t**2)/1000'))
    h  = np.transpose(ne.evaluate('A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - G'))

    # Cv = np.transpose(coeffs[:,0] + coeffs[:,1]*t   + coeffs[:,2]*t**2   + coeffs[:,3]*t**3   + coeffs[:,4]/t**2)
    # u =  np.transpose(coeffs[:,0]*t + coeffs[:,1]*t**2/2 + coeffs[:,2]*t**3/3 + coeffs[:,3]*t**4/4 - coeffs[:,4]/t + coeffs[:,5] - coeffs[:,6])

    return h, Cp


#---------------------------------------------------------------------------------------------------------------------------------
def find_phase(particule,T,gas,system):
    # Cette fonction a pour but de trouver la phase (solide ou liquide) de chacune des espèces dans une particule.
    # Elle renvoie alors un tableau de type ["s","s","s","s"] ou "s" signifie solide, "l" liquide, et "PC" signifie que l'espèce est en cours de changement de phase.
    # Une particule a plusieurs espèces dont on connait la masse.
    # En connaissant la masse, on peut retrouver les deux valeurs d'enthalpie entre lesquelles s'effectue le changement de phase.
    # De cette manière on peut retrouver l'etat de chacune des espèces présente dans la particule.

    phase = particule.phase
    h_p = particule.enthalpy_m

    u_PC_lim_species = get_u_PC_lim_species()
    bool_PC = np.full(system.n_volumes, False)
    for species in gd.ips.keys():
        chron1 = time.time()
        phase[:, gd.ips[species]], bool_PC = find_vec_phase_species(particule, gas, species, u_PC_lim_species, bool_PC)
    return phase, bool_PC


#----------------------------------------------------------------------------------------------------------------------------------
def get_u_PC_lim_species():
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]
    u_min = array_poly_2_Cv_p_multi_species(coeffs_0, gd.MW_gas_vec, T_mid)[0]
    u_max = array_poly_2_Cv_p_multi_species(coeffs_1, gd.MW_gas_vec, T_mid)[0]
    return u_min, u_max


#----------------------------------------------------------------------------------------------------------------------------------
def find_vec_phase_species(particle, gas, name_species, u_PC_lim_species, bool_PC):
        T_PC = gd.thermo_coeff_p[gd.ips[name_species], 0]
        h_p = particle.enthalpy_m

        uh_i_liq = uCv_i_p_vec(T_PC)[0]

        H_s_PC1_species_min = np.sum(np.delete(particle.Y * uh_i_liq,gd.ips[name_species], axis = 1), axis = 1) + particle.Y[:, gd.ips[name_species]] * u_PC_lim_species[0][gd.ips[name_species]]
        H_s_PC1_species_max = np.sum(np.delete(particle.Y * uh_i_liq,gd.ips[name_species], axis = 1), axis = 1) + particle.Y[:, gd.ips[name_species]] * u_PC_lim_species[1][gd.ips[name_species]]

        A = (H_s_PC1_species_min<h_p) & (h_p>H_s_PC1_species_max)
        B = 'PC1'
        C = 's'
        bool_PC_partial_1 = (H_s_PC1_species_min<h_p) & (h_p<H_s_PC1_species_max) & (particle.Y[:, gd.ips[name_species]]>0)
        phase_species = np.where(bool_PC_partial_1, 'PC1', 's')
        bool_PC_partial_2 = np.full(len(particle.temperature), False)
        # if name_species in set(gd.name_gas_species):
        #     A = particle.species_carac[gd.ips[name_species]].T_evap
        #     B = np.interp(gas.pressure,
        #                 gas.T_P_Me_evap[1],
        #                 gas.T_P_Me_evap[0])

        #     T_evap=np.where(A>B, A, B)
        #     # T_evap=ne.evaluate('where(A>B, A, B)')

        #     uh_i_gaz_1 = uCv_i_p_vec(T_evap)[0]
        #     # species_list_gp = list(set(gd.name_gas_species) & set(gd.name_p_species))
        #     uh_i_gaz_2 = uCv_i_g(T_evap, name_species)[0]

        #     H_s_PC2_species_min = np.sum(np.delete(particle.Y * uh_i_gaz_1,gd.ips[name_species], axis = 1), axis = 1) + particle.Y[:, gd.ips[name_species]] * uh_i_gaz_1[:, gd.ips[name_species]]
        #     H_s_PC2_species_max = np.sum(np.delete(particle.Y * uh_i_gaz_1,gd.ips[name_species], axis = 1), axis = 1) + particle.Y[:, gd.ips[name_species]] * uh_i_gaz_2
        #     bool_PC_partial_2 = (H_s_PC2_species_min<h_p) & (h_p>H_s_PC2_species_max) & (particle.Y[:, gd.ips[name_species]]>0)
        #     phase_species = np.where(bool_PC_partial_2, 'PC2', phase_species)

        bool_PC_partial = bool_PC_partial_1 | bool_PC_partial_2
        bool_PC = bool_PC | bool_PC_partial
        return phase_species, bool_PC


#----------------------------------------------------------------------------------------------------------------------------------
def find_phase_0D(particule,T,gas):
    # Cette fonction a pour but de trouver la phase (solide ou liquide) de chacune des espèces dans une particule.
    # Elle renvoie alors un tableau de type ["s","s","s","s"] ou "s" signifie solide, "l" liquide, et "PC" signifie que l'espèce est en cours de changement de phase.
    # Une particule a plusieurs espèces dont on connait la masse.
    # En connaissant la masse, on peut retrouver les deux valeurs d'enthalpie entre lesquelles s'effectue le changement de phase.
    # De cette manière on peut retrouver l'etat de chacune des espèces présente dans la particule.

    phase = particule.phase
    h_p = particule.enthalpy_m

    Y_Al_p    = particule.Y[:,gd.ips["Al"]]
    Y_MeO_p   = particule.Y[:,gd.ips["MeO"]]
    Y_Me_p    = particule.Y[:,gd.ips["Me"]]
    Y_Al2O3_p = particule.Y[:,gd.ips["Al2O3"]]

    Al    = particule.species_carac[gd.ips["Al"]]
    MeO   = particule.species_carac[gd.ips["MeO"]]
    Me    = particule.species_carac[gd.ips["Me"]]
    Al2O3 = particule.species_carac[gd.ips["Al2O3"]]

#------------------------------------------------------------changement de phase Al
    if Y_Al_p>gd.eps10:
        H_s_Al=(h_p)
        H_s_PC1_Al_min=H_tot(Al.T_liq,Al,0)*Y_Al_p+Y_Al2O3_p*H_tot(Al.T_liq,Al2O3,0)#indice 0 car l'al2O3 est solide lors du CP de l'Al. On utilise donc les coeff solides
        H_s_PC1_Al_max=(H_tot(Al.T_liq,Al,0)+Al.h_liq)*Y_Al_p+Y_Al2O3_p*H_tot(Al.T_liq,Al2O3,0)

        if (H_s_Al>H_s_PC1_Al_min and H_s_Al<H_s_PC1_Al_max):
            phase[gd.ips["Al"]]="PC1"
        elif H_s_Al>H_s_PC1_Al_max:
            phase[gd.ips["Al"]]="l"
            T_evap=Al.T_evap
            if particule.temperature>Al.T_evap*0.95:

                T_evap=max(Al.T_evap,np.interp(gas.pressure,gas.T_P_Al_evap[1],gas.T_P_Al_evap[0]))
                # print(T_evap,Al.T_evap)
            H_s_PC2_Al_min=H_tot(T_evap,Al,1)*Y_Al_p+Y_Al2O3_p*H_tot(T_evap,Al2O3,1)#indice 1 car l'al2O3 est liquide lors du CP de l'Al. On utilise donc les coeff liquides
            H_s_PC2_Al_max=(H_tot(T_evap,Al,1)+Al.h_evap)*Y_Al_p+Y_Al2O3_p*H_tot(T_evap,Al2O3,1)
            # save_data_time(H_s_PC2_Al_min,"min_h")
            if (H_s_Al>H_s_PC2_Al_min and H_s_Al<H_s_PC2_Al_max):
                phase[gd.ips["Al"]]="PC2"

        elif H_s_Al<H_s_PC1_Al_min:
            phase[gd.ips["Al"]]="s"
    T_evap=max(Al.T_evap,np.interp(gas.pressure,gas.T_P_Al_evap[1],gas.T_P_Al_evap[0]))
    H_s_PC2_Al_min=H_tot(T_evap,Al,1)*Y_Al_p+Y_Al2O3_p*H_tot(T_evap,Al2O3,1)

#--------------------------------------------------------------changement de phase Al2O3
    if Y_Al2O3_p>gd.eps10 and T>1000:
        H_s_Al2O3=(h_p)
        H_s_PC_Al2O3_min=H_tot(Al2O3.T_liq,Al2O3,0)*Y_Al2O3_p+Y_Al_p*H_tot(Al2O3.T_liq,Al,1)+Y_MeO_p*H_tot(Al2O3.T_liq,MeO,0)+Y_Me_p*H_tot(Al2O3.T_liq,Me,1) #indice 1 car l'alu est liquide lors du CP de l'Al2O3. On utilise donc les coeff liquides
        H_s_PC_Al2O3_max=(H_tot(Al2O3.T_liq,Al2O3,0)+Al2O3.h_liq)*Y_Al2O3_p+Y_Al_p*H_tot(Al2O3.T_liq,Al,1)+Y_MeO_p*H_tot(Al2O3.T_liq,MeO,0)+Y_Me_p*H_tot(Al2O3.T_liq,Me,1)
        if (H_s_Al2O3>H_s_PC_Al2O3_min and H_s_Al2O3<H_s_PC_Al2O3_max):
            phase[gd.ips["Al2O3"]]="PC1"
        elif H_s_Al2O3>H_s_PC_Al2O3_max:
            phase[gd.ips["Al2O3"]]="l"
        elif H_s_Al2O3<H_s_PC_Al2O3_min:
            phase[gd.ips["Al2O3"]]="s"

#--------------------------------------------------------------changement de phase Me
    if Y_Me_p>gd.eps10:
        H_s_Me=(h_p)
        H_s_PC1_Me_min=H_tot(Me.T_liq,Me,0)*Y_Me_p+Y_Al2O3_p*H_tot(Me.T_liq,Al2O3,0)+Y_MeO_p*H_tot(Me.T_liq,MeO,0)
        H_s_PC1_Me_max=(H_tot(Me.T_liq,Me,0)+Me.h_liq)*Y_Me_p+Y_Al2O3_p*H_tot(Me.T_liq,Al2O3,0)+Y_MeO_p*H_tot(Me.T_liq,MeO,0)


        if (H_s_Me>H_s_PC1_Me_min and H_s_Me<H_s_PC1_Me_max):
            phase[gd.ips["Me"]]="PC1"
        elif H_s_Me>H_s_PC1_Me_max:
            phase[gd.ips["Me"]]="l"
            T_evap=Me.T_evap
            if particule.temperature>Me.T_evap*0.95:

                T_evap=max(Me.T_evap,np.interp(gas.pressure,gas.T_P_Me_evap[1],gas.T_P_Me_evap[0]))
            H_s_PC2_Me_min=H_tot(T_evap,Me,1)*Y_Me_p+Y_Al2O3_p*H_tot(T_evap,Al2O3,1)+Y_MeO_p*H_tot(T_evap,MeO,0)
            H_s_PC2_Me_max=(H_tot(T_evap,Me,1)+Me.h_evap)*Y_Me_p+Y_Al2O3_p*H_tot(T_evap,Al2O3,1)+Y_MeO_p*H_tot(T_evap,MeO,0)

            if (H_s_Me>H_s_PC2_Me_min and H_s_Me<H_s_PC2_Me_max):
                phase[gd.ips["Me"]]="PC2"

        elif H_s_Me<H_s_PC1_Me_min:
            phase[gd.ips["Me"]]="s"
    if particule.name == "pMeO":
        H_s_PC2 = H_s_PC2_Me_min
    else:
        H_s_PC2 = H_s_PC2_Al_min

    return phase


#---------------------------------------------------------------------------------------------------------------------------------
def H_tot(T,species,phase_index):
    # cette fonction calcule l'enthalpie totale d'une espèce "i": H_tot_i=H_formation_i+H_sensible_i
    # H_sensible_i est calculée par la fonction "DELTA_H"
    h = DELTA_H(T,species,phase_index)+species.h_form
    # uCv = uCv_i_p(T,species.name)[0]+species.h_form
    return h


#---------------------------------------------------------------------------------------------------------------------------------
def U_tot(T,species,phase_index):
    u=DELTA_U(T,species,phase_index)+species.h_form
    return u


#---------------------------------------------------------------------------------------------------------------------------------
@njit
def energy_int_from_Q(Q):
    velocity_x = Q[:,gd.i_vx] / Q[:,gd.i_d]
    velocity_y = Q[:,gd.i_vy] / Q[:,gd.i_d]
    e_tot      = Q[:,gd.i_e]  / Q[:,gd.i_d]
    e_int      = e_tot - 0.5 * (velocity_x**2 + velocity_y**2)
    return e_int


#---------------------------------------------------------------------------------------------------------------------------------
def TP_from_Q(T_init, density, Q):
    Y = np.transpose(np.transpose(Q[:, gd.i_Y:]) / Q[:, gd.i_d])
    temperature = find_g_temperature(T_init, Q)
    pressure = R * density * temperature * np.sum(Y / gd.MW_gas_vec, axis = 1)
    return temperature, pressure


#---------------------------------------------------------------------------------------------------------------------------------
def RTY_from_Q(T_init, Q, system):
    Y = np.transpose(np.transpose(Q[:, gd.i_Y:]) / Q[:, gd.i_d])
    temperature = find_g_temperature(T_init, Q, system, bool_check_T_range = False)
    return gd.R * temperature * np.sum(Y / gd.MW_gas_vec, axis = 1)

#---------------------------------------------------------------------------------------------------------------------------------
def compute_c(Cv, Y, T):
    Cp = Cv + np.sum(Y / gd.MW_gas_vec, axis = 1) * gd.R
    gamma = Cp / Cv
    R_s = np.sum(Y / gd.MW_gas_vec, axis = 1) * gd.R
    c = (gamma * R_s * T)**0.5
    return c
