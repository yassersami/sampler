
"""
From numbakit looped v1

"""
# PROJECT_PATH_kin = os.path.abspath(os.path.join(os.path.abspath(__file__),
#                                     '..','..'))

import time
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.integrate import BDF

import py0D.global_data as gd
import py0D.kinetics.enthalpy_numba as enth_nb
import py0D.kinetics.functions_numba as func_nb
import py0D.kinetics.int_foo as ifo
import py0D.kinetics.second as sec
from py0D.kinetics.int_foo import add_method, add_attr

# import params_plot
# import functions_assist_plot as fap
# import perso_scipy_jac_modified as psjm
# import nbkode

t = np.transpose


nu_ir   = np.zeros(([gd.nb_gas_species, gd.len_Reac]))
nu_ir_p = np.zeros(([gd.nb_gas_species, gd.len_Reac]))
nu_ir_r = np.zeros(([gd.nb_gas_species, gd.len_Reac]))
k0_r   = np.zeros(gd.len_Reac)
Ea_r   = np.zeros(gd.len_Reac)
T_ex_r = np.zeros(gd.len_Reac)
eff    = np.ones([gd.nb_gas_species, gd.len_Reac]) / gd.nb_gas_species
bool_3b= np.full(gd.len_Reac, False)
n_volumes = 255

for i,r in enumerate(gd.Reac_l):
    for s in gd.name_gas_species:
        if s in r.products.keys():
            nu_ir_p[gd.igs[s],i] = r.products[s]
        else:
            nu_ir_p[gd.igs[s],i] = 0

        if s in r.reactants.keys():
            nu_ir_r[gd.igs[s],i] = r.reactants[s]
        else:
            nu_ir_r[gd.igs[s],i] = 0
        nu_ir[gd.igs[s],i] = nu_ir_p[gd.igs[s],i] - nu_ir_r[gd.igs[s],i]


        if hasattr(r, "efficiencies"):
            if s in r.efficiencies.keys():
                eff[gd.igs[s], i] = r.efficiencies[s]
            else:
                eff[gd.igs[s], i] = 1
            bool_3b[i] = True

    k0_r[i]   = r.rate.pre_exponential_factor       # cantera utilise les kmol comme unité par défaut
    Ea_r[i]   = r.rate.activation_energy             # cantera utilise les kmol comme unité par défaut
    T_ex_r[i] = r.rate.temperature_exponent

nu_ir_r_tiled = t(np.tile(nu_ir_r, (n_volumes , 1, 1)), (0,1,2))
nu_ir_p_tiled = t(np.tile(nu_ir_p, (n_volumes , 1, 1)), (0,1,2))
sum_nu_r = np.sum(nu_ir, axis = 0)
n_volumes = n_volumes
sum_nu_r_dtiled = np.tile(sum_nu_r, (n_volumes, gd.nb_gas_species, 1))
wdot_kin = np.zeros(n_volumes)


@njit
def compute_wdot_nb(T, Y, density, alpha):
    T_tiled  = func_nb.tile_nb(T, gd.len_Reac).T

    # on annule la cinétique des espèces quasi inexistantes
    # Y = np.where(Y<1e-12, 0, Y)
    
    g_i_0       = enth_nb.g_0_vec(T)[0]

    g_i_0_tiled = func_nb.tile_nb(g_i_0, gd.len_Reac).T
    # Calcul de Delta G° et du Kp par réaction
    D_G_r       = np.sum(nu_ir * g_i_0_tiled, axis = 0) # somme sur les espèces
    K_p_r       = t(np.exp(-t(D_G_r) / gd.R / T))
    K_p_r_tiled = func_nb.tile_nb(K_p_r, gd.nb_gas_species)

    C           = t(density * t(Y))/gd.MW_gas_arr / 1000 # pour avoir des kmol comme cantera
    C_tiled     = t(func_nb.tile_nb(C, gd.len_Reac))

    M = np.sum(C_tiled * eff, axis = 0) # reaction 3 corps

    k_0_r_f     = T_tiled**T_ex_r * k0_r * np.exp((-Ea_r) / (gd.R*1e3 * T_tiled)) * ((M-1) * bool_3b + 1)
    k_0_r_f_tiled = func_nb.tile_nb(k_0_r_f, gd.nb_gas_species)

    K_c = ((K_p_r / (gd.R * T_tiled / (gd.P_ref/1e3))**(sum_nu_r) + gd.eps))

    prod_Cnu_r_f_tile = np.zeros((gd.nb_gas_species, gd.len_Reac))
    prod_Cnu_r_b_tile = np.zeros((gd.nb_gas_species, gd.len_Reac))
    for j in range(gd.len_Reac):
        prod_Cnu_r_f_tile[:, j] = np.prod(C_tiled[:,j] ** (nu_ir_r[:,j]))
        prod_Cnu_r_b_tile[:, j] = np.prod(C_tiled[:,j] ** (nu_ir_p[:,j]))

    w_dot_i_r   = k_0_r_f_tiled *((nu_ir_p - nu_ir_r)
                                *(prod_Cnu_r_f_tile - 1/K_c*prod_Cnu_r_b_tile
                                    ))

    w_dot_i = np.sum(w_dot_i_r, axis = 1)

    w_dot_i_mass = w_dot_i * gd.MW_gas_arr * 1e3 # w_dot_i est en kmol, MW_gas_vec est en kg/mol

    return t(t(w_dot_i_mass) * alpha)




@njit
def fun_to_integrate(t, y, dQ_dt_0D = None):
    T = y[0]
    alpha = y[1]
    Q = y[2:]
    density = Q[gd.i_d]/alpha
    Q_tmp = np.zeros((1,Q.shape[0]))
    Q_tmp[0] = Q
    energy = enth_nb.energy_int_from_Q(Q_tmp)[0]
    raY = Q[gd.i_Y:]
    Y = (raY.T / Q[gd.i_d]).T
    if dQ_dt_0D is None:
        dQ_dt_0D = np.zeros(Q.shape)
    volume = 1
    dy_dt = np.zeros(y.shape)
    dy_dt[2:] = dQ_dt_0D

    wdot = compute_wdot_nb(T, Y, density, alpha)
    dYg_dt_tmp =   (((wdot + dQ_dt_0D[gd.i_Y:])
                    - (Y.T * dQ_dt_0D[gd.i_d]).T).T / Q[gd.i_d]).T

    dTg_dt = func_nb.dT_dt_computation_g(Q = Q, Y = Y, dYg_dt = dYg_dt_tmp, dQ_dt = dQ_dt_0D, T = T)
    dy_dt[(gd.i_Y+2):] = wdot + dQ_dt_0D[gd.i_Y:]
    dy_dt[0] = dTg_dt
    return dy_dt


def kin_dQ_dt(alpha, Q, dQ_dt_0D, dt, int_BDF = None, T = None):
    if T is None:
        T = enth_nb.find_g_temperature(np.array([2000]), np.array([Q]))[0]
    y0 = (Q)
    y0 = np.insert(y0, 0, alpha)
    y0 = np.insert(y0, 0, T)

    args = (dQ_dt_0D,)

    int_BDF = ifo.init_BDF(int_BDF, y0, dt, fun_wo_arg = fun_to_integrate, args=args)

###################################################################

    t_BDF, Q_BDF = int_BDF.get_int_Q(int_BDF)

###################################################################

    return t_BDF, Q_BDF, int_BDF


def init_BDF(T, alpha, Q, dQ_dt_0D, dt):
    y0 = (Q)
    y0 = np.insert(y0, 0, alpha)
    y0 = np.insert(y0, 0, T)

    chron_BDF = time.time()
    atol    = np.full(y0.shape, 100)
    rtol    = 1e-4
    atol    = 1e-9

    int_BDF = BDF(fun_to_integrate, 0, y0, dt, atol=np.array(atol), rtol=rtol)
    add_method(int_BDF)
    add_attr(int_BDF, dict(y0_shape = None))

    return int_BDF


def compute_kin(gas, dt, volume, int_BDF = None):
    alpha_save  = copy(gas.alpha)
    gas.alpha   = gas.alpha + gas.d_alpha*0

    Q           = sec.generate_Q(gas.alpha, gas.density, gas.energy, gas.Y)
    dQ_dt_0D = sec.generate_dQ_dt(gas, volume, Q.shape)
    # Q = Q + dQ_dt_0D * dt
    # dQ_dt_0D = np.zeros(dQ_dt_0D.shape)
    if int_BDF is None:
        int_BDF = init_BDF(gas.temperature, gas.alpha, Q, dQ_dt_0D, dt)

    (t_BDF,
        Q_BDF,
        int_BDF) = kin_dQ_dt(gas.alpha,
                                Q,
                                dQ_dt_0D,
                                dt,
                                int_BDF = int_BDF
                                )
    dmi_dt_kin = sec.degenerate_Q(gas.alpha, Q, Q_BDF, dQ_dt_0D, volume, dt)
    # dmi_dt_kin = np.zeros(dmi_dt_kin.shape)
    gas.alpha = alpha_save
    return dmi_dt_kin, int_BDF, t_BDF, Q_BDF


def plot_1(t, Q):
    # fig0 = plt.figure(figsize=params_plot.figsize_square)
    fig0 = plt.figure()
    ax1 = plt.subplot(111)
    ax1.plot(t, enth_nb.find_g_temperature(np.full(Q.shape[0] ,2200),np.array([Q])))
    ax1 = fap.ax_style(ax1, "kinetical T° temporal evolution", "time", "T° (K)")
    ax1.legend()
    plt.tight_layout()
    plt.show()
