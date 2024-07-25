from copy import deepcopy

import numpy as np
from numba import jit, njit
from scipy.linalg import inv
from scipy.linalg import solve_banded
from scipy.sparse import diags

# from assist_functions import vectorize
import py0D.global_data as gd
import py0D.kinetics.enthalpy_numba as enth_nb

R = gd.R
P_atm = 1e5


#---------------------------------------------------------------------------------------------------------------------------------
def V_sphere(r):
    V = 4/3*np.pi*r**3
    return V


#---------------------------------------------------------------------------------------------------------------------------------
def S_sphere(r):
    S = 4*np.pi*r**2
    return S


#---------------------------------------------------------------------------------------------------------------------------------
def mean_x(X):
    X = (X[1:] + X[:-1])/2
    X = np.insert(X, 0, [0], axis = 0)                               # Ajout d'une cellule supplémentaire
    return X


#---------------------------------------------------------------------------------------------------------------------------------
def mean_x_harm(X):
    X = 2/(1/X[1:] + 1/X[:-1])
    X = np.insert(X, 0, [0], axis = 0)                               # Ajout d'une cellule supplémentaire
    return X


#---------------------------------------------------------------------------------------------------------------------------------
def mean_T_harm(Tg, wg, Tp, wp, Tq, wq):
    T_mean = (wg + wp + wq)/(wg/Tg + wp/Tp + wq/Tq)
    return T_mean


#---------------------------------------------------------------------------------------------------------------------------------
def P_evap(T, species):
    return P_atm * np.exp(species.h_evap * species.MW / R * (1/species.T_evap - 1/T))


#---------------------------------------------------------------------------------------------------------------------------------
def Arr(T, k, Ea):
    return k*np.exp(-Ea/R/T)


#---------------------------------------------------------------------------------------------------------------------------------
def flux_prob_evap(species, T):
    E = species.K*T/np.sqrt(2*np.pi*species.MW*R)*np.exp(-(species.h_evap*species.MW)/R/T)*species.MW # kg/m2/s
    return E


#---------------------------------------------------------------------------------------------------------------------------------
def Gauss(base_value, A, mu, sigma, x_vec):
    return (A - base_value) * np.exp(-(x_vec - mu)**2/(2 * sigma**2)) + base_value


#---------------------------------------------------------------------------------------------------------------------------------
def line(A0, A1, x_vec):
    return (x_vec - x_vec[0]) * (A1 - A0) / (x_vec[-1] - x_vec[-0]) + A0


#---------------------------------------------------------------------------------------------------------------------------------
def Square(A, mu, sigma, x_vec):
    return np.where(np.abs(x_vec - mu) < sigma, A, 0)


#---------------------------------------------------------------------------------------------------------------------------------
def dX_dt(X,t):
    dX_dt = (X[1:]-X[:-1])/(t[1:]-t[:-1])
    return dX_dt


#---------------------------------------------------------------------------------------------------------------------------------
def moving_average_weighted(x, window_size, sigma = None, mu = None):
    # https://plotly.com/python/v3/fft-filters/ (pour filtre sinusoidal)
    weights = np.ones(window_size)
    if mu is None:
        mu = int(window_size/2)
    if sigma is None:
        sigma = window_size
        # sigma = np.round(window_size/6)

    for i in range (0, window_size):
        weights[i] = ((1/(sigma * np.sqrt(2*np.pi)))
                     * np.exp(-1/2*(((i-mu)/2)/sigma)**2)
                    )
    weights = weights / np.sum(weights)
    return np.convolve(x, weights, 'valid')


#---------------------------------------------------------------------------------------------------------------------------------
def extrap_x_inv(x, x1):
    """extrapolate f(x)=x function from a given point x1
    with the inverse function f(x) = 1/x
    """
    shift = 1.001
    a = - (x1 - shift)**2
    b = x1 - a / (x1 - shift)
    return (a / (x-shift) + b)


#---------------------------------------------------------------------------------------------------------------------------------
def extrap_x_exp(x, x1, slope = None):
    """extrapolate f(x)=x function from a given point x1
    with the exp function f(x) = exp(x)
    """
    if slope is None:
        slope = 1
    a = 1/np.exp(slope*x1)/slope
    b = x1 - a * np.exp(slope * x1)
    return (a * np.exp(slope * x) + b)


#---------------------------------------------------------------------------------------------------------------------------------
def T_inter(gas, particle, flux):
    """température de l'interface gaz/particule"""
    temperature = np.where(flux>0,
        particle.temperature, # if true
        gas.temperature)      # if false
    return temperature


#---------------------------------------------------------------------------------------------------------------------------------
def HS_vec(x, y_sup, y_inf):
    """Heaviside vectorized"""
    y = np.where(x>0,
                y_sup,       # if true
                y_inf)       # if false
    return y


#---------------------------------------------------------------------------------------------------------------------------------
def get_rho_e(T, P, Y):
    return get_rho(T, P, Y), get_e_g(T, Y)


#---------------------------------------------------------------------------------------------------------------------------------
def get_rho(T, P, Y):
    if isinstance(T, float) or isinstance(T, int):
        ax_correc = 1
    else:
        ax_correc = 0
    density = P / gd.R / T / np.sum(Y / gd.MW_gas_vec, axis = 1 - ax_correc)
    return density


#---------------------------------------------------------------------------------------------------------------------------------
def get_e_g(T, Y):
    uh_i = enth_nb.uCv_i_g_vec2(T)[0]
    energy_int = np.sum(Y * uh_i)
    return energy_int


#---------------------------------------------------------------------------------------------------------------------------------
def get_e_p(T, Y):
    uh_i = enth.uCv_i_p_vec(T)[0]
    energy_int = np.sum(Y * uh_i)
    return energy_int


#---------------------------------------------------------------------------------------------------------------------------------
def get_T(rho, P, Y):
    if isinstance(P, float) or isinstance(P, int):
        ax_correc = 1
    else:
        ax_correc = 0

    T = P / gd.R / rho / np.sum(Y / gd.MW_gas_vec, axis = 1 - ax_correc)
    return T


#---------------------------------------------------------------------------------------------------------------------------------
def get_P(rho, T, Y):
    if isinstance(T, float) or isinstance(T, int):
        ax_correc = 1
    else:
        ax_correc = 0

    P = T * gd.R * rho * np.sum(Y / gd.MW_gas_vec, axis = 1 - ax_correc)
    return P


#---------------------------------------------------------------------------------------------------------------------------------
def equilibrium_0D(Q, Y_tilde, alpha, name_species):
    # ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"]
    MDC = np.zeros(len(Y_tilde))
    ct_composition_input=("O2:"     + str(Y_tilde[gd.igs["O2"]])
                        +",Al:"     + str(Y_tilde[gd.igs["Al"]])
                        +",O:"      + str(Y_tilde[gd.igs["O"]])
                        +",N2:"     + str(Y_tilde[gd.igs["N2"]])
                        +",Al2O:"   + str(Y_tilde[gd.igs["Al2O"]])
                        +",AlO:"    + str(Y_tilde[gd.igs["AlO"]])
                        +",AlO2:"   + str(Y_tilde[gd.igs["AlO2"]])
                        +",Al2O2:"  + str(Y_tilde[gd.igs["Al2O2"]])
                        +",Cu:"     + str(Y_tilde[gd.igs["Cu"]])
                        )

    g = gd.gas_ct
    g.TPY = 300, 1e5, 'O2:1' # initialisation de la variable ct. 
    # Observation: l'objet ct garde en mémoire certaines données, ce
    # qui rend irreproductible des simulations où l'objet ct a un historique
    # différent. En faisant cette pseudo-declaration, on s'assure que
    # l'histoirique sur 1 declaration est le même. (voir étude cantera historique)

    velocity_x = Q[gd.i_vx] / Q[gd.i_d]
    velocity_y = Q[gd.i_vy] / Q[gd.i_d]


    u = Q[gd.i_e] / Q[gd.i_d] - 0.5 * (velocity_x**2 + velocity_y**2)
    v = alpha / Q[gd.i_d]
    g.UVY = u, v, ct_composition_input
    g.equilibrate('UV')

    Y_temp=deepcopy(g.Y)
    MDC_temp = deepcopy(g.mix_diff_coeffs)

    pressure=deepcopy(g.P)
    temperature=deepcopy(g.T)

    lambda_f = deepcopy(g.thermal_conductivity)
    viscosity_kin = deepcopy(g.viscosity) * v # viscosité cinématique = viscosité dynamique * v
    cp = deepcopy(g.cp)

    nb_species = len(name_species)
    for i in range (nb_species):
        Q[gd.i_Y+i] = Q[gd.i_d] * Y_temp[g.species_index(name_species[i])]
        # MDC[i] = MDC_temp[i]
        MDC[i] = MDC_temp[gd.gas_ct.species_index(name_species[i])]

    return pressure, temperature, lambda_f, Q, MDC, viscosity_kin, cp


#---------------------------------------------------------------------------------------------------------------------------------
def equilibrium(Q, Y_tilde, alpha, name_species, len_x):
    # ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"]

    Y_temp = np.zeros(Y_tilde.shape)
    MDC_temp = np.zeros(Q[:,gd.i_Y:].shape)
    pressure = np.zeros(len_x)
    temperature = np.zeros(len_x)
    lambda_f = np.zeros(len_x)
    viscosity_kin = np.zeros(len_x)
    cp = np.zeros(len_x)
    MDC = np.zeros(Q[:,gd.i_Y:].shape)

    for i_x in range(len_x):
        ct_composition_input=("O2:"     + str(Y_tilde[i_x, gd.igs["O2"]])
                            +",Al:"     + str(Y_tilde[i_x, gd.igs["Al"]])
                            +",O:"      + str(Y_tilde[i_x, gd.igs["O"]])
                            +",N2:"     + str(Y_tilde[i_x, gd.igs["N2"]])
                            +",Al2O:"   + str(Y_tilde[i_x, gd.igs["Al2O"]])
                            +",AlO:"    + str(Y_tilde[i_x, gd.igs["AlO"]])
                            +",AlO2:"   + str(Y_tilde[i_x, gd.igs["AlO2"]])
                            +",Al2O2:"  + str(Y_tilde[i_x, gd.igs["Al2O2"]])
                            +",Cu:"     + str(Y_tilde[i_x, gd.igs["Me"]])
                            )

        g = gd.gas_ct
        g.TPY = 300, 1e5, 'O2:1' # initialisation de la variable ct. 
        # Observation: l'objet ct garde en mémoire certaines données, ce
        # qui rend irreproductible des simulations où l'objet ct a un historique
        # différent. En faisant cette pseudo-declaration, on s'assure que
        # l'histoirique sur 1 declaration est le même. (voir étude cantera historique)

        velocity_x = Q[:,gd.i_vx] / Q[:,gd.i_d]
        velocity_y = Q[:,gd.i_vy] / Q[:,gd.i_d]


        u = Q[i_x, gd.i_e] / Q[i_x, gd.i_d] - 0.5 * (velocity_x[i_x]**2 + velocity_y[i_x]**2)
        v = alpha[i_x] / Q[i_x, gd.i_d]
        g.UVY = u, v, ct_composition_input
        g.equilibrate('UV')

        Y_temp[i_x, :]=deepcopy(g.Y)
        MDC_temp[i_x, :] = deepcopy(g.mix_diff_coeffs)

        pressure[i_x]=deepcopy(g.P)
        temperature[i_x]=deepcopy(g.T)

        lambda_f[i_x] = deepcopy(g.thermal_conductivity)
        viscosity_kin[i_x] = deepcopy(g.viscosity) * v # viscosité cinématique = viscosité dynamique * v
        cp[i_x] = deepcopy(g.cp)

    nb_species = len(name_species)
    for i in range (nb_species):
        Q[:, gd.i_Y+i] = Q[:, gd.i_d] * Y_temp[:,g.species_index(name_species[i])]
        # MDC[:, i] = MDC_temp[: ,i]
        MDC[:, i] = MDC_temp[:, gd.gas_ct.species_index(name_species[i])]

    return pressure, temperature, lambda_f, Q, MDC, viscosity_kin, cp


#---------------------------------------------------------------------------------------------------------------------------------
def integrate(f_t0, f_t1, dt):
    return np.trapz([f_t0, f_t1], dx=dt, axis = 0)

#---------------------------------------------------------------------------------------------------------------------------------
# def integrate_axed(f_t0, f_t1, dt):
#     return np.trapz([f_t0, f_t1], dx=dt, axis = 1)

#---------------------------------------------------------------------------------------------------------------------------------
def flux_species_g_p_diff(gas,particle,species_index, *args): # gas index
    Diff=gas.MDC[:, species_index] #Diffusion coeff from ct
    Sh=gas.Sh[:,species_index]

    rho_f=gas.density
    d_p=2*particle.r_ext
    if species_index==0:
        Y_at_s=np.zeros(len(gas.temperature)) #dans le cas de l'oxygène, comparer avec flux dans la particule
    else:
        Y_at_s=np.zeros(len(gas.temperature)) #sous oyxde, concentration nulle à la surface
    if args:
        Y_at_s = args[0]
    Y_g=gas.Y[:,species_index]

    Flux=np.where((Y_g>gd.eps5) | (Y_at_s>Y_g),-np.pi*d_p*rho_f*Diff*Sh*np.log((1-Y_g)/(1-Y_at_s)),0)

    if species_index not in [gd.igs["Al"],gd.igs["Me"]]: # Si on regarde des espèces qui peuvent uniquement aller du gaz vers la particule (AlxOy)
                                                        # on verifie que cette espèce est suffisament présente dans le gaz pour ne pas avoir
                                                        # une concentration négative
        Flux = np.where(Y_g > gd.eps5,                                             # Si la fraction de de AlxOy dans le gaz supérieure à gd.eps5
                     -np.pi*d_p*rho_f*Diff*Sh*np.log((1-Y_g)/(1-Y_at_s)),       #   Alors le flux est non null
                     0)                                                         #   Sinon le flux est nul


    else :                         # Si on a une espèce qui peut aller du gaz vers la particule ou de la particule vers le gaz
                                   # Il faut vérifier que cette espèce est suffisament présente dans sa phase de provenance pour
                                   # ne pas se retrouver avec une concentration négative de cette espèce dans sa phase de provenance
        # species_index_g_p = [None, 0, None, None, None, None, None, None, 2]
        Flux = np.where(((Y_at_s < Y_g) & (Y_g > gd.eps5))                                         # Si condensation, et que la fraction massique dans le gaz > gd.eps5
                    |((particle.Y[:, particle.species_index_g_p[species_index]]>gd.eps5) & (Y_at_s>Y_g)),   # Ou Si vaporisation et la fraction massique dans la particule > gd.eps5
                        -np.pi*d_p*rho_f*Diff*Sh*np.log((1-Y_g)/(1-Y_at_s)),                    #   Alors le flux est non null
                        0)                                                                      #   Sinon le flux est nul

    return Flux


#---------------------------------------------------------------------------------------------------------------------------------
def func_Y_s(Y_s, Ap, Ag, Bp, Bg, d, phi_Ox_condens):
    return (Bg**Ag*Bp**Ap)-(1 - Y_s)**Ag*np.exp(phi_Ox_condens)*(1-1/d * Y_s)**Ap


#---------------------------------------------------------------------------------------------------------------------------------
def analytical_Y_s(Ap, Ag, Bp, Bg,phi_Ox_condens):
    return (1 - Bp**(Ap/(Ap+Ag))
               *Bg**(Ag/(Ap+Ag))
               *np.exp(-phi_Ox_condens/(Ap+Ag)))


#---------------------------------------------------------------------------------------------------------------------------------
# @jit
# def dT_dt_computation_g(Q = None, Y = None, dYg_dt = None, dQ_dt = None, T = None):



#     # chron1 = time.time()
#     uh_i, Cv_i = enth.uCv_i_g_vec2(T)
#     # print(time.time()-chron1)

#     uh = np.sum(Y * uh_i, axis = 1)


#     d_rho_alpha = dQ_dt[gd.i_d]
#     d_rho_alpha_U_x = dQ_dt[gd.i_vx]
#     d_rho_alpha_U_y = dQ_dt[gd.i_vy]
#     d_rho_alpha_etot = dQ_dt[gd.i_e]

#     # grandeurs non actualisée : 
#     velocity_x = Q[gd.i_vx] / Q[gd.i_d]
#     velocity_y = Q[gd.i_vy] / Q[gd.i_d]
#     energy_tot = Q[gd.i_e] / Q[gd.i_d]
#     rho_alpha = Q[gd.i_d]

#     d_velocity_x = (d_rho_alpha_U_x - velocity_x * d_rho_alpha) / rho_alpha
#     d_velocity_y = (d_rho_alpha_U_y - velocity_y * d_rho_alpha) / rho_alpha
#     duh_dt = (d_rho_alpha_etot - energy_tot * d_rho_alpha) / rho_alpha - velocity_x * d_velocity_x - velocity_y * d_velocity_y

#     duh_dt = (d_rho_alpha_etot + 0.5 * d_rho_alpha * velocity_x**2 - velocity_x * d_rho_alpha_U_x - uh * d_rho_alpha) / rho_alpha

#     duh_dt = np.where(np.isnan(duh_dt), 0, duh_dt)

#     """SUPPRIMER LA VARIATION D ENERGIE CINETIQUE en suivant les règles de dérivation !!!"""
#     dYi_dt = dYg_dt

#     duhi_dT = Cv_i

#     dT_dt = (duh_dt - np.sum(dYi_dt * uh_i, axis = 1)) / np.sum(duhi_dT * Y, axis = 1) #celui a utiliser si on se retoruve avec T° négatives

#     return dT_dt[0]


#---------------------------------------------------------------------------------------------------------------------------------
@jit
def dT_dt_computation_g_vec(wdot_kin = 0, Q = None, Y = None, dYg_dt = None, dQ_dt = None, T = None):

    uh_i, Cv_i = enth_nb.uCv_i_g_vec2(T)

    uh = np.sum(Y * uh_i, axis = 1)

    d_rho_alpha = dQ_dt[:,gd.i_d]
    d_rho_alpha_U_x = dQ_dt[:,gd.i_vx]
    d_rho_alpha_U_y = dQ_dt[:,gd.i_vy]
    d_rho_alpha_etot = dQ_dt[:,gd.i_e]

    # grandeurs non actualisée : 
    velocity_x = Q[:,gd.i_vx] / Q[:,gd.i_d]
    velocity_y = Q[:,gd.i_vy] / Q[:,gd.i_d]
    energy_tot = Q[:,gd.i_e] / Q[:,gd.i_d]
    rho_alpha = Q[:,gd.i_d]

    # d_velocity_x = (d_rho_alpha_U_x - velocity_x * d_rho_alpha) / rho_alpha
    # d_velocity_y = (d_rho_alpha_U_y - velocity_y * d_rho_alpha) / rho_alpha
    # duh_dt = (d_rho_alpha_etot - energy_tot * d_rho_alpha) / rho_alpha - velocity_x * d_velocity_x - velocity_y * d_velocity_y

    duh_dt = (d_rho_alpha_etot + 0.5 * d_rho_alpha * velocity_x**2 - velocity_x * d_rho_alpha_U_x - uh * d_rho_alpha) / rho_alpha

    duh_dt = np.where(np.isnan(duh_dt), 0, duh_dt)

    """SUPPRIMER LA VARIATION D ENERGIE CINETIQUE en suivant les règles de dérivation !!!"""
    dYi_dt = dYg_dt

    duhi_dT = Cv_i

    dT_dt = (duh_dt - np.sum(dYi_dt * uh_i, axis = 1)) / np.sum(duhi_dT * Y, axis = 1) #celui a utiliser si on se retoruve avec T° négatives

    return dT_dt

#---------------------------------------------------------------------------------------------------------------------------------
@njit
def dT_dt_computation_g(Q = None, Y = None, dYg_dt = None, dQ_dt = None, T = None):
    Tarr = np.array([T])
    uh_i_tmp, Cv_i_tmp = enth_nb.uCv_i_g_vec2(Tarr)
    uh_i, Cv_i = uh_i_tmp[0], Cv_i_tmp[0]

    uh = np.sum(Y * uh_i)

    d_rho_alpha = dQ_dt[gd.i_d]
    d_rho_alpha_U_x = dQ_dt[gd.i_vx]
    d_rho_alpha_U_y = dQ_dt[gd.i_vy]
    d_rho_alpha_etot = dQ_dt[gd.i_e]

    # grandeurs non actualisée : 
    velocity_x = Q[gd.i_vx] / Q[gd.i_d]
    velocity_y = Q[gd.i_vy] / Q[gd.i_d]
    energy_tot = Q[gd.i_e] / Q[gd.i_d]
    rho_alpha = Q[gd.i_d]

    # d_velocity_x = (d_rho_alpha_U_x - velocity_x * d_rho_alpha) / rho_alpha
    # d_velocity_y = (d_rho_alpha_U_y - velocity_y * d_rho_alpha) / rho_alpha
    # duh_dt = (d_rho_alpha_etot - energy_tot * d_rho_alpha) / rho_alpha - velocity_x * d_velocity_x - velocity_y * d_velocity_y

    duh_dt = (d_rho_alpha_etot + 0.5 * d_rho_alpha * velocity_x**2 - velocity_x * d_rho_alpha_U_x - uh * d_rho_alpha) / rho_alpha

    duh_dt = np.where(np.isnan(duh_dt), 0, duh_dt)

    """SUPPRIMER LA VARIATION D ENERGIE CINETIQUE en suivant les règles de dérivation !!!"""
    dYi_dt = dYg_dt

    duhi_dT = Cv_i

    dT_dt = (duh_dt - np.sum(dYi_dt * uh_i)) / np.sum(duhi_dT * Y) #celui a utiliser si on se retoruve avec T° négatives

    return dT_dt


#---------------------------------------------------------------------------------------------------------------------------------
def dT_dt_computation_p(p, system):
    uh_i, Cp_i = p.uh_i, p.Cp_i

    # grandeurs calculée sur cette itération
    d_rho_alpha = p.d_dt_rho_alpha
    d_rho_alpha_etot = p.d_dt_rho_alpha_etot

    # grandeurs non actualisée : 
    rho_alpha = p.Q[:,gd.i_d]
    rho_alpha_U_x = p.Q[:,gd.i_vx]
    rho_alpha_U_y = p.Q[:,gd.i_vy]
    rho_alpha_Y = p.Q[:,gd.i_Y:]
    energy_tot = p.Q[:,gd.i_e] / p.Q[:,gd.i_d]

    duh_dt = (d_rho_alpha_etot - energy_tot * d_rho_alpha) / rho_alpha
    duh_dt = np.where(np.isnan(duh_dt), 0, duh_dt)

    dYi_dt = p.d_dt_Y

    duhi_dT = Cp_i

    dT_dt = (duh_dt - np.sum(dYi_dt * uh_i, axis = 1)) / np.sum(duhi_dT * p.Y, axis = 1) #celui a utiliser si on se retoruve avec T° négatives

    return dT_dt


#---------------------------------------------------------------------------------------------------------------------------------
def compute_vel_advection(density, dp, eps, mu, L, DELTA_P):
    # ergun equation
    # with singularity drop coefficient: zeta = 1
    zeta = 1
    # A = ( 1.8 * density / dp * (1-eps) / eps**3 + zeta/2 )
    A = ( 1.8 * 1 / dp * (1-eps) / eps**3 + zeta/2/L ) * density
    B = 151.2 * mu / dp**2 * (1 - eps)**2 /  eps**3
    C = - DELTA_P / L
    DELTA = B**2 - 4 * A * C
    velocity = (-B + np.sqrt(DELTA)) / 2/A
    return velocity


#---------------------------------------------------------------------------------------------------------------------------------
def Nb_part(alpha_p, pAl_richness, Y_Al_pAl, volume, m_pAl, m_pMeO_core, r_ext_pAl, r_ext_pMeO):
    '''
    This function calculates the number of Aluminum and Metal Oxide particles
    as a function of the %TMD, the volume, aluminum particle richness and aluminum purity.

    Attributes
    ----------
    alpha_p: float
        %TMD

    pAl_richness: float
        richness of the aluminium

    Y_Al_pAl: float
        aluminium purity

    volume: float

    m_pAl: float
        mass of an aluminium particle

    m_pMeO_core: float
        mass of a metal oxyde particle

    r_ext_pAl: float
        extorior radius of aluminium particle

    r_ext_pMeO: float
        extorior radius of metal oxyde particle

    Returns
    -------
    n_pAl: integer
        number of aluminium particles

    n_pMeO: integer
        number of metal oxyde particles

    alpha_p: float
        %TMD
    '''
    Stoech_mol = 2/3
    MW_Al = gd.MW_dic["Al"]
    MW_CuO = gd.MW_dic["CuO"]
    m_pMeO = m_pMeO_core
    Stoech_mass = Stoech_mol * MW_Al / MW_CuO / Y_Al_pAl
    n_pAl = (alpha_p*(pAl_richness*Stoech_mass*volume*m_pMeO)/(m_pAl*V_sphere(r_ext_pMeO)))/(1+pAl_richness*m_pMeO*Stoech_mass*V_sphere(r_ext_pAl)/(m_pAl*V_sphere(r_ext_pMeO)))
    n_pMeO = (alpha_p*volume - V_sphere(r_ext_pAl)*n_pAl)/V_sphere(r_ext_pMeO)

    return n_pAl, n_pMeO, alpha_p


#---------------------------------------------------------------------------------------------------------------------------------
def poly2_root(A,B,C, bool_return_DELTA = False):
    DELTA = B**2 - 4 * A * C
    if bool_return_DELTA:
        return np.array([(-B - (DELTA)**0.5)/2/A , (-B + (DELTA)**0.5)/2/A, DELTA])
    else:
        return np.array([(-B - (DELTA)**0.5)/2/A , (-B + (DELTA)**0.5)/2/A])


def Cd_compute(gas):
    # https://hal.archives-ouvertes.fr/hal-01706862/document
    Cd_WY = np.where(gas.Re_p < 1000,                                               # Si Rep < 0
                    24/(gas.Re_p + gd.eps) * (1 + 0.15 * gas.Re_p**0.687) * gas.alpha**(-1.7),     # Cd_Wy = ...
                    0.44 * gas.alpha**(-1.7))                                   # Sinon Cd_Wy = ...

    Cd_Er = 200 * (1 - gas.alpha) / (gas.Re_p + gd.eps) + 7/3
    # Cd_Er = 200 * (1 - gas.alpha) / gas.Re_p / gas.alpha + 7/3 / gas.alpha

    Cd = np.where((Cd_WY < Cd_Er) | (gas.alpha > 0.7),                                                # Prendre le min entre Er et WY
                  Cd_WY,
                  Cd_Er)
    Cd = Cd_Er * (gas.bool_ergun_forced) + Cd * (not gas.bool_ergun_forced)
    Cd = np.where(gas.Re_p==0, 0, Cd)
    # Cd = np.where(np.isnan(Cd), 0, Cd)
    return Cd


#---------------------------------------------------------------------------------------------------------------------------------
def compute_conduc_pp(p_Al, p_MeO, gas, particle_name):
    N_coord_th_mono = 6
    N_coord_points_ij = N_coord_th_mono/2
    min_alpha_p = np.where(p_Al.alpha<p_MeO.alpha,
                                p_Al.alpha,
                                p_MeO.alpha)
    alpha_s = (1 - gas.alpha)

    """
     -  min_alpha_p / alpha_s : le nombre de points de coordination diminue lorsqu'une
        phase particulaire devient majoritaire par rapport à la phase solide,
        le maximum de cette grandeur étant 0.5, le nombre de points de contact est
        alors de 6 (théorie pour systeme equi-bidisperse)
     -  alpha_s / alpha_s_max : Lorsque le alpha_s -> 0, le nombre de points de contact
        interphase inter-particules -> 0. D'ailleurs il tend très vite, on pourrait
        rendre cette fonction bien plus rapide, par exemple proportionnelle à Ps.
    """
    alpha_s_max = 0.64
    # N_coord_points_ij = N_coord_th_mono * min_alpha_p / (alpha_s + gd.eps) * (alpha_s / alpha_s_max)
    # if particle_name == "p_Al":
    N_coord_points_ij = N_coord_th_mono * (p_MeO.n_vol / (p_MeO.n_vol + p_Al.n_vol)) * (p_Al.pressure / (1 + p_Al.pressure))
    # N_coord_points_ij = N_coord_th_mono * (p_Al.alpha + p_MeO.alpha) * alpha_s_max
    conduc_per_coord_point = compute_conduc_per_coord_point(p_Al, p_MeO, gas)
    H_cond = p_Al.n_vol * N_coord_points_ij * conduc_per_coord_point
    Q_conduc = H_cond * (p_Al.temperature - p_MeO.temperature)
    return Q_conduc, H_cond


#---------------------------------------------------------------------------------------------------------------------------------
def compute_conduc_per_coord_point(p_Al, p_MeO, gas, overlap = True):
    lambda_species = [233, 33, 401, 30]                                             # conductivité thermique des espèces à T° ambiente
    # i : p_Al
    # j : p_MeO
    r_i = p_Al.r_ext # rayon de la particule i
    lambda_i = p_Al.lambda_p
    # nu_i = # coefficient de poisson de la particule i
    # E_i = # module de Young de la particule i

    r_j = p_MeO.r_ext # rayon de la particule j
    lambda_j = p_MeO.lambda_p
    # nu_j = # coefficient de poisson de la particule j
    # E_j = # module de Young de la particule j

    r_c = compute_r_c(p_Al, p_MeO)
    lambda_g = gas.lambda_f

    lambda_s = lambda_i * lambda_j / (lambda_i + lambda_j) # hypothèse sur le calcul de lambda_s equivalent
    alpha = lambda_s / lambda_g
    R_ij = 2 * r_i * r_j / (r_i + r_j)                      # Rayon équivalent
    KHI = 0.71                                              # Rapport du rayon du cylindre à travers lequel passe le flux sur le rayon équivalent (0.71: Moscardini et Al, 2018)
    R_e_ij = KHI * R_ij
    Beta_ij = alpha * r_c / R_ij
    if ((Beta_ij<0) | np.isnan(Beta_ij)).any():
        print("error on Beta_ij")
        exit()
    C_ct_ij_s = compute_C_ct_s(Beta_ij, R_ij, R_e_ij, lambda_g, alpha, overlap)# Conductance inter-particule par contact ou gap
    C_i_s = np.pi * lambda_i * R_e_ij**2 / r_i              # Conductance dans la particule i (hypothese sur lambda_i != lambda_j)
    C_j_s = np.pi * lambda_j * R_e_ij**2 / r_j              # Conductance dans la particule j (hypothese sur lambda_i != lambda_j)
    C_ij = 1/ (1/C_i_s + 1/C_ct_ij_s + 1/C_j_s)             # Conductance équivalente
    if np.isnan(C_ij).any():
        print("C_ij p nan")
        exit()
    return C_ij


#---------------------------------------------------------------------------------------------------------------------------------
def compute_r_c(p_Al, p_MeO):
    r_i = p_Al.r_ext
    r_j = p_MeO.r_ext
    # E_star = ((1-nu_i**2) / E_i + (1-nu_j**2) / E_j)**(-1) # module d'élasticité effectif (wikipedia: Mécanique des contacts)
    # r_star = (r_i * r_j)**0.5 # moyenne géométrique des rayons (theorie Hertz) (Watson L. Vargas, J.J. McCarthy, 2002)
    # F_n = "???"
    # r_c = 2 * (3 * F_n * r_star / 4 / E_star)**(1/3)
    # r_c = 2 * np.sqrt(np.abs(h_ij) * R_ij / 2) # Moscardini, 2018
    Kc = 0.01
    r_c = Kc * np.min([r_i, r_j], axis = 0)

    return r_c


#---------------------------------------------------------------------------------------------------------------------------------
# def compute_C_ct_s(Beta_ij, R_ij, R_e_ij, lambda_g, alpha, overlap):
#     h_ij = np.nan
#     if overlap :
#         if Beta_ij > 100 :
#             C_ct = Beta_ij_sup_100(Beta_ij, R_ij, lambda_g, alpha)
#         elif Beta_ij < 1 :
#             C_ct = Beta_ij_inf_1(Beta_ij, R_ij, lambda_g, alpha)
#         else:                               # Sinon, on fait l'interpolation entre les deux valeurs: Moscardini, 2018
#             C_ct = np.interp(Beta_ij,[1,100],[Beta_ij_inf_1(1, R_ij, lambda_g, alpha), Beta_ij_sup_100(100, R_ij, lambda_g, alpha)])
#     else:
#         ksi = alpha**2 * h_ij / R_ij
#         if ksi < 0.1:
#             C_ct = np.pi * lambda_g * R_ij * np.log(alpha)
#         else:
#             C_ct = np.pi * lambda_g * R_ij * np.log(1 + R_e_ij**2/R_ij/h_ij)
#     return C_ct

def compute_C_ct_s(Beta_ij, R_ij, R_e_ij, lambda_g, alpha, overlap):
    h_ij = np.nan
    C_ct = np.where(overlap,
                np.where(Beta_ij > 100,
                    Beta_ij_sup_100(Beta_ij, R_ij, lambda_g, alpha),
                    np.where(Beta_ij < 1,
                        Beta_ij_inf_1(Beta_ij, R_ij, lambda_g, alpha),
                        # np.interp(Beta_ij,[1,100],[Beta_ij_inf_1(1, R_ij, lambda_g, alpha), Beta_ij_sup_100(100, R_ij, lambda_g, alpha)]))),
                        interp_multi_dim(Beta_ij,Beta_ij_inf_1, R_ij, lambda_g, alpha))),
                np.where((alpha**2 * h_ij / R_ij)< 0.1,
                    np.pi * lambda_g * R_ij * np.log(alpha),
                    np.pi * lambda_g * R_ij * np.log(1 + R_e_ij**2/R_ij/h_ij))
                    )
    if np.isnan(C_ct).any():
        print("C_ct p nan")
        exit()
    return C_ct


#---------------------------------------------------------------------------------------------------------------------------------
def interp_multi_dim(Beta_ij,Beta_ij_inf_1, R_ij, lambda_g, alpha):

    return [np.interp(Beta_ij[cpt],[1,100],[Beta_ij_inf_1(1, R_ij[cpt], lambda_g[cpt], alpha[cpt]), Beta_ij_sup_100(100, R_ij[cpt], lambda_g[cpt], alpha[cpt])])
    for cpt in range(len(Beta_ij))]


#---------------------------------------------------------------------------------------------------------------------------------
def Beta_ij_inf_1(Beta_ij, R_ij, lambda_g, alpha):
    return np.pi * lambda_g * R_ij *(
            2*Beta_ij/np.pi + 2 * np.log(Beta_ij) + np.log(alpha**2)
        )


#---------------------------------------------------------------------------------------------------------------------------------
def Beta_ij_sup_100(Beta_ij, R_ij, lambda_g, alpha):
    return np.pi * lambda_g * R_ij *(
            0.22*Beta_ij**2 - 0.05 * Beta_ij**2 + np.log(alpha**2)
        )


#---------------------------------------------------------------------------------------------------------------------------------
# def ergun(L, d_p, alpha_g, rho_g, visc_dyn, u):
#     DELTA_P = (150*visc_dyn / d_p**2 * (1-alpha_g)**2 / alpha_g**3 * u
#               + 1.75 * rho_g / d_p * (1 - alpha_g) / alpha_g**3 * u**2) * L
#     return DELTA_P


#---------------------------------------------------------------------------------------------------------------------------------
# def compute_lambda_bed(k_c, alpha_g):
#     return 1 - np.sqrt(1 - alpha_g) + np.sqrt(1 - alpha_g) * k_c


#---------------------------------------------------------------------------------------------------------------------------------
def compute_k_c(alpha_g, lambda_p, lambda_f):
    B = (compute_B_conduc(alpha_g) + gd.eps)
    k_p = lambda_p / lambda_f

    N = 1 - B/k_p

    K1 = + (B / N**2 * (k_p - 1)/k_p * np.log(k_p/B))

    K2 = - (B+1) / 2

    K3 = - (B-1) / N

    k_c = 2 / N * (K1 + K2 + K3)

    return k_c


def compute_B_conduc(alpha_g, Cf = 1.25):
    B = Cf * ((1 - alpha_g) / alpha_g) ** (10/9)
    return B


def compute_tau_th_0D_g(p_Al, p_MeO, gas, system):
    tau_th_1 = (gas.alpha * gas.density * gas.Cv) / (gas.lambda_f * np.pi * (p_Al.r_ext + p_MeO.r_ext) * gas.Nu * (p_Al.n_vol + p_MeO.n_vol)) * system.volume.volume

    T_g_eq = compute_T_g_eq_th(p_Al, p_MeO, gas)
    rho = gas.density
    Cv = gas.Cv
    lambda_g = gas.lambda_f * 10
    p = p_Al
    q = p_MeO
    tau_th_2 = np.ones(system.n_volumes)
    num = gas.alpha * rho * Cv * (T_g_eq - gas.temperature + gd.eps12)
    # num = gas.alpha * rho * Cv * (T_g_eq - gas.temperature)
    denum = lambda_g * np.pi * gas.Nu * (p.r_ext*2 * (p.temperature - gas.temperature + gd.eps12) * p.n_vol
                                       + q.r_ext*2 * (q.temperature - gas.temperature + gd.eps12) * q.n_vol)
                                    #    + gd.eps)
    tau_th_2[gd.N_gc:-gd.N_gc] = (num[gd.N_gc:-gd.N_gc]/denum[gd.N_gc:-gd.N_gc])# * system.volume.volume[gd.N_gc:-gd.N_gc]
    tau_th_2 = np.where(T_g_eq - gas.temperature == 0, 1, tau_th_2)
    # print("T7",min(tau_th_2))
    if (tau_th_2<0).any():
        system.error_quit("tau_th_gas_0D<0")
    return tau_th_2


def compute_tau_th_0D_p(p, q, gas, H_cond_pq, system):
    T_p_eq, H_g, H_q = compute_T_p_eq_th(p, q, gas, H_cond_pq)

    DELTA_T_gp = gas.temperature - p.temperature
    DELTA_T_qp = q.temperature - p.temperature
    DELTA_T_eq = T_p_eq - p.temperature

    C1 = np.abs(DELTA_T_gp)<gd.eps5
    C2 = np.abs(DELTA_T_qp)<gd.eps5
    C3 = np.abs(DELTA_T_eq)<gd.eps3
    C  = C1 & C2 | C3

    DELTA_T_gp = np.where(C1, 0, DELTA_T_gp)
    DELTA_T_qp = np.where(C2, 0, DELTA_T_qp)

    num = p.alpha * p.density * p.Cp * (DELTA_T_eq)
    denum = (H_g * DELTA_T_gp
            +H_q * DELTA_T_qp + gd.eps)

    tau_th = np.ones(system.n_volumes)
    tau_th[gd.N_gc:-gd.N_gc] = num[gd.N_gc:-gd.N_gc] / denum[gd.N_gc:-gd.N_gc]# * system.volume.volume[gd.N_gc:-gd.N_gc]
    # tau_th = np.where(T_p_eq - p.temperature == 0, 1, tau_th)
    tau_th = np.where(C, 1, tau_th)

    return tau_th


def compute_T_g_eq_th(p_Al, p_MeO, gas):
    p = p_Al
    q = p_MeO
    A = - (q.r_ext/(p.r_ext + gd.eps)
          *q.n_vol/(p.n_vol + gd.eps))
    T_p = p.temperature
    T_q = q.temperature
    T_g_eq = (T_p - A * T_q)/(1 - A)
    return T_g_eq


def compute_T_p_eq_th(p, q, gas, H_cond_pq):
    H_g = gas.lambda_f * np.pi * p.r_ext*2 * gas.Nu * p.n_vol
    H_q = H_cond_pq
    norm = np.max([H_g, H_q])
    # T_p_eq = (H_g / norm * gas.temperature + H_q / norm * q.temperature) / (H_g / norm + H_q / norm + gd.eps)
    T_p_eq = (H_g / norm * gas.temperature + H_q / norm * q.temperature) / (H_g / norm + H_q / norm)
    """ reduction erreur numérique en normalisant H par norm"""
    return T_p_eq, H_g, H_q


def compute_vel_input(gas, system, rho_L, alpha_g_L, vel_x_L):
    DELTA_alphaP = (gas.pressure[3] * gas.alpha[3] - gas.pressure[2] * gas.alpha[2])
    A = 1 / (rho_L * alpha_g_L * vel_x_L)
    return gas.velocity_x[2] - DELTA_alphaP * A


# def compute_coll_pressure_granular(system):
#     p_Al, p_MeO = system.p_Al, system.p_MeO
#     alpha_s = p_Al.alpha + p_MeO.alpha
#     alpha_s_max = system.inputs_dic["alpha_s_max"]

#     # https://doi.org/10.1002/aic.690360404
#     # eq 23:
#     g0 = 1 / (1 - (alpha_s / alpha_s_max)**(1/3))
#     rho_s = p_Al.density + p_MeO.density
#     T_s   =(p_Al.density + p_MeO.density) / 2
#     e = 0.8 # coefficient de restitution - estimation random
#     # eq 21:
#     P_s = alpha_s * rho_s * T_s * (
#                                     1 + 2 * (1 + e) * alpha_s * g0
#                                   )
#     return P_s




def get_M_matrix(K1, K2, K3, alpha_s, dt):
    len_diag = len(alpha_s)

    vec_sub_diag = K1 * np.ones(len_diag - 1)
    vec_diag     =-K2 * np.ones(len_diag)
    vec_sup_diag = K3 * np.ones(len_diag - 1)

    k = [vec_sub_diag, vec_diag, vec_sup_diag]
    offset = [-1,0,1]
    K_mat = diags(k,offset).toarray()
    I = np.identity(len_diag)
    M = I - K_mat * dt**2

    # M[:gd.N_gc], M[-gd.N_gc:], M[:,:gd.N_gc], M[:,-gd.N_gc:] = 0,0,0,0
    return M


def get_delta_alpha_P_comp(particle, alpha_s, system):
    grad_P_g_x = system.int_func_g.Grad_centered(system.gas.pressure, system.volume.x)
    # delta_velx = system.dt / particle.Q[:, gd.i_d] * (- particle.Q[:, gd.i_d] * system.inputs_dic["acc_field"] - particle.alpha * grad_P_g_x)
    delta_velx = 0

    div_rho_alpha_delta_vel = system.int_func_g.Grad_centered(particle.density * particle.alpha * delta_velx, system.volume.x)
    delta_alpha_p = system.dt /particle.density * div_rho_alpha_delta_vel

    dPp_dalpha_p = get_dPp_dalpha_s(alpha_s)
    Pp = compute_fric_pressure_granular_second(alpha_s)
    delta_alpha_P = (Pp + particle.alpha * dPp_dalpha_p) * delta_alpha_p
    return delta_alpha_P




def implicit_diff_solver(X_old, C_diff, DELTA_S, DELTA_X, DELTA_t,DELTA_V, grad_0_L=False, grad_0_R=False, C_diff_right=None, B_given = None):
    """solve the equation dX/dt = d/dx (CdX/dx)"""
    len_x = len(X_old)

    M = build_M_mat_diff(C_diff, DELTA_S, DELTA_X, DELTA_t, DELTA_V, len_x, grad_0_L, grad_0_R)
    if C_diff_right is not None:
        M_prime = build_M_mat_diff(C_diff_right, DELTA_S, DELTA_X, DELTA_t, len_x, grad_0_L, grad_0_R)
    else:
        M_prime = np.identity(len_x)
    if B_given is None:
        B_given = np.zeros(X_old.shape)


    X_new = deepcopy(X_old)
    """ résolution du système: My = Ax + B   ==> y = M^(-1) * (Ax+B)"""
    A = M_prime
    B = B_given
    X_new = np.matmul(inv(M), B + np.matmul(M_prime, X_old))
    # X_new[gd.N_gc:-gd.N_gc] = np.matmul(inv(M[gd.N_gc:-gd.N_gc, gd.N_gc:-gd.N_gc]), np.matmul(M_prime[gd.N_gc:-gd.N_gc, gd.N_gc:-gd.N_gc], X_old[gd.N_gc:-gd.N_gc]))
    return X_new



def build_M_mat_diff(C_diff, DELTA_S, DELTA_X, DELTA_t, DELTA_V, len_diag, grad_0_L, grad_0_R):
    C_diff_L_avg = (C_diff[1:-1] + C_diff[:-2]) / 2
    C_diff_R_avg = (C_diff[1:-1] + C_diff[2:]) / 2
    C_diff_L_avg = np.insert(C_diff_L_avg, len(C_diff_L_avg), [None])
    C_diff_R_avg = np.insert(C_diff_R_avg, 0, [None])

    K1 = C_diff_L_avg * DELTA_S[:-1] * DELTA_t / DELTA_X[:-1] / DELTA_V[:-1]
    K2 = C_diff_R_avg * DELTA_S[1:]  * DELTA_t / DELTA_X[1:] / DELTA_V[:-1]
    # K1bis = C_diff_L_avg * DELTA_t / (DELTA_X[:-1])**2
    # K2bis = C_diff_R_avg * DELTA_t / (DELTA_X[1:])**2
    # K1 = C_diff_L_avg * DELTA_t / (DELTA_X[:-1])**2
    # K2 = C_diff_R_avg * DELTA_t / (DELTA_X[1:])**2

    K1 = np.where(K1>1e12, 1e12, K1)
    K2 = np.where(K2>1e12, 1e12, K2)

    # print(K1)
    # K1 = np.ones(DELTA_S[1:].shape)
    # K2 = np.ones(DELTA_S[1:].shape)


    if grad_0_L==True:
        K1[gd.N_gc-1] = 0
    if grad_0_R==True:
        K2[-gd.N_gc] = 0


    K1 = np.insert(K1, 0, [None])
    K2 = np.insert(K2, len(K2), [None])
    K = 1 + K1 + K2
    K1 = np.delete(K1, 0)
    K2 = np.delete(K2, -1)


    vec_sub_diag = - K1 * np.ones(len_diag - 1)
    vec_diag     = K  * np.ones(len_diag)
    vec_sup_diag = - K2 * np.ones(len_diag - 1)

    k = [vec_sub_diag, vec_diag, vec_sup_diag]
    offset = [-1,0,1]
    M = diags(k,offset).toarray()

    BC_M = np.identity(gd.N_gc)
    M[:gd.N_gc,:gd.N_gc] = BC_M
    M[-gd.N_gc:,-gd.N_gc:] = BC_M
    M[gd.N_gc-1,gd.N_gc] = 0
    M[-gd.N_gc,-gd.N_gc-1] = 0
    # print('det=',np.linalg.det(M))
    # print(np.linalg.norm(M) * np.linalg.norm(inv(M)))
    # print("min indenty",np.min(np.matmul(M,inv(M))),"max indenty", np.max(np.matmul(M,inv(M))))
    # print(np.matmul(M,inv(M)))
    return M


def implicit_diff_solver_2(Q_d_new, Q_d_prev, DELTA_t, DELTA_S, DELTA_V):
    """solve the equation A = BX """
    DELTA_Q_d = Q_d_new - Q_d_prev
    B = DELTA_Q_d /DELTA_t

    M = build_M_mat_find_delta_u(Q_d_new)

    # M[0,:], M[-1,:] = 0,0 # suppression de la résolution de la premiere et derniere cellule
    if np.linalg.det(M)!=0:
        u = np.matmul(inv(M), B)
        return u
    else:
        return None

def build_M_mat_find_delta_u(Q_d):
    len_diag = len(Q_d)

    K1 = Q_d[:-1]
    K2 = Q_d[1:]

    # K1 = 1
    # K2 = 2


    # K1 = np.insert(K1, 0, [None])
    # K2 = np.insert(K2, len(K2), [None])
    # K1 = np.delete(K1, 0)
    # K2 = np.delete(K2, -1)


    vec_sub_diag = + K1 * np.ones(len_diag - 1)
    vec_diag     = np.zeros(len_diag)
    vec_sup_diag = - K2 * np.ones(len_diag - 1)

    k = [vec_sub_diag, vec_diag, vec_sup_diag]
    offset = [-1,0,1]
    M = diags(k,offset).toarray()
    # M[0,0] = - M[0,1]
    # M[-1,-1] = - M[-1,-2]

    return M


def implicit_diff_solver_3(size_M, build_M_mat, args=(), kwargs=dict(), X=None, A=None, B=None):
    # chron1 = time.time()
    M = build_M_mat(*args, **kwargs)
    # print("T0", time.time() - chron1)

    """ résolution du système: My = Ax + B   ==> y = M^(-1) * (Ax+B)"""
    if A is None:
        A = np.identity(size_M)
    if X is None:
        X = np.identity(size_M)
    if B is None:
        B = np.zeros(size_M)

    # chron1 = time.time()
    Y = np.matmul(inv(M), B + np.matmul(A, X))
    # print("T1", time.time() - chron1)

    # chron2 = time.time()
    # solve_tridiag_from_mat(size_M, M, B + np.matmul(A, X))
    # print("T2", time.time() - chron2)

    return Y


def build_M_mat_3(len_diag, Ps, density, dt, DELTA_X, flux_BL_0=False, flux_BR_0=False):
    flux_BL_0 = True
    flux_BR_0 = True

    K1 = - Ps[:-2]        / density[1:-1] * dt**2 / DELTA_X[1:-1]**2
    K2 = - Ps[2:]         / density[1:-1] * dt**2 / DELTA_X[1:-1]**2

    if flux_BL_0==True:
        K1[gd.N_gc-1] = 0
    if flux_BR_0==True:
        K2[-gd.N_gc] = 0


    K  = 1 + 2 * Ps[1:-1] / density[1:-1] * dt**2 / DELTA_X[1:-1]**2

    K1 = np.where(np.abs(K1)>1e12, K1/np.abs(K1) * 1e12, K1)
    K2 = np.where(np.abs(K2)>1e12, K2/np.abs(K2) * 1e12, K2)
    K  = np.where(np.abs(K )>1e12, K /np.abs(K ) * 1e12, K )


    K  = np.insert(K, [0], [0])
    K  = np.append(K, 0)
    K1 = np.append(K1, 0)
    K2 = np.insert(K2, 0, [0])



    vec_sub_diag = K1 * np.ones(len_diag - 1)
    vec_diag     = K  * np.ones(len_diag)
    vec_sup_diag = K2 * np.ones(len_diag - 1)

    k = [vec_sub_diag, vec_diag, vec_sup_diag]
    offset = [-1,0,1]
    M = diags(k,offset).toarray()

    BC_M = np.identity(gd.N_gc)
    M[:gd.N_gc,:gd.N_gc] = BC_M
    M[-gd.N_gc:,-gd.N_gc:] = BC_M
    M[gd.N_gc-1,gd.N_gc] = 0
    M[-gd.N_gc,-gd.N_gc-1] = 0
    # print('det=',np.linalg.det(M))
    # print(np.linalg.norm(M) * np.linalg.norm(inv(M)))
    # print("min indenty",np.min(np.matmul(M,inv(M))),"max indenty", np.max(np.matmul(M,inv(M))))
    # print(np.matmul(M,inv(M)))
    return M


# def build_M_mat_4(len_diag, Ps, density, dt, DELTA_X, flux_BL_0=False, flux_BR_0=False):
def build_M_mat_4(len_diag, decent_av = None, decent_not_av = None, cent = None, flux_BL_0=False, flux_BR_0=False):
    if decent_av is None:
        decent_av = np.ones(len_diag)
    if decent_not_av is None:
        decent_not_av = np.ones(len_diag)
    if cent is None:
        cent = np.ones(len_diag)

    """
    cent (A) : value of the cell
    decent_av (B): decentered values averaged on the cell face
    decent_not_av (C) : decentered values taking the other cell values (not conservative)
    cent dx decent_av dx decent_not_av (y) 

    A dx [B dx (C y)]
    """

    flux_null_L = np.ones(len_diag)
    flux_null_R = np.ones(len_diag)


    # flux_BL_0 = True
    # flux_BR_0 = True
    if flux_BL_0==True:
        flux_null_L[gd.N_gc] = 0
    if flux_BR_0==True:
        flux_null_R[-gd.N_gc-1] = 0


    K1 = 0 - cent[1:-1] * decent_not_av[:-2]  *  flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
    K2 = 0 - cent[1:-1] * decent_not_av[2:]   *  flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2
    K  = 1 + cent[1:-1] * decent_not_av[1:-1] * (flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
                                               + flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2)

    K1 = np.where(np.abs(K1)>1e12, K1/np.abs(K1) * 1e12, K1)
    K2 = np.where(np.abs(K2)>1e12, K2/np.abs(K2) * 1e12, K2)
    K  = np.where(np.abs(K )>1e12, K /np.abs(K ) * 1e12, K )

    K  = np.insert(K, [0], [0])
    K  = np.append(K, 0)
    K1 = np.append(K1, 0)
    K2 = np.insert(K2, 0, [0])

    vec_sub_diag = K1 * np.ones(len_diag - 1)
    vec_diag     = K  * np.ones(len_diag)
    vec_sup_diag = K2 * np.ones(len_diag - 1)

    # chron00 = time.time()
    k = [vec_sub_diag, vec_diag, vec_sup_diag]
    offset = [-1,0,1]
    M = diags(k,offset).toarray()
    # print("T00",time.time() - chron00)

    BC_M = np.identity(gd.N_gc)
    M[:gd.N_gc,:gd.N_gc] = BC_M
    M[-gd.N_gc:,-gd.N_gc:] = BC_M
    M[gd.N_gc-1,gd.N_gc] = 0
    M[-gd.N_gc,-gd.N_gc-1] = 0
    # print("\n\n",M,"\n\n")
    return M


def implicit_diff_solver_tridiag(size_M, build_diags, system, args=(), kwargs=dict(), X=None, A=None, B=None):

    """ solve Ax = B with A being a tridiag matrix"""
    """ résolution du système: My = Ax + B   ==> y = M^(-1) * (Ax+B)"""
    # chron1 = time.time()
    if A is None:
        A = np.identity(size_M)
    if X is None:
        X = np.identity(size_M)
    if B is None:
        B = np.zeros(size_M)

    AB = np.zeros((3,size_M))

    AB[2,:-1], AB[1,:], AB[0,1:] = build_diags(*args, **kwargs)

    try:
        Y = solve_banded((1,1), AB, B + np.matmul(A, X))
    except :
        system.error_quit("error_on_solving_matrix")

    return Y



def build_diags_1(len_diag, decent_av = None, decent_not_av = None, cent = None, flux_BL_0=False, flux_BR_0=False):

    if decent_av is None:
        decent_av = np.ones(len_diag)
    if decent_not_av is None:
        decent_not_av = np.ones(len_diag)
    if cent is None:
        cent = np.ones(len_diag)

    """
    cent (A) : value of the cell
    decent_av (B): decentered values averaged on the cell face
    decent_not_av (C) : decentered values taking the other cell values (not conservative)
    cent dx decent_av dx decent_not_av (y) 

    A dx [B dx (C y)]
    """

    flux_null_L = np.ones(len_diag)
    flux_null_R = np.ones(len_diag)


    if flux_BL_0==True:
        flux_null_L[gd.N_gc] = 0
    if flux_BR_0==True:
        flux_null_R[-gd.N_gc-1] = 0


    K1 = 0 - cent[1:-1] * decent_not_av[:-2]  *  flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
    K2 = 0 - cent[1:-1] * decent_not_av[2:]   *  flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2
    K  = 1 + cent[1:-1] * decent_not_av[1:-1] * (flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
                                               + flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2)


    # K1 = 0 - cent[1:-1] * decent_not_av[:-2]  *  flux_null_L[1:-1] * 1/(0.5/decent_av[:-2] + 0.5/decent_av[1:-1])
    # K2 = 0 - cent[1:-1] * decent_not_av[2:]   *  flux_null_R[1:-1] * 1/(0.5/decent_av[2:]  + 0.5/decent_av[1:-1])
    # K  = 1 + cent[1:-1] * decent_not_av[1:-1] * (flux_null_L[1:-1] * 1/(0.5/decent_av[:-2] + 0.5/decent_av[1:-1])
    #                                            + flux_null_R[1:-1] * 1/(0.5/decent_av[2:]  + 0.5/decent_av[1:-1]))


    # if (np.abs(K)>1e12).any():
    #     print()
    # K1 = np.where(np.abs(K1)>1e12, K1/np.abs(K1*(1 + 1e-15)) * 1e12, K1)
    # K2 = np.where(np.abs(K2)>1e12, K2/np.abs(K2*(1 + 1e-15)) * 1e12, K2)
    # K  = np.where(np.abs(K )>1e12, K /np.abs(K *(1 + 1e-15)) * 1e12, K )

    # K1 = np.where(np.abs(K1)>1e5, K1/np.abs(K1*(1 + 1e-15)) * 1e5, K1)
    # K2 = np.where(np.abs(K2)>1e5, K2/np.abs(K2*(1 + 1e-15)) * 1e5, K2)
    # K  = np.where(np.abs(K )>1e5, K /np.abs(K *(1 + 1e-15)) * 1e5, K )

    K  = np.insert(K, [0], [0])
    K  = np.append(K, 0)
    K1 = np.append(K1, 0)
    K2 = np.insert(K2, 0, [0])


    # if np.isnan(K).any() or np.isnan(K1).any() or np.isnan(K2).any():
    #     print()

    # if np.isinf(K).any() or np.isinf(K1).any() or np.isinf(K2).any():
    #     print()

    K[:gd.N_gc], K[-gd.N_gc:] = 1, 1
    # pas de flux entre 2 GC d'une meme BC:
    K1[0] = 0
    K2[0] = 0
    K1[-1] = 0
    K2[-1] = 0

    # Les BC valeurs des BC ne varient pas:
    K2[1] = 0
    K1[-2] = 0



    return K1, K, K2



#---------------------------------------------------------------------------------------------------------------------------------
def implicit_diff_solver_tridiag_fast_alpha(size_M, build_diags, args=(), kwargs=dict(), X=None, A=None, B=None):

    """ solve Ax = B with A being a tridiag matrix"""
    """ résolution du système: My = Ax + B   ==> y = M^(-1) * (Ax+B)"""
    AB = np.empty((3,size_M))
    AB[2,:-1], AB[1,:], AB[0,1:] = build_diags(*args, **kwargs)
    Y = solve_banded((1,1), AB, X)
    return Y


#---------------------------------------------------------------------------------------------------------------------------------
def build_diags_1_fast_alpha(len_diag, decent_av = None, cent = None):

    """
    cent (A) : value of the cell
    decent_av (B): decentered values averaged on the cell face
    decent_not_av (C) : decentered values taking the other cell values (not conservative)
    cent dx decent_av dx decent_not_av (y) 

    A dx [B dx (C y)]
    """

    flux_null_L = np.ones(len_diag)
    flux_null_R = np.ones(len_diag)

    flux_null_L[gd.N_gc] = 0
    flux_null_R[-gd.N_gc-1] = 0


    K1 = 0 - cent[1:-1] *  flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
    K2 = 0 - cent[1:-1] *  flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2
    K  = 1 + cent[1:-1] * (flux_null_L[1:-1] * (decent_av[:-2] + decent_av[1:-1])/2
                                               + flux_null_R[1:-1] * (decent_av[2:]  + decent_av[1:-1])/2)


    K1 = np.where(np.abs(K1)>1e12, K1/np.abs(K1*(1 + 1e-15)) * 1e12, K1)
    K2 = np.where(np.abs(K2)>1e12, K2/np.abs(K2*(1 + 1e-15)) * 1e12, K2)
    K  = np.where(np.abs(K )>1e12, K /np.abs(K *(1 + 1e-15)) * 1e12, K )

    # if np.isnan(K).any() or np.isnan(K1).any() or np.isnan(K2).any():
    #     print()

    K  = np.insert(K, [0], [0])
    K  = np.append(K, 0)
    K1 = np.append(K1, 0)
    K2 = np.insert(K2, 0, [0])


    K[:gd.N_gc], K[-gd.N_gc:] = 1, 1
    # pas de flux entre 2 GC d'une meme BC:
    K1[0] = 0
    K2[0] = 0
    K1[-1] = 0
    K2[-1] = 0

    # Les BC valeurs des BC ne varient pas:
    K2[1] = 0
    K1[-2] = 0

    return K1, K, K2



#---------------------------------------------------------------------------------------------------------------------------------
def Thermal_flux_relative_to_mass_flux_computation(flux_p,source_p,particle,gas, ebull_dic = None):
    len_x = len(particle.temperature)
    flux_per_species=np.zeros([len_x,gas.nb_species,2])
    decomp_Al2O3_species_flux=np.zeros([len_x,2])
    add_heat_flux_AlxOy=np.zeros([len_x,2])
#"O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"

    if ebull_dic == None:
        ebull_dic = dict(Al = False, Cu = False)

    UH_g = enth.uCv_i_g
    UH_p = enth.uCv_i_p

    # Les réactions surfaciques ont elles davantage lieu dans la phase gaz?
    # Attention: si True, un déséquilibre thermique en fin de réaction peut apparaitre
    bool_react_gas = False

    for i in range (0,2):
        if particle.name=="p_Al":
            flux_per_species[:,gd.igs["O2"],i]=(UH_g(T_inter(gas, particle, flux_p[i].O2_ext),"O2")[0])*flux_p[i].O2_ext    #O2
            flux_per_species[:,gd.igs["O"],i]=(UH_g(T_inter(gas, particle, flux_p[i].O_ext),"O")[0])*flux_p[i].O_ext    #O

        elif particle.name=="p_MeO":
            flux_per_species[:,gd.igs["O2"],i]=(UH_g(T_inter(gas, particle, flux_p[i].O2_ext),"O2")[0])*flux_p[i].O2_ext    #O2

        if not ebull_dic["Al"]:
            flux_per_species[:,gd.igs["Al"],i]=(UH_g(T_inter(gas, particle, flux_p[i].Al_ext),"Al")[0])*flux_p[i].Al_ext#+Al.h_evap    #Al
        else:
            flux_per_species[:,gd.igs["Al"],i] = 0

        if not ebull_dic["Cu"]:
            flux_per_species[:,gd.igs["Me"],i]=(UH_g(T_inter(gas, particle, flux_p[i].Me),"Cu")[0])*flux_p[i].Me    #Me
        else:
            flux_per_species[:,gd.igs["Me"],i] = 0
        #additional heat flux related to AlxOy condensation
        add_heat_flux_AlxOy[:,i] = heat_flux_AlxOy(i,UH_g,UH_p,gas,particle,flux_p, bool_react_gas)

        #additional heat flux related to Al2O3 decomposition
        decomp_Al2O3_species_flux[:,i] = heat_flux_decomp(i,UH_g,UH_p,gas,particle, source_p, bool_react_gas)

    return np.sum(flux_per_species,axis=1)[:,1]+add_heat_flux_AlxOy[:,1]+decomp_Al2O3_species_flux[:,1]


def solve_tridiag_from_mat(len_x, A, B):
    AB = np.empty((3, len_x))
    AB[0,1:] = np.diag(A,1)
    AB[1,:] = np.diag(A,0)
    AB[2,:-1] = np.diag(A,-1)
    return y


def get_p_density_from_Q(Q, system):
    # """except error: array of bool where the error sum(Y)!=1 must be
    # considered"""
    # if except_error is None:
    #     except_error = np.full(system.n_volumes, False)

    t = np.transpose
    Y = t(t(Q[:, gd.i_Y:])/Q[:,gd.i_d])
    # i_ee = np.argwhere(except_error)[:,0]# index except error
    # Y[i_ee,:] = t(t(Q[i_ee, gd.i_Y:])/np.sum(Q[i_ee, gd.i_Y:], axis = 1))
    sum_Y = np.sum(Y, axis = 1)
    if np.max(np.abs(sum_Y - 1))>gd.eps10:
        system.error_quit("problem_with_Y_p")
    else:
        Y = np.transpose(np.transpose(Y) / sum_Y)

    rho = 1/np.sum((Y / gd.rho_p_vec), axis = 1)
    return rho




def implicit_drag(p_dic, g_dic, Q_p_adjusted_dic, Q_g_adjusted_dic, system):
    T_dic = dict( g = (    Q_g_adjusted_dic["gas"][:,gd.i_vx] - g_dic["gas"].Q[:, gd.i_vx])/system.dt,
                  p =  ( Q_p_adjusted_dic["p_Al"][:,gd.i_vx] - p_dic["p_Al"].Q[:, gd.i_vx])/system.dt,
                  q =  (Q_p_adjusted_dic["p_MeO"][:,gd.i_vx] - p_dic["p_MeO"].Q[:, gd.i_vx])/system.dt,
                )

    # zeta_p_1 = (p_dic["p_Al"].Q[:, gd.i_vx]  + T_dic["p"] * system.dt)/(1 + system.dt*system.momentum_flux.tau_Fpg_1)
    # zeta_q_1 = (p_dic["p_MeO"].Q[:, gd.i_vx] + T_dic["q"] * system.dt)/(1 + system.dt*system.momentum_flux.tau_Fqg_1)

    if system.inputs_dic["equation_type"]["bool_particles_QDM"]:
        zeta_p_1 = (p_dic["p_Al"].Q[:, gd.i_vx]  + T_dic["p"] * system.dt) * system.momentum_flux.tau_Fpg_1 / (1 + system.dt*system.momentum_flux.tau_Fpg_1)
        zeta_q_1 = (p_dic["p_MeO"].Q[:, gd.i_vx] + T_dic["q"] * system.dt) * system.momentum_flux.tau_Fqg_1 / (1 + system.dt*system.momentum_flux.tau_Fqg_1)

        zeta_p_2 =  Q_p_adjusted_dic["p_Al"][:, gd.i_d] * system.dt * system.momentum_flux.tau_Fpg_1 / (1 + system.dt * system.momentum_flux.tau_Fpg_1)
        zeta_q_2 = Q_p_adjusted_dic["p_MeO"][:, gd.i_d] * system.dt * system.momentum_flux.tau_Fqg_1 / (1 + system.dt * system.momentum_flux.tau_Fqg_1)

        num = g_dic["gas"].Q[:, gd.i_vx] + (T_dic["g"] + zeta_p_1 + zeta_q_1) * system.dt
        denum = Q_g_adjusted_dic["gas"][:, gd.i_d] + (
                                                    Q_p_adjusted_dic["p_Al"][:, gd.i_d] * system.momentum_flux.tau_Fpg_1
                                                + Q_p_adjusted_dic["p_MeO"][:, gd.i_d] * system.momentum_flux.tau_Fqg_1
                                                - zeta_p_2
                                                - zeta_q_2
                                                ) * system.dt


        u_g = num / denum
        u_p = get_u_p( p_dic["p_Al"], T_dic["p"],  Q_p_adjusted_dic["p_Al"], system.momentum_flux.tau_Fpg_1, u_g, system)
        u_q = get_u_p(p_dic["p_MeO"], T_dic["q"], Q_p_adjusted_dic["p_MeO"], system.momentum_flux.tau_Fqg_1, u_g, system)

    else:
        zeta_p_1 = 0
        zeta_q_1 = 0

        zeta_p_2 = 0
        zeta_q_2 = 0
        num = g_dic["gas"].Q[:, gd.i_vx] + (T_dic["g"] + zeta_p_1 + zeta_q_1) * system.dt
        denum = Q_g_adjusted_dic["gas"][:, gd.i_d] + (
                                                    Q_p_adjusted_dic["p_Al"][:, gd.i_d] * system.momentum_flux.tau_Fpg_1
                                                + Q_p_adjusted_dic["p_MeO"][:, gd.i_d] * system.momentum_flux.tau_Fqg_1
                                                - zeta_p_2
                                                - zeta_q_2
                                                ) * system.dt

        u_g = num / denum
        u_p = np.zeros(system.n_volumes)
        u_q = np.zeros(system.n_volumes)


    Q_p_adjusted_dic["p_Al"][gd.N_gc:-gd.N_gc, gd.i_vx]  = u_p[gd.N_gc:-gd.N_gc] * Q_p_adjusted_dic["p_Al"][gd.N_gc:-gd.N_gc, gd.i_d]
    Q_p_adjusted_dic["p_MeO"][gd.N_gc:-gd.N_gc, gd.i_vx] = u_q[gd.N_gc:-gd.N_gc] * Q_p_adjusted_dic["p_MeO"][gd.N_gc:-gd.N_gc, gd.i_d]
    Q_g_adjusted_dic["gas"][gd.N_gc:-gd.N_gc, gd.i_vx]     = u_g[gd.N_gc:-gd.N_gc] * Q_g_adjusted_dic["gas"][gd.N_gc:-gd.N_gc, gd.i_d]

    return Q_p_adjusted_dic, Q_g_adjusted_dic



def get_u_p(p, T, Q_p_adjusted, tau_p_1, u_g, system):
    num = p.Q[:, gd.i_vx] + T * system.dt + Q_p_adjusted[:, gd.i_d] * tau_p_1 * u_g * system.dt
    denum = Q_p_adjusted[:, gd.i_d] * (1 + system.dt * tau_p_1)
    return num/denum



def get_bed_conductivities(p_Al, p_MeO, gas):
    # Evangelos Tsotsas
    # Otto-von-Guericke-Universität Magdeburg, Magdeburg, Germany
    # eq 7
    alpha_g = gas.alpha

    epsilon_p_av = (((  p_Al.alpha *  p_Al.r_ext**2)*p_Al.epsilon
                    + (p_MeO.alpha * p_MeO.r_ext**2)*p_MeO.epsilon)
                    /(( p_Al.alpha *  p_Al.r_ext**2)
                    +( p_MeO.alpha * p_MeO.r_ext**2))
                   )

    T_p_av = (((p_Al.n_vol)*p_Al.temperature
            + (p_MeO.n_vol)*p_MeO.temperature)
             /((p_Al.n_vol)
            + (p_MeO.n_vol))
            )

    lambda_p_av = (((p_Al.n_vol)*p_Al.temperature
                 + (p_MeO.n_vol)*p_MeO.temperature)
                  /((p_Al.n_vol)
                 + (p_MeO.n_vol))
            )

    # moyenne harmonique pondéré par la fraction volumique pour le diametre moyen (cf eq 8 P576 definition de Qi dans ref de la fonction)
    d_p_av  =      1/(((p_Al.alpha)/p_Al.alpha
                    + (p_MeO.alpha)/p_MeO.alpha)
                    #  /((p_Al.alpha)
                    # + (p_MeO.alpha))
            )


    k_G   = 1 # (1 +(l/d))**(-1) # = lambda_g / lambda_f ==> Smoluchowski effect (k_G= 1 ==> not considered)
    k_rad = 4 * gd.sigma / (2/epsilon_p_av - 1) * T_p_av**3 * d_p_av / gas.lambda_f
    phi = 0 # flattening coefficient

    k_p = lambda_p_av / gas.lambda_f
    k_c = compute_k_c_full(alpha_g, k_p, gas.lambda_f, k_rad, k_G)


    k_f_prime = (1 - np.sqrt(1 - alpha_g)) * alpha_g * ((alpha_g - 1 + k_G**(-1))**(-1) + k_rad)
    k_p_prime = np.sqrt(1 - alpha_g) * (phi * k_p + (1 - phi) * k_c)

    lambda_f_prime = k_f_prime * gas.lambda_f
    lambda_p_prime = k_p_prime * gas.lambda_f

    return lambda_p_prime, lambda_f_prime




def compute_k_c_full(alpha_g, k_p, lambda_f, k_rad, k_G):
    B = (compute_B_conduc(alpha_g) + gd.eps)
    N = (1/k_G * (1 + (k_rad - B * k_G)/k_p)
        - B * (1/k_G - 1) * (1 + k_rad/k_p))

    K1 = B * (k_p + k_rad - 1) / (N**2 * k_G * k_p)

    K2 = np.log((k_p + k_rad) / (B * (k_G + (1 - k_G) * (k_p + k_rad))) )

    K3 = (B+1) / (2 * B)

    K4 = k_rad / k_G - B * (1 + (1 - k_G)/k_G * k_rad)

    K5 = (B-1) / (N * k_G)

    k_c = 2/N * (K1 * K2 + K3 * (K4) - K5)

    return k_c


def filter_velocity(Q, system, velocity_x = None):
    if velocity_x is None:
        velocity_x = Q[:, gd.i_vx]/Q[:, gd.i_d]
    velocity_x[:gd.N_gc]  = velocity_x[gd.N_gc]
    velocity_x[-gd.N_gc:] = velocity_x[-gd.N_gc-1]
    velocity_x_filtered = deepcopy(velocity_x)
    window = 3

    sigma = 0.2
    NTF = 1 # cells Not To Filter in the computational domain in addition to the ghost cells
    velocity_x_filtered[int(window/2) + NTF: - int(window/2+NTF)] = moving_average_weighted(velocity_x, window, sigma = sigma)[NTF:-NTF]
    # cond_div_vel_point = np.full(system.n_volumes, False)
    # cond_div_vel_point[1:-1] = (velocity_x_filtered[2:]/velocity_x_filtered[1:-1]<0) | (velocity_x_filtered[:-2]/velocity_x_filtered[1:-1]<0)

    # self.velocity_x = np.where(cond_div_vel_point, 0, velocity_x_filtered)
    Q[:, gd.i_vx] = velocity_x_filtered * Q[:, gd.i_d]
    return velocity_x_filtered, Q


def heat_losses_1(L, lambda_, r1, r2, T1, T2, mesh_volume, n_volumes):
    """
    1) steady state heat flux Q through a cylinder L * r2 with
    two radius r1 & r2, where the temperatures are T(r1) = T1
    and T(r2) = T2 at the inner and outer radius respectively

    2) In the case of the developped code, the length of the cy-
    linder is the mesh size dx.

    3) lambda_ is the conductivity of the material of the cylinder.
    
    4) We assume that the temperature at the inner side is an har-
    monic mean weighted by the density of the different phases.

    5) Considering 1) (steady state) and 2) (T° @ the inner cylin-
    der), note that the heat flux is probably over-estimated.

    6) The heat flux is normalized by the mesh volume in order to
    be directly fitted into the the energy equations.
    Q[W]
    Q_v[W/m3]
    """

    Q = 2 * np.pi * L * lambda_ * (T1 - T2) / np.log(r1/r2)
    Q_v = Q / mesh_volume

    return Q_v



def heat_losses_none(L, lambda_, r1, r2, T1, T2, mesh_volume, n_volumes):
    return np.zeros(n_volumes)


def losses_weight(p_Al, p_MeO, gas, walls, phase_name):
    if phase_name == "p_Al":
        phase = p_Al
    if phase_name == "p_MeO":
        phase = p_MeO
    if phase_name == "gas":
        phase = gas
        w = np.zeros(p_Al.temperature.shape)
    else:
        w = (( np.where(phase.temperature>walls.temperature_in, (phase.temperature - walls.temperature_in), 0) + gd.eps) * phase.Q[:, gd.i_d]/
            (  (np.where(p_Al.temperature>walls.temperature_in, (p_Al.temperature - walls.temperature_in), 0) + gd.eps) * p_Al.Q[:, gd.i_d]
            + (np.where(p_MeO.temperature>walls.temperature_in, (p_MeO.temperature - walls.temperature_in), 0) + gd.eps) * p_MeO.Q[:, gd.i_d]
            +   (np.where(gas.temperature>walls.temperature_in, (gas.temperature - walls.temperature_in), 0) + gd.eps) * gas.Q[:, gd.i_d]
            )
            )
    return w

def P_drop(phase, x_idx, system, coeff_P_drop):
    P_atm = 1e5

    L_tube = system.x_simu * coeff_P_drop
    d_tube = 1e-2

    L_Re  = d_tube
    v_Re  = phase.velocity_x[x_idx]
    # r_Re  = phase.density[x_idx]
    nu_Re = phase.viscosity_kin[x_idx]
    Re = (np.abs(v_Re) + gd.eps) * L_Re / nu_Re

    f = 64/Re

    if (phase.velocity_x[x_idx] < 0 and x_idx == 2) or (phase.velocity_x[x_idx] > 0 and x_idx == -3) :
        Delta_P = f * L_tube / d_tube * phase.density[x_idx] * phase.velocity_x[x_idx]**2/2
    else:
        Delta_P = 0
    P_BC = P_atm + Delta_P
    return P_BC


# def compute_dm_dt_kinetics():
def rel_diff(A,B):
    return (np.max(np.abs((A-B)/(A+gd.eps))),
            np.unravel_index(np.argmax(np.abs((A-B)/(A+gd.eps))), A.shape),
            np.max(np.abs(A-B)),
            np.unravel_index(np.argmax(np.abs((A-B))), A.shape))


def argmax_multiD(A):
    return np.unravel_index(np.argmax(np.abs((A))), A.shape)


def get_ct_val_indexed(func):
    arr = eval("[gd.gas_ct."+ func +"[gd.ct_index[i].astype(int)] for i in range(gd.nb_gas_species)]")
    return np.array(arr)

@njit
def tile_nb(A, size_new_axis):
    if isinstance(A, float) or isinstance(A, int):
        A = np.array(A)
    new_shape = ((size_new_axis,) + A.shape)
    A_tiled = np.zeros((new_shape))
    for i in range(size_new_axis):
        A_tiled[i] = A
    return A_tiled

@njit
def reshape_nb(A, new_shape):
    A_reshaped = np.zeros(new_shape)

@njit
def D2_2_flat_nb(A):
    A_reshaped = np.zeros(A.size)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_reshaped[i*A.shape[1]+j] = A[i,j]
    return A_reshaped

@njit
def flat_2_2D_nb(A, shape):
    A_reshaped = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            A_reshaped[i,j] = A[i*shape[1]+j]
    return A_reshaped


# def prod_nb(A, axis):
#     A_shape = A.shape
#     A_prod_shape = A_shape[:axis] + A_shape[axis+1:]
#     A_prod = np.zeros(A_prod_shape)

#     for idx_ax in len(A_prod_shape):
#         for i in A_prod_shape[idx_ax]:
#             A_prod = np.prod(A[])

#     for i in range(size_new_axis):
#         A_tiled[i] = A
#     return A_tiled
