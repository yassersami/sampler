from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

import simulator_0d.src.py0D.enthalpy as enth
import simulator_0d.src.py0D.kinetics.module_kin as mkin
from simulator_0d.src.py0D.assist_functions import vectorize

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
def P_evap(T, species):
    R = 8.3144626                                           # K/mol/K
    return P_atm * np.exp(species.h_evap * species.MW / R * (1/species.T_evap - 1/T))


#---------------------------------------------------------------------------------------------------------------------------------
def Arr(T, k, Ea):
    R = 8.3144626
    return k*np.exp(-Ea/R/T)


#---------------------------------------------------------------------------------------------------------------------------------
def flux_prob_evap(species, T):
    R = 8.3144626                                           # K/mol/K
    E = species.K*T/np.sqrt(2*np.pi*species.MW*R)*np.exp(-(species.h_evap*species.MW)/R/T)*species.MW # kg/m2/s
    return E


#---------------------------------------------------------------------------------------------------------------------------------
def flux_P_i(species_index, gas):
    R = 8.3144626
    P_i = gas.khi[species_index]*gas.pressure
    species = gas.species_carac[species_index]
    T = gas.temperature
    flux = P_i/np.sqrt(2*np.pi*species.MW*R*T)*species.MW
    return flux


#---------------------------------------------------------------------------------------------------------------------------------
def dX_dt(X,t):
    dX_dt = (X[1:]-X[:-1])/(t[1:]-t[:-1])
    return dX_dt


#---------------------------------------------------------------------------------------------------------------------------------
def T_inter(gas, particle, flux):
    """température de l'interface gaz/particule"""
    if flux>0:
        return particle.temperature
    else:
        return gas.temperature
    # return 1/3 * gas.temperature + 2/3 * particle.temperature
    # DELTA_T = particle.temperature - gas.temperature
    # if DELTA_T < 1200:
    #     return 1/3 * gas.temperature + 2/3 * particle.temperature
    # else:
    #     if flux>0:
    #         return 1/3 * gas.temperature + 2/3 * particle.temperature
    #     else:
    #         return gas.temperature


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_AlxOy(particle, gas):
    species_carac=gas.species_carac

    nb_AlxOy = 4
    flux_AlxOy = np.zeros(gas.nb_species)

    first_index_AlxOy = 4
    for i in range(first_index_AlxOy, first_index_AlxOy+nb_AlxOy):
        flux_AlxOy[i] = flux_species_g_p_diff(gas,particle,i)

    #0)02  1)Al   2)O    3)N2    4)Al2O    5) AlO    6)AlO2   7)Al2O2

    flux_atom_Al = (2 * flux_AlxOy[4]/species_carac[4].MW           # Al2O
                    + 1.0 * flux_AlxOy[5]/species_carac[5].MW       # AlO
                    + 2.0 * flux_AlxOy[7]/species_carac[7].MW       # Al2O2
                    + 1.0 * flux_AlxOy[6]/species_carac[6].MW       # AlO2
                    )

    flux_atom_O2  =(0.5 * flux_AlxOy[4]/species_carac[4].MW         # Al2O
                    + 0.5 * flux_AlxOy[5]/species_carac[5].MW       # AlO
                    + 1.0 * flux_AlxOy[7]/species_carac[7].MW       # Al2O2
                    + 1.0 * flux_AlxOy[6]/species_carac[6].MW       # AlO2
                    )

    if 1/2 * flux_atom_Al < 2/3 * flux_atom_O2:
        source_Al2O3 = - (particle.species_carac[3].MW * ( #signe "-" car on considère ca comme un flux dont la norme et p to g
                     + 1.0 * flux_AlxOy[4]/species_carac[4].MW      # Al2O
                     + 0.5 * flux_AlxOy[5]/species_carac[5].MW      # AlO
                     + 1.0 * flux_AlxOy[7]/species_carac[7].MW      # Al2O2
                     + 0.5 * flux_AlxOy[6]/species_carac[6].MW      # AlO2
                     ))

        flux_O2_cond = (gas.species_carac[0].MW * (
                     - 1.0 * flux_AlxOy[4]/species_carac[4].MW      # Al2O
                     - 1/4 * flux_AlxOy[5]/species_carac[5].MW      # AlO
                     - 1/2 * flux_AlxOy[7]/species_carac[7].MW      # Al2O2
                     + 1/4 * flux_AlxOy[6]/species_carac[6].MW      # AlO2
                     ))

        flux_Al_cond = 0

    elif 1/2 * flux_atom_Al >= 2/3 * flux_atom_O2:
        source_Al2O3 = - (particle.species_carac[3].MW * ( #signe "-" car on considère ca comme un flux dont la norme et p to g
                     + 1/3 * flux_AlxOy[4]/species_carac[4].MW      # Al2O
                     + 1/3 * flux_AlxOy[5]/species_carac[5].MW      # AlO
                     + 2/3 * flux_AlxOy[7]/species_carac[7].MW      # Al2O2
                     + 2/3 * flux_AlxOy[6]/species_carac[6].MW      # AlO2
                     ))

        flux_Al_cond = (particle.species_carac[0].MW * (
                     + 4/3 * flux_AlxOy[4]/species_carac[4].MW      # Al2O
                     + 1/3 * flux_AlxOy[5]/species_carac[5].MW      # AlO
                     + 2/3 * flux_AlxOy[7]/species_carac[7].MW      # Al2O2
                     - 1/3 * flux_AlxOy[6]/species_carac[6].MW      # AlO2
                     ))

        flux_O2_cond = 0

    return (-flux_AlxOy),np.sum(-flux_AlxOy),source_Al2O3,flux_Al_cond,flux_O2_cond


#---------------------------------------------------------------------------------------------------------------------------------
def integrate(f_t0, f_t1, dt):
    return np.trapz([f_t0, f_t1], dx=dt)


#---------------------------------------------------------------------------------------------------------------------------------
def dr_dt_computation(particule, flux, source):
    if particule.name == "p_MeO":
        Y_Me_core = particule.composition[2]/(particule.composition[1] + particule.composition[2])
        Y_MeO_core = particule.composition[1]/(particule.composition[1] + particule.composition[2])

        num1 = Y_Me_core*(source[1].Me - flux[1].Me) + Y_MeO_core*(-source[1].MeO + flux[1].MeO)
        num0 = Y_Me_core*(source[0].Me - flux[0].Me) + Y_MeO_core*(-source[0].MeO + flux[0].MeO)
        denum = 4/3*np.pi*3*particule.r_int**2*(Y_Me_core * (particule.species_carac[1].rho - particule.species_carac[2].rho) + particule.species_carac[2].rho)
        dr_int_dt1 = num1/denum
        dr_int_dt0 = num0/denum


    if particule.name == "p_Al":
        num1 = -flux[1].Al_int-source[1].Al_int
        num0 = -flux[0].Al_int-source[0].Al_int
        denum = 4/3*3*np.pi*particule.r_int**2*particule.species_carac[0].rho
        dr_int_dt1 = num1/denum
        dr_int_dt0 = num0/denum

# ---------------------------------calcul de la variation du rayon intérieur theorique à partir du grossissement de la couche d'alumine vers l'intérieur
# Rappel: la variation du rayon peut etre calculée soit à partir de la variation de la masse du coeur,  soit à partir de la variation de la masse de la coquille
#         le calcul se fait à partir de la variation de la masse du coeur,  puis un ajustement se fait pour le rayon extérieur afin de garder la meme quantité de
#         coquille mais en réduisant le volume total de la particule
    dr_int_dt0_bis = -source[1].Al2O3_int/4*np.pi*particule.species_carac[3].rho*particule.r_int**2
    dr_int_dt1_bis = -source[0].Al2O3_int/4*np.pi*particule.species_carac[3].rho*particule.r_int**2

    dr_ext_dt1 = source[1].Al2O3_ext/(4*np.pi*particule.r_ext**2*particule.species_carac[3].rho)
    dr_ext_dt0 = source[0].Al2O3_ext/(4*np.pi*particule.r_ext**2*particule.species_carac[3].rho)

    dr_ext_dt1_bis = (particule.r_int/particule.r_ext)**2*(dr_int_dt1-dr_int_dt1_bis) # cf equation 50 feuille d'equations (dr(prime) = (rMe/r)^2dr(prime)Me)
    dr_ext_dt0_bis = (particule.r_int/particule.r_ext)**2*(dr_int_dt0-dr_int_dt0_bis)

    return dr_int_dt0, dr_int_dt1, dr_ext_dt0, dr_ext_dt1, dr_ext_dt0_bis, dr_ext_dt1_bis


#---------------------------------------------------------------------------------------------------------------------------------
def flux_species_g_p_diff(gas,particle,species_index, *args): # gas index
    # species=gas.species_carac[species_index]
    # Diff=gaz_kinetic.diff_coeff(gas.temperature, gas.pressure,species.r_VDW,species.MW)*1e0
    Diff=gas.gas_ct.mix_diff_coeffs[gas.gas_ct.species_index(gas.species_name[species_index])] #Diffusion coeff from ct
    rho_f=gas.density
    d_p=2*particle.r_ext
    Sh=2
    if species_index==0:
        Y_at_s=0 #dans le cas de l'oxygène, comparer avec flux dans la particule
    else:
        Y_at_s=0 #sous oyxde, concentration nulle à la surface
    if args:
        Y_at_s = args[0]
    Y_at_p=gas.Y[species_index]
    Flux=-np.pi*d_p*rho_f*Diff*Sh*np.log((1-Y_at_p)/(1-Y_at_s))
    return Flux


#---------------------------------------------------------------------------------------------------------------------------------
def evap_flux(gas, particle, species_index):
    X_s_ths = 0.95
    Y_p_ths = 1e-10
    Y_g_ths = 1e-10
    X_s_max = 1 - 1e-3

    species = gas.species_carac[species_index]

    #Clapeyron :
    P_s = P_evap(particle.temperature, species)

    X_s = P_s / gas.pressure
    ebull_bool_1 = False
    
    if X_s>=X_s_ths:
        """
        X_s > X_s_ths means that the mass transfer follow a boiling regime
        """
        ebull_bool_1 = True

    if X_s>=1:
        X_s = X_s_max

    X_surf = gas.khi/np.sum(np.delete(gas.khi,species_index))*(1-X_s)
    X_surf[species_index] = X_s
    Y_s = X_s * species.MW/(np.sum(gas.MW_vec*X_surf))
    evap_flux = - flux_species_g_p_diff(gas, particle, species_index, Y_s)

    #Ajustement des index en fonction de la phase...
    if species_index == 1:
        p_species_index = 0
    elif species_index == 8:
        p_species_index = 2

    # ponderation par la fraction surfacique (approximée à la fraction massique) de la particule vs. Al2O3
    if particle.composition[p_species_index]>0:
        Surf_ratio = ((particle.composition[p_species_index]/particle.species_carac[p_species_index].rho)/np.sum(particle.composition/vectorize(particle.species_carac[0:4],'rho')))**(2/3)
    else:
        Surf_ratio = 0
    # fraction surfacique uniquement pour la vaporisation (pas pour la condensation)
    ignore_vap_bool = False
    if Surf_ratio < 1e-5:
        """
        If the surface ratio is < 1e-5, we do not consider any mass interphase mass transfer
        The surface ratio is zero, and a boolean is used to cancel the boiling regime if X_s>ths
        """
        ignore_vap_bool = True
        Surf_ratio = 0

    if ebull_bool_1 and not ignore_vap_bool:
        print("py0D.functions.evap_flux -> boiling on particle " + particle.name + " of species " +species.name, end="\r")
    # if ebull_bool_1:
    #     """
    #     if the boiling regime is activated, the evaporation regime is set to 0
    #     """
    #     evap_flux = 0
    if Y_s>0.999999:
        if not ebull_bool_1:
            print("attention, la fraction d'aluminium à la surface est supérieure à 0.999")
    elif Y_s<0 and not ignore_vap_bool:
        print("attention, la fraction d'aluminium à la surface est négative")
    if evap_flux>0:
        evap_flux = evap_flux * Surf_ratio

    return evap_flux, Y_s, ebull_bool_1, ignore_vap_bool


#---------------------------------------------------------------------------------------------------------------------------------
def O_transport_alumina(particle, Y_ext, Diff):
    # avec prise en compte de l'épaisseur d'alumine (Remarque: passé une certaine température l'alumine et l'aluminium
    # sont liquides. la couche d'alumine devient alors un artefact.
    if particle.composition[0]/np.sum(particle.composition)>1e-15:
        rho=particle.species_carac[3].rho
        Y_int = 0
        Flux= - 4*np.pi*rho*Diff*np.log((1-Y_int)/(1-Y_ext)) * (particle.r_ext*particle.r_int/(particle.r_ext-particle.r_int))
    else:
        Flux = 0
    return Flux


#---------------------------------------------------------------------------------------------------------------------------------
def compute_Y_s(species_index, particle, gas, Diff_Al2O3, phi_Ox_condens):
    Diff_g = gas.gas_ct.mix_diff_coeffs[gas.gas_ct.species_index(gas.species_name[species_index])] #Diffusion coeff from ct
    Sh = 2

    Ag = np.pi * particle.r_ext*2 * Sh * gas.density * Diff_g
    Ap = np.pi * 4 * particle.species_carac[3].rho * Diff_Al2O3 * particle.r_ext*particle.r_int/(particle.r_ext - particle.r_int)

    Bg = 1 - gas.Y[species_index]
    Bp = 1

    Y_s_g_anal = analytical_Y_s(Ap, Ag, Bp, Bg,phi_Ox_condens)
    return Y_s_g_anal


#---------------------------------------------------------------------------------------------------------------------------------
def func_Y_s(Y_s, Ap, Ag, Bp, Bg, d, phi_Ox_condens):
    return (Bg**Ag*Bp**Ap)-(1 - Y_s)**Ag*np.exp(phi_Ox_condens)*(1-1/d * Y_s)**Ap


#---------------------------------------------------------------------------------------------------------------------------------
def analytical_Y_s(Ap, Ag, Bp, Bg,phi_Ox_condens):
    return (1 - Bp**(Ap/(Ap+Ag))
               *Bg**(Ag/(Ap+Ag))
               *np.exp(-phi_Ox_condens/(Ap+Ag)))


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
def moving_average_weighted(x, window_size, sigma):
    weights = np.ones(window_size)
    mu = 0
    for i in range (0, window_size):
        weights[i] = ((1/(sigma * np.sqrt(2*np.pi)))
                     * np.exp(-1/2*(((i-mu)/2)/sigma)**2)
                    )
    weights = weights / np.sum(weights)
    return np.convolve(x, weights, 'valid')


#---------------------------------------------------------------------------------------------------------------------------------
def compute_T_derivative(tot_heat_flux_0, tot_heat_flux_1, dm_dt_0, dm_dt_1, gas):

    gamma = np.sum(dm_dt_0+dm_dt_1) / 2

    if gas.transform_type=="P=cste":
        uh_i = np.array([enth.H_ct(gas.temperature,species_name,gas) for species_name in gas.species_name])
        Cvp = np.array([enth.Cp_ct(gas.temperature,species_name,gas) for species_name in gas.species_name])
        duh_dt = ((tot_heat_flux_0 + tot_heat_flux_1) / 2 - gas.enthalpy * gamma) / gas.density / gas.volume

    else:
        uh_i = np.array([enth.U_ct(gas.temperature,species_name,gas) for species_name in gas.species_name])
        Cvp = np.array([enth.Cv_ct(gas.temperature,species_name,gas) for species_name in gas.species_name])
        duh_dt = ((tot_heat_flux_0 + tot_heat_flux_1) / 2 - gas.energy * gamma) / gas.density / gas.volume

    # dYi_dt = ((dm_dt_0 + dm_dt_1) / 2 - gas.Y * gamma) / gas.density / gas.volume
    dYi_dt = gas.dY_dt
    duhi_dT = Cvp
    dT_dt = (duh_dt - np.sum(dYi_dt * uh_i)) / np.sum(duhi_dT * gas.Y) #celui a utiliser si on se retrouve avec T° négatives

    return dT_dt


#---------------------------------------------------------------------------------------------------------------------------------
def Nb_part(alpha_p, pAl_richness, Y_Al_pAl, volume, m_pAl, m_pMeO_core, r_ext_pAl, r_ext_pMeO, MW_dic):
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
    MW_Al = MW_dic['Al']
    MW_CuO = MW_dic['Cu'] + MW_dic['O']
    m_pMeO = m_pMeO_core
    Stoech_mass = Stoech_mol * MW_Al / MW_CuO / Y_Al_pAl
    n_pAl = (alpha_p*(pAl_richness*Stoech_mass*volume*m_pMeO)/(m_pAl*V_sphere(r_ext_pMeO)))/(1+pAl_richness*m_pMeO*Stoech_mass*V_sphere(r_ext_pAl)/(m_pAl*V_sphere(r_ext_pMeO)))
    n_pMeO = (alpha_p*volume - V_sphere(r_ext_pAl)*n_pAl)/V_sphere(r_ext_pMeO)
    alpha_p = (V_sphere(r_ext_pAl)*int(n_pAl) + V_sphere(r_ext_pMeO)*int(n_pMeO))/volume

    return int(n_pAl), int(n_pMeO), alpha_p


#---------------------------------------------------------------------------------------------------------------------------------
def compute_conduc_pp(p_Al, p_MeO, gas):
    N_coord_points_ij = 12/2
    conduc_per_coord_point = compute_conduc_per_coord_point(p_Al, p_MeO, gas)
    H_cond = p_Al.n * N_coord_points_ij * conduc_per_coord_point
    Q_cond = H_cond * (p_Al.temperature - p_MeO.temperature)
    return Q_cond, H_cond


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
    lambda_g = gas.gas_ct.thermal_conductivity

    lambda_s = lambda_i * lambda_j / (lambda_i + lambda_j) # hypothèse sur le calcul de lambda_s equivalent
    alpha = lambda_s / lambda_g
    R_ij = 2 * r_i * r_j / (r_i + r_j)                      # Rayon équivalent
    KHI = 0.71                                              # Rapport du rayon du cylindre à travers lequel passe le flux sur le rayon équivalent (0.71: Moscardini et Al, 2018)
    R_e_ij = KHI * R_ij
    Beta_ij = alpha * r_c / R_ij

    C_ct_ij_s = compute_C_ct_s(Beta_ij, R_ij, R_e_ij, lambda_g, alpha, overlap)# Conductance inter-particule par contact ou gap
    C_i_s = np.pi * lambda_i * R_e_ij**2 / r_i              # Conductance dans la particule i (hypothese sur lambda_i != lambda_j)
    C_j_s = np.pi * lambda_j * R_e_ij**2 / r_j              # Conductance dans la particule j (hypothese sur lambda_i != lambda_j)
    C_ij = 1/ (1/C_i_s + 1/C_ct_ij_s + 1/C_j_s)             # Conductance équivalente
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
    r_c = Kc * min(r_i, r_j)

    return r_c


#---------------------------------------------------------------------------------------------------------------------------------
def compute_C_ct_s(Beta_ij, R_ij, R_e_ij, lambda_g, alpha, overlap):
    h_ij = np.nan
    if overlap :
        if Beta_ij > 100 :
            C_ct = Beta_ij_sup_100(Beta_ij, R_ij, lambda_g, alpha)
        elif Beta_ij < 1 :
            C_ct = Beta_ij_inf_1(Beta_ij, R_ij, lambda_g, alpha)
        else:                               # Sinon, on fait l'interpolation entre les deux valeurs: Moscardini, 2018
            C_ct = np.interp(Beta_ij,[1,100],[Beta_ij_inf_1(1, R_ij, lambda_g, alpha), Beta_ij_sup_100(100, R_ij, lambda_g, alpha)])
    else:
        ksi = alpha**2 * h_ij / R_ij
        if ksi < 0.1:
            C_ct = np.pi * lambda_g * R_ij * np.log(alpha)
        else:
            C_ct = np.pi * lambda_g * R_ij * np.log(1 + R_e_ij**2/R_ij/h_ij)
    return C_ct


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


#-------------------------------------------------Dodds
def compute_coord_points_ij(p_Al, p_MeO):
    X_0 = p_Al.n/(p_Al.n + p_MeO.n)
    # A = np.array([2,2,2,2])
    
    r = np.array([p_Al.r_ext, p_MeO.r_ext])
    # A_0000 = compute_Ai_jkl(r, 0, 0, 0, 0) # tetra full alu
    # A_0001 = compute_Ai_jkl(r, 0, 0, 0, 1) # tetra 3 Alu + 1 MeO
    # A_0011 = compute_Ai_jkl(r, 0, 0, 1, 1) # tetra 2 Alu + 2 MeO
    # A_0111 = compute_Ai_jkl(r, 0, 1, 1, 1) # tetre 1 Alu + 3 MeO
    #   
    # A_1111 = compute_Ai_jkl(r, 1, 
    # 1, 1, 1) # tretra full MeO

    A = np.array([[[[compute_Ai_jkl(r, i, j, k, l) for i in [0,1]] for j in [0,1]] for k in [0,1]] for l in [0,1]])

    # Y_s_= scipy.optimize.bisect(func_Y_s, 0, 1, args=(Ap,Ag,Bp,Bg,d,phi_Ox_condens,))
    # A_bar = scipy.optimize.root(compute_A_i_bar_root_bis, [0.2,0.2], args=(X_0, A,)).x
    A_bar = scipy.optimize.root(compute_A_i_bar_root_bis, [0.01,0.01], args=(X_0, A,)).x
    F = [X_0 / (1-X_0), 1] / A_bar
    # A_bar = scipy.optimize.newton(compute_A_i_bar, x0=[0.5, 0.5], args=(X_0, A), tol=1e-8, maxiter = 10000)
    N_bar = 1/A_bar
    tot_coord_points = (1/2/A_bar + 2)# / 2 # divisé par 2 car le packing n'est pas optimal: Dans un packing optimal: N = 12. Dans un packing aléatoire: N = 12
    list_tetra = ['1111,1112,1122,1222,2222']
    H = np.array([[[[compute_H(r, i, j, k, l) for i in [0,1]] for j in [0,1]] for k in [0,1]] for l in [0,1]])
    H_12 = [0, H[0,0,0,1], H[1,0,0,1], H[1,1,0,1], 0]
    H_12_freq = [0, 3, 4, 3, 0]
    H_tot_12, H_bar_12 = compute_H_bar(F, H_12)
    pij = compute_sum_pij(A_bar, X_0, A, H_tot_12)

    return "a definir"

def compute_Ai_jkl(r,i,j,k,l): # r est un vecteur de taille 2 (i=0: pAl, i=1: pMeO)
    # cos_a = 2*r[i]*(r[i]+r[j]+r[k])/(r[i]+r[j])/(r[i]+r[l])-1
    # cos_b = 2*r[i]*(r[i]+r[j]+r[l])/(r[i]+r[k])/(r[i]+r[l])-1
    # cos_c = 2*r[i]*(r[i]+r[l]+r[k])/(r[i]+r[j])/(r[i]+r[k])-1

    cos_a = 2*r[i]*(r[i]+r[k]+r[j])/(r[i]+r[k])/(r[i]+r[j])-1
    cos_b = 2*r[i]*(r[i]+r[j]+r[l])/(r[i]+r[j])/(r[i]+r[l])-1
    cos_c = 2*r[i]*(r[i]+r[l]+r[k])/(r[i]+r[l])/(r[i]+r[k])-1
    
    print(np.arccos(cos_a))
    print(np.arccos(cos_b))
    print(np.arccos(cos_c))

    sin_a = (1-cos_a**2)**0.5
    sin_b = (1-cos_b**2)**0.5
    sin_c = (1-cos_c**2)**0.5

    cos_alpha = (cos_a - (cos_b * cos_c)) / (sin_b * sin_c)
    cos_beta  = (cos_b - (cos_c * cos_a)) / (sin_c * sin_a)
    cos_gamma = (cos_c - (cos_a * cos_b)) / (sin_a * sin_b)

    # Ai_jkl = 4 * np. pi /(np.arccos(cos_alpha) + np.arccos(cos_beta) + np.arccos(cos_gamma) - np.pi)
    Ai_jkl = (np.arccos(cos_alpha) + np.arccos(cos_beta) + np.arccos(cos_gamma) - np.pi) / (4 * np. pi)

    return Ai_jkl


def compute_A_i_bar_root(A_bar,X_0,A):
    A_0 = A_bar[0]
    A_1 = A_bar[1]
    X_1 = 1 - X_0
    # f1 = A_0 - ((A[0,0,0,0] * k_i(X_0,A_0)**3 + 3*A[0,0,0,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1) + 3*A[0,0,1,1] * k_i(X_0,A_0) * k_i(X_1,A_1)**2 + A[0,1,1,1]*k_i(X_1,A_1)**3)/(k_i(X_0,A_0) + k_i(X_1,A_1))**3)
    # f2 = A_1 - ((A[1,0,0,0] * k_i(X_0,A_0)**3 + 3*A[1,0,0,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1) + 3*A[1,0,1,1] * k_i(X_0,A_0) * k_i(X_1,A_1)**2 + A[1,1,1,1]*k_i(X_1,A_1)**3)/(k_i(X_0,A_0) + k_i(X_1,A_1))**3)


    f1 = A_0 - ((A[0,0,0,0] * k_i(X_0,A_0)**3
             + 3*A[0,0,0,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1) 
             + 3*A[0,0,1,1] * k_i(X_0,A_0) * k_i(X_1,A_1)**2 
             + A[0,1,1,1]*k_i(X_1,A_1)**3)
             /(k_i(X_0,A_0) + k_i(X_1,A_1))**3)


    f2 = A_1 - ((A[1,0,0,0] * k_i(X_0,A_0)**3 
            + 3*A[1,0,0,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1) 
            + 3*A[1,0,1,1] * k_i(X_0,A_0) * k_i(X_1,A_1)**2 
            + A[1,1,1,1]*k_i(X_1,A_1)**3)
            /(k_i(X_0,A_0) + k_i(X_1,A_1))**3)


    # f1 = A_0 - ((A[0,0,0,0] * k_i(X_0,A_0)**3
    #           + (A[0,0,0,1] + A[0,0,1,0] + A[0,1,0,0]) * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**1
    #           + (A[0,0,1,1] + A[0,1,0,1] + A[0,1,1,0]) * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**2
    #            + A[0,1,1,1]*k_i(X_1,A_1)**3)/
    #    (k_i(X_0,A_0) + k_i(X_1,A_1))**3)

    # f2 = A_1 - ((A[1,0,0,0] * k_i(X_0,A_0)**3
    #           + (A[1,0,0,1] + A[1,0,1,0] + A[1,1,0,0]) * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**1
    #           + (A[1,0,1,1] + A[1,1,0,1] + A[1,1,1,0]) * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**2
    #            + A[1,1,1,1]*k_i(X_1,A_1)**3)/
    #    (k_i(X_0,A_0) + k_i(X_1,A_1))**3)
    

    # f1 = A_0 - ((A[0,0,0,0] * k_i(X_0,A_0)**3
    #           + (A[0,0,1,0]) * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**1
    #           + (A[0,1,1,0]) * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**2
    #            + A[0,1,1,1]*k_i(X_1,A_1)**3)/
    #    (k_i(X_0,A_0) + k_i(X_1,A_1))**3)

    # f2 = A_1 - ((A[1,0,0,0] * k_i(X_0,A_0)**3
    #           + (A[1,0,1,0]) * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**1
    #           + (A[1,1,1,0]) * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**2
    #            + A[1,1,1,1]*k_i(X_1,A_1)**3)/
    #    (k_i(X_0,A_0) + k_i(X_1,A_1))**3)


    # f1 = A_0 - ((1*A[0,0,0,0] * k_i(X_0,A_0)**4
    #            + 4*A[0,0,0,1] * k_i(X_0,A_0)**3 * k_i(X_1,A_1)
    #            + 6*A[0,0,1,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**2
    #            + 4*A[0,1,1,1] * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**3)/
    #         (k_i(X_0,A_0) + k_i(X_1,A_1))**4)

    # f2 = A_1 - ((4*A[1,0,0,0] * k_i(X_0,A_0)**3 * k_i(X_1,A_1) 
    #            + 6*A[1,0,0,1] * k_i(X_0,A_0)**2 * k_i(X_1,A_1)**2
    #            + 4*A[1,0,1,1] * k_i(X_0,A_0)**1 * k_i(X_1,A_1)**3
    #            + 1*A[1,1,1,1]*k_i(X_1,A_1)**4)/
    #    (k_i(X_0,A_0) + k_i(X_1,A_1))**4)

    return f1, f2


def k_i(X_i,A_i):
    return X_i/A_i

# def compute_tot_numb_of_tetra(A_bar):
#     return np.sum(A_bar)**4

# def compute_freq_per_tetra()

def compute_A_i_bar_root_bis(A_bar, X_0, A):
    # f     = [nb_p1 / nb_p2, 1]
    f     = [X_0 / (1-X_0), 1]
    print(f)
    N_bar = 1/A_bar          # eq 4 Hogendijk
    F     = [f[0] * N_bar[0] , f[1] * N_bar[1]] # eq 1 Hogendijk

    list_tetra = ['1111,1112,1122,1222,2222']

    A_0 = np.array([A[0,0,0,0],
    A[1,0,0,0],
    A[1,1,0,0],
    A[1,1,1,0],
    0])

    A_1 = np.array([0,
    A[0,0,0,1],
    A[1,0,0,1],
    A[1,1,0,1],
    A[1,1,1,1]
    ])

    n_per_tet_0 = np.array([4, 3, 2, 1, 0])
    n_per_tet_1 = np.array([0, 1, 2, 3, 4])
    tri_Pas     = np.array([1, 4, 6, 4, 1])

    A_0_bar = ((n_per_tet_0[0] * A_0[0] * F[0]**n_per_tet_0[0] * F[1]**n_per_tet_1[0] * tri_Pas[0]
              + n_per_tet_0[1] * A_0[1] * F[0]**n_per_tet_0[1] * F[1]**n_per_tet_1[1] * tri_Pas[1]
              + n_per_tet_0[2] * A_0[2] * F[0]**n_per_tet_0[2] * F[1]**n_per_tet_1[2] * tri_Pas[2]
              + n_per_tet_0[3] * A_0[3] * F[0]**n_per_tet_0[3] * F[1]**n_per_tet_1[3] * tri_Pas[3]
              + n_per_tet_0[4] * A_0[4] * F[0]**n_per_tet_0[4] * F[1]**n_per_tet_1[4] * tri_Pas[4]
             )
             /np.sum(n_per_tet_0 * F[0]**n_per_tet_0 * F[1]**n_per_tet_1 * tri_Pas))

    A_1_bar = ((n_per_tet_1[0] * A_1[0] * F[0]**n_per_tet_0[0] * F[1]**n_per_tet_1[0] * tri_Pas[0]
              + n_per_tet_1[1] * A_1[1] * F[0]**n_per_tet_0[1] * F[1]**n_per_tet_1[1] * tri_Pas[1]
              + n_per_tet_1[2] * A_1[2] * F[0]**n_per_tet_0[2] * F[1]**n_per_tet_1[2] * tri_Pas[2]
              + n_per_tet_1[3] * A_1[3] * F[0]**n_per_tet_0[3] * F[1]**n_per_tet_1[3] * tri_Pas[3]
              + n_per_tet_1[4] * A_1[4] * F[0]**n_per_tet_0[4] * F[1]**n_per_tet_1[4] * tri_Pas[4]
             )
             /np.sum(n_per_tet_1 * F[0]**n_per_tet_0 * F[1]**n_per_tet_1 * tri_Pas))

    return (A_bar[0] - A_0_bar) , (A_bar[1] - A_1_bar)

    # def compute P_ij():



def compute_H(r,i,j,k,l): # r est un vecteur de taille 2 (i=0: pAl, i=1: pMeO)
    a =np.arccos(2*r[i]*(r[i]+r[k]+r[j])/(r[i]+r[k])/(r[i]+r[j])-1)
    b =np.arccos(2*r[i]*(r[i]+r[j]+r[l])/(r[i]+r[j])/(r[i]+r[l])-1)
    c =np.arccos(2*r[i]*(r[i]+r[l]+r[k])/(r[i]+r[l])/(r[i]+r[k])-1)
    S_i = 0.5 * (a + b + c)
    A = np.sin(S_i - a)
    B = np.sin(S_i - b)
    C = np.sin(S_i) * np.sin(S_i - c)
    H = np.arctan((A * B / C)**0.5)*2 / 2 / np.pi # d'où vient le /2/pi ? ==> Hogendijk P121
    return H

def compute_H_bar(F, H_12):
    exp_coeff_0 = np.array([4, 3, 2, 1, 0])
    exp_coeff_1 = np.array([0, 1, 2, 3, 4])
    tri_Pas     = np.array([1, 4, 6, 4, 1])

    H_12_nb = [0, 3, 4, 3, 0] # Hodjenik P121
    H_tot_12 = (H_12[0] * H_12_nb[0] * F[0]**exp_coeff_0[0] * F[1]**exp_coeff_1[0] * tri_Pas[0]
                +H_12[1] * H_12_nb[1] * F[0]**exp_coeff_0[1] * F[1]**exp_coeff_1[1] * tri_Pas[1]
                +H_12[2] * H_12_nb[2] * F[0]**exp_coeff_0[2] * F[1]**exp_coeff_1[2] * tri_Pas[2]
                +H_12[3] * H_12_nb[3] * F[0]**exp_coeff_0[3] * F[1]**exp_coeff_1[3] * tri_Pas[3]
                +H_12[4] * H_12_nb[4] * F[0]**exp_coeff_0[4] * F[1]**exp_coeff_1[4] * tri_Pas[4]
                )

    H_bar_12 = H_tot_12 / np.sum(H_12_nb * F[0]**exp_coeff_0 * F[1]**exp_coeff_1 * tri_Pas)
    return H_tot_12, H_bar_12

def compute_sum_pij(A_bar, X_0, A, H_tot_12):
    # f     = [nb_p1 / nb_p2, 1]
    f     = [X_0 / (1-X_0), 1]
    N_bar = 1/A_bar          # eq 4 Hogendijk
    F     = [f[0] * N_bar[0] , f[1] * N_bar[1]] # eq 1 Hogendijk

    list_tetra = ['1111,1112,1122,1222,2222']

    A_0 = np.array([A[0,0,0,0],
    A[1,0,0,0],
    A[1,1,0,0],
    A[1,1,1,0],
    0])

    A_1 = np.array([0,
    A[0,0,0,1],
    A[1,0,0,1],
    A[1,1,0,1],
    A[1,1,1,1]
    ])

    n_per_tet_0 = np.array([4, 3, 2, 1, 0])
    n_per_tet_1 = np.array([0, 1, 2, 3, 4])
    tri_Pas     = np.array([1, 4, 6, 4, 1])
    m = 0
    m =      (n_per_tet_0[0] * A_0[0] * F[0]**n_per_tet_0[0] * F[1]**n_per_tet_1[0] * tri_Pas[0]
              + n_per_tet_0[1] * A_0[1] * F[0]**n_per_tet_0[1] * F[1]**n_per_tet_1[1] * tri_Pas[1]
              + n_per_tet_0[2] * A_0[2] * F[0]**n_per_tet_0[2] * F[1]**n_per_tet_1[2] * tri_Pas[2]
              + n_per_tet_0[3] * A_0[3] * F[0]**n_per_tet_0[3] * F[1]**n_per_tet_1[3] * tri_Pas[3]
              + n_per_tet_0[4] * A_0[4] * F[0]**n_per_tet_0[4] * F[1]**n_per_tet_1[4] * tri_Pas[4])
  
    m =  m + (n_per_tet_1[0] * A_1[0] * F[0]**n_per_tet_0[0] * F[1]**n_per_tet_1[0] * tri_Pas[0]
              + n_per_tet_1[1] * A_1[1] * F[0]**n_per_tet_0[1] * F[1]**n_per_tet_1[1] * tri_Pas[1]
              + n_per_tet_1[2] * A_1[2] * F[0]**n_per_tet_0[2] * F[1]**n_per_tet_1[2] * tri_Pas[2]
              + n_per_tet_1[3] * A_1[3] * F[0]**n_per_tet_0[3] * F[1]**n_per_tet_1[3] * tri_Pas[3]
              + n_per_tet_1[4] * A_1[4] * F[0]**n_per_tet_0[4] * F[1]**n_per_tet_1[4] * tri_Pas[4])

    nb_tot_tetra = (np.sum(n_per_tet_0 * F[0]**n_per_tet_0 * F[1]**n_per_tet_1 * tri_Pas)
                 + np.sum(n_per_tet_1 * F[0]**n_per_tet_0 * F[1]**n_per_tet_1 * tri_Pas))
    p_12 = H_tot_12/(nb_tot_tetra+m)
    return p_12





def kinetics(gas, volume, dt):
    """
    La cinétique est calculée en considérant un alpha_g constant
    C'est la raison pour laquelle l'état du gas en fin de cinétique
    diffère légèrement de l'état du gaz en fin d'itération
    old: cinétique Euler remplacée par BDF optimisé
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
    return dm_cin_dt, int_BDF


def dt_cin(phi, wdot, rho, alpha, volume, Y, tau, dt):
    A=np.where((phi+wdot)<0, (-(rho*alpha*volume*Y)/(phi+wdot)+1e-50) / 1e5, dt - tau)
    min_dt = dt / 1000
    return np.min((np.min(A), min_dt))



def estim_first_wdot_for_dt_determin(gas, volume, dm_dt_0, dm_dt_1):

    phi = (dm_dt_0 + dm_dt_1)/2                         # flux d'espèce provenant des échanges interphase sur une itération # (kg/s)

    # in fine: dmi_dt_tot = dm_cin_tot_dt + phi

    dm_cin_dt = np.zeros(gas.nb_species)                # variation temporelle finale de l'espece "i" sur une itération # (kg/s)
    density = gas.density                               # densité du gaz (kg/m3)
    energy  = gas.energy                                # energie massique du gas (J/kg)
    phi_ct  = np.zeros(gas.nb_species)
    Y_ct    = np.zeros(gas.nb_species)

    # passage de l'indexation maison à celle de cantera pour les flux et Y d'espèces
    for i,x in enumerate(gas.gas_ct.species_names):
        phi_ct[i] = phi[gas.species_name.index(x)]
        Y_ct[i]   = gas.Y[gas.species_name.index(x)]

    # entrée des grandeurs dans cantera
    gas.gas_ct.UVY = energy, 1/density, Y_ct        # (J/kg), (m3/kg), (kg/kg)
    # Calcul des taux de réaction via cantera
    wdot = gas.gas_ct.net_production_rates * gas.gas_ct.molecular_weights * gas.volume # (kg/s)

    for i,x in enumerate(gas.species_name):
        dm_cin_dt[i]=wdot[gas.gas_ct.species_index(x)]
    
    return dm_cin_dt



#---------------------------------------------------------------------------------------------------------------------------------
def Thermal_flux_relative_to_mass_flux_computation(flux_p,source_p,particle,gas, ebull_dic = None):
    flux_per_species=np.zeros([gas.nb_species,2])
    decomp_Al2O3_species_flux=np.zeros(2)
    add_heat_flux_AlxOy=np.zeros([2])
#"O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"

    if ebull_dic == None:
        ebull_dic = dict(Al = False, Me = False)

    if gas.transform_type=="P=cste":
        UH_ct=enth.H_ct
        UH_tot=enth.H_tot

    if gas.transform_type=="V=cste":
        UH_ct=enth.U_ct
        UH_tot=enth.U_tot

    # Les réactions surfaciques ont elles davantage lieu dans la phase gaz?
    # Attention: si True, un déséquilibre thermique en fin de réaction peut apparaitre
    bool_react_gas = False

    for i in range (0,2):
        if particle.name=="p_Al":
            flux_per_species[0,i]=(UH_ct(T_inter(gas, particle, flux_p[i].O2_ext),"O2",gas))*flux_p[i].O2_ext    #O2
            flux_per_species[2,i]=(UH_ct(T_inter(gas, particle, flux_p[i].O_ext),"O",gas))*flux_p[i].O_ext    #O

        elif particle.name=="p_MeO":
            flux_per_species[0,i]=(UH_ct(T_inter(gas, particle, flux_p[i].O2_ext),"O2",gas))*flux_p[i].O2_ext    #O2

        if not ebull_dic["Al"]:
            flux_per_species[1,i]=(UH_ct(T_inter(gas, particle, flux_p[i].Al_ext),"Al",gas))*flux_p[i].Al_ext#+Al.h_evap    #Al
        else:
            flux_per_species[1,i] = 0

        if not ebull_dic["Me"]:
            flux_per_species[8,i]=(UH_ct(T_inter(gas, particle, flux_p[i].Me),"Cu",gas))*flux_p[i].Me    #Me
        else:
            flux_per_species[8,i] = 0

        #additional heat flux related to AlxOy condensation
        add_heat_flux_AlxOy[i] = heat_flux_AlxOy(i,UH_ct,UH_tot,gas,particle,flux_p, bool_react_gas)

        #additional heat flux related to Al2O3 decomposition
        decomp_Al2O3_species_flux[i] = heat_flux_decomp(i,UH_ct,UH_tot,gas,particle, source_p, bool_react_gas)

    return np.sum(flux_per_species,axis=0)[1]+add_heat_flux_AlxOy[1]+decomp_Al2O3_species_flux[1]


def heat_flux_AlxOy(i,UH_ct,UH_tot,gas,particle,flux_p, bool_react_gas):             #additional heat flux related to AlxOy condensation
    if bool_react_gas: # si l'enthalpie transportée par les transferts est prise dans la phase gaz (attention, apparition d'un déséquilibre thermique)
        add_heat_flux_AlxOy=((UH_ct(T_inter(gas, particle, flux_p[i].AlxOy[4]),gas.species_name[4],gas))*flux_p[i].AlxOy[4]    #Al2O
                            +(UH_ct(T_inter(gas, particle, flux_p[i].AlxOy[5]),gas.species_name[5],gas))*flux_p[i].AlxOy[5]    #AlO
                            +(UH_ct(T_inter(gas, particle, flux_p[i].AlxOy[6]),gas.species_name[6],gas))*flux_p[i].AlxOy[6]    #AlO2
                            +(UH_ct(T_inter(gas, particle, flux_p[i].AlxOy[7]),gas.species_name[7],gas))*flux_p[i].AlxOy[7]    #Al2O2
                            +(UH_ct(T_inter(gas, particle, flux_p[i].O2_condens),gas.species_name[0],gas))*flux_p[i].O2_condens
                            +(UH_ct(gas.temperature, particle.species_name[0],gas))*flux_p[i].Al_condens
                            )
        add_heat_flux_AlxOy=UH_tot(gas.temperature,particle.species_carac[3],1) * flux_p[i].Al2O3_condens
    else:
        add_heat_flux_AlxOy=UH_tot(particle.temperature,particle.species_carac[3],1) * flux_p[i].Al2O3_condens
    return add_heat_flux_AlxOy


def heat_flux_decomp(i,UH_ct,UH_tot,gas,particle, source_p, bool_react_gas): #addition of species thermal flux related to Al2O3 decomposition
    if bool_react_gas: # si l'enthalpie transportée par les transferts est prise dans la phase gaz (attention, apparition d'un déséquilibre thermique)
        decomp_Al2O3_species_flux = 0
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].O2_decomp),"O2",gas))*source_p[i].O2_decomp        #O2
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].Al_decomp),"Al",gas))*source_p[i].Al_decomp        #Al
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].O_decomp),"O",gas))*source_p[i].O_decomp        #O
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].Al2O_decomp),"Al2O",gas))*source_p[i].Al2O_decomp        #Al2O
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].AlO_decomp),"AlO",gas))*source_p[i].AlO_decomp        #AlO
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].AlO2_decomp),"AlO2",gas))*source_p[i].AlO2_decomp        #AlO2
        decomp_Al2O3_species_flux= - decomp_Al2O3_species_flux + (UH_ct(T_inter(gas, particle, source_p[i].Al2O2_decomp),"Al2O2",gas))*source_p[i].Al2O2_decomp        #Al2O2

    else:
        # remarque:
        # attetion au signe "-" car la convention des termes sources sur la particule est "la quantité augmente sur la particule
        # quand le terme source est positif..."
        decomp_Al2O3_species_flux = source_p[i].Al2O3_decomp*UH_tot(particle.temperature,particle.species_carac[3],1)
    return decomp_Al2O3_species_flux



