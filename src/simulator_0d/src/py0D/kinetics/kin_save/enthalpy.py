import os
import sys

import numpy as np
from scipy import optimize

PATH_0D = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..','..'))
sys.path.append(PATH_0D)


import py0D.global_data as gd

#---------------------------------------------------------------------------------------------------------------------------------
def find_p_temperature(particule,gas):
    #Cette fonction a pour but de trouver la temperature de la particule à partir de l'enthalpie de celle-ci en utilisant la méthode de Newton appliquée à l'équation de shomate.
    #Pour cela, on determine d'abord quel jeu de coefficient de shomate on va utiliser en fonction de la phase des espèces dans la particule. L'etat des différentes espèces est ensuite
    #recalculé une fois la temperature trouvée via la fonction"find_phase".
    #Une fois qu'on connait quel jeu de coefficient on va utiliser, on résout l'équation: 0=H_tot-H_sensible(T)-H_formation
    phase, H_s_PC2=find_phase(particule,particule.temperature,gas)

    test_CP=False

    if particule.phase[0]=="PC1":
        test_CP=True
        T=particule.species_carac[0].T_liq

    if particule.phase[2]=="PC1":
        test_CP=True
        T=particule.species_carac[2].T_liq

    if particule.phase[3]=="PC1":
        test_CP=True
        T=particule.species_carac[3].T_liq


    if test_CP is False:
        T = optimize.newton(f_enthalpy_p, particule.temperature, args = (particule,))

    cp_i_vec = uCv_i_p_vec(T)[1]
    cp_part = np.sum(cp_i_vec * particule.composition) / np.sum(particule.composition)
    return T,phase, H_s_PC2, cp_part


#---------------------------------------------------------------------------------------------------------------------------------
def f_enthalpy_p(T, particle):
    #cette fonction est utilisée dans la recherche de la temperature de la particule p_Al en connaissant l'enthalpie de celle-ci. cf: "find_p_temeprature"
    #En effet, sachant que l'enthalpie totale de la particule s'ecrit "H_tot=H_sensible(T)+H_formation", où H_sensible est fonction de la temperature,
    #il suffit de résoudre l'équation "f=0=H_tot-H_sensible(T)-H_formation" pour trouver la temperature à partir de l'enthalpie totale de la particule.
    f=particle.enthalpy - np.sum(particle.composition * [H_tot(T,species,find_phase_index(phase_index)) for species, phase_index in zip(particle.species_carac, particle.phase)])
    return f


#---------------------------------------------------------------------------------------------------------------------------------
def compute_Cp(T,species,phase_index):
    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    # F=species.shomate[phase_index,5]
    # H=species.shomate[phase_index,6]
    t=T/1000
    Cp=A+B*t+C*t**2+D*t**3+E/t**2
    return Cp


#---------------------------------------------------------------------------------------------------------------------------------
def DELTA_H(T,species,phase_index):
    # cette fonction permet d'obtenir l'enthalpie sensible d'une espèce par unité massique en fonction de la temperature. l'index de phase est obtenu par une autre fonction
    # et défini quel jeu de coefficients de shomate on va utiliser. En effet, les coefficients de shomate sont diférrents en fonction de la phase de l'espèce (solide/liquide/gaz)
    # et peuvent également varier dans le cas d'un gaz en fonction de la temperature.
    # pour determiner l'index de phase pour les espèces en phase condensée on utilise la fonction "find_phase"

    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    F=species.shomate[phase_index,5]
    H=species.shomate[phase_index,6]
    t=T/1000
    DELTA_H=(A*t+B*t**2/2+C*t**3/3+D*t**4/4-E/t+F-H)
    return DELTA_H

#
def DELTA_U(T,species,phase_index):
    # cette fonction permet d'obtenir l'enthalpie sensible d'une espèce par unité massique en fonction de la temperature. l'index de phase est obtenu par une autre fonction
    # et défini quel jeu de coefficients de shomate on va utiliser. En effet, les coefficients de shomate sont diférrents en fonction de la phase de l'espèce (solide/liquide/gaz)
    # et peuvent également varier dans le cas d'un gaz en fonction de la temperature.
    # pour determiner l'index de phase pour les espèces en phase condensée on utilise la fonction "find_phase"
    R=8.314462
    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    F=species.shomate[phase_index,5]
    H=species.shomate[phase_index,6]
    t=T/1000
    DELTA_U=(A*t+B*t**2/2+C*t**3/3+D*t**4/4-E/t+F-H)-R*T/species.MW
    return DELTA_U


#----------------------------------------------------------------------------------------------------------------------------------
def c_p(T,species,phase_index):
    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    t=T/1000
    c_p=(A+B*t+C*t**2+D*t**3+E/t**2)/1000
    return c_p


def c_v(T,species,phase_index):
    R=8.314462
    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    t=T/1000
    c_v=(A+B*t+C*t**2+D*t**3+E/t**2)/1000-R/species.MW
    return c_v


#---------------------------------------------------------------------------------------------------------------------------------
def DELTA_H_mol(T,species,phase_index):
    # cette fonction n'est pas utilisée dans le programme. Elle sert simplement de vérification pour obtenir les valeurs d'enthalpie sensible par unité molaire
    # d'une espèce en fonction de la temperature
    A=species.shomate[phase_index,0]
    B=species.shomate[phase_index,1]
    C=species.shomate[phase_index,2]
    D=species.shomate[phase_index,3]
    E=species.shomate[phase_index,4]
    F=species.shomate[phase_index,5]
    H=species.shomate[phase_index,6]
    t=T/1000
    DELTA_H=(A*t+B*t**2/2+C*t**3/3+D*t**4/4-E/t+F-H)/1000*species.MW
    return DELTA_H


#---------------------------------------------------------------------------------------------------------------------------------
def H_ct(T,species_name,gas):
    if species_name == "Me":
        species_name = "Cu"
    species_index = gas.gas_ct.species_index(species_name)
    h_mol = gas.gas_ct.species()[species_index].thermo.h(T)/1000
    molecular_weights = gas.gas_ct.molecular_weights[species_index]/1000
    h_ct = h_mol / molecular_weights  # en J/kg
    return h_ct


#---------------------------------------------------------------------------------------------------------------------------------
def U_ct(T,species_name,gas):
    if species_name == "Me":
        species_name = "Cu"
    R=8.31446261815324
    species_index = gas.gas_ct.species_index(species_name)
    u_mol = gas.gas_ct.species()[species_index].thermo.h(T)/1000-R*T
    molecular_weights = gas.gas_ct.molecular_weights[species_index]/1000
    u_ct = u_mol / molecular_weights  # en J/kg
    return u_ct


#---------------------------------------------------------------------------------------------------------------------------------
def Cp_ct(T,species_name,gas):
    species_index = gas.gas_ct.species_index(species_name)
    cp_mol = gas.gas_ct.species()[species_index].thermo.cp(T)/1000
    molecular_weights = gas.gas_ct.molecular_weights[species_index]/1000
    cp_ct = cp_mol / molecular_weights  # en J/kg/K
    return cp_ct


#---------------------------------------------------------------------------------------------------------------------------------
def Cv_ct(T,species_name,gas):
    R=8.31446261815324
    species_index = gas.gas_ct.species_index(species_name)
    cv_mol = gas.gas_ct.species()[species_index].thermo.cp(T)/1000-R
    molecular_weights = gas.gas_ct.molecular_weights[species_index]/1000
    cv_ct = cv_mol / molecular_weights  # en J/kg/K
    return cv_ct


#---------------------------------------------------------------------------------------------------------------------------------
def find_phase(particule,T,gas):
    # Cette fonction a pour but de trouver la phase (solide ou liquide) de chacune des espèces dans une particule.
    # Elle renvoie alors un tableau de type ["s","s","s","s"] ou "s" signifie solide, "l" liquide, et "PC" signifie que l'espèce est en cours de changement de phase.
    # Une particule a plusieurs espèces dont on connait la masse.
    # En connaissant la masse, on peut retrouver les deux valeurs d'enthalpie entre lesquelles s'effectue le changement de phase.
    # De cette manière on peut retrouver l'etat de chacune des espèces présente dans la particule.

    phase=particule.phase
    H_p=particule.enthalpy
    m_Al=particule.composition[0]
    m_MeO=particule.composition[1]
    m_Me=particule.composition[2]
    m_Al2O3=particule.composition[3]
    Al=particule.species_carac[0]
    MeO=particule.species_carac[1]
    Me=particule.species_carac[2]
    Al2O3=particule.species_carac[3]

#------------------------------------------------------------changement de phase Al
    if m_Al>1e-10:
        H_s_Al=(H_p)
        H_s_PC1_Al_min=H_tot(Al.T_liq,Al,0)*m_Al+m_Al2O3*H_tot(Al.T_liq,Al2O3,0)+m_Me*H_tot(Al.T_liq,Me,0)+m_MeO*H_tot(Al.T_liq, MeO, 0)#indice 0 car l'al2O3 est solide lors du CP de l'Al. On utilise donc les coeff solides
        H_s_PC1_Al_max=(H_tot(Al.T_liq,Al,0)+Al.h_liq)*m_Al+m_Al2O3*H_tot(Al.T_liq,Al2O3,0)+m_Me*H_tot(Al.T_liq,Me,0)+m_MeO*H_tot(Al.T_liq, MeO, 0)

        if (H_s_Al>H_s_PC1_Al_min and H_s_Al<H_s_PC1_Al_max):
            phase[0]="PC1"
        elif H_s_Al>H_s_PC1_Al_max:
            phase[0]="l"
            T_evap=Al.T_evap
            if particule.temperature>Al.T_evap*0.95:

                T_evap=max(Al.T_evap,np.interp(gas.pressure,gas.T_P_Al_evap[1],gas.T_P_Al_evap[0]))
                # print(T_evap,Al.T_evap)
            H_s_PC2_Al_min=H_tot(T_evap,Al,1)*m_Al+m_Al2O3*H_tot(T_evap,Al2O3,1)+m_Me*H_tot(T_evap,Me,1)+m_MeO*H_tot(T_evap, MeO, 0)#indice 1 car l'al2O3 est liquide lors du CP de l'Al. On utilise donc les coeff liquides
            H_s_PC2_Al_max=(H_tot(T_evap,Al,1)+Al.h_evap)*m_Al+m_Al2O3*H_tot(T_evap,Al2O3,1)+m_Me*H_tot(T_evap,Me,1)+m_MeO*H_tot(T_evap, MeO, 0)
            # save_data_time(H_s_PC2_Al_min,"min_h")
            if (H_s_Al>H_s_PC2_Al_min and H_s_Al<H_s_PC2_Al_max):
                phase[0]="PC2"

        elif H_s_Al<H_s_PC1_Al_min:
            phase[0]="s"
    T_evap=max(Al.T_evap,np.interp(gas.pressure,gas.T_P_Al_evap[1],gas.T_P_Al_evap[0]))
    H_s_PC2_Al_min=H_tot(T_evap,Al,1)*m_Al+m_Al2O3*H_tot(T_evap,Al2O3,1)

#--------------------------------------------------------------changement de phase Al2O3
    if m_Al2O3>1e-10 and T>1000:
        H_s_Al2O3=(H_p)
        H_s_PC_Al2O3_min=H_tot(Al2O3.T_liq,Al2O3,0)*m_Al2O3+m_Al*H_tot(Al2O3.T_liq,Al,1)+m_MeO*H_tot(Al2O3.T_liq,MeO,0)+m_Me*H_tot(Al2O3.T_liq,Me,1) #indice 1 car l'alu est liquide lors du CP de l'Al2O3. On utilise donc les coeff liquides
        H_s_PC_Al2O3_max=(H_tot(Al2O3.T_liq,Al2O3,0)+Al2O3.h_liq)*m_Al2O3+m_Al*H_tot(Al2O3.T_liq,Al,1)+m_MeO*H_tot(Al2O3.T_liq,MeO,0)+m_Me*H_tot(Al2O3.T_liq,Me,1)
        if (H_s_Al2O3>H_s_PC_Al2O3_min and H_s_Al2O3<H_s_PC_Al2O3_max):
            phase[3]="PC1"
        elif H_s_Al2O3>H_s_PC_Al2O3_max:
            phase[3]="l"
        elif H_s_Al2O3<H_s_PC_Al2O3_min:
            phase[3]="s"

#--------------------------------------------------------------changement de phase Me
    if m_Me>1e-10:
        H_s_Me=(H_p)
        H_s_PC1_Me_min=H_tot(Me.T_liq,Me,0)*m_Me+m_Al2O3*H_tot(Me.T_liq,Al2O3,0)+m_MeO*H_tot(Me.T_liq,MeO,0)+m_Al*H_tot(Me.T_liq,Al,1)
        H_s_PC1_Me_max=(H_tot(Me.T_liq,Me,0)+Me.h_liq)*m_Me+m_Al2O3*H_tot(Me.T_liq,Al2O3,0)+m_MeO*H_tot(Me.T_liq,MeO,0)+m_Al*H_tot(Me.T_liq,Al,1)


        if (H_s_Me>H_s_PC1_Me_min and H_s_Me<H_s_PC1_Me_max):
            phase[2]="PC1"
        elif H_s_Me>H_s_PC1_Me_max:
            phase[2]="l"
            T_evap=Me.T_evap
            if particule.temperature>Me.T_evap*0.95:

                T_evap=max(Me.T_evap,np.interp(gas.pressure,gas.T_P_Me_evap[1],gas.T_P_Me_evap[0]))
            H_s_PC2_Me_min=H_tot(T_evap,Me,1)*m_Me+m_Al2O3*H_tot(T_evap,Al2O3,1)+m_MeO*H_tot(T_evap,MeO,0)+m_Al*H_tot(T_evap,Al,1)
            H_s_PC2_Me_max=(H_tot(T_evap,Me,1)+Me.h_evap)*m_Me+m_Al2O3*H_tot(T_evap,Al2O3,1)+m_MeO*H_tot(T_evap,MeO,0)+m_Al*H_tot(T_evap,Al,1)

            if (H_s_Me>H_s_PC2_Me_min and H_s_Me<H_s_PC2_Me_max):
                phase[2]="PC2"

        elif H_s_Me<H_s_PC1_Me_min:
            phase[2]="s"
    if particule.name == "pMeO":
        H_s_PC2 = H_s_PC2_Me_min
    else:
        H_s_PC2 = H_s_PC2_Al_min

    return phase, H_s_PC2


#---------------------------------------------------------------------------------------------------------------------------------
def H_tot(T,species,phase_index):
    # cette fonction calcule l'enthalpie totale d'une espèce "i": H_tot_i=H_formation_i+H_sensible_i
    # H_sensible_i est calculée par la fonction "DELTA_H"
    h=DELTA_H(T,species,phase_index)+species.h_form
    return h


#---------------------------------------------------------------------------------------------------------------------------------
def U_tot(T,species,phase_index):
    u=DELTA_U(T,species,phase_index)+species.h_form
    return u


#---------------------------------------------------------------------------------------------------------------------------------
def find_phase_index(phase):
    if phase == 's' or phase == 'PC1':
        phase_index = 0
    else:#if phase == 'l':
        phase_index = 1
    return phase_index


def uCv_i_p_vec(T, phase_index = None):
    len_poly = 7
    T_mid = gd.thermo_coeff_p[:, 0]
    coeffs_0 = gd.thermo_coeff_p[:, 7+1 :]
    coeffs_1 = gd.thermo_coeff_p[:, 1 : 7+1]
    len_x = 1
    cv_ct = np.zeros([gd.nb_p_species, len_x])
    u_ct  = np.zeros([gd.nb_p_species, len_x])
    for i_species in range(gd.nb_p_species):
        u_ct[i_species, :], cv_ct[i_species, :] = np.where(T < T_mid[i_species],
                                                        array_poly_2_Cv_p(coeffs_0[i_species], gd.MW_p_vec[i_species], T),
                                                        array_poly_2_Cv_p(coeffs_1[i_species], gd.MW_p_vec[i_species], T))
    return u_ct, cv_ct


#---------------------------------------------------------------------------------------------------------------------------------
def array_poly_2_Cv_p(coeffs, MW, T):
    """ from NIST coefficients : shomate formulation """
    t = T/1000
    Cv = (coeffs[0] + coeffs[1]*t   + coeffs[2]*t**2   + coeffs[3]*t**3   + coeffs[4]/t**2)/1000
    u =  (coeffs[0]*t + coeffs[1]*t**2/2 + coeffs[2]*t**3/3 + coeffs[3]*t**4/4 - coeffs[4]/t + coeffs[5] - coeffs[6])
    return u, Cv

def test(coeffs, T):
    """ from NIST coefficients : shomate formulation """
    t = T/1000
    Cv = (coeffs[0] + coeffs[1]*t   + coeffs[2]*t**2   + coeffs[3]*t**3   + coeffs[4]/t**2)/1000
    print(Cv)