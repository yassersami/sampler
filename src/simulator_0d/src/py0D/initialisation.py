
import os

import cantera as ct
import numpy as np

import simulator_0d.src.py0D.functions as func
import simulator_0d.src.py0D.global_data as gd
import simulator_0d.src.py0D.species_carac as SC


def initial_condition(type_class, inputs_dic, *args):
    DIR_PATH = os.path.abspath(__file__)
    PROJECT_PATH = os.path.abspath(os.path.join(DIR_PATH, '..','..','..')) 

    # species_name = ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Cu"]
    # name=PROJECT_PATH + "/data/cantera_inp/" + inputs_dic['ct_data_base']
    # MW_dic = {k : gas_ct.molecular_weights[gas_ct.species_index(k)]/1000 for k in species_name}

    gas_ct=ct.Solution(gd.name_ct)
    MW_dic = {k : gas_ct.molecular_weights[gas_ct.species_index(k)]/1000 for k in gd.name_gas_species}


#---------------------------------paramètres du matériau
    rho_Al2O3 = 3950
    rho_Al = 2700
    rho_MeO = 6315
    rho_Me = 8960

#---------------------------------paramètres du système global
    volume = 1.0e-6  # equiv 1cm3
    temperature_init = 300
    alpha_p = inputs_dic['alpha_p']                           # %TMD
    pAl_richness = inputs_dic['pAl_richness']                 # Richesse en aluminium 
    Y_Al_pAl = inputs_dic['Y_Al_pAl']                         # Pureté de l'aluminium

#---------------------------------paramètres des particules
    r_int_pMeO = inputs_dic['r_ext_pMeO']
    Y_MeO_core = 1
    Y_Me_core = 1-Y_MeO_core
    r_ext_pMeO = r_int_pMeO

    r_ext_pAl = inputs_dic['r_ext_pAl']

    # Calcul du rayon intérieur de l'aluminium à partir du rayon extérieur et de la pureté
    if Y_Al_pAl == 1:
        r_int_pAl = r_ext_pAl
    else:
        a = rho_Al2O3*Y_Al_pAl/(1-Y_Al_pAl)
        r_int_pAl = (a*r_ext_pAl**3/(rho_Al+a))**(1/3)

    # Calcul des masses
    m_pAl_core = func.V_sphere(r_int_pAl)*rho_Al
    m_pAl_shell = m_pAl_core*(1-Y_Al_pAl)/Y_Al_pAl
    m_pAl = m_pAl_core + m_pAl_shell

    m_pMeO_core = func.V_sphere(r_int_pMeO)*(Y_Me_core*rho_Me+Y_MeO_core*rho_MeO)
    m_pMeO_shell = (func.V_sphere(r_ext_pMeO)-func.V_sphere(r_int_pMeO))*rho_Al2O3

    # Calcul du nombre de particule
    n_pAl, n_pMeO, alpha_p = func.Nb_part(alpha_p, pAl_richness, Y_Al_pAl, volume, m_pAl, m_pMeO_core, r_ext_pAl, r_ext_pMeO, MW_dic)


#---------------------------------paramètres du gaz
    #0)02  1)Al   2)O    3)N2    4)Al2O    5) AlO    6)AlO2   7)Al2O2
    pressure = 1e5
    nb_gaz_species = 9
    khi = np.zeros(nb_gaz_species)
    khi[0] = 0.21
    khi[3] = 0.79

    Al = SC.species.Al(MW_dic['Al'])
    Me = SC.species.Me(MW_dic['Cu'])
    MeO = SC.species.MeO(1*MW_dic['O']+1*MW_dic['Cu'])
    Al2O3 = SC.species.Al2O3(3*MW_dic['O']+2*MW_dic['Al'])
    O2 = SC.species.Al(MW_dic['O2'])
    O = SC.species.Al(MW_dic['O'])

    transform_type=inputs_dic['transform_type']

    for i in range(len(khi)):
        if khi[i] == 0:
            khi[i] = 1e-10
    
    species_tab_p = [Al, MeO, Me, Al2O3, O2, O]

#-----------------------------définition de la loi P° vapeur saturante
    n_point_interp = int(1e4)

    T_P_Al_evap = np.zeros([2, n_point_interp])
    T_P_Me_evap = np.zeros([2, n_point_interp])

    T_P_Al_evap[0] = np.logspace(np.log10(Al.T_evap), np.log10(7000), n_point_interp)
    T_P_Me_evap[0] = np.logspace(np.log10(Me.T_evap), np.log10(7000), n_point_interp)

    T_P_Al_evap[1] = func.P_evap(T_P_Al_evap[0], Al)
    T_P_Me_evap[1] = func.P_evap(T_P_Me_evap[0], Me)


#-----------------------------retour des valeurs initiales en fonction de la phase
    if type_class == "P_Al_init":
        return (temperature_init, n_pAl, r_int_pAl, r_ext_pAl, m_pAl_core, m_pAl_shell, species_tab_p)

    if type_class == "P_MeO_init":
        return (temperature_init, n_pMeO, r_int_pMeO,r_ext_pMeO, m_pMeO_core, m_pMeO_shell, Y_Me_core, species_tab_p)

    if type_class == "gas":
        species_carac = np.full(nb_gaz_species, None)
        species_carac[0] = SC.species.O2(MW_dic['O2'])
        species_carac[1] = SC.species.Al(MW_dic['Al'])
        species_carac[2] = SC.species.O(MW_dic['O'])
        species_carac[3] = SC.species.N2(MW_dic['N2'])
        species_carac[4] = SC.species.Al2O(MW_dic['Al2O'])
        species_carac[5] = SC.species.AlO(MW_dic['AlO'])
        species_carac[6] = SC.species.AlO2(MW_dic['AlO2'])
        species_carac[7] = SC.species.Al2O2(MW_dic['Al2O2'])
        species_carac[8] = SC.species.Me(MW_dic['Cu'])


        ct_index=np.zeros(nb_gaz_species)
        for i in range (0,nb_gaz_species):
            ct_index[i]=gas_ct.species_index(gd.name_gas_species[i])

        gas_volume = (volume-func.V_sphere(r_ext_pAl)*n_pAl - func.V_sphere(r_ext_pMeO)*n_pMeO)
        alpha=gas_volume/volume

#-------------------------------------spécifique à la phase gaz de la gas_chamber
        if args[0]=="only":
            gas_volume=1e-4
            alpha=1

        return (pressure, temperature_init, khi, species_carac, gas_volume, T_P_Al_evap, T_P_Me_evap,gas_ct,alpha,ct_index,gd.name_gas_species,transform_type)

    if type_class == "system":
        return volume, (n_pAl+n_pMeO), inputs_dic['max_dt'], MW_dic
