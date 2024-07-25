from copy import deepcopy

import numpy as np

import simulator_0d.src.py0D.Balance_eq_source_term as bes
import simulator_0d.src.py0D.enthalpy as enth
import simulator_0d.src.py0D.functions as func
import simulator_0d.src.py0D.global_data as gd


#---------------------------------------------------------------------------------------------------------------------------------
class Flux_static():
    """classe de flux"""
    def __init__(self,gas):
        self.O_ext=0
        self.O_int=0
        self.O2_int=0
        self.O2_ext=0
        self.Al_ext=0
        self._ext_history = np.array([0])
        self._ext_history_wo_mwa = np.array([0])
        self.t_history = np.array([0])
        self.Al_int=0
        self.Me=0
        self.MeO=0
        self.AlxOy=np.zeros(gas.nb_species)                 # ici faire un tableau avec les différentes sous espèces AlxOy
        self.AlxOy_sum=0                                    # somme des flux des sous oxydes (utile uniquement pour le post-traitement)
        self.Al2O3_condens=0
        self.Al_condens=0
        self.O2_condens=0
        self.Y_s_g=np.zeros(gas.nb_species)

#---------------------------------------------------------------------------------------------------------------------------------
    def Flux_update(self,system, particle,gas,source_pAl,source_pMeO,heat_flux,flux, t_t):

        source_pAl_estim = None
        source_pMeO_estim = None

#----------------------particule d'Al
        if particle.name=="p_Al":
#------ Calcul des flux de condensation
            self.AlxOy,self.AlxOy_sum,self.Al2O3_condens,self.Al_condens,self.O2_condens=func.Flux_AlxOy(particle, gas)

#------ Calcul de la diffusion de l'oxygène dans Al
            D_O = func.Arr(particle.temperature, system.inputs_dic['k0_D_Ox'], system.inputs_dic['Ea_D_Ox'])

            self.Y_s_g[0] = func.compute_Y_s(0, particle, gas, D_O, self.O2_condens)
            self.Y_s_g[2] = func.compute_Y_s(2, particle, gas, D_O, 0)

            self.O2_ext = func.O_transport_alumina(particle, self.Y_s_g[0], D_O)
            self.O_ext = func.O_transport_alumina(particle, self.Y_s_g[2], D_O)


            self.O2_ext = min(self.O2_ext, 0)
            self.O_ext = min(self.O_ext, 0)

            if self.O2_ext > 0:
                self.O2_ext = self.O2_ext - self.O2_condens
            if self.O2_ext > 0 : # condition
                print("Attention, la condensation des sous-oxydes est limitée par le flux de dioxygène, mécanisme non implémenté")
                exit()
#------ Calcul du flux de vaporisation/condensation de l'Al et Me

            source_pAl_estim = self.Al_evap(particle, flux, source_pAl, source_pAl_estim, heat_flux, gas, system)
            source_pAl_estim = self.Me_evap(particle, flux, source_pAl, source_pAl_estim, heat_flux, gas, system)


#----------------------particule de Me
        if particle.name == "p_MeO":
            self.AlxOy,self.AlxOy_sum,self.Al2O3_condens,self.Al_condens,self.O2_condens=func.Flux_AlxOy(particle, gas)
            O2_ext = source_pMeO[1].Decomp_MeO(system, particle)[0]

            self.O2_ext = O2_ext
            source_pMeO_estim = self.Me_evap(particle, flux, source_pMeO, source_pMeO_estim, heat_flux, gas, system)
            source_pMeO_estim = self.Al_evap(particle, flux, source_pMeO, source_pMeO_estim, heat_flux, gas, system)


        source_pAl = deepcopy(source_pAl_estim)
        source_pMeO = deepcopy(source_pMeO_estim)
#----------------------multiplication par le nombre de particules
        self.multi_nb_particle(particle.n)

        if particle.name=="p_Al":
            return source_pAl
        if particle.name=="p_MeO":
            return source_pMeO


#---------------------------------------------------------------------------------------------------------------------------------
    def multi_nb_particle(self,nb_particles):
        dic=self.__dict__
        for key in dic:
            if key!="Y_s_g":
                dic[key]=dic[key]*nb_particles


#---------------------------------------------------------------------------------------------------------------------------------
    def Al_evap(self, particle, flux, source_p, source_p_estim, heat_flux, gas, system):
        Al_evap, Y_s, ebull_bool_1, ignore_vap_bool = func.evap_flux(gas, particle, 1)

        Al_ebull, source_p_estim, bool_ebull_neg = species_ebull_compute(self, "Al", particle, flux, source_p, source_p_estim, heat_flux, ignore_vap_bool, gas, system)

        self.Al_ext, source_p = self.ebul_vs_vap(Al_ebull, Al_evap, Y_s, ebull_bool_1, ignore_vap_bool, bool_ebull_neg, source_p_estim)

        return source_p_estim


#---------------------------------------------------------------------------------------------------------------------------------
    def Me_evap(self, particle, flux, source_p, source_p_estim, heat_flux, gas, system):
        Me_evap, Y_s, ebull_bool_1, ignore_vap_bool = func.evap_flux(gas, particle, 8)

        Me_ebull, source_p_estim, bool_ebull_neg = species_ebull_compute(self, "Me", particle, flux, source_p, source_p_estim, heat_flux, ignore_vap_bool, gas, system)

        self.Me, source_p = self.ebul_vs_vap(Me_ebull, Me_evap, Y_s, ebull_bool_1, ignore_vap_bool, bool_ebull_neg, source_p_estim)

        return source_p_estim


#---------------------------------------------------------------------------------------------------------------------------------
    def ebul_vs_vap(self, S_ebull, S_evap, Y_s, ebull_bool_1, ignore_vap_bool, bool_ebull_neg, source_p):
        if ebull_bool_1 and not bool_ebull_neg:
            return S_ebull, source_p
        elif ebull_bool_1 and bool_ebull_neg:
            return S_evap, source_p
        else:
            if S_evap > 0:
                S = min(S_ebull, S_evap)
                if S_ebull > S_evap:
                    return S_evap, source_p
                return S_evap, source_p
            else:
                return S_evap, source_p


#---------------------------------------------------------------------------------------------------------------------------------
def species_ebull_compute(flux_1, species, particle, flux, source_p, source_p_estim, heat_flux, ignore_vap_bool, gas, system):
    """
    if the boiling regime is activated, and the mass transfer is considered:
    """
    
    if gas.transform_type=="P=cste":
        UH_ct=enth.H_ct
        UH_tot=enth.H_tot

    if gas.transform_type=="V=cste":
        UH_ct=enth.U_ct
        UH_tot=enth.U_tot
    flux[1] = deepcopy(flux_1)
    flux[0] = deepcopy(flux[1])
    """
    Fluxes of Me are not considered (neglected) in the energy balance equation to estimate the boiling flux
    of aluminum. Indeed, there are not computed at that stage and it can become complicated to take into
    consideration the fluxes of the others potentially vaporizing species to obtain the boiling flux of
    the current species
    """

    flux[1].multi_nb_particle(particle.n)
    # setattr(flux[1], species, 0)
    if   species == "Al":
        flux[1].Me     = 0
    elif species == "Me":
        flux[1].Al_ext = 0

    if source_p_estim == None:
        source_p = Source_tab_update(system, source_p, flux, particle, source_p_estim)
    else :
        source_p = deepcopy(source_p_estim)

    ebull_dic = dict(Al = False, Me = False)
    ebull_dic[species] = True
    Tp = particle.temperature
    if not ignore_vap_bool:
        species_ebull = -(
            # termes flux d'energie/enthalpie (Q_kappa)
            + bes.dQ_dt_computation_p(heat_flux)[1] - (-heat_flux[1].species_flux_heat)
            # + (- heat_flux[1].thermal)
            # + heat_flux[1].radiative
            # + heat_flux[1].p_to_p
            # + heat_flux[1].source_input
            + (- func.Thermal_flux_relative_to_mass_flux_computation(flux, source_p, particle, gas, ebull_dic = ebull_dic))
        #     # termes sources de la particule
            - (- source_p[1].Al2O3_int*(UH_tot(Tp,particle.species_carac[3],1)))
            - (- flux[1].Al2O3_condens*(UH_tot(Tp,particle.species_carac[3],1)))
            - (- source_p[1].Al2O3_decomp*(UH_tot(Tp,particle.species_carac[3],1)))
            - (- source_p[1].Al_int*(UH_tot(Tp,particle.species_carac[0],1)))

            - (- source_p[1].MeO*(UH_tot(Tp,particle.species_carac[1],1)))
            - (- source_p[1].Me*(UH_tot(Tp,particle.species_carac[2],1)))
        # ) /particle.n/particle.species_carac[0].h_evap
        # ) /particle.n/(UH_tot(Tp,particle.species_carac[gd.ips[species]],1)
        #               -(UH_ct(Tp,species,gas)))
        ) /particle.n/(-(UH_ct(Tp,species,gas))
                    + UH_tot(Tp,particle.species_carac[gd.ips[species]],1))
    else:
        species_ebull = 0
    bool_ebull_neg = False
    if species_ebull<0:
        # species_ebull = 0
        bool_ebull_neg = True

    return species_ebull, source_p, bool_ebull_neg


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_species_tab_init(gas):
    flux=[None,None]
    flux[0]=Flux_static(gas)
    flux[1]=Flux_static(gas)
    return flux


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_species_tab_update(system,flux,particle,gas,source_pAl,source_pMeO,heat_flux, t_t):
    flux[0]=deepcopy(flux[1])
    source_p = flux[1].Flux_update(system,particle,gas,source_pAl,source_pMeO,heat_flux,deepcopy(flux), t_t)
    return flux, source_p


#---------------------------------------------------------------------------------------------------------------------------------
class Source_static():
    """classe des termes sources"""
    def __init__(self,gas):
        self.O_ext=0
        self.O_int=0
        self.O2_ext=0
        self.O2_int=0
        self.Al_ext=0
        self.Al_int=0
        self.Me=0
        self.MeO=0
        self.AlxOy=np.zeros(gas.nb_species) #ici faire un tableau avec les différentes sous espèces AlxOy
        self.Al2O3_ext=0
        self.Al2O3_int=0
        self.Al2O3_decomp=0
        self.Al_decomp=0
        self.O2_decomp=0

        self.O_decomp=0
        self.Al2O2_decomp=0
        self.AlO2_decomp=0
        self.AlO_decomp=0
        self.Al2O_decomp=0

#---------------------------------------------------------------------------------------------------------------------------------
    def Decomp_MeO(self, system, particle):
        #Me_a O_b ==> a Me + b O
        a=1                                                 # pour décomposition CuO==>Cu + O
        b=1                                                 # pour décomposition CuO==>Cu + O

        m_MeO_part = particle.composition[1] / particle.n
        if particle.composition[1]/np.sum(particle.composition)>1e-5:
            MeO=func.Arr(particle.temperature, system.inputs_dic['k0_MeO_decomp'], system.inputs_dic['Ea_MeO_decomp'])*m_MeO_part
            Me=a*(-MeO)/particle.species_carac[1].MW*particle.species_carac[2].MW
        else:
            MeO=0
            Me=0
        MW_O = system.MW_dic['O']
        O2_int=b*(MeO)/particle.species_carac[1].MW*MW_O   # le flux massique considérant l'oxygène atomique ou le dioxygène est le même
        return O2_int,MeO,Me

#---------------------------------------------------------------------------------------------------------------------------------
    def Al2O3_generation(self,system, flux_O2_int,flux_O_int,particle):
        MW_O = system.MW_dic['O']
        MW_O2 = system.MW_dic['O2']


        source_Al2O3_int=(particle.species_carac[3].MW*2/3*(flux_O2_int)/MW_O2
                         +particle.species_carac[3].MW*1/3*(flux_O_int)/MW_O
        )
        source_Al_int=-source_Al2O3_int/particle.species_carac[3].MW*2*particle.species_carac[0].MW
        return source_Al2O3_int,source_Al_int


#----------------------------------------------------------------------------------------------------------------------------------
    def Decomp_Al2O3(self, system, particle, decomp_n_):
        MW_AlO = system.MW_dic['AlO']
        MW_Al2O2 = system.MW_dic['Al2O2']
        MW_AlO2 = system.MW_dic['AlO2']
        MW_O = system.MW_dic['O']
        MW_O2 = system.MW_dic['O2']

        # --------------------  Dans le cas d'une décomposition en Al / O2
        if decomp_n_ == 0:
            source_Al2O3_decomp = func.Arr(particle.temperature,system.inputs_dic['k0_Al2O3_decomp'],
                                             system.inputs_dic['Ea_Al2O3_decomp'])*particle.composition[3]/particle.n             # les sources s'ajoutent sur les particules donc doit être négatif
            source_Al_decomp = 2 * source_Al2O3_decomp/particle.species_carac[3].MW*particle.species_carac[0].MW         # flux positif p to g
            source_O2_decomp = 3/2 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_O2                              # flux positif p to g
            return source_Al2O3_decomp, source_Al_decomp, source_O2_decomp

    # -------------------- Dans le cas d'une décomposition en Al2O2 / O # réaction R77 de Catoire
        if decomp_n_ == 1:
            source_Al2O3_decomp = func.Arr(particle.temperature,0,
                                             system.inputs_dic['Ea_Al2O3_decomp'])*particle.composition[3]/particle.n             # les sources s'ajoutent sur les particules donc doit être négatif
            source_Al2O2_decomp = 1 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_Al2O2                          # flux positif p to g
            source_O_decomp = 1 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_O                                  # flux positif p to g
            return source_Al2O3_decomp, source_Al2O2_decomp, source_O_decomp

    # -------------------- Dans le cas d'une décomposition en AlO2 / AlO # réaction R78 de Catoire
        if decomp_n_ == 2:
            source_Al2O3_decomp = func.Arr(particle.temperature,0,
                                             system.inputs_dic['Ea_Al2O3_decomp'])*particle.composition[3]/particle.n              # les sources s'ajoutent sur les particules donc doit être négatif
            source_AlO2_decomp = 1 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_AlO2                             # flux positif p to g
            source_AlO_decomp = 1 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_AlO                               # flux positif p to g
            return source_Al2O3_decomp, source_AlO2_decomp, source_AlO_decomp

    # -------------------- Dans le cas d'une décomposition en Al2O2 / O2 # réaction maison pour minisation de l'enthalpie de formation transférée au gaz
        if decomp_n_ == 3:
            source_Al2O3_decomp = func.Arr(particle.temperature,0,
                                             system.inputs_dic['Ea_Al2O3_decomp'])*particle.composition[3]/particle.n             # les sources s'ajoutent sur les particules donc doit être négatif
            source_Al2O2_decomp = 1 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_Al2O2                          # flux positif p to g
            source_O2_decomp = 1/2 * source_Al2O3_decomp/particle.species_carac[3].MW*MW_O2                              # flux positif p to g
            return source_Al2O3_decomp, source_Al2O2_decomp, source_O2_decomp



#---------------------------------------------------------------------------------------------------------------------------------
    def Source_update(self,system,flux,particle):
        if particle.name=="p_Al":
            self.Al2O3_int,self.Al_int=self.Al2O3_generation(system,flux[1].O2_ext/particle.n,flux[1].O_ext/particle.n,particle)


        if particle.name=="p_MeO":
            self.O_int,self.MeO,self.Me=self.Decomp_MeO(system, particle)

        Al2O3_decomp = np.zeros(4)
        Al_decomp = np.zeros(1)
        O2_decomp = np.zeros(2)
        Al2O2_decomp = np.zeros(1)
        AlO2_decomp = np.zeros(1)
        AlO_decomp = np.zeros(1)
        O_decomp = np.zeros(1)

        Al2O3_decomp[0], Al_decomp[0], O2_decomp[0] = self.Decomp_Al2O3(system, particle, 0)
        Al2O3_decomp[1], Al2O2_decomp[0], O_decomp[0] = self.Decomp_Al2O3(system, particle, 1)
        Al2O3_decomp[2], AlO2_decomp[0], AlO_decomp[0] = self.Decomp_Al2O3(system, particle, 2)
        Al2O3_decomp[3], Al2O2_decomp[0], O2_decomp[1] = self.Decomp_Al2O3(system, particle, 3)

        self.Al2O3_decomp = np.sum(Al2O3_decomp)
        self.Al_decomp = np.sum(Al_decomp)
        self.O2_decomp = np.sum(O2_decomp)
        self.Al2O2_decomp = np.sum(Al2O2_decomp)
        self.AlO2_decomp = np.sum(AlO2_decomp)
        self.AlO_decomp = np.sum(AlO_decomp)
        self.O_decomp = np.sum(O_decomp)

        self.multi_nb_particle(particle.n)


#---------------------------------------------------------------------------------------------------------------------------------
    def multi_nb_particle(self,nb_particles):
        dic=self.__dict__
        for key in dic:
            dic[key]=dic[key]*nb_particles


#---------------------------------------------------------------------------------------------------------------------------------
    def Source_Al_int(self):
        self.Al_int=0


#---------------------------------------------------------------------------------------------------------------------------------
    def Source_Al_ext(self):
        self.Al_ext=0


#---------------------------------------------------------------------------------------------------------------------------------
    def Source_Al2O3_int(self):
        self.Al2O3_int=0


#---------------------------------------------------------------------------------------------------------------------------------
    def Source_Al2O3_ext(self):
        self.Al2O3_ext=0


#---------------------------------------------------------------------------------------------------------------------------------
def Source_tab_init(gas):
    source=[None,None]
    source[0]=Source_static(gas)
    source[1]=Source_static(gas)
    return source

#---------------------------------------------------------------------------------------------------------------------------------
def Source_tab_update(system,source,flux,particle,source_estim):
    if source_estim == None:
        source[0]=deepcopy(source[1])
        source[1].Source_update(system,flux,particle)
    else:
        source = deepcopy(source_estim)
    return source


#---------------------------------------------------------------------------------------------------------------------------------
class Heat_flux_static():
    """classe des flux thermiques"""
    def __init__(self):
        self.thermal=0
        self.species_flux_heat=0
        self.radiative=0
        self.p_to_p=0
        self.source_input=0
        self.lambda_f = 0
        self.add_values = dict(H_cond_pq = 0)


#---------------------------------------------------------------------------------------------------------------------------------
    def Thermal_flux_update(self,particle,gas):
        Re=0
        Pr=0
        #Gunn correlation
        Nu=(7-10*gas.alpha+5*gas.alpha**2)*(1+0.7*Re**0.2*Pr**(1/3))
        self.lambda_f=gas.gas_ct.thermal_conductivity
        thermal=self.lambda_f*np.pi*2*particle.r_ext*Nu*(particle.temperature-gas.temperature)*particle.n
        self.thermal=thermal


#---------------------------------------------------------------------------------------------------------------------------------
    def radiative_flux_update(self,p_Al,p_MeO,particle_name):
        sigma=5.670374419e-8 # constante de stefan boltzman
        view_factor = p_MeO.n*func.S_sphere(p_MeO.r_ext)/(p_MeO.n*func.S_sphere(p_MeO.r_ext) + p_Al.n*func.S_sphere(p_Al.r_ext))
        num=sigma*(p_Al.temperature**4-p_MeO.temperature**4)
        denum1=(1-p_Al.epsilon)/(func.S_sphere(p_Al.r_ext)*p_Al.n)/p_Al.epsilon
        denum2=1/(func.S_sphere(p_Al.r_ext)*p_Al.n)/view_factor
        denum3=(1-p_MeO.epsilon)/(func.S_sphere(p_MeO.r_ext)*p_MeO.n)/p_MeO.epsilon

        if particle_name=="p_Al":
            self.radiative=-num/(denum1+denum2+denum3)
        elif particle_name=="p_MeO":
            self.radiative=num/(denum1+denum2+denum3)


#---------------------------------------------------------------------------------------------------------------------------------
    def p_to_p_heat_flux(self,p_Al,p_MeO,gas,particle_name):
        flux_tmp, self.add_values["H_cond_pq"] = func.compute_conduc_pp(p_Al, p_MeO, gas)

        if particle_name=="p_Al":
            self.p_to_p = - flux_tmp
        elif particle_name=="p_MeO":
            self.p_to_p =   flux_tmp


#---------------------------------------------------------------------------------------------------------------------------------
    def Thermal_flux_relative_to_mass_flux_computation(self,flux_p,source_p,particle,gas):
    #     flux_per_species=np.zeros([gas.nb_species,2])
    #     decomp_Al2O3_species_flux=np.zeros(2)
    #     add_heat_flux_AlxOy=np.zeros([2])
    # #"O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"

    #     if gas.transform_type=="P=cste":
    #         UH_ct=enth.H_ct
    #         UH_tot=enth.H_tot

    #     if gas.transform_type=="V=cste":
    #         UH_ct=enth.U_ct
    #         UH_tot=enth.U_tot

    #     # Les réactions surfaciques ont elles davantage lieu dans la phase gaz?
    #     # Attention: si True, un déséquilibre thermique en fin de réaction peut apparaitre
    #     bool_react_gas = False

    #     for i in range (0,2):
    #         if particle.name=="p_Al":
    #             flux_per_species[0,i]=(UH_ct(func.T_inter(gas, particle, flux_p[i].O2_ext),"O2",gas))*flux_p[i].O2_ext    #O2
    #             flux_per_species[2,i]=(UH_ct(func.T_inter(gas, particle, flux_p[i].O_ext),"O",gas))*flux_p[i].O_ext    #O

    #         elif particle.name=="p_MeO":
    #             flux_per_species[0,i]=(UH_ct(func.T_inter(gas, particle, flux_p[i].O2_ext),"O2",gas))*flux_p[i].O2_ext    #O2

    #         flux_per_species[1,i]=(UH_ct(func.T_inter(gas, particle, flux_p[i].Al_ext),"Al",gas))*flux_p[i].Al_ext#+Al.h_evap    #Al
    #         flux_per_species[8,i]=(UH_ct(func.T_inter(gas, particle, flux_p[i].Me),"Cu",gas))*flux_p[i].Me    #Me

    #         #additional heat flux related to AlxOy condensation
    #         add_heat_flux_AlxOy[i] = heat_flux_AlxOy(i,UH_ct,UH_tot,gas,particle,flux_p, bool_react_gas)

    #         #additional heat flux related to Al2O3 decomposition
    #         decomp_Al2O3_species_flux[i] = heat_flux_decomp(i,UH_ct,UH_tot,gas,particle, source_p, bool_react_gas)

    #     self.species_flux_heat=np.sum(flux_per_species,axis=0)[1]+add_heat_flux_AlxOy[1]+decomp_Al2O3_species_flux[1]
        self.species_flux_heat=func.Thermal_flux_relative_to_mass_flux_computation(flux_p,source_p,particle,gas)


# def heat_flux_AlxOy(i,UH_ct,UH_tot,gas,particle,flux_p, bool_react_gas):             #additional heat flux related to AlxOy condensation
#     if bool_react_gas: # si l'enthalpie transportée par les transferts est prise dans la phase gaz (attention, apparition d'un déséquilibre thermique)
#         add_heat_flux_AlxOy=((UH_ct(func.T_inter(gas, particle, flux_p[i].AlxOy[4]),gas.species_name[4],gas))*flux_p[i].AlxOy[4]    #Al2O
#                             +(UH_ct(func.T_inter(gas, particle, flux_p[i].AlxOy[5]),gas.species_name[5],gas))*flux_p[i].AlxOy[5]    #AlO
#                             +(UH_ct(func.T_inter(gas, particle, flux_p[i].AlxOy[6]),gas.species_name[6],gas))*flux_p[i].AlxOy[6]    #AlO2
#                             +(UH_ct(func.T_inter(gas, particle, flux_p[i].AlxOy[7]),gas.species_name[7],gas))*flux_p[i].AlxOy[7]    #Al2O2
#                             +(UH_ct(func.T_inter(gas, particle, flux_p[i].O2_condens),gas.species_name[0],gas))*flux_p[i].O2_condens
#                             +(UH_ct(gas.temperature, particle.species_name[0],gas))*flux_p[i].Al_condens
#                             )
#         add_heat_flux_AlxOy=UH_tot(gas.temperature,particle.species_carac[3],1) * flux_p[i].Al2O3_condens
#     else:
#         add_heat_flux_AlxOy=UH_tot(particle.temperature,particle.species_carac[3],1) * flux_p[i].Al2O3_condens
#     return add_heat_flux_AlxOy


# def heat_flux_decomp(i,UH_ct,UH_tot,gas,particle, source_p, bool_react_gas): #addition of species thermal flux related to Al2O3 decomposition
#     if bool_react_gas: # si l'enthalpie transportée par les transferts est prise dans la phase gaz (attention, apparition d'un déséquilibre thermique)
#         decomp_Al2O3_species_flux = 0
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].O2_decomp),"O2",gas))*source_p[i].O2_decomp        #O2
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].Al_decomp),"Al",gas))*source_p[i].Al_decomp        #Al
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].O_decomp),"O",gas))*source_p[i].O_decomp        #O
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].Al2O_decomp),"Al2O",gas))*source_p[i].Al2O_decomp        #Al2O
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].AlO_decomp),"AlO",gas))*source_p[i].AlO_decomp        #AlO
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].AlO2_decomp),"AlO2",gas))*source_p[i].AlO2_decomp        #AlO2
#         decomp_Al2O3_species_flux=decomp_Al2O3_species_flux + (UH_ct(func.T_inter(gas, particle, source_p[i].Al2O2_decomp),"Al2O2",gas))*source_p[i].Al2O2_decomp        #Al2O2

#     else:
#         # remarque:
#         # attetion au signe "-" car la convention des termes sources sur la particule est "la quantité augmente sur la particule
#         # quand le terme source est positif..."
#         decomp_Al2O3_species_flux = - source_p[i].Al2O3_decomp*UH_tot(particle.temperature,particle.species_carac[3],1)
#     return decomp_Al2O3_species_flux


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_heat_tab_init():
    flux=[None,None]
    flux[0]=Heat_flux_static()
    flux[1]=Heat_flux_static()
    return flux


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_heat_tab_update_1(flux,p_Al,p_MeO,particle_name,gas):
    flux[0]=deepcopy(flux[1])
    if particle_name=="p_Al":
        flux[1].Thermal_flux_update(p_Al,gas)
    elif particle_name=="p_MeO":
        flux[1].Thermal_flux_update(p_MeO,gas)
    flux[1].radiative_flux_update(p_Al,p_MeO,particle_name)
    flux[1].p_to_p_heat_flux(p_Al,p_MeO,gas,particle_name)
    return flux


#---------------------------------------------------------------------------------------------------------------------------------
def Flux_heat_tab_update_2(flux_p,source_p,flux,p_Al,p_MeO,particle_name,gas):
    if particle_name=="p_Al":
        flux[1].Thermal_flux_relative_to_mass_flux_computation(flux_p,source_p,p_Al,gas)
    elif particle_name=="p_MeO":
        flux[1].Thermal_flux_relative_to_mass_flux_computation(flux_p,source_p,p_MeO,gas)
    return flux


#---------------------------------------------------------------------------------------------------------------------------------
def Source_input(flux,heat_input):
    flux[1].source_input=heat_input
    return flux


class g_g_flux():
    """flux avec la chambre de decompression (energetiques et massiques)"""
    def __init__(self, gas, system):
        self.DELTA_P=0
        self.mass_flux=0
        self.thermal_flux=0
        self.species_flux_heat=0
        self.cross_area=6 * system.volume**(2/3)
        self.species_flux=np.zeros(gas.nb_species)
        self.grad_Y=0
        self.bool_thermal_flux_gc=system.inputs_dic['bool_thermal_flux_gc']
        self.mlt_thermal_flux=system.inputs_dic['mlt_thermal_flux']
        self.velocity=0
        self.L=system.volume**(1/3)/4
        self.bool_gas_chamber=system.inputs_dic['bool_gas_chamber']

    def Flux_update(self, gas,gas_chamber, system):
        if self.bool_gas_chamber:
            self.DELTA_P = gas.pressure-gas_chamber.pressure
            dp = 2 * ((3*(1-gas.alpha)*system.volume)
                    / (4*np.pi*system.n_p)
                    )**(1/3)                                # pour eviter d'avoir a gerer des particules dans cette classe, on estime le diamètre de la sorte

            if self.DELTA_P >= 0:
                density_flux = gas.density
                gas_ct_flux = gas.gas_ct
                self.velocity = - func.compute_vel_advection(density_flux, dp, gas.alpha, gas_ct_flux.viscosity, self.L, self.DELTA_P)
                flux_Y = gas.Y
            else:
                density_flux = gas_chamber.density
                gas_ct_flux = gas_chamber.gas_ct
                self.velocity = func.compute_vel_advection(density_flux, dp, gas.alpha, gas_ct_flux.viscosity, self.L, - self.DELTA_P)
                flux_Y = gas_chamber.Y

            self.species_flux =  (flux_Y*density_flux*self.cross_area*self.velocity)
            self.mass_flux = np.sum(self.species_flux)      # non utilisé à part dans le post traitement
            lambda_thermal=gas_ct_flux.thermal_conductivity
            if self.bool_thermal_flux_gc:
                self.thermal_flux= - lambda_thermal*(gas.temperature-gas_chamber.temperature)*self.mlt_thermal_flux

            self.species_flux_heat = np.sum(self.species_flux)*gas_ct_flux.int_energy_mass


def Flux_g_g_tab_init(gas,system):
    flux=[None,None]
    flux[0]=g_g_flux(gas, system)
    flux[1]=g_g_flux(gas, system)
    return flux


def Flux_g_g_tab_update(flux, gas, gas_chamber, system):
    flux[0] = deepcopy(flux[1])
    flux[1].Flux_update(gas, gas_chamber, system)
    return flux


class Heat_well():
    """definition de l'objet puit thermique"""
    def __init__(self,temperature):
        self.mass = 1
        self.cp = 0.625
        self.temperature = temperature
        self.enthalpy=self.mass*self.cp*(self.temperature-298.15)

    def update_enthalpy(self, flux_HW, dt):
        self.enthalpy=self.enthalpy - func.integrate(flux_HW[0].thermal,flux_HW[1].thermal,dt)
        self.temperature=self.enthalpy/self.cp/self.mass+298.15

class Heat_well_flux():
    """definition des flux avec le puit thermique"""
    def __init__(self, inputs_dic):
        if inputs_dic['HW'] is True:
            self.conductivity = 100*50
        else:
            self.conductivity = 0
        self.surface=1
        self.thermal=0
        self.HW_temperature=300                             # Cette  grandeur ne sert a rien à part simplifier l'affichage de la T° avec le flux du puit

    def Flux_update(self,phase,heat_well):
        self.thermal = -(phase.temperature - heat_well.temperature) * self.conductivity * self.surface
        self.HW_temperature=heat_well.temperature


def Heat_well_flux_tab_init(inputs_dic):
    flux=[None,None]
    flux[0]=Heat_well_flux(inputs_dic)
    flux[1]=Heat_well_flux(inputs_dic)
    return flux


def heat_well_tab_update(flux, phase, heat_well):
    flux[0] = deepcopy(flux[1])
    flux[1].Flux_update(phase,heat_well)
    return flux
