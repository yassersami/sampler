import numpy as np

import simulator_0d.src.py0D.Balance_eq_source_term as bes
from simulator_0d.src.py0D.enthalpy import find_phase, H_tot
from simulator_0d.src.py0D.functions import integrate  # , dr_dt_computation
from simulator_0d.src.py0D.initialisation import initial_condition


class Particle():
    """objet particule"""
    def __init__(self,name,gas,inputs_dic):
        if name=="p_Al":
            temperature, n,r_int,r_ext,m_pAl_core,m_pAl_shell, species_tab = initial_condition("P_Al_init", inputs_dic)
            composition=np.array([m_pAl_core,0,0,m_pAl_shell])
            epsilon=0.85
        else:
            temperature, n,r_int,r_ext,m_pMeO_core,m_pMeO_shell,Y_Me_core, species_tab= initial_condition("P_MeO_init", inputs_dic)
            Y_MeO_core=1-Y_Me_core
            composition=np.array([0,m_pMeO_core*Y_MeO_core,m_pMeO_core*Y_Me_core,m_pMeO_shell])
            epsilon=0.85

        Al = species_tab[0]
        MeO = species_tab[1]
        Me = species_tab[2]
        Al2O3 = species_tab[3]
        O2 = species_tab[4]
        O = species_tab[5]

        self.name=name
        self.r_int=r_int #metre
        self.r_ext=r_ext #metre
        self.temperature=temperature #K
        self.n=n #nombre de particules
        self.composition=composition*n #kg: [Al,CuO,Cu,Al2O3]
        self.weight=np.sum(self.composition) #kg
        self.species_carac=[Al,MeO,Me,Al2O3] #[Al,CuO,Cu,Al2O3]
        self.species_name=["Al","MeO","Me","Al2O3"]
        self.phase=["s","s","s","s"]
        self.rho = [self.species_carac[species].rho for species in [0,1,2,3]]
        self.lambda_species = [233, 33, 401, 30]                                             # conductivité thermique des espèces à T° ambiente
        self.lambda_p = 0
        self.update_conductivity()
        self.enthalpy = 0
        self.init_enthalpy(gas)
        self.alpha=0
        self.epsilon=epsilon
        self.dm=0
        self.dQ=0
        self.cp = 1e5


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def init_enthalpy(self,gas):
        self.enthalpy=0+\
        self.composition[0]*(H_tot(self.temperature,self.species_carac[0],0))+\
        self.composition[1]*(H_tot(self.temperature,self.species_carac[1],0))+\
        self.composition[2]*(H_tot(self.temperature,self.species_carac[2],0))+\
        self.composition[3]*(H_tot(self.temperature,self.species_carac[3],0))
        self.compute_enthalpy(gas)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def compute_enthalpy(self,gas): #faire attention à l'utilisation de cette fonction, notamment pour la recherche de phase qui utilise l'état d'avant pour computer les enthalpies
        self.phase ,self.h_lim_vap=find_phase(self,self.temperature,gas)

        phase_index=np.zeros(len(self.phase))
        for i in range (0,len(self.phase)):
            if self.phase[i]=="s":
                phase_index[i]=0
            else:
                phase_index[i]=1

        if self.temperature<=933.45:
            phase_index=[0,0,0,0]
        elif (self.temperature>933.45 and self.temperature<=1358):
            phase_index=[1,0,0,0]
        elif (self.temperature>1358 and self.temperature<=2327):
            phase_index=[1,0,1,0]
        else:
            phase_index=[1,0,1,1]

        # print(phase_index)

        self.enthalpy=0+\
        self.composition[0]*(H_tot(self.temperature,self.species_carac[0],int(phase_index[0])))+\
        self.composition[1]*(H_tot(self.temperature,self.species_carac[1],int(phase_index[1])))+\
        self.composition[2]*(H_tot(self.temperature,self.species_carac[2],int(phase_index[2])))+\
        self.composition[3]*(H_tot(self.temperature,self.species_carac[3],int(phase_index[3])))

#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def update_composition(self,flux,source,dt):
        dmAl2O3_dt_0,dmAl2O3_dt_1=bes.dm_dt_computation_solid("Al2O3_p_shell",flux,source)
        dmAl_dt_0,dmAl_dt_1=bes.dm_dt_computation_solid("Al_pAl_core",flux,source)
        dmMeO_dt_0,dmMeO_dt_1=bes.dm_dt_computation_solid("MeO_pMeO_core",flux,source)
        dmMe_dt_0,dmMe_dt_1=bes.dm_dt_computation_solid("Me_pMeO_core",flux,source)

        self.composition[0]=self.composition[0]+integrate(dmAl_dt_0,dmAl_dt_1,dt) #integration variation mAl
        self.composition[1]=self.composition[1]+integrate(dmMeO_dt_0,dmMeO_dt_1,dt) # integration variation MeO
        self.composition[2]=self.composition[2]+integrate(dmMe_dt_0,dmMe_dt_1,dt) # integration variation Me
        self.composition[3]=self.composition[3]+integrate(dmAl2O3_dt_0,dmAl2O3_dt_1,dt) #integration variation mAl2O3

        if (self.composition < 0).any():
            print('Attention: composition de la particule négative')
            # print(self.name, self.composition)
            raise ValueError
        self.dm=np.sum(self.composition)-self.weight
        self.weight=np.sum(self.composition)
        self.update_conductivity()

#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def update_r(self):
        r_ths = 1e-5
#-----------------------------------------------------------------------calcul rayon intérieur MeO
        if self.name=="p_MeO":
            # methode de calcul par integration des flux
            # dr_int_dt0,dr_int_dt1,dr_ext_dt0,dr_ext_dt1,dr_ext_dt0_bis,dr_ext_dt1_bis=dr_dt_computation(self,flux,source)
            # r_int_a=self.r_int+integrate(dr_int_dt0,dr_int_dt1,dt)

            #methode de calcul à partir de la nouvelle masse de la particule
            r_int_b = (3/4/np.pi*np.sum([self.composition[i]/self.rho[i]/self.n for i in [0,1,2]]))**(1/3)
            if self.r_int/self.r_ext < r_ths:
                self.r_int = 0
            else:
                self.r_int=r_int_b

#-----------------------------------------------------------------------calcul rayon intérieur Al
        if self.name=="p_Al":
            # methode de calcul par integration des flux
            # dr_int_dt0,dr_int_dt1,dr_ext_dt0,dr_ext_dt1,dr_ext_dt0_bis,dr_ext_dt1_bis=dr_dt_computation(self,flux,source)
            # r_int_a=self.r_int+integrate(dr_int_dt0,dr_int_dt1,dt)

            #methode de calcul à partir de la nouvelle masse de la particule
            r_int_b = (3/4/np.pi*np.sum([self.composition[i]/self.rho[i]/self.n for i in [0,1,2]]))**(1/3)
            # print("rapport entre methode de calcul rayon Al",r_int_a/r_int_b)
            if self.r_int/self.r_ext < r_ths:
                self.r_int = 0
            else:
                self.r_int=r_int_b

#-----------------------------------------------------------------------calcul rayon extérieur (méthode identique aux deux particules)
        #methode de calcul par integration des flux
        # r_ext_a=self.r_ext+integrate(dr_ext_dt0,dr_ext_dt1,dt)+integrate(dr_ext_dt0_bis,dr_ext_dt1_bis,dt)

        #methode de calcul à partir de la nouvelle masse de la particule
        r_ext_b = (3/4/np.pi*np.sum([self.composition[i]/self.rho[i]/self.n for i in [0,1,2,3]]))**(1/3)
        self.r_ext=max(1e-10,r_ext_b)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def update_enthalpy(self,heat_flux,dt):
        dQ_dt_0, dQ_dt_1 = bes.dQ_dt_computation_p(heat_flux)
        self.dQ=integrate(dQ_dt_0, dQ_dt_1, dt)
        self.enthalpy=self.enthalpy+self.dQ



    def update_conductivity(self):
        self.lambda_p = np.sum(self.lambda_species * self.composition)/np.sum(self.composition)   # conductivité de la particule i (hypothèse: parfaitement mélangée)
