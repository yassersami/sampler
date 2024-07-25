import sys
from copy import deepcopy

import numpy as np

import simulator_0d.src.py0D.Balance_eq_source_term as bes
import simulator_0d.src.py0D.enthalpy as enth
import simulator_0d.src.py0D.functions as func
import simulator_0d.src.py0D.kinetics.module_kin as mkin
from simulator_0d.src.py0D.assist_functions import vectorize
from simulator_0d.src.py0D.initialisation import initial_condition


class Gas():
    """objet gaz"""
    def __init__(self,env_type,inputs_dic):
        R=8.314462
        pressure,temperature,khi,species_carac,volume_gas,T_P_Al_evap,T_P_Me_evap,gas_ct,alpha,ct_index,species_name,transform_type=initial_condition("gas", inputs_dic, env_type)
        self.env_type = env_type
        self.transform_type=transform_type
        self.name="gas"
        self.pressure=pressure                 # pression totale
        self.temperature=temperature         #temperature globale
        self.khi=khi                         #fraction molaire/volumique
        self.gas_ct=gas_ct
        self.ct_index=ct_index
        self.volume=volume_gas
        self.species_carac=species_carac     #caractéristiques des espèces
        self.enthalpy = 0
        self.energy = 0
        self.species_name = species_name
        self.T_P_Al_evap=T_P_Al_evap
        self.T_P_Me_evap=T_P_Me_evap
        self.nb_species=len(self.khi)
        self.MW_vec=vectorize(self.species_carac,"MW")
        self.Y=(self.khi*self.MW_vec)/np.sum(self.khi*self.MW_vec)
        self.Y_save = np.zeros(self.Y.shape)
        self.dY_dt = np.zeros(self.Y.shape)
        self.density=self.pressure/np.sum(self.Y/self.MW_vec)/R/self.temperature
        self.d_alpha=0
        self.gas_ct.TPY=self.temperature,self.pressure,self.ct_composition_input()
        self.cp = gas_ct.cp_mass                 # J/kg/K
        self.cv = gas_ct.cv_mass
        self.compute_H_U()
        self.alpha=alpha
        self.data_El={"Y":self.Y,"T":self.temperature,"P":pressure}
        self.save1 = 0
        self.save1_1 = 0
        self.save1_2 = 0
        self.save2 = 0
        self.save2_1 = 0
        self.save2_2 = 0
        self.bool_kin=inputs_dic["bool_kin"]
        self.dm=np.zeros(self.nb_species)
        self.dQ=0
        self.dm_dt_flux = 0
        self.dQ_dt = 0
        self.enthalpy_save=0
        self.energy_save=0
        self.volume_save=0
        self.density_save=0
        self.tau=0
        self.ct_it=0
        self.lambda_f = self.gas_ct.thermal_conductivity
        self.int_BDF = None


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def compute_H_U(self):
        H=0
        U=0
        for i in range (0,self.nb_species):
            H=H+self.Y[i]*(enth.H_ct(self.temperature,self.species_name[i],self))
            U=U+self.Y[i]*(enth.U_ct(self.temperature,self.species_name[i],self))
        self.enthalpy=H
        self.energy=U


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def update_all(self, dt, volume, *args):
        # flux_pMeO, flux_pAl, source_pMeO,source_pAl,heat_flux_pAl, heat_flux_pMeO
        self.energy_save=self.energy
        self.volume_save=self.volume
        self.density_save=self.density
        self.enthalpy_save=self.enthalpy

        self.flux_dm(dt,volume, args)
        self.flux_dQ(dt,volume, args)
        if  self.bool_kin and self.env_type == "with_p":
            # self.dm_cin_tot_dt, self.int_BDF = func.kinetics(self, volume, dt)
            self.dm_cin_tot_dt, self.int_BDF, t_BDF, Q_BDF = mkin.compute_kin(self, dt, volume, int_BDF = self.int_BDF, bool_get_hist = False)
            # self.dm_cin_tot_dt = np.zeros(self.nb_species)
        else:
            self.dm_cin_tot_dt = np.zeros(self.nb_species)

        self.update_composition(dt)
        self.update_enthalpy_energy(dt, volume, args)

        if self.transform_type=="V=cste":
            self.update_density(volume)
            self.alpha=self.alpha+self.d_alpha
            self.volume=self.alpha*volume
        self.save1=deepcopy(self.density)
        self.save1_1=deepcopy(self.energy)
        self.save1_2=deepcopy(self.Y)
        if self.bool_kin:
            self.not_equilibrium()
        else:
            self.equilibrium()
        self.dY_dt = (self.Y - self.Y_save)/dt

        self.save2=deepcopy(self.gas_ct.density)
        self.save2_1=deepcopy(self.energy)
        self.save2_2=deepcopy(self.Y)
        if self.transform_type=="P=cste":
            self.volume=1/self.density*(self.volume*self.density_save+np.sum(self.dm))
        # self.cp = self.gas_ct.cp

        #remarque: il faut MAJ l'enthalpie avec l'ancienne densité du gaz (pour être cohérent dans le schéma numérique)
        #C'est pourquoi la MAJ de la densité se fait en dernier. Il faut quand meme connaitre les dmi des différentes
        #espèces qui sont donc calculées dans update_composition.
        #Observation: lorsqu'on est en pure explicite, le plot de la conservation d'energie diverge legerement.
        self.ct_it+=1
        # if self.ct_it%100==0:
        #     print({x:self.dm_cin_tot_dt[i] for i,x in enumerate(self.species_name)})


#------------------------------------------------------------------------------------MAJ fractions massiques----------------------------------------------------------------------------------
    def flux_dm(self,dt ,volume,*args):
        #-------------------------Contribution de la cinétique chimique si il y a cinétique chimique
        #pas de cinétique pour le moment
        #--------------------------------Contribution des espèces provenant des particules

        dm_dt_0=np.zeros(self.nb_species)
        dm_dt_1=np.zeros(self.nb_species)
        self.dm=np.zeros(self.nb_species)
        if self.env_type=="with_p":
            flux_pMeO=args[0][0]                                                         # la fonctions précédente n'est pas unpack, d'où le [0]
            flux_pAl=args[0][1]
            source_pMeO=args[0][2]
            source_pAl=args[0][3]
            flux_chamber=args[0][6]
            for i in range (self.nb_species):
                dm_dt_0[i],dm_dt_1[i]=bes.dm_dt_computation_gas(self.species_name[i],flux_pMeO,flux_pAl,source_pMeO,source_pAl)
                self.dm[i]= ( func.integrate(dm_dt_0[i],dm_dt_1[i],dt)
                            + func.integrate(flux_chamber[0].species_flux[i],flux_chamber[1].species_flux[i],dt))
            self.dm_dt_flux=self.dm/dt
        if self.env_type=="only":
            flux_chamber=args[0][0]
            for i in range (self.nb_species):
                self.dm[i] = - func.integrate(flux_chamber[0].species_flux[i],flux_chamber[1].species_flux[i],dt)
            self.dm_dt_flux=self.dm/dt


    def update_composition(self,dt):
        self.Y_save = np.copy(self.Y)
        # self.Y=self.Y+(self.dm-self.Y*np.sum(self.dm))/self.density/self.alpha/volume                                                 #schéma explicite
        self.dm_tot=(self.dm_dt_flux + self.dm_cin_tot_dt) * dt
        self.Y=(self.density*self.volume*self.Y+self.dm_tot)/(self.density*self.volume+np.sum(self.dm_tot))                                     #schéma implicite
        self.khi=(self.Y/(self.MW_vec))/np.sum(self.Y/self.MW_vec)

        for i,x in enumerate(self.Y):
            if self.Y[i]<0  and abs(self.gas_ct.Y[self.gas_ct.species_index(self.species_name[i])]-self.Y[i])<1e-15:
                self.Y[i]=0
        if self.Y.min()<0:
            print(self.env_type)
            print(self.bool_kin)
            print("\n --------------------------ATTENTION: composition du gaz ne peut pas être inferieure à 0: réduire le pas de temps? \n")
            print(self.Y)
            print(np.sum(self.Y))
            print(self.Y.min()*self.density*self.volume/dt)
            raise ValueError("Y<0")
            sys.exit()

#------------------------------------------------------------------------------------MAJ fractions densité----------------------------------------------------------------------------------
    def update_density(self,volume):
           # self.density=self.density+(np.sum(self.dm)/volume-self.density*self.d_alpha)/self.alpha                                         #schéma explicite
        self.density=(self.density*self.volume+np.sum(self.dm))/((self.alpha+self.d_alpha)*volume)                                         #schéma implicite

#------------------------------------------------------------------------------------MAJ enthalpie/energie----------------------------------------------------------------------------------
    def flux_dQ(self,dt, volume, *args):

        if self.env_type=="with_p":
            heat_flux_pAl=args[0][4]
            heat_flux_pMeO=args[0][5]
            flux_chamber=args[0][6]
            dQ_dt_0, dQ_dt_1 = bes.dQ_dt_computation_gas(heat_flux_pAl,heat_flux_pMeO,dt)
            self.dQ=(func.integrate(dQ_dt_0, dQ_dt_1, dt)+func.integrate(flux_chamber[0].species_flux_heat,flux_chamber[1].species_flux_heat,dt)
                                                                           +func.integrate(flux_chamber[0].thermal_flux,flux_chamber[1].thermal_flux,dt))
        elif self.env_type=="only":
            flux_chamber=args[0][0]
            flux_HW=args[0][1]
            self.dQ= (- func.integrate(flux_chamber[0].species_flux_heat,flux_chamber[1].species_flux_heat,dt)
                         - func.integrate(flux_chamber[0].thermal_flux,flux_chamber[1].thermal_flux,dt)
                      + func.integrate(flux_HW[0].thermal,flux_HW[1].thermal,dt)
                      )
        self.dQ_dt = self.dQ / dt

#------------------------------------------------------------------------------------MAJ enthalpie/energie----------------------------------------------------------------------------------
    def update_enthalpy_energy(self,dt, volume, *args):
        if self.transform_type=="P=cste":
            self.enthalpy=(self.density*self.volume*self.enthalpy+self.dQ)/(self.density*self.volume+np.sum(self.dm))                     #schéma implicite
            # self.enthalpy=self.enthalpy+(self.dQ-self.enthalpy*np.sum(self.dm))/(self.alpha-self.d_alpha)/self.density/volume         #schéma explicite
            self.gas_ct.HPY=self.enthalpy,self.pressure,self.ct_composition_input()

        elif self.transform_type=="V=cste":
            self.energy=(self.density*volume*self.energy*self.alpha+self.dQ)/(self.density*volume*self.alpha+np.sum(self.dm))             #schéma implicite
            # self.energy=self.energy+(self.dQ-self.energy*np.sum(self.dm))/self.alpha/self.density/volume                                 #schéma explicite


#------------------------------------------------------------------------------------Equilibre ct----------------------------------------------------------------------------------
    def equilibrium(self):
        # ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"]

        ct_composition_input=("O2:"       + str(self.Y[0])
                              +",Al:"     + str(self.Y[1])
                               +",O:"      + str(self.Y[2])
                               +",N2:"     + str(self.Y[3])
                               +",Al2O:"   + str(self.Y[4])
                               +",AlO:"    + str(self.Y[5])
                               +",AlO2:"   + str(self.Y[6])
                               +",Al2O2:"  + str(self.Y[7])
                               +",Cu:"     + str(self.Y[8])
                             )
        self.gas_ct.TPY = 300, 1e5, 'O2:1'  # Contournement de l'historique stocké dans l'objet ct. 
        # Ainsi, on a toujours le même historique. cf: problématique de l'historique cantera
        if self.transform_type==("V=cste"):
            self.gas_ct.UVY=self.energy,1/self.density,ct_composition_input
            self.data_El={"Y":deepcopy(self.gas_ct.Y),"T":self.gas_ct.T,"P": self.gas_ct.P,"U":self.gas_ct.u,"V":self.gas_ct.v}
            self.gas_ct.equilibrate('UV')

        elif self.transform_type==("P=cste"):
            self.gas_ct.HPY=self.enthalpy,self.pressure,ct_composition_input
            self.data_El={"Y":deepcopy(self.gas_ct.Y),"T":self.gas_ct.T,"P": self.gas_ct.P,"U":self.gas_ct.u,"V":self.gas_ct.v}
            self.gas_ct.equilibrate('HP')

        for i in range (self.nb_species-1):
            self.Y[i]=self.gas_ct.Y[self.gas_ct.species_index(self.species_name[i])]
        self.Y[self.nb_species-1]=self.gas_ct.Y[self.gas_ct.species_index("Cu")]
        self.pressure=self.gas_ct.P
        self.temperature=self.gas_ct.T

        if self.transform_type == "P=cste": # contournement de la non conservation de cantera pour un système à volume constant
            self.density=self.gas_ct.density

        self.cp = self.gas_ct.cp_mass
        self.cv = self.gas_ct.cv_mass
        self.lambda_f = self.gas_ct.thermal_conductivity


#------------------------------------------------------------------------------------Equilibre ct----------------------------------------------------------------------------------
    def ct_composition_input(self):
        ct_composition_input=("O2:"       + str(self.Y[0])
                              +",Al:"     + str(self.Y[1])
                               +",O:"      + str(self.Y[2])
                               +",N2:"     + str(self.Y[3])
                               +",Al2O:"   + str(self.Y[4])
                               +",AlO:"    + str(self.Y[5])
                               +",AlO2:"   + str(self.Y[6])
                               +",Al2O2:"  + str(self.Y[7])
                               +",Cu:"     + str(self.Y[8])
                             )
        return ct_composition_input


#------------------------------------------------------------------------------------Equilibre ct----------------------------------------------------------------------------------
    def not_equilibrium(self):
        # ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Me"]

        Y_ct = np.zeros(self.nb_species)
        for i,x in enumerate(self.gas_ct.species_names):
            Y_ct[i]   = self.Y[self.species_name.index(x)]

        self.gas_ct.UVY=self.energy,1/self.density,Y_ct

        for i in range(self.nb_species-1):
            self.Y[i] = self.gas_ct.Y[self.gas_ct.species_index(self.species_name[i])]
        self.Y[self.nb_species-1] = self.gas_ct.Y[self.gas_ct.species_index("Cu")]
        self.pressure = self.gas_ct.P
        self.temperature = self.gas_ct.T
        self.cp = self.gas_ct.cp_mass
        self.cv = self.gas_ct.cv_mass
        self.lambda_f = self.gas_ct.thermal_conductivity


