import numpy as np
from scipy.interpolate import interp1d

from simulator_0d.src.py0D.functions import integrate, V_sphere
from simulator_0d.src.py0D.initialisation import initial_condition


class System():
    """ objet systeme"""
    def __init__(self,p_Al,p_MeO,gas,gas_chamber,heat_well,dt,t,t_f, inputs_dic):
        volume, n_p, max_dt, MW_dic = initial_condition("system", inputs_dic)
        self.dt=dt
        self.i=0
        self.volume=volume
        self.mass=0
        self.enthalpy=0
        self.energy=0
        self.sum_mass_flux=0
        self.sum_heat_flux_h=0
        self.sum_heat_flux_u=0
        self.t=t
        self.t_f=t_f
        self.heat_input=0
        self.power_input=inputs_dic["power_input"]
        self.power_input_save=0
        self.i_init=0
        self.test_ignition=True
        self.mass_save=0
        self.n_p=n_p
        self.max_dt=max_dt
        self.var_T_g=1e3
        self.var_P_g=1e3
        self.var_T_pAl=1e3
        self.var_h=1e3

        self.t_hist = np.array([0])
        self.T_g_hist = np.array([0])
        self.P_g_hist = np.array([0])
        self.T_pAl_hist = np.array([0])
        self.h_p_hist = np.array([0])
        self.dt_restrict = ('dt', 'species')
        self.inputs_dic=inputs_dic
        self.t_chron = 0
        self.MW_dic = MW_dic
        self.dt_hist= 0


        self.update_system(p_Al,p_MeO,gas,gas_chamber,heat_well,np.array([1]))


    def update_system(self, p_Al,p_MeO,gas,gas_chamber,heat_well,t_t):
        self.mass_save=self.mass
        heat_input_save = self.heat_input
        self.heat_input=self.heat_input+integrate(self.power_input_save,self.power_input, self.dt)
        self.update_mass(p_Al,p_MeO,gas,gas_chamber)
        self.update_enthalpy(p_Al,p_MeO,gas,gas_chamber,heat_well)
        self.update_energy(p_Al,p_MeO,gas,gas_chamber,heat_well)
        self.update_volume(p_Al,p_MeO,gas)
        self.power_input_save=self.power_input
        self.compute_var(gas, p_Al, p_MeO,t_t)
        self.compute_bool_end_simu()
        self.update_test_ignition()


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def update_volume(self,p_Al,p_MeO,gas):
        self.volume=(V_sphere(p_Al.r_ext)*p_Al.n
                    +V_sphere(p_MeO.r_ext)*p_MeO.n
                    +gas.volume)
        gas.alpha=gas.volume/self.volume


    def update_mass(self,p_Al,p_MeO,gas,gas_chamber):
        self.mass=np.sum(p_MeO.composition)+np.sum(p_Al.composition)+gas.density*gas.volume+gas_chamber.density*gas_chamber.volume
        self.sum_mass_flux=self.sum_mass_flux+np.sum(gas.dm)+p_Al.dm+p_MeO.dm+np.sum(gas_chamber.dm)


    def update_enthalpy(self,p_Al,p_MeO,gas,gas_chamber,heat_well):
        self.enthalpy=p_Al.enthalpy+p_MeO.enthalpy+gas.enthalpy*gas.density*gas.volume+gas_chamber.enthalpy*gas_chamber.density*gas_chamber.volume+heat_well.enthalpy
        self.sum_heat_flux_h=self.sum_heat_flux_h+gas.dQ+p_Al.dQ+p_MeO.dQ
        if self.i==self.i_init:
            self.sum_heat_flux_h=self.sum_heat_flux_h-integrate(self.power_input_save,self.power_input,self.dt)


    def update_energy(self,p_Al,p_MeO,gas,gas_chamber,heat_well):
        self.energy=p_Al.enthalpy+p_MeO.enthalpy+gas.energy*gas.density*gas.volume+gas_chamber.energy*gas_chamber.density*gas_chamber.volume+heat_well.enthalpy
        self.sum_heat_flux_u=self.sum_heat_flux_u+gas.dQ+p_Al.dQ+p_MeO.dQ
        if self.i==self.i_init:
            self.sum_heat_flux_u=self.sum_heat_flux_u-self.heat_input*self.dt


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    def Composition_alpha_update(self,p_Al,p_MeO,gas):
        p_Al.alpha=V_sphere(p_Al.r_ext)/self.volume*p_Al.n
        p_MeO.alpha=V_sphere(p_MeO.r_ext)/self.volume*p_MeO.n
        gas.d_alpha=-(gas.alpha-(1-p_Al.alpha-p_MeO.alpha))


    def compute_var(self, gas, p_Al, p_MeO,t_t):
        '''
        This function calculates the variance of the gas
        temperature and the particle enthalpy.
        '''
        CO_T_reach = False
        i_hist_min = 100                                                                              # Nombre minimum d'itération à prendre pour le calcul de la variance
        window = self.inputs_dic["var_window_factor"]
        # i_hist = np.where(int((self.t*window/self.dt)) < i_hist_min, i_hist_min, int((self.t*window/self.dt)))
        tmp = np.argwhere(self.t_hist>= self.t*(1-window))
        size_max_hist = 1e4
        fac_hist_red  = 10
        size_hist = len(self.T_g_hist)

        if len(tmp) == 0:
            i_hist = 0
        else:
            i_hist = len(self.t_hist) - tmp[0,0]
        # print("i_hist = ", -i_hist)
        # print("t_hist = ", t_t[-i_hist], "t = ", self.t)
        # print("T_g_hist = ", self.T_g_hist[-i_hist])
        # print("dt =", self.dt)
        # print("t =", self.t)
        # print("gasT", gas.temperature)
        # print()
        if self.t > self.t_hist[-1] + self.dt_hist:
            self.t_hist = np.append(self.t_hist, np.array([self.t]))
            self.T_g_hist = np.append(self.T_g_hist, np.array([gas.temperature]))
            self.P_g_hist = np.append(self.P_g_hist, np.array([gas.pressure]))
            self.T_pAl_hist = np.append(self.T_pAl_hist, np.array([p_Al.temperature]))
            self.h_p_hist = np.append(self.h_p_hist, np.array([p_Al.enthalpy+p_MeO.enthalpy]))

        if size_hist>size_max_hist:
            self.resize_hist(window, size_max_hist, fac_hist_red)

        if self.i>i_hist and self.test_ignition == False:

            self.var_T_g = np.var(self.T_g_hist[-i_hist:])
            self.var_T_pAl = np.var(self.T_pAl_hist[-i_hist:])
            self.var_P_g = np.var(self.P_g_hist[-i_hist:]/1e5)
            self.var_h = np.var(self.h_p_hist[-i_hist:])
            # if self.var_T_g==0:
            #     print((self.var_T_g, self.var_T_pAl, self.var_h, self.var_P_g))

            # self.var_T_g = np.var(self.T_g_hist[i_hist:])
            # self.var_T_pAl = np.var(self.T_pAl_hist[i_hist:])
            # self.var_h = np.var(self.h_p_hist[i_hist:])

            CO_T_reach = True
        if (CO_T_reach is True) and p_Al.temperature<500:
            self.var_T_g=0
            self.var_P_g=0
            self.var_T_pAl=0
            self.var_h=0
            print("Attention: Il n'y a pas combustion")

        # if self.i >= 4482:
        #     print("T123")

    def update_test_ignition(self):
        self.test_ignition = bool(self.heat_input/self.mass < self.inputs_dic["heat_input_ths"] or self.i<1)


    def compute_bool_end_simu(self):
        '''
        Stop Condition of the while loop.

        ths: threshold, stop condition on the variance

        Return:
        -------
        Booléen
        '''
        var_ths_Pg = 1e0
        var_ths_h = self.inputs_dic['var_ths_h']
        var_ths_T = self.inputs_dic['max_dTg_ite']**2       # Au carré pour être homogène à la variance
        return (self.var_T_g > var_ths_T or self.var_T_pAl > var_ths_T or self.var_h > var_ths_h or self.var_P_g > var_ths_Pg)






    def compute_var2(self, gas, p_Al, p_MeO,t_t):
        '''
        This function calculates the variance of the gas
        temperature and the particle enthalpy.
        '''
        CO_T_reach = False
        i_hist_min = 100                                                                              # Nombre minimum d'itération à prendre pour le calcul de la variance
        window = self.inputs_dic["var_window_factor"]
        size_max_hist = 1e4
        fac_hist_red  = 10
        if size_hist>size_max_hist:
            self.resize_hist(size_max_hist, fac_hist_red)

        self.T_g_hist = np.append(self.T_g_hist, np.array([gas.temperature]))
        self.P_g_hist = np.append(self.P_g_hist, np.array([gas.pressure]))
        self.T_pAl_hist = np.append(self.T_pAl_hist, np.array([p_Al.temperature]))
        self.h_p_hist = np.append(self.h_p_hist, np.array([p_Al.enthalpy+p_MeO.enthalpy]))

        if self.i>i_hist and self.test_ignition == False:

            self.var_T_g = np.var(self.T_g_hist[-i_hist:])
            self.var_T_pAl = np.var(self.T_pAl_hist[-i_hist:])
            self.var_P_g = np.var(self.P_g_hist[-i_hist:]/1e5)
            self.var_h = np.var(self.h_p_hist[-i_hist:])
            # self.var_T_g = np.var(self.T_g_hist[i_hist:])
            # self.var_T_pAl = np.var(self.T_pAl_hist[i_hist:])
            # self.var_h = np.var(self.h_p_hist[i_hist:])

            CO_T_reach = True
        if (CO_T_reach is True) and p_Al.temperature<500:
            self.var_T_g=0
            self.var_P_g=0
            self.var_T_pAl=0
            self.var_h=0
            print("Attention: Il n'y a pas combustion")


    def resize_hist(self, window, size_max_hist, fac_hist_red):
        t_hist0 = self.t * (1 - window)
        new_t_hist = np.linspace(t_hist0, self.t, int(size_max_hist / fac_hist_red))
        f = interp1d(self.t_hist, np.array([self.T_g_hist, self.P_g_hist, self.T_pAl_hist, self.h_p_hist]), fill_value="extrapolate")
        new_val = f(new_t_hist)
        self.t_hist     = new_t_hist
        self.T_g_hist   = new_val[0]
        self.P_g_hist   = new_val[1]
        self.T_pAl_hist = new_val[2]
        self.h_p_hist   = new_val[3]
        self.dt_hist    = (self.t_hist[-1] - self.t_hist[-2])