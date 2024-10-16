import os
import pickle
import sys

import numpy as np

import module_kin as mkin
from functions import kinetics

PATH_0D = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..','..'))
sys.path.append(PATH_0D)

import py0D.global_data as gd

gas = pickle.load(open("gas_0D.pkl", "rb"))
dt = pickle.load(open("dt_0D.pkl", "rb"))/20
volume = pickle.load(open("volume_0D.pkl", "rb"))

def amplify_modif_gas(gas, fac = 1):
    gas.dm_dt_flux = gas.dm_dt_flux * fac
    gas.dQ_dt = gas.dQ_dt * fac
    return gas

def int_gas_0D(gas, dt):
    gas.Y=(gas.density*gas.volume*gas.Y+gas.dm_dt_flux * dt)/(gas.density*gas.volume+np.sum(gas.dm_tot)*dt)                                     #schéma implicite
    gas.density=(gas.density*gas.volume+np.sum(gas.dm_dt_flux)*dt)/((gas.alpha+gas.d_alpha)*volume)                                         #schéma implicite
    gas.energy=(gas.density*volume*gas.energy*gas.alpha+gas.dQ_dt*dt)/(gas.density*volume*gas.alpha+np.sum(gas.dm_dt_flux)*dt)             #schéma implicite

    gas.dm_dt_flux = np.zeros(gas.dm_dt_flux.shape)
    gas.dQ_dt = 0
    return gas

gas = amplify_modif_gas(gas, fac = 5)
# gas = int_gas_0D(gas, dt)


gas.gas_ct = gd.gas_ct
gas.int_BDF = None




dm_cin_dt, int_BDF = kinetics(gas, volume, dt*100)
dmi_dt_kin, int_BDF, t_BDF, Q_BDF = mkin.compute_kin(gas, dt, volume, int_BDF = gas.int_BDF, bool_get_hist = True)

print()


