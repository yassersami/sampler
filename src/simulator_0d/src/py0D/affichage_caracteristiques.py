import numpy as np
from matplotlib import pyplot as plt

# from functions import flux_prob_evap
from py0D.functions import Arr
from py0D.species_carac import species

Al = species.Al()
MeO = species.MeO()
T = np.linspace(0, 1500, 300)
# T = np.linspace(1300+273, 1900+273, 300)
# plt.plot(T, flux_prob_evap(Al, T), label="Al")
# plt.plot(T,flux_prob_evap(MeO,T),label="MeO")
# plt.plot(1000/T,Arr(T,1e-6,50e2))
# plt.plot(T,Arr(T,1e-4,50e3))
plt.plot(T,Arr(T,1000,20e3))
# plt.plot(T,Arr(T,8e-6,50e3))
plt.title("Arr(8e-6,50e3)")
plt.xlabel("T (K)")
# plt.plot(1000/T,Arr(T,6e-2,548e3))
# plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.show()

# soit une fonction Y = A * exp(B/X)
# a partir de deux points (X1, Y1) et (X2, Y2) on veut determiner les coeffs A et B
def B_func(X1, Y1, X2, Y2):
    return np.log(Y1/Y2)*X1*X2/(X2-X1)

def A_func(X1, Y1 , B):
    return Y1/np.exp(B/X1)

def comparaison_PT_ebull():
    R = 8.3144626  # K/mol/K
    P_ref = 101325
    T_ref = 2793
    P_eb_1 = P_ref * np.exp(Al.h_evap/R*Al.MW * (1/T_ref-1/T))
    P_eb_2 = Al.K*(T**(3/2))*np.exp(-Al.h_evap*Al.MW/R/T)
    A = 5.73623
    B = 13204.106
    C = -24.306
    P_eb_3 = 10 ** (A-B/(T+C)) * 1e5
    plt.plot(T, P_eb_1, label = "Clapeyron")
    plt.plot(T, P_eb_2, label = r"$VB_{these}$")
    plt.plot(T, P_eb_3, label = "Antoine")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.title("P° vs. T° d'ebullition de l'aluminium")
    plt.ylabel("P° (Pa)")
    plt.xlabel("T° (K)")
    plt.show()

comparaison_PT_ebull()
