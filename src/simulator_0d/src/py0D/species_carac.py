import numpy as np
# Les masses molaires des espèces condensées sont directements issues des masses molaires cantera

def import_MW_from_gas(gas_ct):
    global MW_Al
    global MW_O
    global MW_Cu
    global MW_N

    MW_Al = gas_ct.molecular_weights[gas_ct.species_index('Al')]/1000
    MW_O = gas_ct.molecular_weights[gas_ct.species_index('O')]/1000
    MW_Cu = gas_ct.molecular_weights[gas_ct.species_index('Cu')]/1000
    MW_N = gas_ct.molecular_weights[gas_ct.species_index('N2')]/ 2 /1000

class species(): # All units SI
    def __init__(self,name,MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T,*args):
        self.name=name
        self.MW=MW #kg/mol
        self.h_form=h_form #J/kg
        self.h_evap=h_evap #J/kg
        self.T_evap=T_evap #K
        self.T_liq=T_liq
        self.h_liq=h_liq
        self.shomate=shomate*1000/self.MW#/self.MW #tableau pour 3 phases des coefficients shomate
        self.K=K #coefficient de decomposition/vaporisation des espèces solides
        self.r_VDW=r_VDW
        if len(args)>0:
            rho=args[0]
            self.rho=rho
        self.gas_phase_T=gas_phase_T #temperatures où les fonctions d'enthalpie/Cp changent pour les gaz. Cf: Nist webbook


    def Al(MW_Al):
        MW = MW_Al
        h_form=0
        h_liq=10.56e3/MW
        T_liq=933.45
        h_evap=294.1433599443601e3/MW
        T_evap=2793
        shomate=np.zeros([3,7])
        K=2.15e5 #evaporation d'Al
        rho=2700

        # shomate solid
        shomate[0,0]=28.08920
        shomate[0,1]=-5.414849
        shomate[0,2]=8.560423
        shomate[0,3]=3.427370
        shomate[0,4]=-0.277375
        shomate[0,5]=-9.147187
        shomate[0,6]=0.000000

        # shomate liquid
        shomate[1,0]=31.75104
        shomate[1,1]=3.935826e-8
        shomate[1,2]=-1.786515e-8
        shomate[1,3]=2.694171e-9
        shomate[1,4]=5.480037e-9
        shomate[1,5]=-0.945684
        shomate[1,6]=10.56201+0.1481671282556043-h_liq*MW/1000 
        #le coefficient 0.148 et l'enthalpie de liquefaction sont ajoutés pour assurer la continuité de l'enthalpie de l'espèce en phase liquide à l'enthalpie de l'espèce en phase solide
        #Ainsi lorsqu'on trace la temperature en fonction de l'enthalpie, la fonction est continue

        # shomate gas
        shomate[2,0]=20.37692
        shomate[2,1]=0.660817
        shomate[2,2]=-0.313631
        shomate[2,3]=0.045106
        shomate[2,4]=0.078173
        shomate[2,5]=323.8575
        shomate[2,6]=0#329.6992 #le zero ici sert a prendre en compte l'enthalpie ajoutée lors de la vaporization et liquéfaction

        gas_phase_T=[0,0]
        #Pour connaitre la fonction de H(T), prendra toujours l'indice "2" pour l'aluminium. Cf: fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        r_VDW=211 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        Al=species("Al",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T,rho)
        return Al


    def Al2O3(MW_Al2O3):
        MW=MW_Al2O3
        h_form=-1675.692e3/MW #A T° ambiente (janaf)
        h_liq=(55e3)/MW #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=2327
        h_evap=None
        T_evap=1e5
        shomate=np.zeros([3,7])
        K=1.04e27
        rho=3950

        # shomate solid
        shomate[0,0]=102.4290
        shomate[0,1]=38.74980
        shomate[0,2]=-15.91090
        shomate[0,3]=2.628181
        shomate[0,4]=-3.007551
        shomate[0,5]=-1717.930
        shomate[0,6]=-1675.690

        # shomate liquid
        shomate[1,0]=192.4640
        shomate[1,1]=9.519856e-8
        shomate[1,2]=-2.858928e-8
        shomate[1,3]=2.929147e-9
        shomate[1,4]=5.599405e-8
        shomate[1,5]=-1757.711
        shomate[1,6]=-1620.568+0.9653980966011773 #cf explication Al

        #we dont use it so we dont care about tha accurency of the
        #radius of the Al2O3 molecule
        r_VDW=878 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        Al2O3=species("Al2O3",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,None,rho)
        return Al2O3


    def MeO(MW_MeO):
        MW=MW_MeO
        h_form=-156.06e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1e5
        h_evap=180e3/MW #somme des énergies des décompositions CuO & Cu2O (UN PEU ALEATOIRE COMME VALEUR)
        T_evap=1e5
        shomate=np.zeros([3,7])
        K=8.69e2*2e6 #on triche un peu en prenant la cinétique la plus lente (entre CuO et Cu2O) de thèse Baijot
        rho=6315

        # shomate solid
        shomate[0,0]=48.56494
        shomate[0,1]=7.498607
        shomate[0,2]=-0.055980
        shomate[0,3]=0.013851
        shomate[0,4]=-0.760082
        shomate[0,5]=-173.4272
        shomate[0,6]=-156.0632

        #we dont use it so we dont care about tha accurency of the
        #radius of the Al2O3 molecule
        r_VDW=352 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        MeO=species("MeO",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,None,rho)
        return MeO


    def Me(MW_Me):
        MW=MW_Me
        h_form=0
        h_liq=11.86e3/MW
        T_liq=1358
        h_evap=301.9505625461548e3/MW #(diff entre h form gaz & liquide)
        T_evap=2843
        shomate=np.zeros([3,7])
        K=1.11e6 #evaporation Cu Vincent Baijot
        K=2.36e5 #evaporation Cu recalculé
        rho=8960

        # shomate solid
        shomate[0,0]=17.72891
        shomate[0,1]=28.09870
        shomate[0,2]=-31.25289
        shomate[0,3]=13.97243
        shomate[0,4]=0.068611
        shomate[0,5]=-6.056591
        shomate[0,6]=0.000000

        # shomate liquid
        shomate[1,0]=32.84450
        shomate[1,1]=-0.000084
        shomate[1,2]=0.000032
        shomate[1,3]=-0.000004
        shomate[1,4]=-0.000028
        shomate[1,5]=-1.804901
        shomate[1,6]=11.85730-h_liq/1000*MW+1.2723428764294056

        # shomate gas    
        shomate[2,0]=-80.48635
        shomate[2,1]=49.35865
        shomate[2,2]=-7.578061
        shomate[2,3]=0.404960
        shomate[2,4]=133.3382
        shomate[2,5]=519.9331
        shomate[2,6]=0#337.6003 #le zero ici sert a prendre en compte l'enthalpie ajoutée lors de la vaporization et liquéfaction

        gas_phase_T=[0,0]
        #Pour connaitre la fonction de H(T), prendra toujours l'indice "2" pour le metal. Cf: fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        r_VDW=200 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        Me=species("Me",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T,rho)
        return Me


    def O2(MW_O2):
        MW=MW_O2
        h_form=0 #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None        

        #T=100-700
        shomate[0,0]=31.32234    
        shomate[0,1]=-20.23531    
        shomate[0,2]=57.86644    
        shomate[0,3]=-36.50624    
        shomate[0,4]=-0.007374    
        shomate[0,5]=-8.903471    
        shomate[0,6]=0

        #T=700-2000
        shomate[1,0]=30.03235
        shomate[1,1]=8.772972
        shomate[1,2]=-3.988133
        shomate[1,3]=0.788313
        shomate[1,4]=-0.741599
        shomate[1,5]=-11.32468
        shomate[1,6]=0.0

        #T=2000-6000
        shomate[2,0]=20.91111
        shomate[2,1]=10.72071
        shomate[2,2]=-2.020498
        shomate[2,3]=0.146449
        shomate[2,4]=9.245722
        shomate[2,5]=5.337651
        shomate[2,6]=0

        gas_phase_T=[700,2000]
        #700 et 2000 sont les temperatures auxquelles les coefficients de shomate changent pour cette espèce gazeuse. Pour trouver quels coefficients on prend en fonction de la temperature,
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_O2 = 2 x r_O
        r_VDW=304 # Van Der Waals Radii in picometer #Bondi, 1964

        O2=species("O2",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return O2


    def N2(MW_N2):
        MW=MW_N2
        h_form=-0 #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        #100-500
        shomate[0,0]=28.98641
        shomate[0,1]=1.853978
        shomate[0,2]=-9.647459
        shomate[0,3]=16.63537
        shomate[0,4]=0.000117
        shomate[0,5]=-8.671914
        shomate[0,6]=0

        #500-2000
        shomate[1,0]=19.50583
        shomate[1,1]=19.88705
        shomate[1,2]=-8.598535
        shomate[1,3]=1.369784
        shomate[1,4]=0.527601
        shomate[1,5]=-4.935202
        shomate[1,6]=0

        # 2000-6000
        shomate[2,0]=35.51872
        shomate[2,1]=1.128728
        shomate[2,2]=-0.196103
        shomate[2,3]=0.014662
        shomate[2,4]=-4.553760
        shomate[2,5]=-18.97091
        shomate[2,6]=0

        gas_phase_T=[500,2000]
        #500 et 2000 sont les temperatures auxquelles les coefficients de shomate changent pour cette espèce gazeuse. Pour trouver quels coefficients on prend en fonction de la temperature,
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_N = 2 x r_N
        r_VDW=310 # Van Der Waals Radii in picometer #Bondi, 1964

        N2=species("N2",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return N2


    def Al2O(MW_Al2O):
        MW=MW_Al2O
        h_form=-145.185e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        shomate[2,0]=58.99022
        shomate[2,1]=2.945512
        shomate[2,2]=-0.861624
        shomate[2,3]=0.085153
        shomate[2,4]=-0.725284
        shomate[2,5]=-165.3270
        shomate[2,6]=-145.1852

        gas_phase_T=[0,0]
        #ce gaz n'a qu'un seul jeu de coefficients de shomate en fonction de la T°. ces coefficients sont stockés dans l'index "2". Lors de la verification de l'index de shomate,
        #la temperature sera forcément > 0. Donc on prendra bien le coefficient "2"
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_Al2O = 2*r_Al+r_O
        r_VDW=574 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        Al2O=species("Al2O",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return Al2O


    def AlO(MW_AlO):
        MW=MW_AlO
        h_form=66.944e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        # 298-2000K
        shomate[1,0]=35.53572
        shomate[1,1]=-3.947982
        shomate[1,2]=7.684670
        shomate[1,3]=-1.626841
        shomate[1,4]=-0.371449
        shomate[1,5]=55.19826
        shomate[1,6]=66.94400

        #2000-6000K
        shomate[2,0]=74.71327
        shomate[2,1]=-9.426846
        shomate[2,2]=0.739790
        shomate[2,3]=0.011019
        shomate[2,4]=-55.39407
        shomate[2,5]=-27.74339
        shomate[2,6]=66.94400

        gas_phase_T=[0,2000]
        #0 et 2000 sont les temperatures auxquelles les coefficients de shomate changent pour cette espèce gazeuse. Pour trouver quels coefficients on prend en fonction de la temperature,
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_AlO = r_Al+r_O
        r_VDW=363 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        AlO=species("AlO",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return AlO


    def AlO2(MW_AlO2):
        MW=MW_AlO2
        h_form=-86.19e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        # 298-1000
        shomate[1,0]=39.43040
        shomate[1,1]=58.35630
        shomate[1,2]=-57.07000
        shomate[1,3]=19.94580
        shomate[1,4]=-0.213463
        shomate[1,5]=-100.7920
        shomate[1,6]=-86.19040

        #1000-6000K
        shomate[2,0]=65.18890
        shomate[2,1]=-2.675200
        shomate[2,2]=0.788830
        shomate[2,3]=-0.053813
        shomate[2,4]=-2.866541
        shomate[2,5]=-112.9770
        shomate[2,6]=-86.19040

        gas_phase_T=[0,1000]
        #0 et 1000 sont les temperatures auxquelles les coefficients de shomate changent pour cette espèce gazeuse. Pour trouver quels coefficients on prend en fonction de la temperature,
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_AlO2 = r_Al + 2*r_O
        r_VDW=515 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        AlO2=species("AlO2",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return AlO2


    def Al2O2(MW_Al2O2):
        MW=MW_Al2O2
        h_form=-394.551e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        shomate[2,0]=80.97797
        shomate[2,1]=1.839923
        shomate[2,2]=-0.497222
        shomate[2,3]=0.042360
        shomate[2,4]=-1.303940
        shomate[2,5]=-423.1447
        shomate[2,6]=-394.5516
        gas_phase_T=[0,0]
        #ce gaz n'a qu'un seul jeu de coefficients de shomate en fonction de la T°. ces coefficients sont stockés dans l'index "2". Lors de la verification de l'index de shomate,
        #la temperature sera forcément > 0. Donc on prendra bien le coefficient "2"
        #on utilise la fonction "trouver indice de shomate de la phase gaz" appelée dans le programme: "find_gas_phase_index"

        #appriximation: r_Al2O2 = sqrt(2)*(r_Al+r_O) (square form of the molecule)
        r_VDW=513 # Van Der Waals Radii in picometer #Van der Waals Radii of element

        Al2O2=species("Al2O2",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return Al2O2


    def O(MW_O):
        MW=MW_O
        h_form=249.173e3/MW #A T° ambiente (janaf)
        h_liq=None #égal à la diff entre enthalpie de formation solide et liquide (nist)
        T_liq=1
        h_evap=None
        T_evap=1
        shomate=np.zeros([3,7])
        K=None

        shomate[2,0]=21
        #Pas de fonction de H en fonction de T pour l'oxygene. On considère Cp=21J/mol/K (source?). Cela revient à dire que tous les coefficients de shomate sont nuls à l'exception de A.
        #en effet shomate:Cp=A + Bt + Ct^2 + Dt^3 + E/t^2
        #Question: peut-on appliquer alors l'equation de l'enthalpie uniquement sur le coefficient "A"??? 
        gas_phase_T=[0,0]

        r_VDW=152 # Van Der Waals Radii in picometer #Bondi, 1964

        O=species("O",MW,h_form,h_liq,T_liq,h_evap,T_evap,shomate,K,r_VDW,gas_phase_T)
        return O




