import os

import cantera as ct
import numpy as np

from simulator_0d.src.py0D.species_carac import species

DIR_PATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.abspath(os.path.join(DIR_PATH, '..','..','..')) 

eps = 1e-15
R = 8.31446261815324
P_ref = 101325

# it does not matter which cantera data we use as it is only used to compute
# molar weights
name_ct = PROJECT_PATH + "/data/cantera_inp/" + "AlxOy_N2_Cu_O_and_Catoire_kin.yaml"
gas_ct = ct.Solution(name_ct)

Reac_l = ct.Reaction.listFromFile(name_ct, gas_ct)
len_Reac = len(Reac_l)

name_gas_species = ["O2","Al","O","N2","Al2O","AlO","AlO2","Al2O2","Cu"]
name_p_species = ["Al","CuO","Cu","Al2O3"]


nb_gas_species = len(name_gas_species)

MW_dic = {k : gas_ct.molecular_weights[gas_ct.species_index(k)]/1000 for k in name_gas_species}
MW_dic.update({"Al2O3": MW_dic['O'] * 3 + MW_dic['Al'] * 2,
               "CuO"  : MW_dic['O'] * 1 + MW_dic['Cu'] * 1})





nb_p_species = 4

species_tab_tmp = [species.Al(MW_dic["Al"]), species.MeO(MW_dic["CuO"]), species.Me(MW_dic["Cu"]), species.Al2O3(MW_dic["Al2O3"])]
for species in species_tab_tmp:
    species.shomate[0,-2] = species.shomate[0,-2] + species.h_form
    species.shomate[1,-2] = species.shomate[1,-2] + species.h_form

thermo_coeff_p = np.array([np.concatenate([[species.T_liq], species.shomate[1], species.shomate[0]]) for species in species_tab_tmp])


ct_index = np.zeros(nb_gas_species)
for i in range (0,nb_gas_species):
    ct_index[i] = gas_ct.species_index(name_gas_species[i])

thermo_coeff_g = np.array([gas_ct.species()[int(i)].thermo.coeffs for i in ct_index]) 



list_of_element_tmp = []
dic_element_idx     = dict()
for species in gas_ct.species():
    for key in species.composition:
        list_of_element_tmp.append(key)

list_of_element = set(list_of_element_tmp)
for i,element in enumerate(list_of_element):
    dic_element_idx[element] = i
len_list_of_element = len(list_of_element)
# species
element_species_array = np.zeros((len(list_of_element), nb_gas_species))
for i in range (len(gas_ct.species())):
    for key in gas_ct.species()[int(ct_index[i])].composition:
        element_species_array[dic_element_idx[key], i] = gas_ct.species()[int(ct_index[i])].composition[key]

el_sorted_arg_species = np.zeros(element_species_array.shape)
bool_el_ratio_sup1 = np.full(element_species_array.shape, False)
for i, element in enumerate(list_of_element):
    var = np.delete(element_species_array, i, axis=0)
    sum_var = np.sum(var, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = element_species_array[i] / sum_var
    el_sorted_arg_species[i] = np.argsort(ratio)[::-1]
    bool_el_ratio_sup1[i] = ratio > 1

idx_where_el_ratio_sup1 = np.argwhere(bool_el_ratio_sup1)
el_ratio_sup1_list = []
els = np.array([])
i = 0
while i<idx_where_el_ratio_sup1.shape[0]:
    els = np.array([idx_where_el_ratio_sup1[i,1]])
    if i!=idx_where_el_ratio_sup1.shape[0]-1:
        while idx_where_el_ratio_sup1[i,0]==idx_where_el_ratio_sup1[i+1,0]:
            els = np.append(els, idx_where_el_ratio_sup1[i+1,1])
            i = i+1
            if i==idx_where_el_ratio_sup1.shape[0]-1:
                break
    # if idx_where_el_ratio_sup1[i,0]!=idx_where_el_ratio_sup1[i+1,0]:
    i = i+1
    el_ratio_sup1_list.append(els)

MW_p_vec   = [MW_dic[k] for k in name_p_species]
MW_gas_vec = [MW_dic[k] for k in name_gas_species]

MW_gas_arr = np.array(MW_gas_vec)

ips = dict(Al    = 0,
           MeO   = 1,
           Me    = 2,
           Al2O3 = 3
          )

igs = dict(O2    = 0,
           Al    = 1,
           O     = 2,
           N2    = 3,
           Al2O  = 4,
           AlO   = 5,
           AlO2  = 6,
           Al2O2 = 7,
           Cu    = 8
          )

i_np = 0
i_d  = 1
i_vx = 2
i_vy = 3
i_e  = 4
i_Y  = 5

len_Q_g = 5 + nb_gas_species




def adjust_Y_0(density, Y, Y_ths = 1e-14):
    idx_s_small = np.argwhere(Y<Y_ths)
    C = (density * Y.T).T/MW_gas_arr
    C_el = np.zeros(len_list_of_element)
    adjusted_C = np.zeros(C.shape)
    C_new = np.zeros(C.shape)
    if len(idx_s_small)!=0:
        C_el = np.sum(C[idx_s_small[0]] * element_species_array[:, idx_s_small[0]], axis = 1)
        species_maj_per_el_ratio_sup1_idx = np.empty(len_list_of_element, dtype=int)
        for i_el in range(len_list_of_element):
            arg_C_max_tmp = np.argmax(C[el_ratio_sup1_list[i_el]])
            species_maj_per_el_ratio_sup1_idx[i_el] = el_ratio_sup1_list[i_el][arg_C_max_tmp]

        el_species_mat = element_species_array[:, species_maj_per_el_ratio_sup1_idx]
        adjusted_C[species_maj_per_el_ratio_sup1_idx] = C_el @ np.linalg.inv(el_species_mat)
        adjusted_C[idx_s_small[0]] = - C[idx_s_small[0]]
    C_new = adjusted_C + C
    Y_new = C_new * MW_gas_arr / density
    # Y_new = np.where((Y_new<0) & (Y_new>-1e-20), 0, Y_new)
    return Y_new

    # def adjust_Y_0(density, Y):
    # idx_s_small = np.argwhere(Y<1e-14)
    # C = (density * Y.T).T/gd.MW_gas_arr
    # C_el = np.zeros(gd.len_list_of_element)
    # adjusted_C = np.zeros(C.shape)
    # C_new = np.zeros(C.shape)
    # if len(idx_s_small)!=0:
    #     C_el = np.sum(C[idx_s_small[0]] * gd.element_species_array[:, idx_s_small[0]], axis = 1)
    #     species_maj_per_el_ratio_sup1_idx = np.empty(gd.len_list_of_element, dtype=int)
    #     for i_el in range(gd.len_list_of_element):
    #         arg_C_max_tmp = np.argmax(C[gd.el_ratio_sup1_list[i_el]])
    #         species_maj_per_el_ratio_sup1_idx[i_el] = gd.el_ratio_sup1_list[i_el][arg_C_max_tmp]

    #     el_species_mat = gd.element_species_array[:, species_maj_per_el_ratio_sup1_idx]
    #     adjusted_C[species_maj_per_el_ratio_sup1_idx] = C_el @ np.linalg.inv(el_species_mat)
    #     adjusted_C[idx_s_small[0]] = - C[idx_s_small[0]]
    # C_new = adjusted_C + C
    # Y_new = C_new * gd.MW_gas_arr / density
    # # Y_new = np.where((Y_new<0) & (Y_new>-1e-20), 0, Y_new)
    # return Y_new