import json
import os
import pickle
import time
from configparser import ConfigParser
from copy import deepcopy

import numpy as np


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def open_data_time(data_name):
    data = np.loadtxt("/data/"+str(data_name)+".txt", delimiter="\n")
    open("data/"+str(data_name)+".txt",  'w').close()
    return data


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_data_time(data,data_name):
    VS_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..'))

    fichier=open(str(VS_PATH)+"/data/saved_additional_data/"+str(data_name)+".txt",'a')
    fichier.write(str(data)+"\n")
    fichier.close()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def print_data(gas,p_Al,p_MeO,system,ite_print,*args):
    print("gas:")
    print("            Density           :%.4e kg/m3" %gas.density,                       "               alpha=%.1f " %(gas.alpha*100))
    print("            Pressure          :%.4e Pa   " %gas.pressure,                      "            enthalpy=%.4e J/kg" %gas.enthalpy   )
    print("            Temperature       :%.5f K    " %gas.temperature,                   "              energy=%.4e J/kg" %gas.energy     )
    if args:
        gas_chamber=args[0]
        heat_well=args[1]
        print("gas chamber:")
        print("            Density           :%.4e kg/m3" %gas_chamber.density,           "         temperature=%.1f " %(gas_chamber.alpha*100))
        print("            Pressure          :%.4e Pa   " %gas_chamber.pressure,          "            enthalpy=%.4e J/kg" %gas_chamber.enthalpy)
        print("            Temperature       :%.1f K    " %gas_chamber.temperature,       "              energy=%.4e J/kg" %gas_chamber.energy)
        print("heat well:")
        print("            Temperature       :%.1f K    " %heat_well.temperature)
    print()
    print("particles:")
    print("            P Al temperature  :%.1f K" %p_Al.temperature)
    print("            P MeO temperature :%.1f K" %p_MeO.temperature)
    print()
    print("system:")
    print("            total mass        :%.5e kg" %system.mass,                          "             log(sum_mass_flux)=%d" %np.log10(np.abs(system.sum_mass_flux)).astype(int))
    print("            total volume      :%.5e m3" %system.volume)
    print("            total enthalpy    :%.5e J" %system.enthalpy,                       "             log(sum_heat_flux_h)=%d" %np.log10(np.abs(system.sum_heat_flux_h)).astype(int))
    print("            total energy      :%.5e J" %(system.energy),                         "             log(sum_heat_flux_u)=%d" %np.log10(np.abs(system.sum_heat_flux_u)).astype(int))
    print("            dt limittant, species limittant      :%s, %s  " %system.dt_restrict)
    print("            var Tg, var TpAl, var h, var P      :%.2e" %system.var_T_g , "%.2e" %system.var_T_pAl, "%.2e" %system.var_h, "%.2e" %system.var_P_g)
    print("time to compute %d iteration : %.5f" %(ite_print, time.time()-system.t_chron))
    system.t_chron = time.time()
    print("-------")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def print_start_loop(system, ite_print):
    if system.i % ite_print == 0:
        print("\n--------  %.1e"%system.t_f,
        "start of iteration: ",system.i,
        "  ----  avancement:%.1f" %(system.t/system.t_f*100),
        "------t=%.5e" %system.t ,
        "------dt=%.1e" %system.dt ,
        "-----------")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def time_save_dic(p_Al,p_MeO,gas,system,p_Al_d,p_MeO_d,gas_d,system_d,*args):
    gas_keys_no_save={"name","species_carac", "gas_ct","int_BDF",
                      "T_P_Al_evap","T_P_Me_evap",
                      "nb_species","Lagr_multiplier", "MW_vec", 
                      "Y_save", "dY_dt"}

    particle_keys_no_save={"name","species_carac", "dm", "dh", "lambda_species", "lambda_p"}

    system_keys_no_save={"var_T_g", "var_T_pAl", "var_h", "inputs_dic", "T_g_hist", "T_pAl_hist", "h_p_hist"}

    gas_d.append(deepcopy(without_keys(gas.__dict__,gas_keys_no_save)))
    p_Al_d.append(deepcopy(without_keys(p_Al.__dict__,particle_keys_no_save)))
    p_MeO_d.append(deepcopy(without_keys(p_MeO.__dict__,particle_keys_no_save)))
    system_d.append(deepcopy(without_keys(system.__dict__,system_keys_no_save)))
    if args:
        gas_chamber=args[0]
        gas_chamber_d=args[1]
        heat_well=args[2]
        heat_well_d=args[3]
        gas_chamber_d.append(deepcopy(without_keys(gas_chamber.__dict__,gas_keys_no_save)))

        # exit()
        heat_well_d.append(deepcopy(heat_well.__dict__))


def time_save_flux_dic(flux_pAl,flux_pMeO,source_pAl,source_pMeO,heat_flux_pAl,heat_flux_pMeO,flux_chamber,flux_HW,
                    flux_pAl_d,flux_pMeO_d,source_pAl_d,source_pMeO_d,heat_flux_pAl_d,heat_flux_pMeO_d,flux_chamber_d,flux_HW_d):

    flux_keys_no_save={"_ext_history","_ext_history_wo_mwa", "t_history"}
    flux_pAl_d.append(deepcopy(without_keys(flux_pAl[1].__dict__,flux_keys_no_save)))
    flux_pMeO_d.append(deepcopy(without_keys(flux_pMeO[1].__dict__,flux_keys_no_save)))
    source_pAl_d.append(deepcopy(source_pAl[1].__dict__))
    source_pMeO_d.append(deepcopy(source_pMeO[1].__dict__))
    heat_flux_pAl_d.append(deepcopy(heat_flux_pAl[1].__dict__))
    heat_flux_pMeO_d.append(deepcopy(heat_flux_pMeO[1].__dict__))
    flux_chamber_d.append(deepcopy(flux_chamber[1].__dict__))
    flux_HW_d.append(deepcopy(without_keys(flux_HW[1].__dict__,{"conductivity"})))


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# def save_pickle(*args):
#     pickle.dump(args[0],open(path_pickle+'/'+"t_t",'wb'))
#     pickle.dump(args[1],open(path_pickle+'/'+"system_d",'wb'))
#     pickle.dump(args[2],open(path_pickle+'/'+"p_Al_d",'wb'))
#     pickle.dump(args[3],open(path_pickle+'/'+"p_MeO_d",'wb'))
#     pickle.dump(args[4],open(path_pickle+'/'+"p_gas_d",'wb'))
#     pickle.dump(args[5],open(path_pickle+'/'+"gas_chamber_d",'wb'))
#     pickle.dump(args[6],open(path_pickle+'/'+"heat_well_d",'wb'))


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def import_pickle(*args):
    VS_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..'))

    if args:
        PATH_simu=str("/data/sauv_simus/"+args[0]+"/pickle")
    else:
        PATH_simu="/data/pickle"

    path_pickle=str(VS_PATH+PATH_simu)
    t_t=pickle.load(open(path_pickle+'/'+"t_t",'rb'))
    p_Al_d=pickle.load(open(path_pickle+'/'+"p_Al_d",'rb'))
    p_MeO_d=pickle.load(open(path_pickle+'/'+"p_MeO_d",'rb'))
    gas_d=pickle.load(open(path_pickle+'/'+"p_gas_d",'rb'))
    gas_chamber_d=pickle.load(open(path_pickle+'/'+"gas_chamber_d",'rb'))
    system_d=pickle.load(open(path_pickle+'/'+"system_d",'rb'))
    heat_well_d=pickle.load(open(path_pickle+'/'+"heat_well_d",'rb'))

    return t_t,system_d,p_Al_d,p_MeO_d,gas_d,gas_chamber_d,heat_well_d

# def save_flux_pickle(*args):
#     pickle.dump(args[0],open(path_pickle+'/'+"flux_pAl_d",'wb'))
#     pickle.dump(args[1],open(path_pickle+'/'+"flux_pMeO_d",'wb'))
#     pickle.dump(args[2],open(path_pickle+'/'+"source_pAl_d",'wb'))
#     pickle.dump(args[3],open(path_pickle+'/'+"source_pMeO_d",'wb'))
#     pickle.dump(args[4],open(path_pickle+'/'+"heat_flux_pAl_d",'wb'))
#     pickle.dump(args[5],open(path_pickle+'/'+"heat_flux_pMeO_d",'wb'))
#     pickle.dump(args[6],open(path_pickle+'/'+"flux_chamber_d",'wb'))
#     pickle.dump(args[7],open(path_pickle+'/'+"flux_HW_d",'wb'))

def import_flux_pickle(*args):
    VS_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..'))
    if args:
        PATH_simu=str("/data/sauv_simus/"+args[0]+"/pickle")
    else:
        PATH_simu="/data/pickle"

    path_pickle=str(VS_PATH+PATH_simu)
    flux_pAl_d=pickle.load(open(path_pickle+'/'+"flux_pAl_d",'rb'))
    flux_pMeO_d=pickle.load(open(path_pickle+'/'+"flux_pMeO_d",'rb'))
    source_pAl_d=pickle.load(open(path_pickle+'/'+"source_pAl_d",'rb'))
    source_pMeO_d=pickle.load(open(path_pickle+'/'+"source_pMeO_d",'rb'))
    heat_flux_pAl_d=pickle.load(open(path_pickle+'/'+"heat_flux_pAl_d",'rb'))
    heat_flux_pMeO_d=pickle.load(open(path_pickle+'/'+"heat_flux_pMeO_d",'rb'))
    flux_chamber_d=pickle.load(open(path_pickle+'/'+"flux_chamber_d",'rb'))
    flux_HW_d=pickle.load(open(path_pickle+'/'+"flux_HW_d",'rb'))
    return flux_pAl_d,flux_pMeO_d,source_pAl_d,source_pMeO_d,heat_flux_pAl_d,heat_flux_pMeO_d,flux_chamber_d,flux_HW_d

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def vectorize(state, var):
    if isinstance(getattr(state[0], var),  np.ndarray):
        vect_var = np.zeros([len(state), len(getattr(state[0], var))])
    else:
        vect_var = np.zeros(len(state))
    for i in range(0, len(state)):
        vect_var[i] = getattr(state[i], var)
    return vect_var

def vectorize_dic(list_dic, var,begin=0):
    if isinstance(list_dic[begin][var],  np.ndarray):
        vect_var = np.zeros([len(list_dic)-begin, len(list_dic[begin][var])])
    elif isinstance(list_dic[begin][var],  list):
        vect_var = np.full([len(list_dic)-begin, len(list_dic[begin][var])], '___')
    elif isinstance(list_dic[begin][var],dict):
        vect_var={k:[] for k in list_dic[begin][var]}
        for i in range (begin,len(list_dic)):
            for k in list_dic[i][var]:
                if k not in vect_var:
                    vect_var[k]=[0 for j in range(i-1)]
                vect_var[k].append(list_dic[i][var][k])

        return vect_var
    elif isinstance(list_dic[begin][var],tuple):
        vect_var=np.full((len(list_dic),2),'________')
    else:
        vect_var = np.zeros(len(list_dic)-begin)
    for i in range(begin, len(list_dic)):
        vect_var[i] = list_dic[i][var]
    return vect_var


def save_ite(p_Al, p_MeO, gas, system, gas_chamber, heat_well,
            flux_pAl,flux_pMeO,source_pAl,source_pMeO,heat_flux_pAl,heat_flux_pMeO,flux_chamber,flux_HW):
    VS_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..'))
    path_pickle=str(VS_PATH+"/data/pickle_save_ite")
    pickle.dump(p_Al,open(path_pickle+'/'+"p_Al",'wb'))
    pickle.dump(p_MeO,open(path_pickle+'/'+"p_MeO",'wb'))
    pickle.dump(gas,open(path_pickle+'/'+"gas",'wb'))
    pickle.dump(system,open(path_pickle+'/'+"system",'wb'))
    pickle.dump(gas_chamber,open(path_pickle+'/'+"gas_chamber",'wb'))
    pickle.dump(heat_well,open(path_pickle+'/'+"heat_well",'wb'))
    pickle.dump(heat_well,open(path_pickle+'/'+"heat_well",'wb'))

    pickle.dump(flux_pAl,open(path_pickle+'/'+"flux_pAl",'wb'))
    pickle.dump(flux_pMeO,open(path_pickle+'/'+"flux_pMeO",'wb'))
    pickle.dump(source_pAl,open(path_pickle+'/'+"source_pAl",'wb'))
    pickle.dump(source_pMeO,open(path_pickle+'/'+"source_pMeO",'wb'))
    pickle.dump(heat_flux_pAl,open(path_pickle+'/'+"heat_flux_pAl",'wb'))
    pickle.dump(heat_flux_pMeO,open(path_pickle+'/'+"heat_flux_pMeO",'wb'))
    pickle.dump(flux_chamber,open(path_pickle+'/'+"flux_chamber",'wb'))
    pickle.dump(flux_HW,open(path_pickle+'/'+"flux_HW",'wb'))


def load_ite():
    VS_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                           '..', '..'))
    path_pickle=str(VS_PATH+"/data/pickle_save_ite")
    p_Al=pickle.load(open(path_pickle+'/'+"p_Al",'rb'))
    p_MeO=pickle.load(open(path_pickle+'/'+"p_MeO",'rb'))
    gas=pickle.load(open(path_pickle+'/'+"gas",'rb'))
    system=pickle.load(open(path_pickle+'/'+"system",'rb'))
    gas_chamber=pickle.load(open(path_pickle+'/'+"gas_chamber",'rb'))
    heat_well=pickle.load(open(path_pickle+'/'+"heat_well",'rb'))

    flux_pAl=pickle.load(open(path_pickle+'/'+"flux_pAl",'rb'))
    flux_pMeO=pickle.load(open(path_pickle+'/'+"flux_pMeO",'rb'))
    source_pAl=pickle.load(open(path_pickle+'/'+"source_pAl",'rb'))
    source_pMeO=pickle.load(open(path_pickle+'/'+"source_pMeO",'rb'))
    heat_flux_pAl=pickle.load(open(path_pickle+'/'+"heat_flux_pAl",'rb'))
    heat_flux_pMeO=pickle.load(open(path_pickle+'/'+"heat_flux_pMeO",'rb'))
    flux_chamber=pickle.load(open(path_pickle+'/'+"flux_chamber",'rb'))
    flux_HW=pickle.load(open(path_pickle+'/'+"flux_HW",'rb'))

    return (p_Al, p_MeO, gas, system, gas_chamber, heat_well,
            flux_pAl,flux_pMeO,source_pAl,source_pMeO,heat_flux_pAl,heat_flux_pMeO,flux_chamber,flux_HW)




def json_file_wo_coments(str_var):
    str_out = ''
    lines = str_var.split('\n')
    for line in lines:
        str_out += line.split('#')[0].strip()
    return str_out


def dump_json_to_file(dic, json_filename):
    """si génération du fichier json à partir de inputs"""
    dic = dict()
    # for key in inputs.__dict__:
    #     if key[0]!='_':
    #         dic[key] = inputs.__dict__[key]

    with open(json_filename, 'w') as json_file:
        json.dump(dic, json_file, indent=4)


def load_json_from_file(json_filename):
    with open(json_filename) as json_file:
        str_var = json_file.read()
    dic = json.loads(json_file_wo_coments(str_var))
    return dic


# storepath = 'test_config.ini'
# write_config(storepath)
# config_dict = read_config(storepath)


    # def ct_composition_input(Y):
    #     ct_composition_input=("O2:"       + str(Y[0])
    #                             +",Al:"     + str(Y[1])
    #                             +",O:"      + str(Y[2])
    #                             +",N2:"     + str(Y[3])
    #                             +",Al2O:"   + str(Y[4])
    #                             +",AlO:"    + str(Y[5])
    #                             +",AlO2:"   + str(Y[6])
    #                             +",Al2O2:"  + str(Y[7])
    #                             +",Cu:"     + str(Y[8])
    #                             )
    #     return ct_composition_input


# WRITING
# def write_config(storepath):
#     config_object = ConfigParser()
#     inputs_data = dict()
#     config_object.optionxform = str
#     #
#     for key in inputs.__dict__:
#         if key[0]!='_':
#             inputs_data[key] = inputs.__dict__[key]
#     config_object['MODEL_CONFIG'] = inputs_data
#     #
#     with open(storepath, 'w') as conf:
#         config_object.write(conf)


def del_inline_coments(str_var): 
    return str_var.split('#')[0].strip()

# READING
def read_config(storepath):
    config_object = ConfigParser()
    config_object.optionxform = str
    config_object.read(storepath)
    config_object.read(storepath)
    #
    config_dict = dict(config_object["MODEL_CONFIG"])
    for k in config_dict:
        config_dict[k] = json.loads(json.dumps((del_inline_coments(config_dict[k]).lower())))
    return config_dict