import os

import pandas as pd

# from simulator_0d.src.py0D import assist_functions
from simulator_0d.src.py0D.map_generator import label, map_generator
from simulator_0d.src.pypp.launch import set_inputs_dic, adapt_inputs_dic  # compute_Y_Al_pAl


def map_creation(df: pd.DataFrame, map_dir: str = 'default'):
    df["workdir"] = df.apply(label, axis=1)

    constants = {
        "alpha_p": 0.3,
        "th_Al2O3": 5.12281599140251e-08,
        "heat_input_ths": 556000,
        "power_input": 20000000,
        "coeff_contact": 0.01,
        "Ea_D_Ox": 50000,
        "Ea_Al2O3_decomp": 400000,
        "Ea_MeO_decomp": 50000,
        "k0_D_Ox": 0.000008,
        "k0_Al2O3_decomp": 1520000,
        "k0_MeO_decomp": 30000000,
        "bool_kin": 'false'
    }
    
    for col, val in constants.items(): # don't set a feature constant if it's an input # ? rs_inputs 
        if col not in df.columns:
            df[col] = val

    if map_dir == 'default':
        df.apply(map_generator, axis=1)
    else:
        df.apply(lambda sample: map_generator(sample, map_dir), axis=1)


def define_in_out(input_dir: str):
    inputs_dic = set_inputs_dic(input_dir)
    inputs_dic = adapt_inputs_dic(inputs_dic)
    output_dir = os.path.abspath(os.path.join(input_dir, 'outputs'))
    return inputs_dic, output_dir


# def join_path(a, *p):
#     return os.path.abspath(os.path.join(a, *p))


# def set_inputs_dic(input_dir):
#     PATH_JSON_var = join_path(input_dir, 'inputs_var.json')
#     PATH_JSON_fix = join_path(input_dir, 'inputs_fix.json')
#     inputs_dic = assist_functions.load_json_from_file(PATH_JSON_var)
#     inputs_dic.update(assist_functions.load_json_from_file(PATH_JSON_fix))
#     return inputs_dic


# def adapt_inputs_dic(inputs_dic):
#     if "th_Al2O3" in [*inputs_dic]:
#         inputs_dic["Y_Al_pAl"] = compute_Y_Al_pAl(
#             r_ext_pAl=inputs_dic["r_ext_pAl"],
#             th_Al2O3=inputs_dic.pop("th_Al2O3")
#         )
#     return inputs_dic


# def output_dir(simu_dir):
#     return os.path.abspath(os.path.join(simu_dir, 'outputs'))
