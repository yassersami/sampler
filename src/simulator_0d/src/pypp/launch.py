if __name__ == '__main__':
    # Set module path if testing script
    import os
    import sys
    PATH_src = os.path.abspath(os.path.join(
        os.path.abspath(__file__),
        '..', '..'
    ))
    if PATH_src not in sys.path:
        sys.path.append(PATH_src)
    # print('sys.path:', *sys.path, sep='\n')

import json
import os

import numpy as np

import simulator_0d.src.py0D.assist_functions as assist_functions
import simulator_0d.src.py0D.main as main
import simulator_0d.src.pypp.get_opt as get_opt
from simulator_0d.src.py0D.functions_pp import DB_simu_pp


def join_path(a, *p): return os.path.abspath(os.path.join(a, *p))


def set_inputs_dic(input_dir):
    '''
    Set simulation environment
    
    Arguments
    ---------
    input_dir: string
        Path of simulation directory containing json input files and future
        outputs.
 
    Return
    ------
    inputs_dic: dict
        dict of all necessary arguments for simulator
    '''

    # Default paths
    PATH_JSON_var = join_path(input_dir, 'inputs_var.json')
    PATH_JSON_fix = join_path(input_dir, 'inputs_fix.json')

    # print(f'input_dir: {input_dir}')
    # print(f'PATH_JSON_var: {PATH_JSON_var}')
    # print(f'PATH_JSON_fix: {PATH_JSON_fix}')

    inputs_dic = assist_functions.load_json_from_file(PATH_JSON_var)
    inputs_dic.update(assist_functions.load_json_from_file(PATH_JSON_fix))

    return inputs_dic


def compute_Y_Al_pAl(r_ext_pAl, th_Al2O3):
    rho_Al = 2700  # [kg.m^-3]
    rho_Al2O3 = 3950  # [kg.m^-3]
    r_int_pAl = r_ext_pAl - th_Al2O3
    V_Al = 4*np.pi/3 * r_int_pAl**3
    V_Al2O3 = 4*np.pi/3 * (r_ext_pAl**3-r_int_pAl**3)
    return rho_Al*V_Al / (rho_Al*V_Al + rho_Al2O3*V_Al2O3)


def compute_th_Al2O3(r_ext_pAl, Y_Al_pAl):
    rho_Al = 2700  # [kg.m^-3]
    rho_Al2O3 = 3950  # [kg.m^-3]
    a = rho_Al2O3*Y_Al_pAl/(1-Y_Al_pAl)
    r_int_pAl = (a*r_ext_pAl**3/(rho_Al+a))**(1/3)
    return r_ext_pAl - r_int_pAl


def adapt_inputs_dic(inputs_dic):
    if "Y_Al_pAl" not in [*inputs_dic]: # ? rs_inputs
        if "th_Al2O3" in [*inputs_dic]:
            inputs_dic["Y_Al_pAl"] = compute_Y_Al_pAl(
                r_ext_pAl=inputs_dic["r_ext_pAl"],
                th_Al2O3=inputs_dic.pop("th_Al2O3")
            )
    return inputs_dic


if __name__ == '__main__':
    '''
    Launch it with:
        python
        src/pypp/launch.py
        -d
        /net/phorcys/data/oxyde/ysami/data/trials_optuna_1/trial_00000
    '''
    # Get terminal options
    options = get_opt.for_dir(sys.argv)
    print(
        '\n options: \n'
        + json.dumps(options, indent=2)
    )
    # Set environment
    input_dir = get_opt.input_dir(
        local_on=options["local_on"],
        dir=options["dir"]
    )
    inputs_dic = set_inputs_dic(input_dir)
    inputs_dic = adapt_inputs_dic(inputs_dic)
    output_dir = get_opt.output_dir(input_dir)
    print(
        '\n inputs_dic: \n'
        + json.dumps(inputs_dic, indent=4)
    )
    # Launch simulation
    db_simu = main.main(inputs_dic)
    # Start post process
    db_simu_pp = DB_simu_pp(**db_simu.__dict__)
    # Create outputs
    db_simu_pp.manage(output_dir=output_dir)
