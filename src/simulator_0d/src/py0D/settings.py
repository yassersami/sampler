import os
import pathlib
import sys

import cantera as ct

PROJECT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                    '..', '..'))

ct_thermo_data = "AlxOy_N2_Cu_O_and_Catoire_kin.yaml"
gas_thermo_data = ct.Solution(PROJECT_PATH + "/data/cantera_inp/" + ct_thermo_data)

if len(sys.argv) == 2:
    PATH_simu = sys.argv[1]
    print(PATH_simu)
    # os.mkdir(PATH_simu + '/data')
    # os.mkdir(PATH_simu + '/data/pickle')
    pathlib.Path(PATH_simu + '/data/pickle').mkdir(  # Create subdir
    parents=True,
    exist_ok=True)
    PATH_pickle=str(PATH_simu + '/data/pickle')
    # PATH_compare = PROJECT_PATH + '/data/compare_folder'
else:
    PROJECT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                '..', '..'))
    PATH_simu = PROJECT_PATH + "/data/inputs_folder"
    PATH_pickle=str(PROJECT_PATH+"/data/pickle")
    # PATH_compare = PROJECT_PATH + '/data/compare_folder'


PATH_JSON_var = PATH_simu + "/inputs_var.json"
PATH_JSON_fix = PATH_simu + "/inputs_fix.json"
