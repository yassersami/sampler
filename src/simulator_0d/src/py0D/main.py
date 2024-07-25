import pickle

# import settings
import time

import numpy as np

import simulator_0d.src.py0D.Flux_Source as Flux_Source
import simulator_0d.src.py0D.assist_functions as assist_functions
from simulator_0d.src.py0D.class_db_simu import DB_simu
from simulator_0d.src.py0D.dt_func import dt_determination
from simulator_0d.src.py0D.enthalpy import find_p_temperature
from simulator_0d.src.py0D.phase_gas import Gas
from simulator_0d.src.py0D.phase_particle import Particle
from simulator_0d.src.py0D.phase_system import System


def main(inputs_dic):

    dt = 2e-8
    t = 0  # np.array([0])
    ite_print = int(1e2)
    t_f = inputs_dic["t_f"]
    CO_T = inputs_dic["CO_T"]
    compute_time_0 = time.time()

    #  -----------------------------------------------Si on prend des données déja existantes
    if inputs_dic["load_last_data"]:
        system.inputs_dic = inputs_dic

        t_t, system_d, p_Al_d, p_MeO_d, gas_d, gas_chamber_d, heat_well_d = (
            assist_functions.import_pickle()
        )
        (
            flux_pAl_d,
            flux_pMeO_d,
            source_pAl_d,
            source_pMeO_d,
            heat_flux_pAl_d,
            heat_flux_pMeO_d,
            flux_chamber_d,
            flux_HW_d,
        ) = assist_functions.import_flux_pickle()

    else:
        #  -----------------------------------------------Sinon, initialisation des phases
        gas = Gas("with_p", inputs_dic)
        p_Al = Particle("p_Al", gas, inputs_dic)
        p_MeO = Particle("p_MeO", gas, inputs_dic)
        gas_chamber = Gas("only", inputs_dic)
        heat_well = Flux_Source.Heat_well(300)
        system = System(
            p_Al, p_MeO, gas, gas_chamber, heat_well, dt, t, t_f, inputs_dic
        )
        gas_d = []
        gas_chamber_d = []
        p_Al_d = []
        p_MeO_d = []
        heat_well_d = []
        system_d = []

        # ---------------------------------------------------initialisation des flux
        flux_pAl = Flux_Source.Flux_species_tab_init(gas)
        flux_pMeO = Flux_Source.Flux_species_tab_init(gas)
        flux_pAl_d = []
        flux_pMeO_d = []

        # --------------------------------------------------initialisation des sources
        source_pAl = Flux_Source.Source_tab_init(gas)
        source_pMeO = Flux_Source.Source_tab_init(gas)
        source_pAl_d = []
        source_pMeO_d = []

        # ---------------------------------------------------initialisation des flux themriques
        heat_flux_pAl = Flux_Source.Flux_heat_tab_init()
        heat_flux_pMeO = Flux_Source.Flux_heat_tab_init()
        heat_flux_pAl_d = []
        heat_flux_pMeO_d = []

        # -----------------------------------------------------initialisation des flux avec la chambre gaz
        flux_chamber = Flux_Source.Flux_g_g_tab_init(gas, system)
        flux_chamber_d = []

        # -----------------------------------------------------initialisation des flux le puit thermique
        flux_HW = Flux_Source.Heat_well_flux_tab_init(inputs_dic)
        flux_HW_d = []

        system.Composition_alpha_update(p_Al, p_MeO, gas)

        t_t = np.array([t])
        assist_functions.time_save_dic(
            p_Al,
            p_MeO,
            gas,
            system,
            p_Al_d,
            p_MeO_d,
            gas_d,
            system_d,
            gas_chamber,
            gas_chamber_d,
            heat_well,
            heat_well_d,
        )
        assist_functions.time_save_flux_dic(
            flux_pAl,
            flux_pMeO,
            source_pAl,
            source_pMeO,
            heat_flux_pAl,
            heat_flux_pMeO,
            flux_chamber,
            flux_HW,
            flux_pAl_d,
            flux_pMeO_d,
            source_pAl_d,
            source_pMeO_d,
            heat_flux_pAl_d,
            heat_flux_pMeO_d,
            flux_chamber_d,
            flux_HW_d,
        )

    # -----------------------------------------------------------------------------------------------------------------start of temporal loop----------------------------------------
    elapsed_time = 0
    while system.compute_bool_end_simu() and (
        elapsed_time < system.inputs_dic["max_simu_time"]
    ):
        system.i = system.i + 1
        # assist_functions.print_start_loop(system, ite_print)

        # -----------------------------------------------------injection de puissance pour l'initiation
        if system.test_ignition is True:
            system.i_init = int(system.i / inputs_dic["sample_period"])
        else:
            system.power_input = 0

        heat_flux_pMeO = Flux_Source.Source_input(
            heat_flux_pMeO, system.power_input / 2
        )
        heat_flux_pAl = Flux_Source.Source_input(heat_flux_pAl, system.power_input / 2)

        # ------------------------------------------------------calcul des flux thermiques non relatif au déplacement d'espèce
        heat_flux_pAl = Flux_Source.Flux_heat_tab_update_1(
            heat_flux_pAl, p_Al, p_MeO, "p_Al", gas
        )
        heat_flux_pMeO = Flux_Source.Flux_heat_tab_update_1(
            heat_flux_pMeO, p_Al, p_MeO, "p_MeO", gas
        )

        # ------------------------------------------------------calcul des flux d'espèce
        flux_pAl, source_pAl_estim = Flux_Source.Flux_species_tab_update(
            system, flux_pAl, p_Al, gas, source_pAl, source_pMeO, heat_flux_pAl, t_t
        )
        flux_pMeO, source_pMeO_estim = Flux_Source.Flux_species_tab_update(
            system, flux_pMeO, p_MeO, gas, source_pAl, source_pMeO, heat_flux_pMeO, t_t
        )

        # ------------------------------------------------------calcul des termes sources d'espèce
        source_pAl = Flux_Source.Source_tab_update(
            system, source_pAl, flux_pAl, p_Al, source_pAl_estim
        )
        source_pMeO = Flux_Source.Source_tab_update(
            system, source_pMeO, flux_pMeO, p_MeO, source_pMeO_estim
        )

        # ------------------------------------------------------calcul des flux thermiques relatif au déplacement d'espèce
        heat_flux_pAl = Flux_Source.Flux_heat_tab_update_2(
            flux_pAl, source_pAl, heat_flux_pAl, p_Al, p_MeO, "p_Al", gas
        )
        heat_flux_pMeO = Flux_Source.Flux_heat_tab_update_2(
            flux_pMeO, source_pMeO, heat_flux_pMeO, p_Al, p_MeO, "p_MeO", gas
        )

        # -----------------------------------------------------calcul des flux avec la chambre gaz et du puit thermique
        flux_chamber = Flux_Source.Flux_g_g_tab_update(
            flux_chamber, gas, gas_chamber, system
        )
        flux_HW = Flux_Source.heat_well_tab_update(flux_HW, gas_chamber, heat_well)
        system.dt = dt_determination(
            system,
            gas,
            p_Al,
            p_MeO,
            flux_pAl,
            flux_pMeO,
            source_pAl,
            source_pMeO,
            heat_flux_pAl,
            heat_flux_pMeO,
            flux_chamber,
            flux_HW,
        )

        # ------------------------------------------------------MAJ des phases
        p_Al.update_composition(flux_pAl, source_pAl, system.dt)
        p_MeO.update_composition(flux_pMeO, source_pMeO, system.dt)

        # ------------------------------------------------------MAJ des géométries
        p_MeO.update_r()
        p_Al.update_r()  #  Si on considère le rayon constant,  ne pas mettre à jour!
        system.Composition_alpha_update(p_Al, p_MeO, gas)

        # ------------------------------------------------------MAJ de l'energie
        p_Al.update_enthalpy(heat_flux_pAl, system.dt)
        p_MeO.update_enthalpy(heat_flux_pMeO, system.dt)

        gas.update_all(
            system.dt,
            system.volume,
            flux_pMeO,
            flux_pAl,
            source_pMeO,
            source_pAl,
            heat_flux_pAl,
            heat_flux_pMeO,
            flux_chamber,
        )
        gas_chamber.update_all(system.dt, gas_chamber.volume, flux_chamber, flux_HW)
        heat_well.update_enthalpy(flux_HW, dt)

        # ------------------------------------------------------MAJ de la T°
        p_Al.temperature, p_Al.state, p_Al.h_lim_vap, p_Al.cp = find_p_temperature(
            p_Al, gas
        )
        p_MeO.temperature, p_MeO.state, p_MeO.h_lim_vap, p_MeO.cp = find_p_temperature(
            p_MeO, gas
        )

        system.update_system(
            p_Al, p_MeO, gas, gas_chamber, heat_well, t_t
        )  # MAJ du systeme

        # ------------------------------------------------------enregistrement temporel
        system.t = system.t + system.dt
        if (
            system.i % inputs_dic["sample_period"] == 0
        ):  # Enregistrement des données toute les sample_period itérations
            t_t = np.append(t_t, [system.t])
            assist_functions.time_save_dic(
                p_Al,
                p_MeO,
                gas,
                system,
                p_Al_d,
                p_MeO_d,
                gas_d,
                system_d,
                gas_chamber,
                gas_chamber_d,
                heat_well,
                heat_well_d,
            )
            assist_functions.time_save_flux_dic(
                flux_pAl,
                flux_pMeO,
                source_pAl,
                source_pMeO,
                heat_flux_pAl,
                heat_flux_pMeO,
                flux_chamber,
                flux_HW,
                flux_pAl_d,
                flux_pMeO_d,
                source_pAl_d,
                source_pMeO_d,
                heat_flux_pAl_d,
                heat_flux_pMeO_d,
                flux_chamber_d,
                flux_HW_d,
            )

        elapsed_time = time.time() - compute_time_0
    # ------------------------------------------------------affichage
    # if system.i % ite_print == 0:
    #     assist_functions.print_data(gas,p_Al,p_MeO,system,ite_print,gas_chamber,heat_well)

    # -----------------------------------------------------------------------------------------------------------------end of temporal loop----------------------------------------

    # assist_functions.print_start_loop(system, ite_print)
    # assist_functions.print_data(gas,p_Al,p_MeO,system,ite_print,gas_chamber,heat_well)

    print("py0D.main.main -> computational time=%.2f" % elapsed_time)
    compute_time_save0 = time.time()
    db_simu = DB_simu(
        t_t,
        system_d,
        p_Al_d,
        p_MeO_d,
        gas_d,
        gas_chamber_d,
        heat_well_d,
        flux_pAl_d,
        flux_pMeO_d,
        source_pAl_d,
        source_pMeO_d,
        heat_flux_pAl_d,
        heat_flux_pMeO_d,
        flux_chamber_d,
        flux_HW_d,
    )
    db_simu.DOI["sim_time"] = elapsed_time / 60  # (minutes)

    # Fix to tag timed out (uncompleted) simulations
    db_simu.DOI["timed_out"] = elapsed_time >= system.inputs_dic["max_simu_time"]

    # print("save time=%.2f" %(time.time()-compute_time_save0))
    if inputs_dic["save_last_data"]:
        delattr(gas, "gas_ct")
        delattr(gas_chamber, "gas_ct")
        assist_functions.save_ite(
            p_Al,
            p_MeO,
            gas,
            system,
            gas_chamber,
            heat_well,
            flux_pAl,
            flux_pMeO,
            source_pAl,
            source_pMeO,
            heat_flux_pAl,
            heat_flux_pMeO,
            flux_chamber,
            flux_HW,
        )

    # print("\nvar_T_g = ", system.var_T_g, "\nvar_T_pAl = ", system.var_T_pAl, "\nvar_h = ", system.var_h, "\nsystem.i = ", system.i, "\n")

    return db_simu


# if __name__ == '__main__':
#     import os
#
#
#     def join_path(a, *p): return os.path.abspath(os.path.join(a, *p))
#
#     # Debug paths
#     PATH_simu=settings.PATH_simu
#     PATH_JSON_fix=settings.PATH_JSON_fix
#     PATH_JSON_var=settings.PATH_JSON_var
#     print(f'PATH_simu: {PATH_simu}')
#     print(f'PATH_JSON_var: {PATH_JSON_var}')
#     print(f'PATH_JSON_fix: {PATH_JSON_fix}')
#
#     inputs_dic = assist_functions.load_json_from_file(PATH_JSON_var)
#     inputs_dic.update(assist_functions.load_json_from_file(PATH_JSON_fix))
#    # print(
#   #      '\n inputs_dic: \n'
#  #       + json.dumps(inputs_dic, indent=4)
# #    )
#
#     db_simu = main(inputs_dic)
#     db_simu_pp=DB_simu_pp(db_simu)
#     db_simu_pp.manage(join_path(PATH_JSON_var,'..','data'))
#
#     # Create outputs
#     output_dir=join_path(PATH_JSON_var,'..','data','pickle')
#     pickle.dump(db_simu,open(output_dir+"/db_simu.pkl",'wb'))
