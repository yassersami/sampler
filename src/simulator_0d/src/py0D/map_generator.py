#--------------------------------------------------------------------
#----------------- Post-traitement des données 0D -------------------
#--------------------------------------------------------------------

#------------------- Importation des librairies ---------------------
# ------------------- Importation des librairies ---------------------
import json
import os
import pathlib
import shutil

import pandas as pd


#------------------- Déclaration des différents chemins -------------------
def join_path(a, *p): 
    return os.path.abspath(os.path.join(a, *p))

PATH_proj = join_path(os.path.abspath(__file__), '..', '..', '..')
MAP_DIR = join_path(PATH_proj, 'data', 'data_map')
global_data_path = join_path(MAP_DIR, '00_global_data')


#------------------- Fonction pour nettoyer la DB -------------------
def clean_map(MAP_DIR):
    for file in os.listdir(MAP_DIR):
        full_path = os.path.join(os.path.abspath(MAP_DIR), file)

        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


#---- Fonction pour associer simulation/dossier : création du nom 0000X -------
def label(sample):
    return f'simu_{sample.name:05}'


#------------ Génération de la DB : creation des dossiers avec inputs_var.json et inputs_fix.json dedans --------------
def map_generator(sample, map_dir=MAP_DIR):

    # Création des dossiers simu_0000X
    simu_dir = join_path(map_dir, str(sample.loc["workdir"]))
    pathlib.Path(simu_dir).mkdir(
        parents=False,
        exist_ok=True
    )

    # Mise en place des fichiers inputs_fix.json et inputs_var.json dedans
    inputs_fix_path = join_path(simu_dir, 'inputs_fix.json')        # Création du chemins pour inputs_fix.json
    inputs_var_path = join_path(simu_dir, 'inputs_var.json')        # Création du chemins pour inputs_var.json
    inputs_fix_ref_path = join_path(PATH_proj, 'data', 'inputs_fix', 'inputs_fix.json') # Récupération du chemin pour le fichier inputs_fix.json de référence
    shutil.copyfile(inputs_fix_ref_path, inputs_fix_path)           # Copie du fichier inputs_fix.json de référence
    with open(inputs_var_path, "w") as json_file:                   # Ecriture du fichier inputs_var.json pour chaque répertoire
        json.dump(sample.to_dict(), json_file, indent=4)


#---- Fonction pour créer une liste à partir d'arguments np -------
def np_to_list(dic):
    return {k : list(v) for k,v in dic.items()}

#----------------------- Création de la DB ------------------------
def data_map_creation(fichier):

    # Lecture du fichier .csv/data_frame
    fichier_path = join_path(PATH_proj + fichier)
    df = pd.read_csv(fichier_path)

    # Rajout des colonnes manquantes à la dataFrame 
    df["workdir"] = df.apply(label, axis=1)
    df["alpha_p"] = 0.3
    df["th_Al2O3"] = 5.12281599140251e-08
    df["heat_input_ths"] = 556000
    df["power_input"] = 20000000
    df["coeff_contact"] = 0.01
    df["Ea_D_Ox"] = 50000
    df["Ea_Al2O3_decomp"] = 400000
    df["Ea_MeO_decomp"] = 50000
    df["k0_D_Ox"] = 0.000008
    df["k0_Al2O3_decomp"] = 1520000
    df["k0_MeO_decomp"] = 30000000
    df["bool_kin"] = 'false'

    # Nettoyage du dossier data_map
    clean_map(MAP_DIR)

    # Création de l'ensemble des dossiers de simulations
    df.apply(map_generator, axis=1)
