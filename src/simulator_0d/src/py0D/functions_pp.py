# --------------------------------------------------------------------
# ----------------- Post-traitement des données 0D -------------------
# --------------------------------------------------------------------

# ------------------- Importation des librairies ---------------------
import json
import pathlib

import numpy as np
import pandas as pd

from simulator_0d.src.py0D.assist_functions import vectorize_dic
from simulator_0d.src.py0D.class_db_simu import DB_simu


# -------------------- Fonction dérivée dX/dt -----------------------
def derive(db_simu, X_key, nb_points):

    # Création des dérivées temporelles
    if isinstance(X_key, str):
        X = vectorize_dic(db_simu.gas_d, X_key)
    else:
        X = X_key
    i = 0
    dX_dt_short = []
    dX_dt = []

    # On calcule la dérivée sur les "nb_points" (dX = nb_points points)
    while i * nb_points < (len(X) - nb_points):
        if (db_simu.t_t[(i + 1) * nb_points] - db_simu.t_t[i * nb_points]) == 0:
            dX_dt_short.append(dX_dt_short[-1])
        else:
            dX_dt_short.append(
                (X[(i + 1) * nb_points] - X[i * nb_points])
                / (db_simu.t_t[(i + 1) * nb_points] - db_simu.t_t[i * nb_points])
            )
        j = 0
        while j < nb_points:
            dX_dt.append(dX_dt_short[i])
            j = j + 1
        i = i + 1

    # On prend la derniere valeur sur le reste des valeurs
    if (db_simu.t_t[-1] - db_simu.t_t[-nb_points]) == 0:
        dX_dt_short.append(dX_dt_short[-1])
    else:
        dX_dt_short.append(
            (X[-1] - X[-nb_points]) / (db_simu.t_t[-1] - db_simu.t_t[-nb_points])
        )
    j = i * nb_points
    while j < len(X):
        dX_dt.append(dX_dt_short[-1])
        j = j + 1
    i = i + 1

    return dX_dt


# ------------- Création de la classe pour post-traiter les données ---------------
class DB_simu_pp(DB_simu):

    # Initialisation de la classe
    def __init__(self, *args, **kwargs):

        # Rajout d'attributs à DB_simu
        DOI = kwargs.pop("DOI")

        # Définition des attributs donnés à DB_simu comme des arguments
        super().__init__(*args, **kwargs)

        # Définitions des attributs internes à la classe DB_simu
        self.dP_dt = []
        self.dT_dt = []
        self.ddP_ddt = []
        self.ddT_ddt = []
        self.aux_data = dict(
            key_except=["DOI"],
            nb_points_deriv=2,
            i_reduced_vec=[],
            nb_points_reduce=300,
            dt_data=np.nan,
        )
        self.DOI = DOI

    # Programme principal : attribution des DOI
    def manage(self, output_dir):

        # Calcul des dérivées premières et secondes pour la température et pression
        self.dP_dt = derive(self, "pressure", self.aux_data["nb_points_deriv"])
        self.dT_dt = derive(self, "temperature", self.aux_data["nb_points_deriv"])
        self.ddP_ddt = derive(self, self.dP_dt, self.aux_data["nb_points_deriv"])
        self.ddT_ddt = derive(self, self.dT_dt, self.aux_data["nb_points_deriv"])

        # Calcul des différents DOI
        self.def_DOI()

        # Création du dossier "outputs" pour stockage des données
        if True:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Sauvegarde des données DOI.json et du fichier temporel
        self.save_DOI(output_dir)
        # print("time simulation = ", self.DOI['sim_time'])
        # print("T_Tmax = ", self.DOI['Tg_Tmax'])
        # print("Pg_f = ", self.DOI['Pg_f'])
        # print("Pg_rate = ", self.DOI['Pg_rate'])
        # print("Y_O2_f = ", self.DOI['Y_O2_f'])
        self.csv_file(output_dir)
        return {
            "sim_time": self.DOI["sim_time"],
            "Tg_Tmax": self.DOI["Tg_Tmax"],
            "Pg_f": self.DOI["Pg_f"],
            "Pg_rate": self.DOI["Pg_rate"],
            "Y_O2_f": self.DOI["Y_O2_f"],
        }

    # Permet de trouver les indices d'intérêt (Tmax, Pmax, f, init)
    def find_iOIs(self):

        i_0 = np.argwhere(self.t_t > self.t_t[-1] / 20)[0][0]
        iOIs = []
        ddT_ddt_max = max(self.ddT_ddt[i_0:])
        ddP_ddt_max = max(self.ddP_ddt[i_0:])
        T = vectorize_dic(self.gas_d, "temperature")
        T_max = max(T)
        P = vectorize_dic(self.gas_d, "pressure")
        P_max = max(P)
        iOIs.append(
            i_0 + np.argwhere(self.ddT_ddt[i_0:] > 0.2 * ddT_ddt_max)[0][0]
        )  # Premier indice : i_init
        iOIs.append(np.where(T == T_max)[0][0])  # Indice pour T_max
        iOIs.append(np.where(P == P_max)[0][0])  # Indice pour P_max
        iOIs.append(len(self.t_t) - 1)  # Indice pour t_f

        return iOIs

    # Permet d'attribuer/de stocker les DOI
    def def_DOI(self):

        # Liste des indices sur lequel on se focalise
        list_iOIs = self.find_iOIs()  # Liste des indices
        list_suffix = ["init", "Tmax", "Pmax", "f"]  # Liste des suffixes

        # Dictionnaire temporaire dans lequel sont stockées les données sous forme de liste
        dict_tmp = dict()
        dict_tmp["Tg"] = vectorize_dic(self.gas_d, "temperature")[list_iOIs]
        dict_tmp["Pg"] = vectorize_dic(self.gas_d, "pressure")[list_iOIs]
        dict_tmp["t"] = np.array(self.t_t)[list_iOIs]

        # Dictionnaire final
        self.DOI.update(
            {
                (key + "_" + suffix): dict_tmp[key][index]
                for index, suffix in enumerate(list_suffix)
                for key in [*dict_tmp]
            }
        )

        # Rajout de DOI calculés au dictionnaire final
        self.DOI["Pg_rate"] = (self.DOI["Pg_Pmax"] - 101325) / (self.DOI["t_Pmax"] - 0)
        self.DOI["Y_O2_f"] = vectorize_dic(self.gas_d, "Y")[-1, 0]

    # Sauvegarde en un fichier .json des données DOI
    def save_DOI(self, output_dir):
        json_filename = output_dir + "/DOI.json"
        with open(json_filename, "w") as json_file:
            json.dump(self.DOI, json_file, indent=4)

    # Mise en place d'un fichier .pkl pour stocker les données temporelles
    def csv_file(self, output_dir):

        # Recuperation des données temporelles d'intérêt
        hist_params = [
            dict(
                obj=self.system_d,
                cols=["t"],
                expand=None,
                rename=["t"],
            ),
            dict(
                obj=self.p_Al_d,
                cols=["composition"],
                expand=True,
                rename=["comp_pAl_Al", "comp_pAl_MeO", "comp_pAl_Me", "comp_pAl_Al2O3"],
            ),
            dict(
                obj=self.p_Al_d,
                cols=["temperature", "r_ext"],
                expand=None,
                rename=["T_pAl", "r_pAl"],
            ),
            dict(
                obj=self.p_MeO_d,
                cols=["composition"],
                expand=True,
                rename=[
                    "comp_pMeO_Al",
                    "comp_pMeO_MeO",
                    "comp_pMeO_Me",
                    "comp_pMeO_Al2O3",
                ],
            ),
            dict(
                obj=self.p_MeO_d,
                cols=["temperature", "r_ext"],
                expand=None,
                rename=["T_pMeO", "r_pMeO"],
            ),
            dict(
                obj=self.gas_d,
                cols=["pressure", "temperature"],
                expand=None,
                rename=["P_g", "T_g"],
            ),
            dict(
                obj=self.gas_d,
                cols=["Y"],
                expand=True,
                rename=[
                    "Y_O2",
                    "Y_Al",
                    "Y_O",
                    "Y_N2",
                    "Y_Al2O",
                    "Y_AlO",
                    "Y_AlO2",
                    "Y_Al2O2",
                    "Y_Me",
                ],
            ),
        ]

        # Mise sous forme de DataFrame
        df_list = []
        for dic in hist_params:
            df = pd.DataFrame(dic["obj"], columns=dic["cols"])
            if dic["expand"] is not None:
                df = df[dic["cols"][0]].apply(pd.Series)
            df.columns = dic["rename"]
            df_list.append(df)
        df_global = pd.concat(df_list, axis=1)

        # Sauvegarde en fichier .pkl
        df_global.to_pickle(output_dir + "/DOI_temporal_vectors.pkl")
