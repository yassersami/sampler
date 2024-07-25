from datetime import datetime

class DB_simu():
    '''
    This class is used to store simulation main physical quantities

    Attributes
    ----------
    aux_data: dict
        all potential usefull data

    '''
    def __init__(
        self, t_t, system_d, p_Al_d, p_MeO_d, gas_d, gas_chamber_d, heat_well_d,
        flux_pAl_d, flux_pMeO_d, source_pAl_d, source_pMeO_d, heat_flux_pAl_d, heat_flux_pMeO_d, flux_chamber_d, flux_HW_d
    ):
        self.t_t = t_t
        self.system_d = system_d
        self.p_Al_d = p_Al_d
        self.p_MeO_d = p_MeO_d
        self.gas_d = gas_d
        self.gas_chamber_d = gas_chamber_d
        self.heat_well_d = heat_well_d
        self.flux_pAl_d = flux_pAl_d
        self.flux_pMeO_d = flux_pMeO_d
        self.source_pAl_d = source_pAl_d
        self.source_pMeO_d = source_pMeO_d
        self.heat_flux_pAl_d = heat_flux_pAl_d
        self.heat_flux_pMeO_d = heat_flux_pMeO_d
        self.flux_chamber_d = flux_chamber_d
        self.flux_HW_d = flux_HW_d
        self.DOI = dict(  # dictionnaire stockant les grandeurs d'interet (DOI = Data Of interest)
            sim_time=0,
            date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
