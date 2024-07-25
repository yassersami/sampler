i1 = 1
i0 = 1


#---------------------------------------------------------------------------------------------------------------------------------
def dm_dt_computation_solid(name_of_matter, flux, source):
    dm_dt0, dm_dt1 = 0, 0
    # variation d'aluminium dans la particule d'aluminium dans le coeur
    if name_of_matter == "Al_pAl_core":
        dm_dt1 = - flux[i1].Al_ext - source[i1].Al_int
        dm_dt0 = - flux[i0].Al_ext - source[i0].Al_int

    # variation d'alumine dans les deux particules à l'interface
    if name_of_matter == "Al2O3_p_shell":
        dm_dt1 = - source[i1].Al2O3_int - flux[i1].Al2O3_condens - source[i1].Al2O3_decomp
        dm_dt0 = - source[i0].Al2O3_int - flux[i0].Al2O3_condens - source[i0].Al2O3_decomp

    # variation d'oxyde métallique dans la particule d'oxyde métallique dans le coeur
    if name_of_matter == "MeO_pMeO_core":
        dm_dt1 = - source[i1].MeO
        dm_dt0 = - source[i0].MeO

    # variation de metal dans la particule d'oxyde métallique dans le coeur
    if name_of_matter == "Me_pMeO_core":
        dm_dt1 = - source[i1].Me - flux[i1].Me
        dm_dt0 = - source[i0].Me - flux[i0].Me

    return dm_dt0,  dm_dt1


#---------------------------------------------------------------------------------------------------------------------------------
def dm_dt_computation_gas(name_of_matter, flux_pMeO, flux_pAl, source_pMeO, source_pAl):     #0)02  1)Al   2)O    3)N2    4)Al2O    5) AlO    6)AlO2   7)Al2O2
    if name_of_matter == "O2":
        dm_dt1 = flux_pMeO[i1].O2_ext + flux_pAl[i1].O2_ext + flux_pAl[i1].O2_condens + flux_pMeO[i1].O2_condens + source_pAl[i1].O2_decomp + source_pMeO[i1].O2_decomp
        dm_dt0 = flux_pMeO[i0].O2_ext + flux_pAl[i0].O2_ext + flux_pAl[i0].O2_condens + flux_pMeO[i0].O2_condens + source_pAl[i0].O2_decomp + source_pMeO[i0].O2_decomp
    if name_of_matter == "Al":
        dm_dt1 = + flux_pAl[i1].Al_ext + flux_pMeO[i1].Al_ext + flux_pAl[i1].Al_condens + flux_pMeO[i1].Al_condens + source_pAl[i1].Al_decomp + source_pMeO[i1].Al_decomp
        dm_dt0 = + flux_pAl[i0].Al_ext + flux_pMeO[i0].Al_ext + flux_pAl[i0].Al_condens + flux_pMeO[i0].Al_condens + source_pAl[i0].Al_decomp + source_pMeO[i0].Al_decomp
    if name_of_matter == "O":
        dm_dt1 = + flux_pAl[i1].O_ext + source_pAl[i1].O_decomp + source_pMeO[i1].O_decomp
        dm_dt0 = + flux_pAl[i0].O_ext + source_pAl[i0].O_decomp + source_pMeO[i0].O_decomp
    if name_of_matter == "N2":
        dm_dt1 = 0
        dm_dt0 = 0
    if name_of_matter == "Al2O":
        dm_dt1 = + flux_pAl[i1].AlxOy[4] + flux_pMeO[i1].AlxOy[4]
        dm_dt0 = + flux_pAl[i0].AlxOy[4] + flux_pMeO[i0].AlxOy[4]
    if name_of_matter == "AlO":
        dm_dt1 = + flux_pAl[i1].AlxOy[5] + flux_pMeO[i1].AlxOy[5] + source_pAl[i1].AlO_decomp + source_pMeO[i1].AlO_decomp
        dm_dt0 = + flux_pAl[i0].AlxOy[5] + flux_pMeO[i0].AlxOy[5] + source_pAl[i0].AlO_decomp + source_pMeO[i0].AlO_decomp
    if name_of_matter == "AlO2":
        dm_dt1 = + flux_pAl[i1].AlxOy[6] + flux_pMeO[i1].AlxOy[6] + source_pAl[i1].AlO2_decomp + source_pMeO[i1].AlO2_decomp
        dm_dt0 = + flux_pAl[i0].AlxOy[6] + flux_pMeO[i0].AlxOy[6] + source_pAl[i0].AlO2_decomp + source_pMeO[i0].AlO2_decomp
    if name_of_matter == "Al2O2":
        dm_dt1 = + flux_pAl[i1].AlxOy[7] + flux_pMeO[i1].AlxOy[7] + source_pAl[i1].Al2O2_decomp + source_pMeO[i1].Al2O2_decomp
        dm_dt0 = + flux_pAl[i0].AlxOy[7] + flux_pMeO[i0].AlxOy[7] + source_pAl[i0].Al2O2_decomp + source_pMeO[i0].Al2O2_decomp
    if name_of_matter == "Cu":
        dm_dt1 = flux_pMeO[i1].Me + flux_pAl[i1].Me
        dm_dt0 = flux_pMeO[i0].Me + flux_pAl[i0].Me

    return dm_dt0, dm_dt1


#---------------------------------------------------------------------------------------------------------------------------------
def dQ_dt_computation_gas(heat_flux_pAl,heat_flux_pMeO,dt):
    dQ_dt_0 = + heat_flux_pAl[i0].thermal + heat_flux_pMeO[i0].thermal + heat_flux_pMeO[i0].species_flux_heat + heat_flux_pAl[i0].species_flux_heat
    dQ_dt_1 = + heat_flux_pAl[i1].thermal + heat_flux_pMeO[i1].thermal + heat_flux_pMeO[i1].species_flux_heat + heat_flux_pAl[i1].species_flux_heat
    return dQ_dt_0, dQ_dt_1


#---------------------------------------------------------------------------------------------------------------------------------
def dQ_dt_computation_p(heat_flux):
    dQ_dt_0 = - heat_flux[i0].thermal - heat_flux[i0].species_flux_heat + heat_flux[i0].radiative + heat_flux[i0].p_to_p + heat_flux[i0].source_input
    dQ_dt_1 = - heat_flux[i1].thermal - heat_flux[i1].species_flux_heat + heat_flux[i1].radiative + heat_flux[i1].p_to_p + heat_flux[i1].source_input
    return dQ_dt_0, dQ_dt_1
