tmelt = 273.15 # As in mo_physical_constants not dependent on pressure
cpd = 1004.64 # J/(kg K) specific heat capacity of air for const. pressure
g = 9.80665 # m/s2 gravitational accelaration
dT = 3600 # s model timestep
EPSILON = 1e-6 # constant value to add to logarithm computation to avoid ln(0) in ln(cloud_area_frac)

# Same definitions as in mo_cumastr
lh_vap   = 2.5008e6 # J/kg   latent heat for vaporisation
lh_sub   = 2.8345e6 # J/kg   latent heat for sublimation
lh_fus   = lh_sub - lh_vap # J/kg   latent heat for fusion

label_remapping = {'subg_flux_qv': r'$F^\mathrm{sg}_{q_v}$',
                   'subg_flux_qc': r'$F^\mathrm{sg}_{q_l}$',
                   'subg_flux_qi': r'$F^\mathrm{sg}_{q_i}$',
                   'subg_flux_qr': r'$F^\mathrm{sg}_{q_r}$',
                   'subg_flux_qs': r'$F^\mathrm{sg}_{q_s}$',
                   'subg_flux_h': r'$F^\mathrm{sg}_h$',
                   'subg_flux_u': r'$F^\mathrm{sg}_u$',
                   'subg_flux_v': r'$F^\mathrm{sg}_v$',
                   'clt': 'clt',
                   'cltp': 'cltp',
                   'liq_detri': '$q_{l,\mathrm{detr}}$',
                   'ice_detri': '$q_{i,\mathrm{detr}}$',
                   'tot_prec': 'prec',
                   'pr': 'prec'}

input_var_remapping = {'w_fl': r'$w$',
                       'qv': r'$q_v$',
                       'qc': r'$q_l$',
                       'qi': r'$q_i$',
                       'qr': r'$q_r$',
                       'qs': r'$q_s$',
                       'u': r'$u$',
                       'v': r'$v$',
                       'h': r'$h$'}