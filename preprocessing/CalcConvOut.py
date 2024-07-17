import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import psyplot.project as psy
import psyplot
import os
import sys
from pprint import pprint
from numba import jit, njit
from convection_param.HelperFuncs import *
from convection_param.ConvParamFuncs import get_convective_thetav_buoyancy
from convection_param.Constants import *#,EPSILON,tmelt
from multiprocessing import Pool

def rename_to_standard(da, names2d, names3d):
    if len(da.dims) == 2:
        rename_dict = {old: new for old,new in zip(da.dims, names2d)}
    elif len(da.dims) == 3:
        rename_dict = {old: new for old,new in zip(da.dims, names3d)}
    return da.rename(rename_dict)

def get_cloud_tops(ccells):
    # ccells = ccells.astype(bool)
    last_convective = np.argmax(ccells, axis=1)
    maxvals = np.max(ccells, axis=1)
    last_convective[maxvals==0] = ccells.shape[1] - 1 # if no convective cell is found return klevm1 as in mo_cuinitialize
    return last_convective

def pick_with_idx_array_dimm1(vals, idx_arr, axis):
    ''' pick values from array <vals> based on the indices in array <idx_arr>, this array
    must not have the axis <axis> from vals.
    '''
    return np.take_along_axis(vals, np.expand_dims(idx_arr, axis=axis), axis).squeeze(axis=axis)

def get_frac_detr_theta_excess(virt_temp_excess, z):
    '''According to [Baba 2018]'''
    frac_detr = -grad_along_height_dim_fast(np.log(virt_temp_excess), z)
    frac_detr[frac_detr<0] = 0
    return frac_detr

def get_frac_detr_cloud_area(cloud_area_frac, z):
    '''According to [Nordeng 1994]'''
    frac_detr = -grad_along_height_dim_fast(np.log(cloud_area_frac+EPSILON), z)
    frac_detr[frac_detr<0] = 0
    return frac_detr

def get_cloud_detr(frac_detr, M, temp):
    # return frac_detr * mass_flux
    water_detr = frac_detr*M
    t_gt_melt_mask = temp > tmelt # Corresponds to llo1 in mo_cufluxdts
    liquid_detr = water_detr * t_gt_melt_mask # Corresponds to MERGE(0.0_wp,plude(jl,jk),llo1)
    ice_detr = water_detr * ~t_gt_melt_mask
    return liquid_detr, ice_detr # Corresponds to zxtecl, zxteci

def get_cloud_detr_int(liquid_detr, ice_detr, delta_z):
    liquid_detr_int = np.sum(liquid_detr*-delta_z, axis=1)
    ice_detr_int = np.sum(ice_detr*-delta_z, axis=1)
    return liquid_detr_int, ice_detr_int

def get_tr_tend_liq_ice(liquid_detr, ice_detr, air_layer_density): # air_layer_density corresponding to mref?
    liquid_tend = np.maximum(0, liquid_detr/air_layer_density)
    ice_tend = np.maximum(0, liquid_detr/air_layer_density)
    return liquid_tend, ice_tend

def get_dry_st_energy(T,z):
    return cpd*T + z*g # J/kg

def get_moist_st_energy(T,z,qc,qr,qi,qs,qg):
    # liquid/ice water static energy [=cpT + gz - Lc(qc + qr) - Ls(qi + qs + qg)]
    return get_dry_st_energy(T,z) - lh_vap * (qc + qr) - lh_sub * (qi + qs + qg)

# def get_wind_tend(wu, w, u, z, rho):
    # subg_mom_flux = rho*(wu - w*u)
    # sgmf_grad = grad_along_heigt_dim_fast(subg_mom_flux, z)
# #     return -1/rho*w*rho * sgmf_grad
    # return w * sgmf_grad

def get_q_tend(rho, w, q, z):
    q_grad = grad_along_height_dim_fast(q, z)
#     return 1/(rho*cpd) * w*rho * q_grad
    return 1/cpd * w * q_grad

def get_temp_tend(rho, w, s, z):
    s_grad = grad_along_height_dim_fast(s, z)

    M = w*rho
    return 1/(rho*cpd) * M *s_grad

def get_sfc_rain_snow_flux_massflux(qr, qs, rho, w_fl_appr):
    idx = np.s_[:,-1,:]               # surface layer index
    rho_w = rho[idx]*w_fl_appr[idx]
    qr_flux = qr[idx]*rho_w
    qs_flux = qs[idx]*rho_w
    return qr_flux, qs_flux


def get_sfc_rain_snow_flux_detrainment(liq_detr, ice_detr):
    idx = np.s_[:,-1,:]               # surface layer index
    return liq_detr[idx], ice_detr[idx]


@njit
def np_interp_along_height(x, xp, fp):
    result = np.empty_like(x)
    for i in range(x.shape[1]):
        result[:,i,:] = np_interp_invert_extralr_nongu(x, xp, fp)
    return result


def create_conv_out_ds(ds, ds_theta, ds_cloud, vars_to_add, da_prec, v_interpolation='full_lvl'):

    # pool = Pool()
    # result1 = pool.apply_async(solve1, [A])    # evaluate "solve1(A)" asynchronously
    # result2 = pool.apply_async(solve2, [B])    # evaluate "solve2(B)" asynchronously
    # answer1 = result1.get(timeout=10)
    # answer2 = result2.get(timeout=10)

    # Get input variables
    rho_vals = ds.rho.values
    pres_vals = ds.pres.values
    pres_height_axis = ds.pres.dims.index('height')
    z_ifc_vals = ds.z_ifc.values  # Half level geometric height
    theta_v_vals = ds.theta_v.values
    w_vals = ds.w.values
    w_fl_appr = (w_vals[:,:-1,:] + w_vals[:,1:,:])/2
    u_vals = ds.u.values
    v_vals = ds.v.values
    z_fl_appr = (z_ifc_vals[:-1,:] + z_ifc_vals[1:,:])/2
    delta_z = np.diff(z_ifc_vals, axis=0)

    temp_vals = ds.temp.values
    qv_vals = ds.qv.values # spec. humidity
    qc_vals = ds.qc.values # spec. cloud water
    qi_vals = ds.qi.values # ice
    qr_vals = ds.qr.values # rain
    qs_vals = ds.qs.values # snow
    qg_vals = ds_cloud.qg.values # graupel

    # Vertical distance between layers (bottom to top):
    # dz = -np.diff(z_ifc_vals, axis=0)  # negative so that dz is positive
    # Air density (w.o. water content)
    # air_density = rho_vals*(1-qc_vals) # or -(qc_vals + qi_vals + qr_vals + qs_vals + qg_vals) ?
    # Air layer density
    # air_layer_density = air_density*dz

    theta_v_e_vals = ds_theta.theta_v_e.values
    cloud_area_frac_vals = ds_cloud.clc.values / 100

    # Calc helper variables
    # frac_detr = get_frac_detr_theta_excess(theta_v_vals-theta_v_e_vals, z_fl_appr)
    frac_detr = get_frac_detr_cloud_area(cloud_area_frac_vals, z_fl_appr)
    # s = get_dry_st_energy(temp_vals, z_fl_appr)
    h = get_moist_st_energy(temp_vals, z_fl_appr, qc_vals, qr_vals, qi_vals, qs_vals, qg_vals)

    if v_interpolation == 'full_lvl':
        w_interp = w_fl_appr
        u_interp = u_vals
        v_interp = v_vals
        rho_interp = rho_vals
        temp_interp = ds.temp.values
        qv_interp = qv_vals
        qc_interp = qc_vals
        qi_interp = qi_vals
        qr_interp = qr_vals
        qs_interp = qs_vals
        qg_interp = qg_vals
        h_interp = h

    elif v_interpolation == 'half_lvl':
        w_interp = w_vals
        u_interp = np_interp_along_height(z_ifc, z_fl_appr, u_vals)
        v_interp = np_interp_along_height(z_ifc, z_fl_appr, v_vals)
        rho_interp = np_interp_along_height(z_ifc, z_fl_appr, rho_vals)
        qv_interp = np_interp_along_height(z_ifc, z_fl_appr, qv_vals)
        qc_interp = np_interp_along_height(z_ifc, z_fl_appr, qc_vals)
        qi_interp = np_interp_along_height(z_ifc, z_fl_appr, qi_vals)
        qr_interp = np_interp_along_height(z_ifc, z_fl_appr, qr_vals)
        qs_interp = np_interp_along_height(z_ifc, z_fl_appr, qs_vals)
        qg_interp = np_interp_along_height(z_ifc, z_fl_appr, qg_vals)
        h_interp = np_interp_along_height(z_ifc, z_fl_appr, h)

    else:
        raise Exception(f'v_interpolation method: {v_interpolation} not understood. Use \'full_lvl\' or \'half_lvl\'')

    rhowu = rho_interp * w_interp * u_interp
    rhowv = rho_interp * w_interp * v_interp
    rhowh = rho_interp * w_interp * h_interp
    rhowqv = rho_interp * w_interp * qv_interp
    rhowqc = rho_interp * w_interp * qc_interp
    rhowqi = rho_interp * w_interp * qi_interp
    rhowqr = rho_interp * w_interp * qr_interp
    rhowqs = rho_interp * w_interp * qs_interp
    rhowqg = rho_interp * w_interp * qg_interp

    wh = w_interp * h_interp
    wu = w_interp * u_interp
    wv = w_interp * v_interp
    wqv = w_interp * qv_interp
    wqc = w_interp * qc_interp
    wqi = w_interp * qi_interp
    wqr = w_interp * qr_interp
    wqs = w_interp * qs_interp
    wqg = w_interp * qg_interp

    # rhow = rho_interp * w_interp
    # rhoh = rho_interp * h_interp
    # rhou = rho_interp * u_interp
    # rhov = rho_interp * v_interp
    # rhoqv = rho_interp * qv_interp
    # rhoqc = rho_interp * qc_interp
    # rhoqi = rho_interp * qi_interp
    # rhoqr = rho_interp * qr_interp
    # rhoqs = rho_interp * qs_interp
    # rhoqg = rho_interp * qg_interp

    # Not needed anymore
    del z_ifc_vals, w_vals # ,dz

    # Calc observables
    ccells = get_convective_thetav_buoyancy(theta_v_vals, theta_v_e_vals, w_fl_appr, qc_vals, qi_vals)
    cl_tops_idx = get_cloud_tops(ccells)
    # cl_tops = pick_with_idx_array_dimm1(z_fl_appr[None,:,:], cl_tops_idx, pres_height_axis)
    cl_top_pres = pick_with_idx_array_dimm1(pres_vals, cl_tops_idx, pres_height_axis) # in mo_cumastr set cltp=99999 for no cloud?
    liq_detr, ice_detr = get_cloud_detr(frac_detr, w_fl_appr*rho_vals, temp_vals)
    liq_detr_int, ice_detr_int = get_cloud_detr_int(liq_detr, ice_detr, delta_z) # TODO: for implementation in icon this has to be multiplied by the integration timestep (expects kg m-2 units)
    # liq_detr_int *= dT # int_detr has units of kg m-2 s-1 so multiply by timestep
    # ice_detr_int *= dT # int_detr has units of kg m-2 s-1 so multiply by timestep
    # tr_tend_liq, tr_tend_ice = get_tr_tend_liq_ice(liq_detr, ice_detr, air_layer_density)
    # temp_tend = get_temp_tend(rho_vals, w_fl_appr, s, z_fl_appr)
    # layer_heating = temp_tend*cpd*rho_vals*-delta_z*dT # temp_tend*cpd*dT
    # qv_tend = get_q_tend(rho_vals, w_fl_appr, qv_vals, z_fl_appr)
    # rain_sfc_fl, snow_sfc_fl = get_sfc_rain_snow_flux(qr_vals, qs_vals, rho_vals, w_fl_appr)
    rain_sfc_fl, snow_sfc_fl = get_sfc_rain_snow_flux_detrainment(liq_detr, ice_detr)

    # Create DataArray and save DataSet
    ds_fine_attrs = ds.attrs
    coords3d_fl = ds.u.coords
    coords3d_hl = ds.w.coords
    coords2d = ds.pres_sfc.coords
    dims3d_fl = ds.u.dims
    dims3d_hl = ds.w.dims
    dims2d = ds.pres_sfc.dims
    create_3d_fl_data_array = lambda data, name, st_name, l_name, units: create_data_array(data, name, st_name, l_name, units, coords3d_fl, dims3d_fl) # for full levels
    create_3d_hl_data_array = lambda data, name, st_name, l_name, units: create_data_array(data, name, st_name, l_name, units, coords3d_hl, dims3d_hl) # for half levels
    create_2d_data_array = lambda data, name, st_name, l_name, units: create_data_array(data, name, st_name, l_name, units, coords2d, dims2d)

    v_interpolation_desc = ''
    create_flux_data_array = None
    if v_interpolation == 'full_lvl':
        v_interpolation_desc = ' (on full lvls)'
        create_flux_data_array = create_3d_fl_data_array
    elif v_interpolation == 'half_lvl':
        v_interpolation_desc = ' (on half lvls)'
        create_flux_data_array = create_3d_hl_data_array

    print(f'create_flux_data_array: {create_flux_data_array}')
    d_arrs = []
    # dims_height_cells = tuple(dim for dim in ds.u.dims[1:])
    # coords_height = xr.core.coordinates.DataArrayCoordinates(ds.u.coords['height'])
    # d_arrs.append(create_data_array(z_fl_appr, 'z_fl', 'z_full_level_interp', 'Vertical coordinate linearly interpolated to full levels', 'm', coords_height, dims_height_cells))
    d_arrs.append(create_3d_fl_data_array(w_fl_appr, 'w_fl', 'w_full_level_interp', 'Vertical velocity linearly interpolated to full levels', 'm s-1'))
    d_arrs.append(create_3d_fl_data_array(h, 'h', 'moist_static_energy', 'Moist static energy', 'J kg-1'))

    d_arrs.append(create_2d_data_array(np.sum(ccells, axis=pres_height_axis), 'convsum', 'conv_sum', 'Sum of convective cells along height', '1'))
    d_arrs.append(create_2d_data_array(cl_tops_idx, 'clt', 'cloud_top_idx', 'Cloud Top Index', '1'))
    d_arrs.append(create_2d_data_array(cl_top_pres, 'cltp', 'cloud_top_pressure', 'Cloud Top Pressure', 'Pa'))
    d_arrs.append(create_2d_data_array(liq_detr_int, 'liq_detri', 'liquid_detr_int', 'Vertically Integrated Liquid Detrainment', 'kg m-2 s-1'))
    d_arrs.append(create_2d_data_array(ice_detr_int, 'ice_detri', 'ice_detr_int', 'Vertically Integrated Ice Detrainment', 'kg m-2 s-1'))
    d_arrs.append(create_2d_data_array(rain_sfc_fl, 'sfc_rain_fl', 'conv_sfc_rain_flux', 'Convective Surface Rain Flux', 'kg m-2 s-1'))
    d_arrs.append(create_2d_data_array(snow_sfc_fl, 'sfc_snow_fl', 'conv_sfc_snow_flux', 'Convective Surface Snow Flux', 'kg m-2 s-1'))
    d_arrs.append(create_2d_data_array(da_prec.values, 'tot_prec', 'tot_prec', 'total precip', 'kg m-2'))
    # d_arrs.append(create_3d_fl_data_array(tr_tend_liq, 'tr_tend_liq', 'liquid_tracer_tendency', 'Liquid Tracer Tendency', 's-1'))
    # d_arrs.append(create_3d_fl_data_array(tr_tend_ice, 'tr_tend_ice', 'ice_tracer_tendency', 'Ice Tracer Tendency', 's-1'))
    # d_arrs.append(create_3d_fl_data_array(qv_tend, 'qv_tend', 'qv_tendency', 'Water Vapor Tendency', 's-1'))
    # d_arrs.append(create_3d_fl_data_array(layer_heating, 'q_heat', 'q_heating', 'Layer Heating', 'J kg-1')) # Layer heating in W/m2 or K/s??
    # d_arrs.append(create_3d_fl_data_array(s, 's', 'dry_static_energy', 'Dry static energy', 'J kg-1'))
    d_arrs.append(create_flux_data_array(rhowu, 'rhowu', 'rho*w*u', f'Density times vertical velocity times zonal wind{v_interpolation_desc}', 'kg m-1 s-2'))
    d_arrs.append(create_flux_data_array(rhowv, 'rhowv', 'rho*w*v', f'Density times vertical velocity times meriodional wind{v_interpolation_desc}', 'kg m-1 s-2'))
    d_arrs.append(create_flux_data_array(rhowh, 'rhowh', 'rho*w*h', f'Density times vertical velocity times moist static energy{v_interpolation_desc}', 'm-2 s-1 J'))
    d_arrs.append(create_flux_data_array(rhowqv, 'rhowqv', 'rho*w*qv', f'Density times vertical velocity times specific humidity{v_interpolation_desc}', 'kg m-2 s-1'))
    d_arrs.append(create_flux_data_array(rhowqc, 'rhowqc', 'rho*w*qc', f'Density times vertical velocity times specific cloud water content{v_interpolation_desc}', 'kg m-2 s-1'))
    d_arrs.append(create_flux_data_array(rhowqi, 'rhowqi', 'rho*w*qi', f'Density times vertical velocity times specific cloud ice content{v_interpolation_desc}', 'kg m-2 s-1'))
    d_arrs.append(create_flux_data_array(rhowqr, 'rhowqr', 'rho*w*qr', f'Density times vertical velocity times rain mixing ratio{v_interpolation_desc}', 'kg m-2 s-1'))
    d_arrs.append(create_flux_data_array(rhowqs, 'rhowqs', 'rho*w*qs', f'Density times vertical velocity times snow mixing ratio{v_interpolation_desc}', 'kg m-2 s-1'))
    d_arrs.append(create_flux_data_array(rhowqg, 'rhowqg', 'rho*w*qg', f'Density times vertical velocity times specific graupel content{v_interpolation_desc}', 'kg m-2 s-1'))

    d_arrs.append(create_flux_data_array(wh, 'wh', 'w*h', f'Vertical velocity times moist static energy{v_interpolation_desc}', 'm s-1 J kg-1'))
    d_arrs.append(create_flux_data_array(wu, 'wu', 'w*u', f'Vertical velocity times zonal wind{v_interpolation_desc}', 'm2 s-2'))
    d_arrs.append(create_flux_data_array(wv, 'wv', 'w*v', f'Vertical velocity times meridional wind{v_interpolation_desc}', 'm2 s-2'))
    d_arrs.append(create_flux_data_array(wqv, 'wqv', 'w*qv', f'Vertical velocity times specific specific humidity{v_interpolation_desc}', 'm s-1 kg kg-1'))
    d_arrs.append(create_flux_data_array(wqc, 'wqc', 'w*qc', f'Vertical velocity times specific specific cloud water content{v_interpolation_desc}', 'm s-1 kg kg-1'))
    d_arrs.append(create_flux_data_array(wqi, 'wqi', 'w*qi', f'Vertical velocity times specific specific cloud ice content{v_interpolation_desc}', 'm s-1 kg kg-1'))
    d_arrs.append(create_flux_data_array(wqr, 'wqr', 'w*qr', f'Vertical velocity times rain mixing ratio{v_interpolation_desc}', 'm s-1 kg kg-1'))
    d_arrs.append(create_flux_data_array(wqs, 'wqs', 'w*qs', f'Vertical velocity times snow mixing ratio{v_interpolation_desc}', 'm s-1 kg kg-1'))
    d_arrs.append(create_flux_data_array(wqg, 'wqg', 'w*qg', f'Vertical velocity times specific graupel content{v_interpolation_desc}', 'm s-1 kg kg-1'))

    # d_arrs.append(create_flux_data_array(rhow, 'rhow', 'rho*h', f'Density times moist static energy{v_interpolation_desc}', 'kg m-2 s-1'))
    # d_arrs.append(create_flux_data_array(rhoh, 'rhoh', 'rho*h', f'Density times moist static energy{v_interpolation_desc}', 'kg m-3 J kg-1'))
    # d_arrs.append(create_flux_data_array(rhou, 'rhou', 'rho*u', f'Density times zonal wind{v_interpolation_desc}', 'kg m-2 s-1'))
    # d_arrs.append(create_flux_data_array(rhov, 'rhov', 'rho*v', f'Density times meridional wind{v_interpolation_desc}', 'kg m-2 s-1'))
    # d_arrs.append(create_flux_data_array(rhoqv, 'rhoqv', 'rho*qv', f'Density times specific specific humidity{v_interpolation_desc}', 'kg m-3 kg kg-1'))
    # d_arrs.append(create_flux_data_array(rhoqc, 'rhoqc', 'rho*qc', f'Density times specific specific cloud water content{v_interpolation_desc}', 'kg m-3 kg kg-1'))
    # d_arrs.append(create_flux_data_array(rhoqi, 'rhoqi', 'rho*qi', f'Density times specific specific cloud ice content{v_interpolation_desc}', 'kg m-3 kg kg-1'))
    # d_arrs.append(create_flux_data_array(rhoqr, 'rhoqr', 'rho*qr', f'Density times rain mixing ratio{v_interpolation_desc}', 'kg m-3 kg kg-1'))
    # d_arrs.append(create_flux_data_array(rhoqs, 'rhoqs', 'rho*qs', f'Density times snow mixing ratio{v_interpolation_desc}', 'kg m-3 kg kg-1'))
    # d_arrs.append(create_flux_data_array(rhoqg, 'rhoqg', 'rho*qg', f'Density times specific graupel content{v_interpolation_desc}', 'kg m-3 kg kg-1'))

    d_arrs.append(da_prec)

#     d_arrs.extend([rename_to_standard(ds[var], dims2d, dims3d) for var in vars_to_add])
    d_arrs.append(ds_cloud['qg'])
    d_arrs.extend(ds[var] for var in vars_to_add)

    ds_result_attrs = ds.attrs.copy()
    ds_result_attrs['history'] = ''
    ds_result = xr.Dataset({x.name: x for x in d_arrs},
                           attrs = ds_result_attrs)
    return ds_result


# dates = ["2016080100","2016080900","2016081200","2016081800","2016082400","2016082500","2016083000"]
# date = '2016081800'#'2013120100'#

# date = dates[int(sys.argv[1]) // 37]
date = sys.argv[1]
work_path = sys.argv[2]
print(f'date: {date}', flush=True)
region = 'DOM01'
root_path = f'/scratch/b/b309215/HErZ-NARVALII/DATA/{date}/'
theta_v_path = os.path.join(work_path, 'VirtPotTempRemapcon')
tot_prec_path = os.path.join(work_path, 'TotPrec')
target_path = os.path.join(work_path, 'ParamPrep', 'HighRes')
# if not os.path.exists(target_path):
os.makedirs(target_path, exist_ok=True)

vars_to_add = ['w','rho','theta_v','qv','qc','qi','qr','qs','tke','u','v','temp','pres']
# vars_to_add = ['h']

fg_files = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if 'fg' in f and region in f])# and 'merged' not in f and 'frland' not in f])
cloud_files = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if 'cloud' in f and region in f])
theta_v_files = sorted([os.path.join(theta_v_path, f) for f in os.listdir(theta_v_path) if 'fg' in f and region in f and 'remaplaf' not in f])
tot_prec_file = [os.path.join(tot_prec_path, f) for f in os.listdir(tot_prec_path) if region in f]
assert len(tot_prec_file) == 1, 'Only expecting one tot_prec file'
tot_prec_file = tot_prec_file[0]

# fg_files, cloud_files, theta_v_files = fg_files[1:], cloud_files[1:], theta_v_files[1:] # Skip first timestep
# print(tot_prec_file, flush=True)

# pprint(fg_files)
# pprint(cloud_files)
# pprint(theta_v_files)

# file_path = fg_files[int(sys.argv[2]) % 37]
# theta_v_file_path = theta_v_files[int(sys.argv[2]) % 37]
# cloud_file_path = cloud_files[int(sys.argv[2]) % 37]
file_path = fg_files[int(sys.argv[3])]
theta_v_file_path = theta_v_files[int(sys.argv[3])]
cloud_file_path = cloud_files[int(sys.argv[3])]
print(f'Using file {file_path}', flush=True)
print(f'and for theta_v {theta_v_file_path}', flush=True)
print(f'and for clc {cloud_file_path}', flush=True)
ds = xr.open_dataset(file_path)

ds_theta = xr.open_dataset(theta_v_file_path)
ds_cloud = xr.open_dataset(cloud_file_path)
da_prec_list = xr.open_dataset(tot_prec_file).tot_prec[::2] # ::2 range for only taking full hours (precip data has 30min time steps)
da_prec = da_prec_list[int(sys.argv[3])]
da_prec = da_prec.expand_dims({'time':1})
print(f'and time {da_prec.time.values}', flush=True)
print(f'Calculating from high-res data...', flush=True)
ds_out = create_conv_out_ds(ds, ds_theta, ds_cloud, vars_to_add, da_prec, v_interpolation='half_lvl')

print('Saving to: ', os.path.join(target_path, os.path.basename(file_path).replace('.nc', '_conv_out.nc')), '...', flush=True)
ds_out.to_netcdf(os.path.join(target_path, os.path.basename(file_path).replace('.nc', '_conv_out.nc')))
