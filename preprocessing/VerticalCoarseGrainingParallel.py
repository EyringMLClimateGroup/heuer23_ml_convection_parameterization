#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from numba import njit
from convection_param.HelperFuncs import create_data_array
import dask


@njit
def vert_coarse_grain_fl(var, zh_lr_vals, zh_hr_vals):
    result = np.full((var.shape[0], zh_lr_vals.shape[0]-1, var.shape[2]), np.nan)
    for j in range(zh_lr_vals.shape[0]-1):
        z_u = zh_lr_vals[j,:]
        z_l = zh_lr_vals[j+1,:]
        weights = np.maximum(np.minimum(z_u, zh_hr_vals[:-1]) - np.maximum(zh_hr_vals[1:], z_l), 0)

        # Following implementations are equivalent
        # result[0,j,:] = np.einsum('ij,ij->j', weights, var[0])/(z_u - z_l)
        # result[0,j,:] = np.diag(weights.T @ var[0])/(z_u - z_l)
        # result[0,j,:] = np.array([np.sum(np.array([weights[k,l]*var[0,k,l] for k in range(weights.shape[0])])) for l in range(var.shape[2])])/(z_u - z_l)
        result[0,j,:] = np.sum(np.multiply(weights, var[0]), axis=0)/(z_u - z_l)

        result[0,j] = np.where((np.abs((z_u - z_l) - np.sum(weights, axis = 0))) >= 0.5, np.nan, result[0,j])

    return result

@njit
def vert_coarse_grain_hl(var, zh_lr_vals, zf_hr_vals):
    result = np.full((var.shape[0], zh_lr_vals.shape[0], var.shape[2]), np.nan)
    result[:,0,:] = 0 # Boundary condition for half lvl fluxes
    result[:,-1,:] = 0 # Boundary condition for half lvl fluxes

    zf_lr_vals = (zh_lr_vals[1:,:] + zh_lr_vals[:-1,:]) / 2
    for j in range(zf_lr_vals.shape[0]-1):
        z_u = zf_lr_vals[j,:]
        z_l = zf_lr_vals[j+1,:]
        weights = np.maximum(np.minimum(z_u, zf_hr_vals[:-1]) - np.maximum(zf_hr_vals[1:], z_l), 0)

        # Following implementations are equivalent
        # result[0,j+1,:] = np.einsum('ij,ij->j', weights, var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.diag(weights.T @ var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.array([np.sum(np.array([weights[k,l]*var[0,k+1,l] for k in range(weights.shape[0])])) for l in range(var.shape[2])])/(z_u - z_l)
        result[0,j+1,:] = np.sum(np.multiply(weights, var[0,1:-1]), axis=0)/(z_u - z_l)

        result[0,j+1] = np.where((np.abs((z_u - z_l) - np.sum(weights, axis = 0))) >= 0.5, np.nan, result[0,j+1])

    return result

def coarse_grain(da, da_nan, nnan_cell_mask):
    print(da.name)
    if 'height' in list(da.dims): # full lvl case
        vals = da_nan.values[...,nnan_cell_mask]
        if da.dims == ('height', 'cell'):
            print('In notime branch now')
            vals = vals[None,...]
            vals_lr = vert_coarse_grain_fl(vals, zh_lr_vals, zh_hr_vals)[0,...]
            coords = lr_vfl_zf_coords_notime
        elif da.dims == ('time', 'height', 'cell'):
            coords = lr_vfl_coords
            vals_lr = vert_coarse_grain_fl(vals, zh_lr_vals, zh_hr_vals)
        return create_data_array(vals_lr, da.name, da.standard_name, da.long_name, da.units, coords, da.dims)

    elif 'height_2' in list(da.dims): # half lvl case
        vals = da_nan.values[...,nnan_cell_mask]
        if da.dims == ('time', 'height_2', 'cell'):
            coords = lr_vhl_coords
            vals_lr = vert_coarse_grain_hl(vals, zh_lr_vals, zf_hr_vals)
        return create_data_array(vals_lr, da.name, da.standard_name, da.long_name, da.units, coords, da.dims)

    elif all(['height' not in dim for dim in da.dims]): # also append 2d data
        return da

    else:
        return None

if __name__ == '__main__':
    in_file = sys.argv[2]
    print(f'Coarse graining {in_file}')
    base_path_project = '/work/bd1179/b309215'

    if sys.argv[1] == 'R02B04':
        ds_lr = xr.open_dataset(f'{base_path_project}/R02B04VertGrid/zghalf_icon-a_capped.nc')
        # zh_hr_vals = xr.open_dataset(f'{base_path_project}/heuer22_convection_parameterization/Processed/2013120100/ParamPrep/LowRes/R02B04/dei4_NARVALI_2013120100_fg_DOM01_ML_0000_conv_out_R02B04_m2degr.nc')['z_ifc'].values
        zh_hr_vals = xr.open_dataset(f'{base_path_project}/R02B04VertGrid/dei4_NARVALI_2013120100_fg_DOM01_ML_0000_conv_out_R02B04_m2degr_zifc.nc')['z_ifc'].values
    elif sys.argv[1] == 'R02B05':
        ds_lr = xr.open_dataset(f'{base_path_project}/R02B05VertGrid/zghalf_icon-a_capped_R02B05.nc')
        # zh_hr_vals = xr.open_dataset(f'{base_path_project}/heuer22_convection_parameterization/Processed/2013120100/ParamPrep/LowRes/R02B05/dei4_NARVALI_2013120100_fg_DOM01_ML_0000_conv_out_R02B05_m2degr.nc')['z_ifc'].values
        zh_hr_vals = xr.open_dataset(f'{base_path_project}/R02B05VertGrid/dei4_NARVALI_2013120100_fg_DOM01_ML_0000_conv_out_R02B05_m2degr_zifc.nc')['z_ifc'].values
    elif sys.argv[1] == 'R02B06':
        ds_lr = xr.open_dataset(f'{base_path_project}/R02B06VertGrid/zghalf_icon-a_0021_R02B06-capped.nc')
        zh_hr_vals = xr.open_dataset(f'{base_path_project}/R02B06VertGrid/dei4_NARVALI_2013123100_fg_DOM01_ML_0000_zifc_R02B06_0021_m2degr.nc')['z_ifc'].values

    zh_lr_vals = ds_lr['zghalf'].values
    zh_lr_vals_zifc = ds_lr.zghalf.rename('z_ifc') # get lr half levels and rename so it is consistent with hr data

    # with open('./vertc_in_files.txt', 'r') as f:
        # files = f.read().splitlines()

    # in_file = files[int(sys.argv[1])]
    high_res_ds = xr.open_dataset(in_file)
    test_field = high_res_ds['u'].values
    nnan_cell_mask = ~np.isnan(np.sum(test_field, axis=(0,1)))
    # print(f'shape of non nan cell mask: {nnan_cell_mask.shape}')
    # print(f'Found {nnan_cell_mask.sum()} not nan cells')
    # test_field = test_field[...,nnan_cell_mask]
    zh_hr_vals = zh_hr_vals[...,nnan_cell_mask]
    zh_lr_vals = zh_lr_vals[...,nnan_cell_mask]
    zf_hr_vals = (zh_hr_vals[1:,:] + zh_hr_vals[:-1,:]) / 2
    high_res_ds_nna = high_res_ds.copy(deep=True).dropna('cell', subset=['u']) # drop all nan cells based on zonal wind
    # high_res_cell_sort = np.argsort(high_res_ds_nna.u, )
    # print(f'Shape of test field: {test_field.shape}')

    zh_lr_vals_size = zh_lr_vals.shape[0]
    zf_lr_size = zh_lr_vals_size - 1

    # Construct low-res coordinates (with and without time)
    u = high_res_ds_nna.u.copy(deep=True)
    u = u.sel(height=slice(0,zf_lr_size))
    lr_vfl_coords = u.coords

    coord_dict_vfl = dict(high_res_ds_nna.u.coords)
    del coord_dict_vfl['time']
    lr_vfl_zf_coords_notime = xr.Dataset(coord_dict_vfl).coords

    # for vertical half lvls
    rhowu = high_res_ds_nna.rhowu.copy(deep=True)
    rhowu = rhowu.sel(height_2=slice(0,zh_lr_vals_size))
    lr_vhl_coords = rhowu.coords

    coord_dict_vhl = dict(high_res_ds_nna.rhowu.coords)
    del coord_dict_vhl['time']
    lr_vhl_zf_coords_notime = xr.Dataset(coord_dict_vhl).coords


    from dask.distributed import Client
    n_da = len(high_res_ds.values())
    n_workers = int(sys.argv[4])
    client = Client(n_workers=n_workers)#processes=False)

    # results = Parallel(n_jobs=10)(delayed(coarse_grain)(da) for da in high_res_ds_nna.values())
    print(f'Constructing lazy_results for {n_da} data arrays')
    lazy_results = [dask.delayed(coarse_grain)(da, da_nan, nnan_cell_mask) for da, da_nan in zip(high_res_ds_nna.values(), high_res_ds.values())]
    print('Constructed lazy_results: ', lazy_results)
    results = dask.compute(*lazy_results)
    # results = [coarse_grain(da) for da in high_res_ds_nna.values()]
    print('Done with computation: ', results)
    lowres_das = [result for result in results if result is not None]
    print('Data Arrays: ', lowres_das)

    ds_out = xr.Dataset({x.name: x for x in lowres_das}, attrs = high_res_ds_nna.attrs)
    # out_file = in_file.replace('.nc', '_vertc.nc')
    out_file = sys.argv[3]
    print(f'Writing vertically coarse grained file to {out_file} ...')
    ds_out.to_netcdf(out_file)
