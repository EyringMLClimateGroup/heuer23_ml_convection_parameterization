import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
# import psyplot.project as psy
# import psyplot
import os
import sys
import datetime
from itertools import chain
from convection_param.HelperFuncs import *
from convection_param.ConvParamFuncs import *
from convection_param.Constants import cpd, dT
# import warnings
# warnings.filterwarnings('error')

def get_adv_subgrid_tend(wvar, w, var, z):
    subg_flux = rho*(wvar - w*var)
    sgmf_grad = grad_along_height_dim_fast(subg_flux, z)
    return -1/rho * sgmf_grad # TODO: - sign?

def get_adv_subgrid_layer_heat(ws, w, s, z, dT, delta_z, rho):
    subg_flux = rho*(ws - w*s)
    sgmf_grad = grad_along_height_dim_fast(subg_flux, z)
    return -delta_z*dT*sgmf_grad

def add_tke_fl(ds):
    tke_fl_appr = (ds['tke'].values[:,1:,:] + ds['tke'].values[:,:-1,:])/2
    tke_fl_da = create_data_array(tke_fl_appr, 'tke_fl', 'specific_kinetic_energy_of_air_full_level_interpolated', 'Turbulent Kinetic Energy Full Level Interpolated', 'm2 s-2', ds.v.coords, ds.v.dims)
    ds = ds.assign({tke_fl_da.name: tke_fl_da})
    return ds

# def add_subg_mom_tends(ds):
    # z_ifc_vals = ds.z_ifc.values  # Half level geometric height
    # delta_z = np.diff(z_ifc_vals, axis=0)
    # z_fl_appr = ds.z_fl.values
    # w_fl_appr = ds.w_fl.values
    # rho_vals = ds.rho.values
    # variables = ['u','v','qv','qc','qi','qr','qs']
    # res_names = ['u_tend', 'v_tend', 'qv_tend', 'qc_tend', 'qi_tend', 'qr_tend', 'qs_tend']
    # res_descs = ['zonal_wind_tendency', 'merid_wind_tendency', 'specific_humidity_cont_tend', 'specific_cloud_water_content_tend', 'specific_cloud_ice_content_tend', 'cloud_rain_mixing_ratio_tend', 'cloud_snow_mixing_ratio_tend']
    # res_long_descs = ['Zonal Wind Tendency', 'Meridional Wind Tendency', 'Specific Humidity Content Tendency', 'Specific Cloud Water Content Tendency', 'Specific Cloud Ice Content Tendency', 'Cloud Rain Mixing Ratio Tendency', 'Cloud Snow Mixing Ratio Tendency']
    # res_units = ['m s-2','m s-2','s-1','s-1','s-1','s-1','s-1']

    # d_arrs = []
    # for var, res_name, res_desc, res_long_desc, res_unit in zip(variables, res_names, res_descs, res_long_descs, res_units):
        # wvar = 'w' + var
        # var_field = ds[var].values
        # wvar_field = ds[wvar].values

        # subgrid_tend = get_adv_subgrid_tend(wvar_field, w_fl_appr, var_field, z_fl_appr, rho_vals)
        # d_arrs.append(create_data_array(subgrid_tend, res_name, res_desc, res_long_desc, res_unit, ds.v.coords, ds.v.dims))

    # ds = ds.assign({x.name: x for x in d_arrs})


    # layer_heating = get_adv_subgrid_layer_heat(ds['ws'].values, ds['w_fl'].values, ds['s'].values, z_fl_appr, dT, delta_z, rho_vals)
    # layer_heat_da = create_data_array(layer_heating, 'q_heat', 'q_heating', 'Layer Heating', 'J m-2', ds.v.coords, ds.v.dims) # Rather switch to W m-2 ?
    # ds = ds.assign({layer_heat_da.name: layer_heat_da})

    # return ds

def get_adv_subgrid_flux(wvar, w, var, rho):
    subg_flux = rho*(wvar - w*var)
    return subg_flux


def get_adv_subgrid_flux_w_rho(rhowvar, w, var, rho):
    rhowvar_coarse = rho*w*var
    rhowvar_coarse_hl = np.zeros_like(rhowvar)
    #todo: interp here
    # interp_func = lambda x: (x, )
    # np.apply_along_axis(, axis=1, arr=rhowvar_coarse)
    rhowvar_coarse_hl[:,1:-1,:] = (rhowvar_coarse[:,1:,:] + rhowvar_coarse[:,:-1,:])/2
    subg_flux = (rhowvar - rhowvar_coarse_hl)
    return subg_flux

def add_subg_mom_fluxes(ds):
    variables = ['u','v','h','qv','qc','qi','qr','qs']
    res_names = ['subg_flux_u',
                'subg_flux_v',
                'subg_flux_h',
                'subg_flux_qv',
                'subg_flux_qc',
                'subg_flux_qi',
                'subg_flux_qr',
                'subg_flux_qs']
    res_descs = ['subg_zonal_momentum_flux',
                'subg_merid_momentum_flux',
                'subg_moist_stat_energy_flux',
                'subg_specific_humidity_cont_flux',
                'subg_specific_cloud_water_content_flux',
                'subg_specific_cloud_ice_content_flux',
                'subg_cloud_rain_mixing_ratio_flux',
                'subg_cloud_snow_mixing_ratio_flux']
    res_long_descs = ['Subgrid Zonal Momentum Flux',
                'Subgrid Meridional Momentum Flux',
                'Subgrid Moist Static Energy Flux',
                'Subgrid Specific Humidity Content Flux',
                'Subgrid Specific Cloud Water Content Flux',
                'Subgrid Specific Cloud Ice Content Flux',
                'Subgrid Cloud Rain Mixing Ratio Flux',
                'Subgrid Cloud Snow Mixing Ratio Flux']
    res_units = ['kg m-1 s-2',
                'kg m-1 s-2',
                'J m-2 s-1',
                'kg m-2 s-1',
                'kg m-2 s-1',
                'kg m-2 s-1',
                'kg m-2 s-1',
                'kg m-2 s-1']


    d_arrs = []
    for var, res_name, res_desc, res_long_desc, res_unit in zip(variables, res_names, res_descs, res_long_descs, res_units):
        w_fl_appr = ds.w_fl.values
        rho_field = ds.rho.values

#         wvar = 'w' + var
#         var_field = ds[var].values
#         wvar_field = ds[wvar].values

#         subgrid_flux = get_adv_subgrid_flux(wvar_field, w_fl_appr, var_field, rho_field)
#         d_arrs.append(create_data_array(subgrid_flux, res_name, res_desc, res_long_desc, res_unit, ds.v.coords, ds.v.dims))

        rhowvar = 'rhow' + var
        var_field = ds[var].values
        rhowvar_field = ds[rhowvar].values

        subgrid_flux = get_adv_subgrid_flux_w_rho(rhowvar_field, w_fl_appr, var_field, rho_field)
        d_arrs.append(create_data_array(subgrid_flux, res_name, res_desc, res_long_desc, res_unit, ds[rhowvar].coords, ds[rhowvar].dims))

    ds = ds.assign({x.name: x for x in d_arrs})

    return ds

def gather_training_data(path, X_vars, Y_vars, conv_col_threshold, unconv_frac, exclude_fst_step, time_av, return_fluxes, test_data=False):
    '''
    conv_col_threshold: minimum number of convective cells that must be present in column (averaged over region)
    '''
    X_all_list = []
    Y_all_list = []
    files = sorted([d for d in os.listdir(path) if '.nc' in d])
    # print(files)
    fst_file = [f for f in files if '0000' in f]
    if exclude_fst_step:
        if len(fst_file) == 1:
            files.remove(fst_file[0])
        elif len(fst_file) == 0:
            print('No first timestep found')
        else:
            raise Exception('There should only be one first timestep nc file present.')
    else:
        if len(fst_file) == 0:
            raise Exception('No first timestep found')

    idx0 = time_av
    idx1 = len(files) - time_av
    ds_to_av = []

    def open_preprocess_dataset(path):
        ds = xr.open_dataset(path)
        ds = ds.dropna('cell')
        if return_fluxes:
            ds = add_subg_mom_fluxes(ds)
        else:
            raise Exception('Tendency mode currently not supported')
            # ds = add_subg_mom_tends(ds)
        # ds = add_tke_fl(ds)
        return ds

    for i,f in enumerate(files[idx0:idx1]):
        print(f'File: {f}')
        ds = open_preprocess_dataset(os.path.join(path, f))
        ds_to_av = []
        for j in list(range(i-time_av,i))+list(range(i+1,i+time_av+1)):
            ds_to_av.append(open_preprocess_dataset(os.path.join(path, files[j])))

        ccols = ds.convsum.values >= conv_col_threshold
        print(f'ccols mask shape: {ccols.shape}')
        print(f'Found {ccols.sum()} convective columns')

        nzero = np.transpose(np.nonzero(~ccols))

        if test_data:
            rand_false_idx = nzero
        else:
            percent_to_add = unconv_frac   # Add unconv_frac percent of unconvective cells
            n_unconvective = int(ccols.sum() * percent_to_add/(1-percent_to_add))
            n_unconvective = min(n_unconvective, len(nzero)) # only take unconv_frac % if there are enough unconvective cells, else take less
            rand_false_idx = nzero[npr.choice(nzero.shape[0], size=n_unconvective, replace=False),:]

        ccols[tuple(rand_false_idx.T)] = True
        rnd_false_mask = np.zeros(ccols.shape, dtype=bool)
        rnd_false_mask[tuple(rand_false_idx.T)] = True

        ds_X_vals = [(ds[a].values+sum([ds_to_av[k][a].values for k in range(2*time_av)]))/(2*time_av+1) for a in X_vars]
        ds_Y_vals = [(ds[a].values+sum([ds_to_av[k][a].values for k in range(2*time_av)]))/(2*time_av+1) for a in Y_vars]

        ds_X_vals = [vals if len(vals.shape)==3 else vals[:,None,:] if len(vals.shape)==2 else vals[None,None,:] for vals in ds_X_vals]
        ds_Y_vals = [vals if len(vals.shape)==3 else vals[:,None,:] if len(vals.shape)==2 else vals[None,None,:] for vals in ds_Y_vals]

        # append file data to lists (concatenate along feature dimension)
        # Take transpose to (time,cell,height) because ccols has shape (time, cell)
        X_all_list.append(np.concatenate([np.transpose(vals, (0,2,1))[ccols] for vals in ds_X_vals], axis=1))
        # --- try with cyclic daytime encoding ---
        # daytime = ds.time.dt.hour.values[0]
        # X_all_list.append(np.concatenate([np.transpose(vals, (0,2,1))[ccols] for vals in ds_X_vals] + [np.full(ccols.shape, np.sin(2*np.pi*daytime/24)),
        #                                                                                                np.full(ccols.shape, np.cos(2*np.pi*daytime/24))], axis=1))
        # X_var_expl = list(chain(*[[[var, i+1] for i in range(ds[var].shape[1])] if len(ds[var].shape)==3 else [[var, i] for i in range(1)] for var in X_vars] +
        # [['hour_sin', 0],['hour_cos', 0]]))
        # --- end: try with cyclic daytime encoding ---

        Y_to_concat = []
        for vals in ds_Y_vals:
            valsT = np.transpose(vals, (0,2,1))
            valsT[rnd_false_mask] = 0
            Y_to_concat.append(valsT[ccols])
        Y_all_list.append(np.concatenate(Y_to_concat, axis=1))


    # Return lists concatenated along sample dimension
    result_X = np.concatenate(X_all_list, axis=0)
    result_Y = np.concatenate(Y_all_list, axis=0)
    X_var_expl = list(chain(*[[[var, i+1] for i in range(ds[var].shape[1])] if len(ds[var].shape)==3 else [[var, i] for i in range(1)] for var in X_vars]))
    Y_var_expl = list(chain(*[[[var, i+1] for i in range(ds[var].shape[1])] if len(ds[var].shape)==3 else [[var, i] for i in range(1)] for var in Y_vars]))
    return result_X, result_Y, X_var_expl, Y_var_expl


def extract_vars_fl_height(ds, variables):
    fl_height = len(ds['height'])
    hl_height = fl_height + 1
    result = {}
    for var in variables:
        da = ds[var]
        if len(da.values.shape) == 3 and da.values.shape[1] == fl_height:
            result[var] = da.values
        elif len(da.values.shape) == 3 and da.values.shape[1] == hl_height:
            print(f'Interpolating to full levels for variable {var}')
            result[var] = (da.values[:,1:,:] + da.values[:,:-1,:])/2
        elif len(da.values.shape) == 2:
            result[var] = da.values
        else:
            print(f'Found variable with non hl/fl height with shape {da.values.shape}; fl height is {fl_height}')

    return result


def read_coarse_data(path, target_format, X_vars, Y_vars, conv_col_threshold, unconv_frac, exclude_fst_step=True, time_av=0, return_fluxes=0): #conv_col_threshold=2, unconv_frac=0.1
    npz_files = [d.path for d in os.scandir(path) if d.name.endswith('.npz')
                                                  and target_format in d.name
                                                  and f'ct{conv_col_threshold}' in d.name # conv_col_threshold in data name
                                                  and f'uf{unconv_frac}' in d.name # unconv_frac in data name
                                                  and f'exfs{int(exclude_fst_step)}' in d.name # exclude_fst_step in data name
                                                  and f'tav{time_av}' in d.name # time_average length in data name
                                                  and f'retflux{return_fluxes}' in d.name]

    if target_format=='seq':
        if len(npz_files) == 0:
            X_all, Y_all, X_expl, Y_expl = gather_training_data(path, X_vars, Y_vars, conv_col_threshold, unconv_frac, exclude_fst_step, time_av, return_fluxes)
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(path, f'seq_{now}_ct{conv_col_threshold}_uf{unconv_frac}_exfs{int(exclude_fst_step)}_tav{time_av}_retflux{return_fluxes}.npz')
            print(f'Saving to {file_path}')
            np.savez(file_path, X_all=X_all, Y_all=Y_all, X_expl=X_expl, Y_expl=Y_expl)
        elif len(npz_files) == 1:
            print(f'Loading from npz file: {npz_files[0]}')
            data = np.load(npz_files[0], mmap_mode='r')
            X_all, Y_all, X_expl, Y_expl = data['X_all'], data['Y_all'], data['X_expl'], data['Y_expl']
            X_expl, Y_expl = X_expl.tolist(), Y_expl.tolist()
        else:
            raise Exception(f'Found multiple npz files: {npz_files}')
        output = (X_all, Y_all, X_expl, Y_expl)
    else:
        raise Exception(f'target_format {target_format} not found, only <seq> is supported for now')

    return output
