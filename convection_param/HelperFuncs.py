import xarray as xr
import numpy as np
from numba import jit,njit
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
from scipy.stats import pearsonr


def assign_bnd_coords(ds):
    bnd_var_dir = {var: ds[str(var)] for var in ds.keys() if 'bnd' in var}
    return ds.assign_coords(bnd_var_dir)


def format_date_str(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8]


def create_data_array(data, name, st_name, l_name, units, coords, dims):
    return xr.DataArray(data,
                     coords=coords,
                     dims=dims,
                     name=name,
                     attrs={'standard_name': st_name,
                            'long_name': l_name,
                            'units': units})


def grad_along_height_dim(vals3d, z, edge_order=1):
    grad = np.empty_like(vals3d)
    # Vertical axis has different orientation but gradient is invariant
    for jc in range(vals3d.shape[2]):
        grad[:,:,jc] = np.gradient(vals3d[:,:,jc], z[:,jc], axis=1, edge_order=edge_order)

    return grad

# Version without using np.gradient so that numba can be used.
# Code adapted from https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/function_base.py#L945-L1288

@njit
def grad_along_height_dim_fast(vals3d, z, edge_order=1):
    grad = np.empty_like(vals3d)
    dx = z[1:,:] - z[:-1,:]
    dx_shape = (vals3d.shape[1]-2, vals3d.shape[2], vals3d.shape[0])
    dx1 = np.transpose(dx[:-1,:].repeat(vals3d.shape[0]).reshape(dx_shape), (2,0,1))
    dx2 = np.transpose(dx[1:,:].repeat(vals3d.shape[0]).reshape(dx_shape), (2,0,1))
    a = -(dx2)/(dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    grad[:,1:-1,:] = a*vals3d[:,:-2,:] + b*vals3d[:,1:-1,:] + c*vals3d[:,2:,:]
    if edge_order == 1:
        grad[:,0,:] = (vals3d[:,1,:] - vals3d[:,0,:])/(z[1,:] - z[0,:])
        grad[:,-1,:] = (vals3d[:,-2,:] - vals3d[:,-1,:])/(z[-2,:] - z[-1,:])
    elif edge_order == 2:
        print('WARNING: With edge_order=2 numpy throws an error that a test arrays gradient with this function\
        is not equal to np.gradient(*, edge_order=2) but at least the same with an absolute tolerance of 1e-6')
        dx_0 = dx1[:,0,:]
        dx_1 = dx1[:,1,:]
        a = -(2. * dx_0 + dx_1)/(dx_0 * (dx_0 + dx_1))
        b = (dx_0 + dx_1) / (dx_0 * dx_1)
        c = - dx_0 / (dx_1 * (dx_0 + dx_1))
        grad[:,0,:] = a * vals3d[:,0,:] + b * vals3d[:,1,:] + c * vals3d[:,2,:]
        
        dx_0 = dx2[:,-2,:]
        dx_1 = dx2[:,-1,:]
        a = (dx_1) / (dx_0 * (dx_0 + dx_1))
        b = - (dx_1 + dx_0) / (dx_0 * dx_1)
        c = (2. * dx_1 + dx_0) / (dx_1 * (dx_0 + dx_1))
        grad[:,-1,:] = a * vals3d[:,-3,:] + b * vals3d[:,-2,:] + c * vals3d[:,-1,:]
    else:
        print(f'edge_order must be one of 1,2 but is {edge_order}')
        return None
#         ex = Exception(f'edge_order must be one of 1,2 but is {edge_order}')
#         raise ex
    
    return grad


def get_flattened_idx(var_expl, var):
    var_idx = [i for i,element in enumerate(var_expl) if element[0] == var]
    return min(var_idx), max(var_idx)+1


def unique_unsorted(array, return_counts=False):
    uniq_result = np.unique(array, return_index=True, return_counts=return_counts)
    if return_counts:
        uniq, index, counts = uniq_result
        idx = index.argsort()
        return uniq[idx], counts[idx]
    else:
        uniq, index = uniq_result
        idx = index.argsort()
        return uniq[idx]

    
def check_ordered(expls, variables):
    expl_vars = np.array([e[0] for e in expls])
    unique = unique_unsorted(expl_vars)
    print(unique)
    if np.all(unique == np.array(variables)):
        return True
    return False


def get_sorted_idx_from_expl_vars(expls, variables):
    expl_vars = np.array([e[0] for e in expls])
    to_concat = []
    for var in variables:
        to_concat.append(np.where(expl_vars==var)[0])
    idx_sorted = np.concatenate(to_concat)
    return idx_sorted


def reorder_output(X_all, Y_all, X_expl, Y_expl, X_vars, Y_vars):
    if not check_ordered(X_expl, X_vars) or not check_ordered(Y_expl, Y_vars):
        print('Data is not in order of X_vars / Y_vars, ordering ...')
        X_idx_sorted = get_sorted_idx_from_expl_vars(X_expl, X_vars)
        Y_idx_sorted = get_sorted_idx_from_expl_vars(Y_expl, Y_vars)

        X_expl = [X_expl[idx] for idx in X_idx_sorted]
        Y_expl = [Y_expl[idx] for idx in Y_idx_sorted]
        X_all = X_all[:,X_idx_sorted]
        Y_all = Y_all[:,Y_idx_sorted]
    else:
        print('Data is already in order of X_vars / Y_vars')
    return X_all, Y_all, X_expl, Y_expl


class StandardScaler3d():
    def __init__(self):
        self.scalar = StandardScaler()

    def fit(self, X, y=None):
        self.scalar.fit(X.reshape(X.shape[0], -1))

    def transform(self, X):
        return self.scalar.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    def inverse_transform(self, X_std):
        return self.scalar.inverse_transform(X_std.reshape(X_std.shape[0], -1)).reshape(X_std.shape)

    
class StandardScalerOneVar():
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X.reshape((X.size,1)))
    
    def transform(self, X):
        return self.scaler.transform(X.reshape((X.size,1))).reshape(X.shape)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X.reshape((X.size, 1))).reshape(X.shape)

    
def rad2degr(x):
    return x*180/np.pi


def plot_icon_tricolor(ax, clon_bnds, clat_bnds, vals, **kwargs):
    plot = ax.tripcolor(rad2degr(clon_bnds.flatten()),
                        rad2degr(clat_bnds.flatten()),
                        np.arange(clon_bnds.size).reshape(clon_bnds.shape),
                        # test['mean_rel_err'].values.squeeze(),
                        vals,
                        **kwargs)
    
    return plot


def calc_correlation(x1,x2, mode='custom'):
    if mode == 'custom':
        return np.mean((x1 - np.mean(x1))*(x2 - np.mean(x2)) / (np.std(x1)*np.std(x2)))
    elif mode == 'numpy':
        return np.corrcoef(x1,x2)[0,1]
    elif mode == 'scipy':
        return pearsonr(x1,x2).statistic
    else:
        raise Exception(f'Mode argument: {mode} not supported')


def compute_correlation_per_var(Y_true, Y_pred, multioutput='raw_values'):
    n_vars = Y_true.shape[-1]
    rs = np.empty(n_vars)
    for i in range(n_vars):
        r = calc_correlation(Y_true[...,i], Y_pred[...,i], mode='custom')
        # print(i, r)
        rs[i] = r
    
    if multioutput == 'raw_values':
        return rs
    elif multioutput == 'uniform_average':
        return np.mean(rs)
    elif multioutput == 'variance_weighted':
        return np.average(rs, weights=np.std(Y_true, axis=0))
    else:
        raise Exception(f'multioutput argument: {multioutput} not supported')
    
    return rs


def precip_rescaling(tp, eps=1):
    return np.log(1 + tp/eps)


def inv_precip_rescaling(tp_r, eps=1):
    return (np.exp(tp_r) - 1) * eps


@njit
def np_interp_invert_extralr_nongu(x, xp, fp):
    result = np.interp(x[::-1], xp[::-1], fp[::-1])[::-1]
    m0 = (fp[1] - fp[0])/(xp[1] -  xp[0])
    result[0] = m0*(x[0] -  xp[0]) + fp[0]
    m1 = (fp[-1] - fp[-2])/(xp[-1] -  xp[-2])
    result[-1] = m1*(x[-1] -  xp[-1]) + fp[-1]
    return result
    # return x[:]
