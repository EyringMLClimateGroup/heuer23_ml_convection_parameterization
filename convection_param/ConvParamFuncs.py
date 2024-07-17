import numpy as np

def get_convective(w_hl, w_fl, ql, qi, a=0, b=0.1):
    '''
    parameters:
        a (float): threshold for individual cell vertical velocity to count as convective
                    (default 0)
        b (float): cells in a column only count as convective if there is at least one
                    cell in the column with w > b m/s (default 0.1)
    Use w_hl for choosing columns with w>1 m s-1 and interpolated
    w_fl for choosing cells with w>0.5 m s-1, another idea would be
    to use 1d "max pooling" along the height dimension
    '''
    water_mask = (ql+qi)*1e3 > 0.01
    w_cell_mask = w_fl > a    # Choose lower thresholds (see UpwardVelocity.ipynb)
    conv_mask = w_cell_mask & water_mask

    w_col_max = np.max(w_hl, axis=1)
    w_col_max_mask = w_col_max > b    # Choose lower thresholds (see UpwardVelocity.ipynb)
    w_col_max_mask3d = np.tile(w_col_max_mask[:,None,:], [1,conv_mask.shape[1],1]) # Repeat mask along height dimension
    conv_mask[w_col_max_mask3d] = 0

    return conv_mask

def get_convective_thetav_grad(theta_v, z):
    thetav_grad = grad_along_height_dim2(theta_v, z)
    return thetav_grad < 0

def get_convective_thetav_buoyancy(theta_v, theta_v_bar, w, qc, qi):
    q_mask = qc + qi > 1e-5
    w_mask = w > 0
    theta_mask = theta_v > theta_v_bar
    return q_mask & w_mask & theta_mask

def get_convective_col_depr(w_hl, w_fl, ql, qi, b=0.1):
    '''
    parameters:
        b (float): cells in a column only count as convective if there is at least one
                    cell in the column with w > b m/s (default 0.1)
    '''
    water_col_mask = np.any((ql+qi)*1e3 > 0.01, axis=1)
    w_col_max = np.max(w_hl, axis=1)
    w_col_max_mask = w_col_max > b    # Choose lower thresholds (see UpwardVelocity.ipynb)

    return w_col_max_mask & water_col_mask
