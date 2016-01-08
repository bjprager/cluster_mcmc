import numpy as np
from math import fabs
from scipy.interpolate import interp2d
from scipy.special import hyp2f1 as HYP2F1

def find_nearest_idx(array,value):
    """ Fast way to find the nearest value in a sorted array. """

    idx = np.searchsorted(array, value, side='left')

    try:
        if fabs(value-array[idx-1]) < fabs(value-array[idx]):
            return idx-1
        else:
            return idx
    except IndexError:
        return -999

def log_prior_fnc(theta,lb,ub):
    """ Calculate the prior of the distributions. """

    if (theta>lb).all() and (theta<ub).all():
        return 0
    else:
        return -np.inf

def log_l_pos_fnc(theta,rp_l_2,r,rc_grid,alpha_grid,lookup):
    """ Get the probability of having a given l value from Phinney 3.7 """

    # Unpack values
    r_c      = theta[1]
    alpha    = theta[2]
    r_c_sqrd = r_c*r_c

    # Calculate the numerator
    if alpha > 1 and alpha < 7:
        numerator   = r*np.sqrt((rp_l_2+r_c_sqrd)**(-alpha))
    else:
        return -np.inf

    # Lookup the best possible denominator from our lookup table
    idx_r       = find_nearest_idx(rc_grid,r_c)
    idx_a       = find_nearest_idx(alpha_grid,alpha)
    denominator = lookup[:,idx_r,idx_a]

    # Get the probability given by 3.7
    prob = numerator*denominator

    if np.greater(prob,0).all():
        return np.sum(np.log(prob))
    else:
        return -np.inf

def model_accel_fnc(theta,r,l_values):
    """
    Return the model acceleration.
    Note: This is the slowest part of the code. The hypergeometric function is slow.
    """

    rho_prefac   = -2.79e-10*theta[0]
    r_rc         = r/theta[1]
    alpha        = theta[2]

    return rho_prefac*HYP2F1(1.5,.5*alpha,2.5,-r_rc**2)*l_values

def log_likelihood_accel_pbdot_fnc(y_model,y_measured,y_measured_var_div):
    """ Return the log likelihood values for a normal distribution of accelerations around a model for the binary systems with a measured PBDOT. """

    dy           = y_measured-y_model
    logL1        = dy*dy*y_measured_var_div

    return -.5*np.sum(logL1-np.log(y_measured_var_div))

def prob_b_field_atnf(B):
    """ Return the probability of finding a b field as defined by a log-normal distribution of ATNF psrs. """

    B8log       = np.log(1e-8*B)
    dB          = B8log-1.23297934
    sigma_inv_2 = 3.4388                                # sigma = 0.53925782330903649
    return -(0.5*(dB*dB*sigma_inv_2)+B8log)

def log_likelihood_accel_iso_fnc(y_model,y_measured,P0):
    """ Return the log likelihood values for a distribution of accelerations around the predicted flat B-field distribution. (Isolated pulsars) """

    dy         = np.abs(y_measured-y_model)
    Bvals      = np.sqrt(1.25e31*dy)*P0

    return np.sum(prob_b_field_atnf(Bvals))

def density_at_r(rho_c,r_rc,alpha):
    """ Return the density at a given location in the cluster. """
    return rho_c*(1+r_rc**2)**(-0.5*alpha)

def log_jerk_mf_fnc(theta,r,vmin_ind,vmax_ind,j_measured,j_measured_var_div):
    """ Find the approximate jerk from the mean field. """

    # Unpact values
    rho_c = theta[0]
    r_rc  = r/theta[1]
    alpha = theta[2]

    # Get the density at each pulsar
    rho_psr = density_at_r(rho_c,r_rc,alpha)

    # Get the predicted mean field jerk
    j_mf = 2.79e-10*rho_psr*theta[vmin_ind:vmax_ind]

    return j_mf

def log_jerk_neigh_fnc(theta,r,j_measured,jerk_mass_grid,jerk_rc_grid,neigh_jerk_lookup):
    """ Find the approximate jerk from a neighboring star. """

    # Unpack values
    rho_c = theta[0]
    rc    = theta[1]
    rc_pc = 3.24e-17*rc
    r_rc  = r/rc
    Mtot  = 5e-31*theta[3]

    # Derive the PDF
    idx_M  = find_nearest_idx(jerk_mass_grid,Mtot)
    idx_rc = find_nearest_idx(jerk_rc_grid,rc_pc)

    if idx_M == -999 or idx_rc == -999:
        return -np.inf

    fits  = neigh_jerk_lookup[:,idx_M,idx_rc]
    mu    = fits[0]*r_rc+fits[1]
    sigma = fits[2]*r_rc+fits[3]

    # Get the log likelihood for neighbors
    jlog      = np.log(j_measured)
    dJ        = jlog-mu
    log_neigh = np.sum(-0.5*(dJ*dJ/(sigma*sigma))-jlog)

    return log_neigh

def log_likelihood(theta,args):
    """ Calculate the likelihood of our distributions. """

    # Unpack values
    rperp              = args[0]
    P0                 = args[1]
    y_measured         = args[2]
    y_measured_var_div = args[3]
    j_measured         = args[4]
    j_measured_var_div = args[5]
    zmin_ind           = args[6]
    zmax_ind           = args[7]
    ISO_INDS           = args[8]
    PBDOT_INDS         = args[9]
    rc_grid            = args[10]
    alpha_grid         = args[11]
    lookup             = args[12]
    l_signs            = args[13]
    J_INDS             = args[14]
    jerk_mass_grid     = args[15]
    jerk_rc_grid       = args[16]
    neigh_jerk_lookup  = args[17]
    vmin_ind           = args[18]
    vmax_ind           = args[19]
    jflag              = args[20]

    # Calculate values
    l_values       = theta[zmin_ind:zmax_ind]
    rp_l_2         = rperp*rperp+l_values*l_values
    r              = np.sqrt(rp_l_2)
    accels         = model_accel_fnc(theta,r,l_values)
    log_pbdot      = log_likelihood_accel_pbdot_fnc(accels[PBDOT_INDS],y_measured[PBDOT_INDS],y_measured_var_div[PBDOT_INDS])
    log_iso        = log_likelihood_accel_iso_fnc(accels[ISO_INDS],y_measured[ISO_INDS],P0)
    log_l_pos      = log_l_pos_fnc(theta,rp_l_2,r,rc_grid,alpha_grid,lookup)

    if jflag:
        j_mf     = log_jerk_mf_fnc(theta,r[J_INDS],vmin_ind,vmax_ind,j_measured,j_measured_var_div)
        log_jerk = log_jerk_neigh_fnc(theta,r[J_INDS],np.fabs(j_measured-j_mf),jerk_mass_grid,jerk_rc_grid,neigh_jerk_lookup)
    else:
        log_jerk = 0

    return log_pbdot+log_iso+log_l_pos+log_jerk

def log_prior(theta,args):

    # Unpack values
    lb = args[21]
    ub = args[22]
    return log_prior_fnc(theta,lb,ub)
