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

def bh_influence(rho_c,rc,BHMass):
    """
    Calculate the radius of influence and tidal radius for a central black hole.
    The tidal radius is calculated for a solar mass and solar radius star. 
    """
    numerator   = 3*BHMass
    denominator = 8*np.pi*rho_c*rc*rc
    r_influence = numerator/denominator
    r_tidal     = .057*(BHMass**(0.33))                       # Coefficient is taken from Eqn 1 Baumgardt 2004 for a solar mass solar radius star.

    return r_tidal,r_influence

def rho_at_r_influence(rho_c, r_tidal, r_influence):
    """ Find the density at the boundary of the BHs influence. """

    normalization = rho_c*(r_tidal**(1.75))
    return normalization,normalization*(r_influence**(-1.75))

def model_accel_fnc(theta,r,l_values,jflag,bhflag):
    """
    Return the model acceleration.
    Note: This is the slowest part of the code. The hypergeometric function is slow.
    """

    rho_c = theta[0]
    rc    = theta[1]
    alpha = theta[2]
    r_rc  = r/rc

    if not bhflag:
        return -2.79e-10*rho_c*HYP2F1(1.5,.5*alpha,2.5,-r_rc**2)*l_values
    else:
        # Find radii and relevant indices
        BHMass               = theta[3]
        r_tidal, r_influence = bh_influence(rho_c,rc,BHMass)
        r_influence_rc       = r_influence/rc
        rshift_rc            = (r-r_influence)/rc
        accel                = np.zeros(r.size)
        inds_king            = (r>r_influence)
        inds_bh              = np.invert(inds_king)
        r3                   = 1/(r*r*r)

        # Mass of BH influenced region
        normalization    = rho_c*(r_influence**1.55)
        mass_bh_region   = 8.66*normalization*(r_influence**(1.45)-r_tidal**(1.45))
        mass_bh_region_r = 8.66*normalization*(r**(1.45)-r_tidal**(1.45))

        # Mass outside BH region
        king_region_const = -4.19*rho_c*(r_influence**3)*HYP2F1(1.5,.5*alpha,2.5,-r_influence_rc**2)
        mass_king_region  = 4.19*rho_c*(r**3)*HYP2F1(1.5,.5*alpha,2.5,-rshift_rc**2)+king_region_const

        # Find the total acceleration
        accel[inds_king] = -6.67e-11*(BHMass+mass_bh_region+mass_king_region[inds_king])*l_values[inds_king]*r3[inds_king]
        accel[inds_bh]   = -6.67e-11*(BHMass+mass_bh_region_r[inds_bh])*l_values[inds_bh]*r3[inds_bh]

        return accel

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

def log_jerk_analytical(theta,r,vmin_ind,vmax_ind,j_measured,bhflag):

    # Unpack the values
    rho_c = theta[0]
    rc    = theta[1]
    alpha = theta[2]
    vel   = np.fabs(theta[vmin_ind:vmax_ind])

    # Scaled radius
    r_rc  = r/rc

    # Get the density at each pulsar
    if not bhflag:
        rho = density_at_r(rho_c,r_rc,alpha)
    else:
        BHMass               = theta[3]
        rho                  = np.zeros(r.size)
        r_tidal, r_influence = bh_influence(rho_c,rc,BHMass)
        rshift_rc            = (r-r_influence)/rc
        inds_king            = (r>r_influence)
        inds_bh              = np.invert(inds_king)
        normalization        = rho_c*(r_influence**1.55)
        rho[inds_king]       = density_at_r(rho_c,rshift_rc[inds_king],alpha)
        rho[inds_bh]         = normalization*r[inds_bh]**(-1.55)

    # Get the characteristic jerk f0
    f0   = 2.45e-10*rho*vel

    # Get the likelihood
    denom   = np.pi*(j_measured**2+f0**2)
    loglike = np.sum(np.log(f0/denom))

    return loglike 

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
    vmin_ind           = args[15]
    vmax_ind           = args[16]
    jflag              = args[17]
    bhflag             = args[18]

    # Calculate values
    l_values       = theta[zmin_ind:zmax_ind]
    rp_l_2         = rperp*rperp+l_values*l_values
    r              = np.sqrt(rp_l_2)
    accels         = model_accel_fnc(theta,r,l_values,jflag,bhflag)
    log_pbdot      = log_likelihood_accel_pbdot_fnc(accels[PBDOT_INDS],y_measured[PBDOT_INDS],y_measured_var_div[PBDOT_INDS])
    log_iso        = log_likelihood_accel_iso_fnc(accels[ISO_INDS],y_measured[ISO_INDS],P0)
    log_l_pos      = log_l_pos_fnc(theta,rp_l_2,r,rc_grid,alpha_grid,lookup)

    if jflag:
        log_jerk = log_jerk_analytical(theta,r[J_INDS],vmin_ind,vmax_ind,j_measured,bhflag)
    else:
        log_jerk = 0

    return log_pbdot+log_iso+log_l_pos+log_jerk

def log_prior(theta,args):

    # Unpack values
    jflag  = args[17]
    bhflag = args[18]
    lb     = args[19]
    ub     = args[20]

    # Calculate the priors
    log_prior = log_prior_fnc(theta,lb,ub)

    if not bhflag:
        return log_prior
    else:
        bhmass           = theta[3]
        log_bhmass_prior = -np.log(bhmass)

        return log_prior+log_bhmass_prior
