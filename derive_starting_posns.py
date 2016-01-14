import numpy as np
import pylab as PLT
from optparse import OptionParser
from astropy import units,constants
from scipy.special import hyp2f1 as HYP2F1

def bh_influence(rho_c,r_c,BHMass):
    """
    Calculate the radius of influence and tidal radius for a central black hole.
    The tidal radius is calculated for a solar mass and solar radius star. 
    """

    numerator   = 3*BHMass
    denominator = 8*np.pi*rho_c*r_c*r_c
    r_influence = numerator/denominator
    r_tidal     = .057*(BHMass**(0.33))                       # Coefficient is taken from Eqn 1 Baumgardt 2004 for a solar mass solar radius star.

    return r_tidal,r_influence

def bh_accel(r,r_rc,l_values,rho_c,r_c,alpha,BHMass):
    """ Returns the expected accelerations if there is a central black hole. """

    # Find radii and relevant indices
    r_tidal, r_influence = bh_influence(rho_c,r_c,BHMass)
    rshift_rc            = (r-r_influence)/r_c
    r_influence_rc       = r_influence/r_c
    accel                = np.zeros(r.size)
    inds_king            = (r>r_influence)
    inds_bh              = np.invert(inds_king)

    # Mass of BH influenced region
    normalization    = rho_c*(r_influence**1.55)
    mass_bh_region   = 8.66*normalization*(r_influence**(1.45)-r_tidal**(1.45))
    mass_bh_region_r = 8.66*normalization*(r**(1.45)-r_tidal**(1.45))

    # Mass outside BH region
    king_region_const = -4.19*rho_c*(r_influence**3)*HYP2F1(1.5,.5*alpha,2.5,-r_influence_rc**2)
    mass_king_region  = 4.19*rho_c*(r**3)*HYP2F1(1.5,.5*alpha,2.5,-rshift_rc**2)+king_region_const

    # Find the total acceleration
    accel[inds_king] = -6.67e-11*(BHMass+mass_bh_region+mass_king_region[inds_king])*l_values[inds_king]*constants.pc.value/(r[inds_king]**3)
    accel[inds_bh]   = -6.67e-11*(BHMass+mass_bh_region_r[inds_bh])*l_values[inds_bh]*constants.pc.value/(r[inds_bh]**3)

    return accel

def king_accel(rho_c,r_rc,alpha,l_values):
    """ Returns the accelerations at a given r for the King model. """

    return -2.79e-10*rho_c*HYP2F1(1.5,.5*alpha,2.5,-r_rc**2)*l_values*constants.pc.value

def king_accel_los_posn(psr_data,rho_c,r_c,alpha,BHMass,BHflag):
    """ Calculate the best line of sight position for a given pulsar acceleration and given cluster parameters. """

    # Output array with [L1,L2,L1_prob,L2_prob]
    output = np.zeros((psr_data.shape[0],4))

    # Iterate over pulsars
    for idx,ipsr in enumerate(psr_data):
        if np.sign(ipsr[1]) == 1:
            l_values = np.sort(-np.logspace(-4,2,10000))
        else:
            l_values = np.logspace(-4,2,10000)

        # Calculate the radii and accelerations for model values
        r_values = np.sqrt(ipsr[0]**2+l_values**2)*constants.pc.value
        r_rc     = r_values/r_c
        if not BHflag:
            accels = king_accel(rho_c,r_rc,alpha,l_values)
        else:
            accels = bh_accel(r_values,r_rc,l_values,rho_c,r_c,alpha,BHMass)

        # Find where the maximum acceleration is
        accel_max_ind = np.argmax(np.fabs(accels))

        # Find the best solution for L1 and L2
        if np.sign(accels[accel_max_ind]) == 1:
            L1 = l_values[np.argmin(np.fabs(accels[accel_max_ind:]-ipsr[1]))+accel_max_ind]*constants.pc.value
            L2 = l_values[np.argmin(np.fabs(accels[:accel_max_ind]-ipsr[1]))]*constants.pc.value
        else:
            L1 = l_values[np.argmin(np.fabs(accels[:accel_max_ind]-ipsr[1]))]*constants.pc.value
            L2 = l_values[np.argmin(np.fabs(accels[accel_max_ind:]-ipsr[1]))+accel_max_ind]*constants.pc.value

        # Get the probabilities
        L1_prob = prob_l(ipsr[0],L1,np.sort(np.append(-l_values,l_values))*constants.pc.value,r_c,alpha)
        L2_prob = prob_l(ipsr[0],L2,np.sort(np.append(-l_values,l_values))*constants.pc.value,r_c,alpha)

        output[idx] = [L1,L2,L1_prob,L2_prob]

    return output

def npsr_fnc(rperp,l,r_c,alpha):
    """ Return the pulsar density """

    return (r_c**2+rperp**2+l**2)**(-alpha/2)

def prob_l(rperp,l_psr,l_values,r_c,alpha):

    npsr          = npsr_fnc(rperp,l_psr,r_c,alpha)
    npsr_l_values = npsr_fnc(rperp,l_values,r_c,alpha)
    numerator     = npsr*np.sqrt(rperp**2+l_psr**2)
    denominator   = npsr_l_values*np.sqrt(rperp**2+l_values**2)

    return numerator/np.trapz(denominator,x=l_values)

def get_los_posn_prob(psr_data,rho_c,r_c,alpha,BHMass,bhflag):
    """ Return an array with pulsar positions and probabilities. """

    los_data = king_accel_los_posn(psr_data,rho_c,r_c,alpha,BHMass,BHflag=bhflag)
    los_data = np.hstack((psr_data,los_data))

    return los_data

if __name__ == "__main__":

    ### Define Command Line Options
    parser = OptionParser()
    parser.add_option("-f", "--psrdata", action="store", type="string", dest="psrdata", help="Saved numpy array with Nx2 array for N pulsars with [position (pc),measured accel (m/s^2)]")
    parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", default=None, help="File name for the output numpy array with pulsar los positions and probabilities.")
    parser.add_option("--blackhole", action="store_true", dest="bhflag", default=False, help="Flag to find acceleration with a central black hole.")
    parser.add_option("--rho", action="store", type="float", dest="rho", default=.9, help="Central density in units of 10^6 Msun/pc^3.")
    parser.add_option("--rc", action="store", type="float", dest="rc", default=.18, help="Core density in units of pc.")
    parser.add_option("--alpha", action="store", type="float", dest="alpha", default=2.5, help="Spectral index")
    parser.add_option("--BHmass", action="store", type="float", dest="BHMass", default=1000, help="Black Hole Mass in units of solar masses.")
    (options, args) = parser.parse_args()

    # Read in the pulsar data
    psr_data = np.load(options.psrdata)

    # Convert cluster parameters into mks units
    rho_c  = (1e6*options.rho*units.solMass/units.pc**3).decompose().value
    r_c    = (options.rc*units.pc).decompose().value
    alpha  = options.alpha
    BHMass = (options.BHMass*units.solMass).decompose().value

    # Get the positions and probabilities
    los_data = get_los_posn_prob(psr_data,rho_c,r_c,alpha,BHMass,options.bhflag)

    # Save the output
    if options.outfile == None:
        if options.bhflag:
            fname = 'position_probabilities_bh.npy'
        else:
            fname = 'position_probabilities.npy'
    else:
        fname = options.outfile
    np.save(fname,los_data)
