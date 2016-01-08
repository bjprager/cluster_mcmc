import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import os
import corner
import numpy as np
import pylab as PLT
from sys import exit
from glob import glob
from math import fabs
from psr_utils import *
import mcmc_fncs as mcmcfnc
from scipy.special import gamma
from scipy.integrate import quad
from emcee import PTSampler,utils
from parfile_units import psr_par
from optparse import OptionParser
from astropy import units as units
from presto import sphere_ang_diff
from scipy.special import factorial
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp2d
from astropy import constants as const
from scipy.special import hyp2f1 as HYP2F1

def optlist(option, opt, value, parser):
    """
    Split up option parser values into a list of floats.
    """
    setattr(parser.values, option.dest, N.asarray([float(x) for x in value.split(',')]))

class massagedata:

    def __init__(self,files,cluster):
        """ Define a few things for prepping the data. """
	self.files            = files                        # List of par files
	self.cluster          = cluster                      # Which cluster are we working with
	self.KMPERKPC         = const.pc.to(units.km)
        self.SECPERJULYR      = (1.*units.year).to(units.s)
        self.tauc             = 10000000000.0*units.year     # Characteristic Age
        self.r_c_offset_error = 0.0*units.arcsec             # Offset error in arcsec
	self.c                = const.c

    class DataBunch(dict):
        """
        Creates a simple class instance db = DataBunch(a=1, b=2,....) 
        that has attributes a and b, which are callable and update-able 
        using either syntax db.a or db['a'].
        """
        def __init__(self, **kwds):
            dict.__init__(self, kwds)
            self.__dict__ = self

    def read_parfiles(self):
        """
        Read in all parfiles and store relevant fields.
        """

        ### Useful cluster parameters. This should be updated to reflect individual clusters being examined.
        cluster = {"Ter5": ["Terzan5", "17:48:04.85", "-24:46:44.6", 7.0, 7.9/60.0, 0.83, 13.27, 7.60, 8.97, 20.33, 5.06, 3.8, 1.7, 17.0, 21.72, 5.0], \
                   "M28": ['M28', "18:24:32.89", "-24:52:11.4", 7.0, 7.9/60.0, 0.83, 13.27, 7.60, 8.97, 20.33, 5.06, 8.14, -6.18, 17.0, 21.72, 5.0], \
                   "47TUC": ['47TUC', "00:24:5.29", "-72:04:52.3", 7.0, 7.9/60.0, 0.83, 13.27, 7.60, 8.97, 20.33, 5.06, 305.89, -44.88, 17.0, 21.72, 5.0]}

        ### Create an easy to use dictionary with simple attributes
        params = self.DataBunch(R=[],ACCEL=[],ACCEL_ERR=[],P0=[],P0_ERR=[],P1=[],P1_ERR=[],P2=[],P2_ERR=[],P3=[],P3_ERR=[],A1=[],A1_ERR=[], \
                                PB=[],PB_ERR=[],PBDOT=[],PBDOT_ERR=[],ACCEL_BINARY=[],ACCEL_BINARY_ERR=[],FNAME=[],PSRNAME=[],ACCEL_CLUSTER=[],ACCEL_CLUSTER_ERR=[], \
                                RA_OFFSET=[],DEC_OFFSET=[],DM=[])

	for ipar in self.files:
            psr         = psr_par(ipar,unit_flag=True)
            psr.cluster = self.cluster

            ### Calculate the Surface Density and its Error
            psr.D                = cluster[psr.cluster][3]
            psr.l                = (cluster[psr.cluster][11]*units.deg).to(units.rad)
            psr.b                = (cluster[psr.cluster][12]*units.deg).to(units.rad)
            psr.cluster_RA       = ra_to_rad(cluster[psr.cluster][1])*units.rad
            psr.cluster_DEC      = dec_to_rad(cluster[psr.cluster][2])*units.rad
            psr.core_ang_asec    = (sphere_ang_diff(psr.RA_RAD.value, psr.DEC_RAD.value, psr.cluster_RA.value, psr.cluster_DEC.value)*units.rad).to(units.arcsec)+self.r_c_offset_error
            R0                   = (8.34*units.kpc).to(units.m)
            R0_error             = (.16*units.kpc).to(units.m)
            Omega0               = (240.0*units.km/units.s).to(units.m/units.s)
            Omega0_error         = (8.0*units.km/units.s).to(units.m/units.s)
            d                    = (5.9*units.kpc).to(units.m)
            d_error              = (.5*units.kpc).to(units.m)
            beta                 = (d/R0)*np.cos(psr.b)-np.cos(psr.l)
            beta_error           = np.sqrt(((d_error/R0)*np.cos(psr.b))**2.+((R0_error*d/R0**2.)*np.cos(psr.b))**2.)
            psr.abyc_gal         = (-np.cos(psr.b)*((Omega0**2.)/R0)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/const.c
            psr.abyc_gal_error   = np.sqrt((((-np.cos(psr.b)*((Omega0**2.)/R0))/const.c)*((np.sin(psr.l)**2.-beta**2.)/(np.sin(psr.l)**2.+beta**2.)**2.)*beta_error)**2.
                                  +((-np.cos(psr.b)*((Omega0_error*Omega0)/R0)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/const.c)**2.
                                  +((-np.cos(psr.b)*((R0_error*Omega0**2.)/R0**2.)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/const.c)**2.)
            psr.abyc_meas        = psr.P1/psr.P0
            psr.abyc_meas_error  = np.sqrt((psr.P1_ERR/psr.P0)**2.+(psr.P0_ERR*psr.P1/psr.P0**2.)**2.)
            psr.abyc_clust       = psr.abyc_meas - psr.abyc_gal
            psr.abyc_clust_error = np.sqrt(psr.abyc_meas_error**2.+psr.abyc_gal_error**2.)

            ### Store important attributes to params
            params.R.append((psr.core_ang_asec).to(units.arcmin))
            params.ACCEL.append(psr.abyc_clust*self.c)
            params.ACCEL_ERR.append(psr.abyc_clust_error*self.c)
            params.P0.append(psr.P0)
            params.P0_ERR.append(psr.P0_ERR)
            params.P1.append(psr.P1)
            params.P1_ERR.append(psr.P1_ERR)
            params.FNAME.append(psr.FILE.split('.par')[0])
            params.PSRNAME.append(psr.FILE.split('.par')[0].split('-2446')[-1])
            params.ACCEL_CLUSTER.append(0.*units.m/(units.s**2.))
            params.ACCEL_CLUSTER_ERR.append(0.*units.m/(units.s**2.))
            params.RA_OFFSET.append((psr.RA_RAD-psr.cluster_RA).to(units.arcsec))
            params.DEC_OFFSET.append((psr.DEC_RAD-psr.cluster_DEC).to(units.arcsec))
            params.DM.append(psr.DM)

            if hasattr(psr, 'F2'):
		params.P2.append(-1.0*((psr.F2/psr.F0**2)-2.0*psr.F1**2/psr.F0**3))
        	params.P2_ERR.append(((psr.P0_ERR*(2*psr.P0*psr.F2+2*(psr.P1**2)/(psr.P0**2)))**2+(psr.F2_ERR*psr.P0**2)**2+(4*psr.P1*psr.P1_ERR/psr.P0)**2)**.5)
            else:
                unit = 1./units.s
        	params.P2.append(0*unit)
        	params.P2_ERR.append(0*unit)

            if hasattr(psr, 'F3'):
		dF0 = 2.0*(psr.F3*psr.F0**2+12.0*psr.F1**3-9.0*psr.F0*psr.F1*psr.F2)/psr.F0**5
		dF1 = 6.0*(psr.F0*psr.F2-3.0*psr.F1**2)/psr.F0**4
		dF2 = 6.0*psr.F1/psr.F0**3
		dF3 = -1.0/psr.F0**2
		err = np.sqrt((dF0*psr.F0_ERR)**2+(dF1*psr.F1_ERR)**2+(dF2*psr.F2_ERR)**2+(dF3*psr.F3_ERR)**2)

		params.P3.append(-1.0*(psr.F3*psr.F0**2+6.0*psr.F1**3-6.0*psr.F0*psr.F1*psr.F2)/psr.F0**4)
        	params.P3_ERR.append(err)
            else:
	        unit = 1./units.s**2.
        	params.P3.append(-999*unit)
        	params.P3_ERR.append(-999*unit)

            if hasattr(psr, 'PBDOT'):
        	params.A1.append(psr.A1*self.c)
        	params.A1_ERR.append(psr.A1_ERR*self.c)
        	params.PB.append(psr.PB.to(units.s))
        	params.PB_ERR.append(psr.PB_ERR.to(units.s))
        	params.PBDOT.append(psr.PBDOT*1e-12)
        	params.PBDOT_ERR.append(psr.PBDOT_ERR*1e-12)
        	params.ACCEL_BINARY.append((psr.PBDOT*1e-12/(psr.PB.to(units.s))-psr.abyc_gal)*self.c)
        	params.ACCEL_BINARY_ERR.append(np.sqrt((psr.PBDOT_ERR*1e-12/(psr.PB.to(units.s)))**2+(psr.PBDOT*1e-12*psr.PB_ERR.to(units.s)/(psr.PB.to(units.s))**2)**2)*self.c)
            else:
	        unit = units.m
        	params.A1.append(0*unit)
        	params.A1_ERR.append(0*unit)

                unit = units.s
        	params.PB.append(0*unit)
        	params.PB_ERR.append(0*unit)

                unit = units.s/units.s
        	params.PBDOT.append(0*unit)
        	params.PBDOT_ERR.append(0*unit)
		
                params.ACCEL_BINARY.append(0.*units.m/(units.s**2.))
                params.ACCEL_BINARY_ERR.append(0.*units.m/(units.s**2.))

	# Clean up the units a bit. Not good to have them IN the list.
	for ikey in params.keys():
            try:
                unit = params[ikey][0].unit
	        tmp  = []
	        for ivalue in params[ikey]:
	            tmp.append(ivalue.value)
	        params[ikey] = np.asarray(tmp)*unit
	    except AttributeError:
	        params[ikey] = np.asarray(params[ikey])

	### Return values in params sorted by projected distance from the core. (Returns as numpy array).
	idx = np.argsort(params.R)
	for i in params.keys():
	    params[i] = params[i][idx]

        return params

class mcmc_fit:

    def __init__(self,params,options):
        self.params  = params
        self.options = options
        self.rperp   = (params.R.value*5900*const.pc.value*60./206265.)
        self.bmin    = options.bmin
        self.bmax    = options.bmax

        # Initial Conditions
        self.theta_init     = np.copy(options.theta_init)
        self.theta_init[0]  = (self.theta_init[0]*units.solMass/units.pc**3).decompose().value
        self.theta_init[1] *= const.pc.value
        self.theta_init[3] *= const.M_sun.value

        if options.jerkflag:
            self.nparam = 4
        else:
            self.nparam     = 3
            self.theta_init = self.theta_init[:-1]

        # Define the indices bounding nuisance parameters to make grabbing things easy
        self.PBDOT_INDS = (self.params.ACCEL_BINARY.value!=0)
        self.ISO_INDS   = np.invert(self.PBDOT_INDS)
        self.J_INDS     = (self.params.P2.value!=0)
        self.rsize      = params.R.size
        self.isosize    = np.invert(self.params.ACCEL_BINARY.value!=0).sum()
        self.jsize      = (self.params.P2.value!=0).sum()
        self.zmin_ind   = self.nparam
        self.zmax_ind   = self.nparam+self.rsize
        self.bmin_ind   = self.nparam+self.rsize
        self.bmax_ind   = self.nparam+self.rsize+self.isosize
        self.vmin_ind   = self.nparam+self.rsize+self.isosize
        self.vmax_ind   = self.nparam+self.rsize+self.isosize+self.jsize
        self.ndim       = self.nparam+self.rsize+self.isosize

        # Add velocities if using jerk flag
        if options.jerkflag:
            self.ndim    += self.jsize

        # Define some variables for the mcmc code
        self.rc_max     = 2.*const.pc.value
        self.Mtot_min   = 50000
        self.Mtot_max   = 1000000
        self.jerk_pdf_grid()

    def jerk_pdf_grid(self,dr=0.5):
        """ Create a list with functions for the mean and standard deviation of jerk. """

        nmass                = 500
        nrc                  = 500
        self.jerk_mass_grid  = np.linspace(np.amin(self.Mtot_min),np.amax(self.Mtot_max),nmass)
        self.jerk_rc_grid    = np.linspace(0,self.rc_max/const.pc.value,nrc)

        if os.path.exists('neigh.jerk.lookup.npy'):
            self.neigh_jerk_lookup = np.load('neigh.jerk.lookup.npy')
        else:
            print "Making the lookup table for neighbor jerks."

            # Load data from simulations
            mu_contour     = np.load('mu.grid.0.5rc.npy')
            sigma_contour  = np.load('sigma.grid.0.5rc.npy')
            mu_fnc_list    = []
            sigma_fnc_list = []

            for ii in range(mu_contour.shape[0]):
                mus    = mu_contour[ii]
                sigmas = sigma_contour[ii]
                masses = mus[0,:,0]
                rcs    = mus[1,0,:]

                mu_fnc_list.append(interp2d(rcs, masses, mus[2]))
                sigma_fnc_list.append(interp2d(rcs, masses, sigmas[2]))

            # Make the lookup table
            jerk_radii             = 0.5*np.arange(mu_contour.shape[0])+0.25
            self.neigh_jerk_lookup = np.zeros((4,nmass,nrc))

            for ii,imass in enumerate(self.jerk_mass_grid):
                for jj,irc in enumerate(self.jerk_mass_grid):
                    mu        = np.vstack([fnc(irc,imass) for fnc in mu_fnc_list])
                    sigma     = np.vstack([fnc(irc,imass) for fnc in sigma_fnc_list])
                    fit_mu    = np.polyfit(jerk_radii,mu.T[0],1)
                    fit_sigma = np.polyfit(jerk_radii,sigma.T[0],1)
                    self.neigh_jerk_lookup[0,ii,jj] = fit_mu[0]
                    self.neigh_jerk_lookup[1,ii,jj] = fit_mu[1]
                    self.neigh_jerk_lookup[2,ii,jj] = fit_sigma[0]
                    self.neigh_jerk_lookup[3,ii,jj] = fit_sigma[1]

            np.save('neigh.jerk.lookup.npy',self.neigh_jerk_lookup)

    def l_pos_denom_lookup(self):
        """ Make a lookup table for the l pos denominator according to Phinney 1993 Eqn 3.7. """

        # Define the grid of lookup values
        n_rc            = 2000
        n_alpha         = 2000
        self.rc_grid    = np.linspace(0,5,n_rc)*const.pc.value
        self.alpha_grid = np.linspace(0,8,n_alpha) 

        if os.path.exists(self.options.lookup):
            self.lookup = np.load(self.options.lookup)
        else:
            n_l         = 1000
            l_grid      = np.linspace(0,150,n_l)*const.pc.value
            l_grid_sqrd = l_grid*l_grid
            self.lookup = np.zeros((self.rperp.size,n_rc,n_alpha))

            # Create the denominator for Eqn 3.7 from Phinney 1993
            for idx_rperp in range(self.rperp.size):
                rp2_l2      = self.rperp[idx_rperp]**2+l_grid_sqrd
                rp2_l2_sqrt = np.sqrt(rp2_l2)

                for idx_rc in range(n_rc):
                    n_psr_base = self.rc_grid[idx_rc]**2+rp2_l2

                    for idx_alpha in range(n_alpha):
                        n_psr                                   = n_psr_base**(-0.5*self.alpha_grid[idx_alpha])
                        self.lookup[idx_rperp,idx_rc,idx_alpha] = 2*np.trapz(n_psr*rp2_l2_sqrt,x=l_grid)
            np.save(self.options.lookup, self.lookup)

        # Invert the array to save time during the simulations
        self.lookup = 1/self.lookup

    def accel_lookup(self):
        """
        Make a lookup table for the model acceleration. This gets around slow calculation time for HYP2F1.
        Not currently used because the lookup time more or less equals the time to calculate.
        """

        # Make the r/r_c grid
        n_rrc         = 2000
        self.rrc_grid = np.linspace(0,100,n_rrc)

        if os.path.exists(self.options.hyp2f1):
            self.hyp2f1 = np.load(self.options.hyp2f1)
        else:
            self.hyp2f1 = np.zeros((n_rrc,self.alpha_grid.size))

            for idx,ialpha in enumerate(self.alpha_grid):
                for idy,irrc in enumerate(self.rrc_grid):
                    self.hyp2f1[idy,idx] = HYP2F1(1.5,.5*ialpha,2.5,-irrc**2)
            np.save(self.options.hyp2f1, self.hyp2f1)

    def rough_l_guess(self,y_measured,shape):
        """ Derive some initial guesses on l. """

        if os.path.exists('Ter5_psr_starting_posn.npy'):
            posn_data = np.load('Ter5_psr_starting_posn.npy')
            locations = np.ones(shape)

            for ii in range(shape[1]):
                relative_probability = posn_data[ii,5]/posn_data[ii,4]
                for jj in range(shape[0]):
                    z_prob           = np.random.random_sample()
                    if z_prob < relative_probability:
                        locations[jj,ii] = posn_data[ii,3]
                    else:
                        locations[jj,ii] = posn_data[ii,2]
            return locations
        else:
            l_out  = np.zeros(self.rperp.size)
            da_out = np.zeros(self.rperp.size)
            signs  = -1*np.sign(y_measured)
            l_grid = np.logspace(12,17,20000)
            for idx,ipsr in enumerate(self.rperp):
                igrid       = signs[idx]*l_grid
                r           = np.sqrt(ipsr**2+igrid**2)
                rho_prefac  = -2.79e-10*self.theta_init[0]
                r_rc        = r/self.theta_init[1]
                accel       = rho_prefac*HYP2F1(1.5,.5*self.theta_init[2],2.5,-r_rc**2)*igrid
                d_accel     = np.fabs(accel-y_measured[idx])
                best_ind    = np.argmin(d_accel)
                l_out[idx]  = igrid[best_ind]
                da_out[idx] = d_accel[best_ind]
            return np.ones(shape)[:]*l_out

    def log_normal(self,x,mu,sigma):
        return (1./(sigma*x*np.sqrt(2*np.pi))*np.exp(-((np.log(x)-mu)/(np.sqrt(2)*sigma))**2))

    def bfield_dist(self,bmin,bmax,mu=1.23297934,sigma=0.53925782330903649,shape=(1,)):
        b_guesses = (10**np.random.uniform(bmin,bmax,size=shape))*1e-8
        probs     = self.log_normal(b_guesses,mu,sigma)
        guesses   = np.random.random_sample(size=probs.shape)

        while np.greater(guesses,probs).any() == True:
            inds            = (guesses>probs)
            new_b_guesses   = (10**np.random.uniform(bmin,bmax,size=shape))*1e-8
            b_guesses[inds] = new_b_guesses[inds]
            probs           = self.log_normal(b_guesses,mu,sigma)
            new_guesses     = np.random.random_sample(size=probs.shape)
            guesses[inds]   = new_guesses[inds]

        return b_guesses

    def plot_starting_posn(self,starting_guesses):
        """ Plot histogram of starting positions for some sanity checks. """

        #### Plot the histogram of starting positions
        dpi     = 100
        width   = 600/dpi
        height  = 600/dpi
        fig     = PLT.figure(figsize=(width,height),dpi=dpi)
        gs      = gridspec.GridSpec(12,1)
        gs_yind = 4
        data    = np.load('ratio.npy')
        data    = data[1:]
        intersection_ratio_2 = np.interp(2,data[:,1],data[:,0])

        # Main body of the plot
        ax1   = fig.add_subplot(gs[gs_yind:,0])
        posns = np.unique(np.sqrt(starting_guesses[:,:,self.zmin_ind:self.zmax_ind]**2+self.rperp**2))/self.theta_init[1]
        pbin  = np.logspace(np.log10(np.amin(posns)),np.log10(np.amax(posns)),50)
        ax1.hist(posns,bins=pbin,facecolor='k',alpha=.5)
        ax1.vlines(intersection_ratio_2,ax1.get_ylim()[0],ax1.get_ylim()[1],color='r',linestyle='--')
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$\frac{|r|}{r_c}$',fontsize=18)
        ax1.set_ylabel(r'Counts',fontsize=18)

        # Plot the top plot with ratios
        ax2  = fig.add_subplot(gs[:gs_yind,0],sharex=ax1)

        for idx in range(data.shape[1]-1):
            ax2.plot(data[:,0],data[:,idx+1])

        ax2.hlines(2,1e-1,1e2,color='r',linestyle='--')
        ax2.vlines(intersection_ratio_2,ax2.get_ylim()[0],ax2.get_ylim()[1],color='r',linestyle='--')
        ax2.set_xlim([2e-1,1e2])
        ax2.set_yticks([1,3,5,7])
        ax2.set_ylabel(r'$\frac{v_{\rm pred}}{v_{\rm meas}}$', fontsize=20)
        PLT.setp(ax2.get_xticklabels(), visible=False)

        fig.tight_layout()
        PLT.savefig('starting_posn_dist.png')

    def make_prior_array(self,l_signs):

        # Out a better way to handle priors quickly
        lb = np.array([5e-15,0,1,self.Mtot_min*const.M_sun.value])
        ub = np.array([5e-13,self.rc_max,8,self.Mtot_max*const.M_sun.value])

        if not self.options.jerkflag:
            lb = lb[:-1]
            ub = ub[:-1]

        for idx in range(l_signs.size):
            if l_signs[idx] == 1:
                lb = np.append(lb,0)
                ub = np.append(ub,np.inf)
            else:
                lb = np.append(lb,-np.inf)
                ub = np.append(ub,0)

        for idx in range(l_signs[self.ISO_INDS].size):
            lb = np.append(lb,self.bmin)
            ub = np.append(ub,self.bmax)

        if self.options.jerkflag:
            for idx in range(l_signs[self.J_INDS].size):
                lb = np.append(lb,-50000)
                ub = np.append(ub,50000)

        return lb,ub

    def call_mcmc(self):
        """ Start running the emcee code. """

        # Get the data and the intrinsic spin-down values
        y_measured,y_measured_error       = self.params.ACCEL.value,self.params.ACCEL_ERR.value
        y_measured[self.PBDOT_INDS]       = self.params.ACCEL_BINARY[self.PBDOT_INDS].value
        y_measured_error[self.PBDOT_INDS] = self.params.ACCEL_BINARY_ERR[self.PBDOT_INDS].value
        y_measured_var                    = y_measured_error**2
        y_measured_var_div                = (1/y_measured_error**2)
        j_measured                        = (self.params.P2[self.J_INDS].value/self.params.P0[self.J_INDS].value)*const.c.value
        j_measured_var                    = ((self.params.P2_ERR[self.J_INDS].value/self.params.P0[self.J_INDS].value)*const.c.value)**2
        j_measured_var_div                = (1/j_measured_var)
        P0                                = self.params.P0[self.ISO_INDS].value

        # MCMC setup parameters
        nwalkers = int(self.options.nwalker)
        nsteps   = int(self.options.nchain)
        ntemps   = int(self.options.ntemp)
        nthreads = int(self.options.nthread)
        nburn    = int(self.options.nburn)
        nglide   = int(self.options.nglide)
        nthin    = int(self.options.nthin)

        # Get initial guesses for l
        l_guesses       = self.rough_l_guess(y_measured,(nwalkers,self.rsize))
        l_guesses_scale = .1*np.fabs(l_guesses)
        l_signs         = np.sign(l_guesses[0])

        # Boundaries for the flat prior
        lb,ub = self.make_prior_array(l_signs)

        # Initial guesses for walkers
        if os.path.exists('starting.guesses.npy'):
            starting_guesses = np.load('starting.guesses.npy')
        else:
            np.random.seed(0)
            starting_guesses = np.zeros((ntemps,nwalkers,self.ndim))
            for ii in range(ntemps):
                starting_guesses[ii,:,:self.nparam]                = np.random.normal(self.theta_init, .1*self.theta_init, (nwalkers,self.nparam))
                starting_guesses[ii,:,self.zmin_ind:self.zmax_ind] = np.random.normal(l_guesses, l_guesses_scale, (nwalkers,self.rsize))
                starting_guesses[ii,:,self.bmin_ind:self.bmax_ind] = self.bfield_dist(np.log10(self.bmin),np.log10(self.bmax),shape=(nwalkers,self.isosize))*1e8

                # Get the initial velocity guesses if needed
                if self.options.jerkflag:
                    v_inits  = np.sqrt(4*np.pi*const.G.value*starting_guesses[ii,:,0]/3.)*starting_guesses[ii,:,1]
                    v_inits  = np.repeat(v_inits,self.jsize).reshape(-1,self.jsize)
                    j_signs  = np.sign(j_measured)
                    v_inits *= j_signs
                    v_inits  = np.random.normal(v_inits,.05*np.fabs(v_inits))
                    starting_guesses[ii,:,self.vmin_ind:self.vmax_ind] = v_inits

            np.save('starting.guesses.npy',starting_guesses)

            # Plot distribution of starting positions
            self.plot_starting_posn(starting_guesses)

        # Make the argument list to pass to the MCMC handler
        args = [self.rperp,P0,y_measured,y_measured_var_div,j_measured,j_measured_var_div,self.zmin_ind,self.zmax_ind \
                ,self.ISO_INDS,self.PBDOT_INDS,self.rc_grid,self.alpha_grid,self.lookup,l_signs,self.J_INDS \
                ,self.jerk_mass_grid,self.jerk_rc_grid,self.neigh_jerk_lookup,self.vmin_ind,self.vmax_ind,self.options.jerkflag,lb,ub]

        # Sample the distribution
        try:
            sampler = PTSampler(ntemps, nwalkers, self.ndim, mcmcfnc.log_likelihood, mcmcfnc.log_prior, loglargs=[args], logpargs=[args],threads=nthreads)
        except AssertionError:
            print "Incorrect number of walkers for the given dimensions. Minimum number of walkers needed: %d." %(2*self.ndim)
            exit()

        # Burning in the data
        print "Beginning the burn-in."
        try:
            for p, lnprob, lnlike in sampler.sample(starting_guesses, iterations=nburn):
                pass
            sampler.reset()
            print "Finished the burn in. Starting the sampling"
        except ValueError:
            print "Starting Guess array dimensions do not match input. Please delete this file and rerun the script to generate new starting positions"
            exit()

        for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=nsteps, thin=nthin):
            pass

        # Save the array
        np.save(self.options.outfile, sampler.chain)
        np.save('chain.p.npy',p)
        np.save('chain.lbprob.npy',lnprob)
        np.save('chain.lnlike.npy',lnlike)

    #################################
    ####### Plotting Routines #######
    #################################

    def tri_plot_init(self,h=600.,w=600.,nsubplot=311):
        """ Initialize plotting environment """
        dpi      = 100
        width    = w/dpi
        height   = h/dpi
        self.fig = PLT.figure(figsize=(width,height),dpi=dpi)
        self.ax  = self.fig.add_subplot(nsubplot)

    def chains_to_dict(self,names,full_chain):
        chains   = [full_chain[:,:,ii].T for ii in range(len(names))]
        return dict(zip(names,chains))

    def plot_mcmc(self):
        """ Plot the mcmc results in a corner plot. """

        # Dimensions of the simulation
        nwalkers   = self.options.nwalker
        nsteps     = self.options.nchain
        nburn      = self.options.nburn
        glide      = self.options.nglide
        label      = [r'$10^6\rho_c (M_\odot pc^{-3})$', r'$r_c$ (pc)',r'$\alpha$',r'$M_{\rm tot} (10^6 M_\odot)$']
        label      = label[:self.nparam]

        if self.options.corner:

            if os.path.exists('corner_%s' %(self.options.outfile)):
                full_chain         = np.load('corner_%s' %(self.options.outfile))
            else:
                full_chain         = np.load(self.options.outfile)[0]
                full_chain         = full_chain[:,:,:self.nparam]
                full_chain[:,:,0] *= 1e-6*(1.*units.kg/units.m**3).to(units.solMass/units.pc**3).value
                full_chain[:,:,1] /= const.pc.value

                if self.options.jerkflag:
                    full_chain[:,:,3] /= (1e6*const.M_sun.value)

                np.save('corner_%s' %(self.options.outfile), full_chain)

            full_chain = full_chain.reshape(-1,self.nparam)
            truth      = np.copy(self.theta_init)
            truth[0]  *=  1e-6*(1.*units.kg/units.m**3).to(units.solMass/units.pc**3).value
            truth[1]  /= const.pc.value

            print "Plotting the cluster parameter corner plot."
            fig = corner.corner(full_chain, bins=60, labels=label, truths=truth)
            fig.tight_layout()
            fig.savefig("mcmc_cluster_params.png")
            PLT.close('all')

        if self.options.chain or self.options.histogram:
            glide = self.options.nglide

            # Massage the data and downsample it to plot faster
            if os.path.exists('glide_%s' %(self.options.outfile)):
                full_chain = np.load('glide_%s' %(self.options.outfile))
            else:
                full_chain                 = np.load(self.options.outfile)[0]
                full_chain                 = full_chain[:,::glide,:]
                full_chain[:,:,0]         *= 1e-6*(1.*units.kg/units.m**3).to(units.solMass/units.pc**3).value
                full_chain[:,:,1]         /= const.pc.value
                full_chain[:,:,self.zmin_ind:self.zmax_ind] /= const.pc.value

                if self.options.jerkflag:
                    full_chain[:,:,3] /= (1e6*const.M_sun.value)
                    full_chain[:,:,self.vmin_ind:self.vmax_ind] /= 1e3

                np.save('glide_%s' %(self.options.outfile), full_chain)

            # Make the labels
            for idx in range(self.rsize):
                label.append(r'$z_{\rm %s}$ (pc)' %(self.params.FNAME[idx].split('-2446')[-1]))

            ISO_FNAME = self.params.FNAME[self.ISO_INDS]
            for idx in range(self.isosize):
                label.append(r'$B_{\rm %s}$ (G)' %(ISO_FNAME[idx].split('-2446')[-1]))

            if self.options.jerkflag:
                JERK_FNAME = self.params.FNAME[self.J_INDS]
                for idx in range(self.jsize):
                    label.append(r'$v_{\rm %s}$ (km/s)' %(JERK_FNAME[idx].split('-2446')[-1]))

            # Get the data dict
            chain_dict = self.chains_to_dict(label,full_chain)

            # Re-order the keys
            bidx      = 0
            vidx      = 0
            keys      = np.array(chain_dict.keys())
            key_order = np.zeros(keys.size)
            for idx,ikey in enumerate(keys):
                if ikey[1] == '1' and ikey[0] != 'B':
                    key_order[0] = idx
                elif ikey[1] == 'r':
                    key_order[1] = idx
                elif ikey[1] == '\\':
                    key_order[2] = idx
                elif ikey[1] == 'M' and self.options.jerkflag:
                    key_order[3] = idx
                elif ikey[1] == 'z':
                    psr_name             = ikey.split('rm ')[-1].split('}')[0]
                    psr_idx              = np.where(self.params.PSRNAME==psr_name)[0][0]
                    key_order[self.nparam+psr_idx] = idx
                elif ikey[1] == 'B':
                    key_order[self.bmin_ind+bidx]  = idx
                    bidx                          += 1
                elif ikey[1] == 'v' and self.options.jerkflag:
                    key_order[self.vmin_ind+vidx]  = idx
                    vidx                          += 1

            keys = keys[key_order.astype('int')]

            # Plot the chains if requested
            if self.options.chain:
                # Plot the chains
                print "Plotting the chains."
                fnum   = 0
                for idx,iname in enumerate(keys):
                    if idx%3 == 0:
                        self.tri_plot_init()
                        self.ax.plot(np.arange(chain_dict[iname].shape[0])*glide,chain_dict[iname],color='k',alpha=0.02)
                        self.ax.set_ylabel(iname, fontsize=16)
                        self.ax.set_xlim([0,chain_dict[iname].shape[0]])
                        PLT.setp(self.ax.get_xticklabels(), visible=False)

                    if idx%3 == 1:
                        self.ax2 = self.fig.add_subplot(312,sharex=self.ax)
                        self.ax2.plot(np.arange(chain_dict[iname].shape[0])*glide,chain_dict[iname],color='k',alpha=0.02)
                        self.ax2.set_ylabel(iname, fontsize=16)
                        PLT.setp(self.ax2.get_xticklabels(), visible=False)

                    if idx%3 == 2:
                        self.ax3 = self.fig.add_subplot(313,sharex=self.ax)
                        self.ax3.plot(np.arange(chain_dict[iname].shape[0])*glide,chain_dict[iname],color='k',alpha=0.02)
                        self.ax3.set_ylabel(iname, fontsize=16)
                        self.ax3.set_xlabel('Steps')

                    if idx%3 == 2:
                        self.fig.tight_layout()
                        PLT.savefig('cluster_chains%02d.png' %(fnum))
                        PLT.close('all')
                        fnum += 1

                PLT.savefig('cluster_chains%02d.png' %(fnum))


            if self.options.histogram:
                psr_posn = []
                # Plot the histogram
                for idx,iname in enumerate(keys):
                    nbin  = 60
                    scale = 'linear'
                    data  = chain_dict[iname].flatten()

                    # Massage the data slightly and make filenames
                    if idx == 0:
                        fname = 'histogram_rho_core.png'
                    elif idx == 1:
                        fname = 'histogram_core_radius.png'
                    elif idx == 2:
                        fname = 'histogram_alpha.png'
                    else:
                        psr_name = iname.split('rm ')[-1].split('}')[0]
                        ptype    = iname[1]
                        fname    = 'histogram_%s_%s.png' %(ptype,psr_name)

                        if ptype == 'B':
                            scale = 'log'
                            nbin  = np.logspace(np.log10(np.amin(data)),np.log10(np.amax(data)),40)
                        elif ptype == 'z':
                            sign  = np.sign(data.mean())
                            data  = np.fabs(data)
                            scale = 'log'
                            nbin  = np.logspace(np.log10(np.amin(data)),np.log10(np.amax(data)),40)
                            if sign == -1:
                                iname = '|'+iname+'|'

                    # Plot the histograms
                    self.tri_plot_init(nsubplot=111)
                    counts,edges,inds = self.ax.hist(data, bins=nbin, facecolor='black', alpha=0.75)
                    self.ax.set_xlabel(iname, fontsize=16)
                    self.ax.set_ylabel('Counts')
                    self.ax.set_xscale(scale)

                    # Special handling of magnetic field histograms and positions
                    if idx > 2:
                        if ptype == 'B':
                            ticks  = np.around(1e-8*np.logspace(np.log10(self.bmin),np.log10(self.bmax),6),3)
                            labels = []
                            for ival in ticks:
                                base  = ival*10**(-np.floor(np.log10(ival)))
                                power = int(8+np.floor(np.log10(ival)))
                                labels.append(r'$%2.1f\times10^{%d}$' %(base,power))

                            self.ax.set_xticks(1e8*ticks)
                            self.ax.get_xaxis().get_major_formatter().labelOnlyBase = False
                            self.ax.set_xticklabels(labels)
                            self.ax.set_xlim([0.99*self.bmin,1.01*self.bmax])
                        elif ptype == 'z':
                            self.ax.set_xlim([1e-4,1e2])

                    PLT.savefig(fname)
                    PLT.close('all')

                    # Print the median and +/- confidence intervals
                    mids   = 0.5*(edges[:-1]+edges[1:])
                    cdf    = np.cumsum(counts)/np.sum(counts).astype('float')
                    argmed = np.argmin(np.fabs(cdf-0.5))
                    arglo  = np.argmin(np.fabs(cdf-0.16))
                    arghi  = np.argmin(np.fabs(cdf-0.84))
                    if idx == 0:
                        print "Label Median -1sigma +1sigma"
                    print '%s %4.3f %4.3f %4.3f' %(iname,mids[argmed],mids[argmed]-mids[arglo],mids[arghi]-mids[argmed])

                    # Save the median los position of each pulsar to use for 3d simulation
                    if iname[1] == 'z' or iname[2] == 'z':
                        psr_posn.append(sign*mids[argmed])
                np.save('psr_fit_posns.npy',np.array(psr_posn))

def main():
    """ Calls the main body of the code. Also allows for interactive profiling in ipython if called as a function. """

    ### Define Command Line Options
    parser = OptionParser()
    parser.add_option("-f", "--file", action="store", type="string", dest="parfiles", help="Alternate Input File with list of PAR files to examine.")
    parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", default='mcmc_results.npy', help="Base name for the output files.")
    parser.add_option("-l", "--lookup", action="store", type="string", dest="lookup", default='lookup.npy', help="Filename with lookup info for l position normalization.")
    parser.add_option("--hyp2f1", action="store", type="string", dest="hyp2f1", default='hyp2f1.npy', help="Filename with lookup info for hypergeometric function.")
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir", default='/nimrod2/bprager/TER5/PARFILES/', help="Directory to search for parfiles in.")
    parser.add_option("--jerks", action="store_true", dest="jerkflag", default=False, help="Flag to analyze the jerks as well as the accelerations.")
    parser.add_option("--cluster", action='store', type='string', dest="cluster", default='Ter5', help="Globular Cluster name. Default = Ter5")
    parser.add_option("--init", type='string', dest="theta_init", default=np.asarray([9e5,.16,2.4,5e5]), action='callback', callback=optlist, \
                      help="Initial Guesses for cluster params. [Density(Msun/pc^3),rc(pc),alpha,Mtot(Msun)]. Fourth argument only used if jerk flag is turned on.")
    parser.add_option("--dval", action="store", type="float", dest="dval", default=5900, help="Assumed distance to cluster.")
    parser.add_option("--bmin", action="store", type="float", dest="bmin", default=1e7, help="Minimum magnetic field strength to test for in Gauss.")
    parser.add_option("--bmax", action="store", type="float", dest="bmax", default=1e10, help="Maximum magnetic field strength to test for in Gauss.")
    parser.add_option("--nchain", action="store", type="float", dest="nchain", default=200000, help="Number of chains.")
    parser.add_option("--nburn", action="store", type="float", dest="nburn", default=20000, help="Number of burns.")
    parser.add_option("--nthin", action="store", type="float", dest="nthin", default=10, help="Number of thinning steps to hand to PTSampler.")
    parser.add_option("--nglide", action="store", type="float", dest="nglide", default=5, help="Number of steps to glide over in the plot.")
    parser.add_option("--nwalker", action="store", type="float", dest="nwalker", default=128, help="Number of walkers.")
    parser.add_option("--ntemp", action="store", type="float", dest="ntemp", default=16, help="Number of temperatures.")
    parser.add_option("--nthread", action="store", type="float", dest="nthread", default=1, help="Number of threads. (Default=1 due to errors on development system. Try at your own risk.)")
    parser.add_option("--plot", action="store_true", dest="plot", default=False, help="Make a corner plot.")
    parser.add_option("--corner", action="store_true", dest="corner", default=False, help="Make a corner plot.")
    parser.add_option("--chain", action="store_true", dest="chain", default=False, help="Make a chain plot.")
    parser.add_option("--histogram", action="store_true", dest="histogram", default=False, help="Make marginalized distribution plots.")
    parser.add_option("--threedplot", action="store_true", dest="threedplot", default=False, help="Make a file with info for producing 3d plots of results.")
    (options, args) = parser.parse_args()

    ### Define list of par files to fit
    if options.parfiles:
        input = np.loadtxt(options.parfiles,dtype=str)
    else:
        input = glob("%s*par" %(options.dir))

    ### Read in the data ###
    p      = massagedata(input,options.cluster)
    params = p.read_parfiles()

    # Get/Create Model
    p = mcmc_fit(params,options)

    if not os.path.exists(options.outfile):
        p.l_pos_denom_lookup()
        p.call_mcmc()

    if options.plot:
        p.plot_mcmc()

if __name__ == "__main__":
    main()
