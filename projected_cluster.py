import os,mpmath
import numpy as np
import pylab as PLT
from glob import glob
import logprob_integrator as LI
from parfile_units import psr_par
from optparse import OptionParser
from presto import sphere_ang_diff
from astropy import units,constants
from scipy.special import hyp2f1 as HYP2F1
from psr_utils import ra_to_rad,dec_to_rad

def optlist(option, opt, value, parser):
    """
    Split up option parser values into a list of floats.
    """
    setattr(parser.values, option.dest, np.asarray([float(x) for x in value.split(',')]))

def optlist_int(option, opt, value, parser):
    """
    Split up option parser values into a list of integers.
    """
    setattr(parser.values, option.dest, np.asarray([int(x) for x in value.split(',')]))


class massagedata:

    def __init__(self,files,cluster):
        """ Define a few things for prepping the data. """
	self.files            = files                        # List of par files
	self.cluster          = cluster                      # Which cluster are we working with
	self.KMPERKPC         = constants.pc.to(units.km)
        self.SECPERJULYR      = (1.*units.year).to(units.s)
        self.tauc             = 10000000000.0*units.year     # Characteristic Age
        self.r_c_offset_error = 0.0*units.arcsec             # Offset error in arcsec
	self.c                = constants.c

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
        cluster = {"Ter5": ["Terzan5", "17:48:04.85", "-24:46:44.6", 5.9, 7.9/60.0, 0.83, 13.27, 7.60, 8.97, 20.33, 5.06, 3.8, 1.7, 17.0, 21.72, 5.0], \
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
            psr.abyc_gal         = (-np.cos(psr.b)*((Omega0**2.)/R0)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/constants.c
            psr.abyc_gal_error   = np.sqrt((((-np.cos(psr.b)*((Omega0**2.)/R0))/constants.c)*((np.sin(psr.l)**2.-beta**2.)/(np.sin(psr.l)**2.+beta**2.)**2.)*beta_error)**2.
                                  +((-np.cos(psr.b)*((Omega0_error*Omega0)/R0)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/constants.c)**2.
                                  +((-np.cos(psr.b)*((R0_error*Omega0**2.)/R0**2.)*(np.cos(psr.l)+beta/(np.sin(psr.l)**2.+beta**2.)))/constants.c)**2.)
            psr.abyc_meas        = psr.P1/psr.P0
            psr.abyc_meas_error  = np.sqrt((psr.P1_ERR/psr.P0)**2.+(psr.P0_ERR*psr.P1/psr.P0**2.)**2.)
            psr.abyc_clust       = psr.abyc_meas - psr.abyc_gal
            psr.abyc_clust_error = np.sqrt(psr.abyc_meas_error**2.+psr.abyc_gal_error**2.)

            ### Store important attributes to params
            params.R.append((psr.core_ang_asec)*psr.D*1000/206265)
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

class make_models:

    def __init__(self,params,options):
        self.params   = params
        self.options  = options
	self.filename = "IntegratedKing_%4.2f_%4.2f_%d_%4.2f_%4.2f_%d_fit" %(options.mns[0],options.mxs[0],options.nbins[0],options.mns[1],options.mxs[1],options.nbins[1])

    def bfield_pdot_rmv(self):
	"""
	Remove intrinsic spin down as much as possible using known MSP Bfields to estimate Pdot.
	"""

        # Important values from ATNF
	B_atnf_mean         = 2.35e8*units.G                 # Obtained using psr catalog and finding mean B field of MSPs
        B_atnf_mean_slow    = 2.35e10*units.G                # Obtained using psr catalog and finding mean B field of slow MSPs (30 ms - 100ms)
	B_atnf_error        = 1.25e8*units.G                 # Approximately 1 sigma spread on mean B field
	B_atnf_error_slow   = 5e9*units.G                    # Approximately 1 sigma spread on mean B field of slow MSPs (30 ms - 100ms)
        Bfield_nomalization = 6.14e19*units.G                # Normalization Factor used to convert Bfield to Pdot

        # Make the magnetic fields for each pulsar
        ind_slow                   = (self.params.P0.value>30)
        B_estimate                 = B_atnf_mean*np.ones(self.params.R.size)
        B_error_estimate           = B_atnf_error*np.ones(self.params.R.size)
        B_estimate[ind_slow]       = B_atnf_mean_slow*np.ones(self.params.R[ind_slow].size)
        B_error_estimate[ind_slow] = B_atnf_error_slow*np.ones(self.params.R[ind_slow].size)

        ### Calculate the intrinsic spindown
        Pdint_coeff   = 1.0*units.s
	Pdint         = (Pdint_coeff/self.params.P0)*(B_estimate/Bfield_nomalization)**2
	Pdint_err     = ((2*Pdint_coeff*B_estimate*B_error_estimate)/(self.params.P0*Bfield_nomalization**2))**2
	Pdint_err    += ((Pdint_coeff*self.params.P0_ERR/self.params.P0**2.)*(B_estimate/Bfield_nomalization)**2)**2.
	Pdint_err     = np.sqrt(Pdint_err)
	accel_int     = (Pdint/self.params.P0)*constants.c
	accel_int_err = np.sqrt((Pdint_err/self.params.P0)**2.+(self.params.P0_ERR*Pdint/self.params.P0**2.)**2.)*constants.c

        # Get the cluster acceleration
	self.params.ACCEL_CLUSTER                                       = self.params.ACCEL - accel_int
	self.params.ACCEL_CLUSTER_ERR                                   = np.sqrt(self.params.ACCEL_ERR**2+accel_int_err**2)
	self.params.ACCEL_CLUSTER[(params.ACCEL_BINARY.value != 0)]     = self.params.ACCEL_BINARY[(params.ACCEL_BINARY.value != 0)]
	self.params.ACCEL_CLUSTER_ERR[(params.ACCEL_BINARY.value != 0)] = self.params.ACCEL_BINARY_ERR[(params.ACCEL_BINARY.value != 0)]

    def model_params(self,flatten=False):
        """
        Create coordinate grid for given model.
        """
        xgrid        = np.linspace(self.options.mns[0],self.options.mxs[0],self.options.nbins[0])
        ygrid        = np.linspace(self.options.mns[1],self.options.mxs[1],self.options.nbins[1])
	xgrid_error  = xgrid[1]-xgrid[0]
	ygrid_error  = ygrid[1]-ygrid[0]

        if not flatten:
            xgrid, ygrid = np.meshgrid(xgrid, ygrid)

        return xgrid,ygrid,xgrid_error,ygrid_error

    def model_eqn(self,model_input,model_input_error,R):
        """
        Return the acceleration profile
        """

        def accel(r,rc):
            return ((4/3.)*np.pi*constants.G.value)*model_input[1]*rc*HYP2F1(1.5,0.5*self.options.alpha,2.5,-(r/rc)**2)

        rc      = model_input[0]*constants.pc.value
        r       = np.sqrt(model_input[0]**2+R**2)*constants.pc.value
        profile = accel(r,rc)

        return profile

    def fit_model(self,save_flag=True):
        """
        Fit the data to a given model using the CDF for the cluster, pulsar position, and the cluster model.
        """

        if not os.path.exists('%s.npy' %(self.filename)):
            # Get the data
    	    X    = self.params.R.value
            Y    = np.fabs(self.params.ACCEL_CLUSTER.value)
            YERR = self.params.ACCEL_CLUSTER_ERR.value

            # Make the output arrays
            best                = np.zeros((self.options.nbins[1],self.options.nbins[0]))
            profile_offset_mask = np.zeros((self.options.nbins[1],self.options.nbins[0]))
    
            # Make the grid of model points
            xgrid,ygrid,xgrid_error,ygrid_error = self.model_params()
    
            # Loop over each pulsar
            for ii in range(len(X)):
                YARR    = np.fabs(Y[ii])*np.ones((self.options.pbin,self.options.nbins[1],self.options.nbins[0]))
                YERRARR = np.fabs(YERR[ii])*np.ones((self.options.pbin,self.options.nbins[1],self.options.nbins[0]))
                profile = self.model_eqn([xgrid,ygrid],[xgrid_error,ygrid_error],X[ii])
                PROB_Y  = np.linspace(Y[ii]-3.0*YERR[ii],Y[ii]+3.0*YERR[ii],self.options.pbin).reshape((self.options.pbin,1,1))
    
                # Mark bad models in the mask
                bad_inds                      = (profile<(Y[ii]-2.0*YERR[ii]))
                profile_offset_mask[bad_inds] = 1
    
                # Calculate the log likelihoods
                profile  = (np.asarray([profile,]*self.options.pbin))
                a        = (PROB_Y/profile)
                P_phin   = (a**(self.options.alpha-2)/np.sqrt(1-a**2.))*((1.-np.sqrt(1.-a**2.))**(1-0.5*self.options.alpha)+(1.+np.sqrt(1.-a**2.))**(1-0.5*self.options.alpha))/profile
                P_phin   = np.nan_to_num(P_phin)
                P_meas   = (1.0/(YERRARR*np.sqrt(2*np.pi)))*np.exp(-(PROB_Y-YARR)**2/(2*YERRARR**2))
                prob     = P_phin*P_meas
                prob     = np.trapz(prob,axis=0)
                prob[(prob<1e-322)]= 1e-322
                best    += np.log(prob)
    
            if save_flag:
                np.save("%s" %(self.filename),best)
    	    np.save("%s_mask" %(self.filename),profile_offset_mask)

    def best_solution(self,in_array):
        xgrid,ygrid,xgrid_error,ygrid_error = self.model_params(flatten=True)
        ygrid                               = 1e-6*(ygrid*units.kg/units.m**3).to(units.solMass/units.pc**3).value
        self.yid,self.xid                   = np.unravel_index(in_array.argmax(), in_array.shape)
	self.rc                             = xgrid[self.xid]
	self.rho_c                          = ygrid[self.yid]

    def label(self,fnames,xs,ys,sz=16,yoff=1.01,xoff=1.01,axis=None):
        """
        Label a plot with the pulsar names.
        """

        for label, x, y in zip(fnames, xs, ys):
            label = label.strip("1748-2446")
            if axis == None:
                PLT.annotate(label, xy = (xoff*x, yoff*y),size=sz)
            else:
                axis.annotate(label, xy = (xoff*x, yoff*y),size=sz)

    def plot_loglike(self):
        """
        Plot model with certain options.
        """

        # Load some of the needed values
        loggrid                             = np.load('%s.npy' %(self.filename))
        logmask                             = np.load('%s_mask.npy' %(self.filename))
        loggrid_masked                      = np.ma.masked_array(loggrid,mask=logmask)
        self.best_solution(loggrid_masked)
        xgrid,ygrid,xgrid_error,ygrid_error = self.model_params(flatten=True)
        ygrid                               = 1e-6*(ygrid*units.kg/units.m**3).to(units.solMass/units.pc**3).value

        # Get the probabilities back out to calculate the confidence intervals
        problist = []
        for irow in range(loggrid.shape[0]):
            row_list = []
            for icol in range(loggrid.shape[1]):
                val = mpmath.exp(loggrid[irow,icol])
                row_list.append(val)
            problist.append(row_list)
        probs = np.array(problist)

        # Calculate the probabilities
        probs_masked             = np.ma.masked_array(probs,mask=logmask,fill_value=0.0)
        probs_masked_filled      = probs_masked.filled()
        probs_masked_filled     /= np.sum(probs_masked_filled)
        probs_masked_filled      = probs_masked_filled.astype('float')

        # Calculate the sigma values using circles around max
        C = LI.circle_sums(probs_masked_filled,self.xid,self.yid)
        one_sigma_mask,one_sigma_radius   = C.sum_interior_radii(.68)
        two_sigma_mask,two_sigma_radius   = C.sum_interior_radii(.95)
        three_sigma_mask,two_sigma_radius = C.sum_interior_radii(.99)

        # Get the minimum non zero entry from each mask
        one_sigma_min   = np.amin(one_sigma_mask[(one_sigma_mask>0.0)])
        two_sigma_min   = np.amin(one_sigma_mask[(two_sigma_mask>0.0)])
        three_sigma_min = np.amin(one_sigma_mask[(three_sigma_mask>0.0)])

        # Plot the solutions
        dpi                 = 100
        fig                 = PLT.figure(figsize=(800./dpi,800./dpi),dpi=dpi)
        ax                  = fig.add_subplot(111)
        ll                  = ax.imshow(probs_masked_filled,origin='lower',aspect='auto',cmap='binary', extent=(xgrid[0],xgrid[-1],ygrid[0],ygrid[-1]))
        one_sigma_contour   = ax.contour(one_sigma_mask,[one_sigma_min],origin='lower',aspect='auto',extent=(xgrid[0],xgrid[-1],ygrid[0],ygrid[-1]),colors='k')
        two_sigma_contour   = ax.contour(two_sigma_mask,[two_sigma_min],origin='lower',aspect='auto',extent=(xgrid[0],xgrid[-1],ygrid[0],ygrid[-1]),colors='k')
        three_sigma_contour = ax.contour(three_sigma_mask,[three_sigma_min],origin='lower',aspect='auto',extent=(xgrid[0],xgrid[-1],ygrid[0],ygrid[-1]),colors='k')

        # Get the one sigma errorbars
        try:
            vertices         = one_sigma_contour.collections[0].get_paths()[0].vertices
            self.rc_error    = (np.amax(vertices[:,0])-np.amin(vertices[:,0]))/2.
            self.rho_c_error = (np.amax(vertices[:,1])-np.amin(vertices[:,1]))/2.
        except IndexError:
            PLT.show()
            exit()

        # Best fit label
        alabel = r'$\rho_c$ = %3.2f $\pm$ %3.2f $\times$ 10$^6$ M$_\odot$ pc$^{-3}$' '\n' r'r$_{c}$ = %3.2f $\pm$ %3.2f pc' %(self.rho_c, self.rho_c_error, self.rc, self.rc_error)
        ax.annotate(alabel, (.05, .15), xycoords="axes fraction", va="top", ha="left", bbox=dict(boxstyle="square, pad=1", fc="w"),fontsize=13)

        ax.set_ylabel(r'$\rho_c$ [10$^6$ M$_\odot$ pc$^{-3}$]',fontsize=18)
        ax.set_xlabel(r'r$_{c}$ [pc]', fontsize=18)
        fig.tight_layout()

        # Add label to 1 sigma contour
        fmt  = {}
        strs = [r'1$\sigma$']
        for l,s in zip(one_sigma_contour.levels, strs):
            fmt[l] = s
        pos = [(.15,1.6)]
        PLT.clabel(one_sigma_contour,one_sigma_contour.levels[::1],inline=1,fmt=fmt,fontsize=16,manual=pos)

        # Add label to 2 sigma contour
        fmt  = {}
        strs = [r'2$\sigma$']
        for l,s in zip(two_sigma_contour.levels, strs):
            fmt[l] = s
        pos = [(.15,1.8)]
        PLT.clabel(two_sigma_contour,two_sigma_contour.levels[::1],inline=1,fmt=fmt,fontsize=16,manual=pos)

        # Add label to 3 sigma contour
        fmt  = {}
        strs = [r'3$\sigma$']
        for l,s in zip(three_sigma_contour.levels, strs):
            fmt[l] = s
        pos = [(.15,2.)]
        PLT.clabel(three_sigma_contour,three_sigma_contour.levels[::1],inline=1,fmt=fmt,fontsize=16,manual=pos)

        PLT.savefig('projected_loglike.png')
        PLT.close('all')

    def plot_accels(self):
        """ Plot the acceleration profile. """

        # Get the best solution
        loggrid                             = np.load('%s.npy' %(self.filename))
        logmask                             = np.load('%s_mask.npy' %(self.filename))
        loggrid_masked                      = np.ma.masked_array(loggrid,mask=logmask)
        self.best_solution(loggrid_masked)
        best_fit                            = [self.rc,(1e6*self.rho_c*units.solMass/units.pc**3).decompose().value]
	best_fit_error                      = [self.rc_error,(1e6*self.rho_c_error*units.solMass/units.pc**3).decompose().value]

        # Start plotting environment
        dpi = 100
        fig = PLT.figure(figsize=(800./dpi,800./dpi),dpi=dpi)
        ax  = fig.add_subplot(111)

        ### Acceleration Profile
        smoothX = np.linspace(self.params.R[0].value,self.params.R[-1].value,1000)
        profile = self.model_eqn(best_fit,best_fit_error,smoothX)*1e9

        # Plot the acceleration profile
        prof, = PLT.plot(smoothX,profile,color='k')
        prof, = PLT.plot(smoothX,-profile,color='k')

	# Plot Pulsars without PBDOT
        inds = (params.ACCEL_BINARY.value == 0)
        x    = self.params.R[inds].value
        y    = self.params.ACCEL_CLUSTER[inds].value*1e9
        yerr = self.params.ACCEL_CLUSTER_ERR[inds].value*1e9
	iso  = ax.errorbar(x,y,yerr=yerr,fmt='bo',ms=6)

	# Plot Pulsars with PBDOT
        inds   = (params.ACCEL_BINARY.value != 0)
        x      = self.params.R[inds].value
        y      = self.params.ACCEL_CLUSTER[inds].value*1e9
        yerr   = self.params.ACCEL_CLUSTER_ERR[inds].value*1e9
	binary = ax.errorbar(x,y,yerr=yerr,fmt='go',ms=6)

	# Plot Pulsar measured accel
        x    = self.params.R.value
        y    = self.params.ACCEL.value*1e9
	meas = ax.scatter(x,y,marker='v',color='r')
	self.label(self.params.PSRNAME,x,y,axis=ax)

        # Add a top axis to show as a function of core radius
        ax2 = ax.twiny()
	ax2.plot(smoothX/self.rc,profile,color='k',alpha=0.0)
        ax2.set_xlabel(r'r$_{\perp}/r_c$', fontsize=18)

        # Legends and labels
	ax.legend([prof,meas,binary,iso],['Best Fit Profile','Original Measurement','Measurement from Binary orbit',r'Measurements after $\dot{P}$ removal.'],loc=4,prop={'size':15})
	alabel = r'r$_{c}$    = %3.2f $\pm$ %3.2f pc' '\n' r'$\rho_{c}$    = %3.2f $\pm$ %3.2f [10$^6$ M$_\odot$ pc$^{-3}$]' %(self.rc,self.rc_error,self.rho_c,self.rho_c_error)
	ax.annotate(alabel, (.45, .925), xycoords="axes fraction", va="top", ha="left", bbox=dict(boxstyle="square, pad=1", fc="w"),fontsize=13)

        # Make the plot pretty
	ax.set_xlabel(r'r$_{\perp}$ [arcmin]', fontsize=18)
	ax.set_ylabel(r'a [10$^{-9}$ m s$^{-2}$]',fontsize=18)
        ax.set_ylim([-1.25*np.amax(profile),1.25*np.amax(profile)])
        ax.set_xlim([0.*np.amin(self.params.R).value,1.025*np.amax(self.params.R).value])

        fig.tight_layout()
        PLT.savefig('projected_accel_profile.png')

if __name__ == "__main__":

    ### Define Command Line Options
    parser = OptionParser()
    parser.add_option("-f", "--file", action="store", type="string", dest="parfiles", help="Alternate Input File with list of PAR files to examine.")
    parser.add_option("-L", "--LogLike", action="store_true", dest="logcalc", default=False, help="Fit for a King Model with Log Likelihood Analysis.")
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir", default='/nimrod2/bprager/TER5/PARFILES/', help="Directory to search for parfiles inp.")
    parser.add_option("--alpha", type='float', dest="alpha", default=2.5, action='store', help="Pulsar spectral index for density (alpha).")
    parser.add_option("--nbins", type='string', dest="nbins", default=np.asarray([125,125]), action='callback', callback=optlist_int, help="List of number of bins in each axes.")
    parser.add_option("--mns", type='string', dest="mns", default=np.asarray([.27,.35]), action='callback', callback=optlist, help="List of minimum value along each axes.")
    parser.add_option("--mxs", type='string', dest="mxs", default=np.asarray([.37,0.8]), action='callback', callback=optlist, help="List of maximum value along each axes.")
    parser.add_option("--pbin", type='float', dest="pbin", default=750, action='store', help="List of maximum value along each axes.")
    parser.add_option("--cluster", action='store', type='string', dest="cluster", default='Ter5', help="Globular Cluster name. Default = Ter5")
    (options, args) = parser.parse_args()

    ### Define list of par files to fit
    if options.parfiles:
        input = np.loadtxt(options.parfiles,dtype=str)
    else:
        input = glob("%s*par" %(options.dir))

    # Convert central density into mks
    options.mns[1] = 1e6*(options.mns[1]*units.solMass/units.pc**3).decompose().value
    options.mxs[1] = 1e6*(options.mxs[1]*units.solMass/units.pc**3).decompose().value

    ### Read in the data ###
    p      = massagedata(input,options.cluster)
    params = p.read_parfiles()

    ### Initialize the model making class
    m = make_models(params,options)
    m.bfield_pdot_rmv()

    ### Fit the model
    m.fit_model()

    # Plot values
    m.plot_loglike()
    m.plot_accels()
