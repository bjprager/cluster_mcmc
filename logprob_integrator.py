import numpy as N
import scipy as SP
import pylab as PLT
from matplotlib.path import Path
from scipy.optimize import minimize_scalar,curve_fit

def integrate_contour_area(data, lvl, maskflag=False):
    """ Turns a set of contours into a series of vertices. You can then get the integrated area inside. """

    # Create a contour at the given level
    contours = PLT.contour(data,N.array([lvl]))
    PLT.close('all')

    # Get the needed path
    try:
        vertices = contours.collections[0].get_paths()[0].vertices
    except IndexError:
        return 0.0

    # Find the x and y extent
    xmin = N.amin(vertices[:,0])
    xmax = N.amax(vertices[:,0])
    ymin = N.amin(vertices[:,1])
    ymax = N.amax(vertices[:,1])

    # Because a path connects end point to start point, and we want the entire area inside, add origin
    # This is not correct for all setups. Fix later
    if vertices[0,1] == 0.0 and vertices[-1,0] == 0.0:
        vertices=N.vstack((N.array([0,0]),vertices))
        vertices=N.vstack((vertices,N.array([0,0])))
    path     = Path(vertices, closed=True)

    # Make an array of points to compare to the path
    ny,nx        = data.shape
    ygrid, xgrid = SP.mgrid[:ny, :nx]
    xypix        = SP.vstack((xgrid.ravel(), ygrid.ravel())).T

    # Make the mask
    mask = N.array([])
    for ipix in xypix:
        mask = N.append(mask,path.contains_point(ipix))
    mask = mask.reshape(data.shape)

    # Get the masked data
    masked_data     = mask*data
    masked_data_sum = N.sum(masked_data)

    if maskflag:
        return mask,xmin,xmax,ymin,ymax
    else:
        return masked_data_sum

def fit_contour_level_minimizer(lvl, data, user_required_level):
    """ To be used with a minimizer to get out the best level."""
    return N.fabs(integrate_contour_area(data, lvl)-user_required_level)

def fit_contour_level(data, user_required_level):
    """ Fit for a user inputted level for the sum inside a contour. Good for finding sigma values. """

    bound_min    = N.amin(data[(data>0.0)])
    bound_max    = N.amax(data)
    dbound       = N.fabs(bound_max-bound_min)
    bound_center = (bound_max+bound_min)/2.

    ### Brute force approach
    num_shrink = 5
    num_soln   = 20
    print "Solving for the %3.2f contour level sum." %(user_required_level)
    for i in range(num_shrink):
        print "Iteration:",i+1

        lhs     = max([bound_min,bound_center-dbound/(2.0*(i+1))])
        rhs     = min([bound_max,bound_center+dbound/(2.0*(i+1))])
        inputs  = N.linspace(lhs,rhs,num_soln)
        outputs = N.zeros(num_soln)
        for j in range(num_soln):
            outputs[j] = fit_contour_level_minimizer(inputs[j],data,user_required_level)
        idx          = N.argmin(outputs)
        bound_center = inputs[idx]
    tmp = integrate_contour_area(data,bound_center)

    print "User required sum:",user_required_level
    print "Actual sum:",tmp
    print "Level at:",bound_center
    print "~~~~~~~~~~~~~~~"
    return bound_center

class circle_sums:

    def __init__(self,data,xid,yid):
        self.data   = data
        yy,xx       = N.mgrid[:data.shape[0],:data.shape[1]]
        self.circle = N.sqrt((xx-xid)**2.+(yy-yid)**2.)

    def sum_interior_radii_minimizer(self,x,r):
        circle_mask        = (self.circle>=r)
        prob_circle_masked = N.ma.masked_array(self.data,mask=circle_mask,fill_value=0.0).filled()
        return N.fabs(self.level-N.sum(prob_circle_masked))

    def sum_interior_radii_bruteforce(self):
        """ Brute force solver because curve_fit is a finnicky b***h. """
        n_trial = N.sqrt(self.data.shape[0]**2.+self.data.shape[1]**2.).astype('int')
        output  = N.zeros(n_trial)

        for i_trial in range(n_trial):
            iradius            = i_trial+1
            circle_mask        = (self.circle>=iradius)
            prob_circle_masked = N.ma.masked_array(self.data,mask=circle_mask,fill_value=0.0).filled()
            output[i_trial]    = N.fabs(self.level-N.sum(prob_circle_masked))
        return N.argmin(output)+1

    def sum_interior_radii_best(self,r):
        circle_mask        = (self.circle>=r)
        prob_circle_masked = N.ma.masked_array(self.data,mask=circle_mask,fill_value=0.0).filled()
        return prob_circle_masked,N.sum(prob_circle_masked)

    def sum_interior_radii(self,user_lvl):
        """ Add up to consecutively larger radii until the interior sum equals the user level. """
        self.level       = user_lvl
        r                = self.sum_interior_radii_bruteforce()
        masked_data,psum = self.sum_interior_radii_best(r)
        print "Radius of %d pixels gives a summed interior of %4.3f" %(r,psum)

        return N.ma.masked_array(masked_data,mask=(masked_data==0.0),fill_value=0.0).filled(),r
