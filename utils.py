__author__ = "Hengrui Luo, Justin Strait"
__copyright__ = "© 2023. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so."
__license__ = "BSD-3"
__version__ = "1"
__maintainer__ = "Justin Strait (CCS-6)"
__email__ = "jstrait@lanl.gov"

#########################
## Auxiliary functions ##
#########################
import numpy as np
print('utils VERSION: last updated 2023-11-27')

# This function computes the total length of a closed curve, represented as (x,y) coordinates
# Does so by summing up L2 difference between consecutive points
def xy_length(ordered_sample_points):
    if ordered_sample_points.ndim <=1:
        return 0
    else:
        N = ordered_sample_points.shape[1]
        res = np.zeros((N,1))
        for s in range(1,N):
            res[s] = res[s-1] + np.sqrt( np.sum( np.power( ordered_sample_points[:,s]-ordered_sample_points[:,s-1],2 ),0 ) )
        res = res.reshape(-1,1)
        res = np.max(res)
        
        # Add distance between final point and starting point
        # Note that if these are represented as same point, closing_dist = 0
        closing_dist = np.sqrt( np.sum( np.power( ordered_sample_points[:,N-1]-ordered_sample_points[:,0],2 ),0 ) )
        return res+closing_dist

# This function converts (x,y) coordinates into arc-length parameters with respect to the true curve
# Does so by constructing piecewise linear segments between each pair of sample points in order, and identifying the
# closest point on piecewise linear curve to xy_coords
# The point to curve distance is generally estimated better as Nb increases
def xy_to_arc_param_new(xy_coords, ordered_sample_points, Nb=19):
    def xy_to_arc_param_single_new(xy_coords, ordered_sample_points=ordered_sample_points, Nb=Nb):
        ## Remove replicated final point of curve if closed
        if ( ordered_sample_points[:,0]==ordered_sample_points[:,-1]).all():
            ordered_sample_points = ordered_sample_points[:,:-1]
        N = ordered_sample_points.shape[1]
        
        ## Form "over-sampled" piecewise linear curve
        # Nb = number of points to sample between each pair of sample points for "over-sampled" piecewise linear curve
        opl_curve = np.zeros((2,N*(Nb+1)))
        tb = np.linspace(1/(Nb+1),Nb/(Nb+1),num=Nb)
        
        # Fill in each segment between neighboring sample points with straight line
        for s in range(0,N-1):
            opl_curve[:,s*(Nb+1)] = ordered_sample_points[:,s]  # fill in with sample point
            vecd = ordered_sample_points[:,s+1]-ordered_sample_points[:,s]  # direction to move to next sample point
            ctr = 0
            for t in range(0,Nb):
                opl_curve[:,s*(Nb+1)+ctr+1] = opl_curve[:,s*(Nb+1)]+tb[ctr]*vecd
                ctr = ctr+1
        opl_curve[:,(N-1)*(Nb+1)] = ordered_sample_points[:,N-1]
        
        # Connect final sample point to starting sample point (since curves are closed)
        vecd = ordered_sample_points[:,0]-ordered_sample_points[:,N-1]
        ctr = 0
        for t in range(0,Nb):
            opl_curve[:,(N-1)*(Nb+1)+ctr+1] = opl_curve[:,(N-1)*(Nb+1)]+tb[ctr]*vecd
            ctr = ctr+1
        
        ## Compute distance between xy_coords and each point in "over-sampled" piecewise linear curve
        dist = np.sqrt( np.sum( np.power( xy_coords-np.transpose(opl_curve), 2), 1 ) )

        ## Compute parameter value
        # Arc-length parameter values can be computed only using ordered_sample_points
        res = np.zeros((N,1))
        for s in range(1,N):
            res[s] = res[s-1] + np.sqrt( np.sum( np.power( ordered_sample_points[:,s]-ordered_sample_points[:,s-1],2 ),0 ) )
        res = res.reshape(-1,1)
        
        # Identify point along "over-sampled" piecewise linear curve that is closest to xy_coords
        cl_oversample_dist = np.min(dist)
        cl_oversample_pt = np.argmin(dist)
        
        # Identify immediately preceding sample point to xy_coords along piecewise linear curve and compute parameter value
        prev_sample_pt = int(cl_oversample_pt/(Nb+1))
        return res[prev_sample_pt] + np.sqrt( np.sum( np.power( xy_coords-ordered_sample_points[:,prev_sample_pt],2 ),0 ) )
    if xy_coords.ndim <=1:
        return xy_to_arc_param_single_new(xy_coords)
    else:
        res_list = []
        for k in range(xy_coords.shape[1]):
            res_list.append(xy_to_arc_param_single_new(xy_coords[:,k])[0])
        return res_list
        
# This function convert the arc-length parameters into (x,y) coordinates with respect to the true curve
def arc_to_xy_param_new(arc_param, ordered_sample_points):
    # arc_param must be something of form like [0.1] or [0.1,0.2,0.3]
    def arc_to_xy_param_single_new(arc_param, ordered_sample_points=ordered_sample_points):
        ## Remove replicated final point of curve if closed
        if ( ordered_sample_points[:,0]==ordered_sample_points[:,-1]).all():
            ordered_sample_points = ordered_sample_points[:,:-1]
        N = ordered_sample_points.shape[1]
        
        ## Compute arc-length parameter values based on ordered_sample_points
        # res[N] is the total length of the curve (max parameter value)
        res = np.zeros((N+1,1))
        for s in range(1,N):
            res[s] = res[s-1] + np.sqrt( np.sum( np.power( ordered_sample_points[:,s]-ordered_sample_points[:,s-1],2 ),0 ) )
        res[N] = res[N-1] + np.sqrt( np.sum( np.power( ordered_sample_points[:,0]-ordered_sample_points[:,N-1],2 ),0 ) )
        
        ## Identify sample point with arc-length parameter just preceding arc_param (or equal to)
        dist = arc_param-res
        prev_sample_pt = np.where(np.diff(np.signbit(dist), axis=0))[0][0]  # this gets index preceding or equal to arc_param

        ## Linear interpolation based on arc-length parameter distance to preceding sample point
        prev_dist = np.abs(arc_param-res[prev_sample_pt])
        interval_dist = res[prev_sample_pt+1]-res[prev_sample_pt]
        ratio = prev_dist/interval_dist
        if prev_sample_pt < N-1:
            xy_c = ordered_sample_points[:,prev_sample_pt] + ratio*(ordered_sample_points[:,prev_sample_pt+1] - ordered_sample_points[:,prev_sample_pt])
        else:  # preceding point is the final unique sampled point before wrapping around to "starting point"
            xy_c = ordered_sample_points[:,prev_sample_pt] + ratio*(ordered_sample_points[:,0] - ordered_sample_points[:,prev_sample_pt])
        return xy_c
    
    if len(arc_param) <=1:
        return arc_to_xy_param_single_new(arc_param)
    else:
        res_list = np.zeros((2,len(arc_param)))
        for k in range(0,len(arc_param)):
            res_list[:,k] = arc_to_xy_param_single_new([arc_param[k]])
        return res_list