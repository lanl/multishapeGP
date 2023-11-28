__author__ = "Hengrui Luo, Justin Strait"
__copyright__ = "© 2023. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so."
__license__ = "BSD-3"
__version__ = "1"
__maintainer__ = "Justin Strait (CCS-6)"
__email__ = "jstrait@lanl.gov"

##################################
## GP modeling of closed curves ##
##################################
#import pkg_resources

# Comment out?
#pkg_resources.require("numpy==1.21.2")
#pkg_resources.require("fdasrsf==2.3.4")
#pkg_resources.require("matplotlib==3.5.1")

from utils import *
import numpy as np
import copy
import gpflow
from gpflow import set_trainable
from gpflow.kernels import Coregion, SquaredExponential, Matern12, Matern32, Matern52, ArcCosine
from gpflow.utilities import print_summary
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import fdasrsf as fs
from fdasrsf.curve_stats import fdacurve
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import collections
#from kernelClass import *

print('curveGP VERSION: last updated 2023-11-27')


## Function to bound length scale kernel hyperparameter
# Source: https://stackoverflow.com/questions/59504125/bounding-hyperparameter-optimization-with-tensorflow-bijector-chain-in-gpflow-2
def bounded_lengthscale(low, high, lengthscale):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # low, high = lower, upper bound (respectively) to restrict length scale parameter in kernel 
    # lengthscale = initial
    
    ## Output ##
    # parameter = GPflow parameter object
    # when calling a kernel's length scale parameter, can set equal to output of this function to constrain that parameter
    sigmoid = tfb.Sigmoid(low, high)
    parameter = gpflow.Parameter(lengthscale, transform=sigmoid, dtype='double')
    return parameter

## Different function to bound parameter (for bounding noise variance)
def bounded_parameter(low, high, default):
    """Make lengthscale tfp Parameter within optimization bounds."""
    affine_scale = tfb.Scale(scale=tf.cast(high-low, tf.float64))
    affine_shift = tfb.Shift(shift=tf.cast(low, tf.float64))
    sigmoid = tfb.Sigmoid()
    logistic = tfb.Chain([affine_shift, affine_scale, sigmoid])
    parameter = gpflow.Parameter(default, transform=logistic, dtype=tf.float64)
    return parameter

## Modified versions of code from fdasrsf module for registration
def find_rotation_and_seed_unique_warp(q1, q2, closed=1, rotation=True, method='DP'):
    n, T = q1.shape
    scl = 1
    minE = 1000
    if closed==1:
        end_idx = int(np.floor(T/scl))
    else:
        end_idx = 0
    
    for ctr in range(0,end_idx+1):
        if closed==1:
            q2n = cf.shift_f(q2,scl*ctr)
        else:
            q2n = q2.copy()
        
        if rotation:
            q2new, R = cf.find_best_rotation(q1,q2n)
        else:
            q2new = q2n
            R = np.eye(n)

        # Reparam
        if np.linalg.norm(q1-q2new,'fro') > 0.0001:
            gam = cf.optimum_reparam_curve(q2new, q1, 0.0, method)
            gamI = uf.invertGamma(gam)
            p2n = cf.q_to_curve(q2n)
            p2n = cf.group_action_by_gamma_coord(p2n,gamI)
            q2new = cf.curve_to_q(p2n)[0]
            if closed==1:
                q2new = cf.project_curve(q2new)
        else:
            gamI = np.linspace(0,1,T)
        
        tmp = cf.innerprod_q2(q1,q2new)
        if tmp>1:
            tmp = 1
        if tmp<-1:
            tmp = -1
        Ec = np.arccos(tmp)
        if Ec<minE:
            Rbest = R
            q2best = q2new
            gamIbest = gamI
            ctrbest = scl*ctr
            minE = Ec

    return (q2best, Rbest, gamIbest, ctrbest)

def find_rotation_and_seed_unique_warpv2(q1, q2, closed=1, rotation=True, method='DP'):
    n, T = q1.shape
    scl = 1
    minE = 1000
    if closed==1:
        end_idx = int(np.floor(T/scl))
    else:
        end_idx = 0
    
    for ctr in range(0,end_idx+1):
        if closed==1:
            q2n = cf.shift_f(q2,scl*ctr)
        else:
            q2n = q2.copy()
        
        if rotation:
            q2new, R = cf.find_best_rotation(q1,q2n)
        else:
            q2new = q2n
            R = np.eye(n)
        
        tmp = cf.innerprod_q2(q1,q2new)
        if tmp>1:
            tmp = 1
        if tmp<-1:
            tmp = -1
        Ec = np.arccos(tmp)
        if Ec<minE:
            Rbest = R
            q2best = q2new
            ctrbest = scl*ctr
            minE = Ec

    # Reparam
    if np.linalg.norm(q1-q2best,'fro') > 0.0001:
        gam = cf.optimum_reparam_curve(q2best, q1, 0.0, method)
        gamIbest = uf.invertGamma(gam)
        p2best = cf.q_to_curve(q2best)
        p2best = cf.group_action_by_gamma_coord(p2best,gamIbest)
        q2best = cf.curve_to_q(p2best)[0]
        if closed==1:
            q2best = cf.project_curve(q2best)
    else:
        gamIbest = np.linspace(0,1,T)

    return (q2best, Rbest, gamIbest, ctrbest)

def find_rotation_and_seed_unique_nowarp(q1, q2, closed=1, rotation=True):
    n, T = q1.shape
    scl = 1
    minE = 1000
    if closed==1:
        end_idx = int(np.floor(T/scl))
    else:
        end_idx = 0
    
    for ctr in range(0,end_idx+1):
        if closed==1:
            q2n = cf.shift_f(q2,scl*ctr)
        else:
            q2n = q2.copy()
        
        if rotation:
            q2new, R = cf.find_best_rotation(q1,q2n)
        else:
            q2new = q2n
            R = np.eye(n)
        
        tmp = cf.innerprod_q2(q1,q2new)
        if tmp>1:
            tmp = 1
        if tmp<-1:
            tmp = -1
        Ec = np.arccos(tmp)
        if Ec<minE:
            Rbest = R
            q2best = q2new
            ctrbest = scl*ctr
            minE = Ec

    return (q2best, Rbest, ctrbest)

class fdacurve_v2:
    def __init__(self, beta, mode='C', N=200, scale=False):
        self.mode = mode
        self.scale = scale

        K = beta.shape[2]
        n = beta.shape[0]
        q = np.zeros((n,N,K))
        beta1 = np.zeros((n,N,K))
        cent1 = np.zeros((n,K))
        len1 = np.zeros(K)
        lenq1 = np.zeros(K)
        for ii in range(0,K):
            if False:#beta.shape[1]!=N:
                beta1[:,:,ii] = cf.resamplecurve(beta[:,:,ii], N, mode=mode)
            else:
                beta1[:,:,ii] = beta[:,:,ii]
            a = -cf.calculatecentroid(beta1[:,:,ii])
            #print('a',a)
            a = a*0.
            beta1[:,:,ii] += np.tile(a,(N,1)).T
            q[:,:,ii], len1[ii], lenq1[ii] = cf.curve_to_q(beta1[:,:,ii], mode)
            cent1[:,ii] = -a

        self.q = q
        self.beta = beta1
        self.cent = cent1
        self.len = len1
        self.len_q = lenq1
        
    def karcher_mean(self, rotation=True, parallel=False, cores=-1, method="DP"):
        n, T, N = self.beta.shape

        modes = ['O','C']
        mode = [i for i, x in enumerate(modes) if x==self.mode]
        if len(mode)==0:
            mode = 0
        else:
            mode = mode[0]

        # Initialize mu as one of the shapes
        mu = self.q[:,:,0]
        betamean = self.beta[:,:,0]
        itr = 0

        gamma = np.zeros((T,N))
        maxit = 20

        sumd = np.zeros(maxit+1)
        v = np.zeros((n,T,N))
        sumv = np.zeros((n,T))
        normvbar = np.zeros(maxit+1)

        delta = 0.5
        tolv = 1e-4
        told = 5*1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr<maxit:
            print("updating step: %d" % (itr+1))

            if iter==maxit:
                print("maximal number of iterations reached")

            mu = mu/np.sqrt(cf.innerprod_q2(mu,mu))
            if mode==1:
                self.basis = cf.find_basis_normal(mu)
            else:
                self.basis = []

            sumv = np.zeros((n,T))
            sumd[0] = np.inf
            sumd[itr+1] = 0
            out = Parallel(n_jobs=cores)(delayed(karcher_calc)(mu, self.q[:,:,n], self.basis, 
                                                               mode, rotation, method) for n in range(N))
            v = np.zeros((n,T,N))
            gamma = np.zeros((T,N))
            for i in range(0, N):
                v[:,:,i] = out[i][0]
                gamma[:,i] = out[i][1]
                sumv += v[:,:,i]
                sumd[itr+1] = sumd[itr+1]+out[i][2]**2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv/float(N)

            normvbar[itr] = np.sqrt(cf.innerprod_q2(vbar,vbar))
            normv = normvbar[itr]

            if (sumd[itr]-sumd[itr+1])<0:
                break
            elif (normv>tolv and np.fabs(sumd[itr+1]-sumd[itr])>told):
                # Update mu in direction of vbar
                mu = np.cos(delta*normvbar[itr])*mu + np.sin(delta*normvbar[itr])*vbar/normvbar[itr]

                if mode==1:
                    mu = cf.project_curve(mu)

                x = cf.q_to_curve(mu)
                a = -1*cf.calculatecentroid(x)
                betamean = x + np.tile(a,[T, 1]).T
            else:
                break

            itr += 1

        # compute average length
        if self.scale:
            self.mean_scale = (np.prod(self.len))**(1./self.len.shape[0])
            self.mean_scale_q = (np.prod(self.len_q))**(1./self.len.shape[0])
            betamean = self.mean_scale*betamean
        
        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0:(itr+1)]
        self.E = normvbar[0:(itr+1)]

        return

    def srvf_align(self, rotation=True, parallel=False, cores=-1, method="DP"):
        n, T, N = self.beta.shape

        modes = ['O','C']
        mode = [i for i, x in enumerate(modes) if x==self.mode]
        if len(mode)==0:
            mode = 0
        else:
            mode = mode[0]

        # find mean
        if not hasattr(self,'beta_mean'):
            self.karcher_mean()

        self.qn = np.zeros((n,T,N))
        self.betan = np.zeros((n,T,N))
        self.O = np.zeros((n,n,N))
        self.gams = np.zeros((T,N))
        self.ctr = np.zeros((N,1))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - np.tile(centroid2,[T, 1]).T

        # align to mean
        out = Parallel(n_jobs=-1)(delayed(find_rotation_and_seed_unique_warp)(self.q_mean,
                                           self.q[:,:,n], mode, rotation, method) for n in range(N))
        for ii in range(0,N):
            self.gams[:,ii] = out[ii][2]
            self.qn[:,:,ii] = out[ii][0]
            self.O[:,:,ii] = out[ii][1]
            self.ctr[ii] = int(out[ii][3])
            tmp = cf.shift_f(self.beta[:,:,ii], out[ii][3])
            btmp = out[ii][1].dot(tmp)
            self.betan[:,:,ii] = cf.group_action_by_gamma_coord(btmp,out[ii][2])        

        return
    
def karcher_calc(mu, q, basis, closed, rotation, method):
    # Compute shooting vector from mu to q_i
    qn_t, R, gamI, ctr = find_rotation_and_seed_unique_warp(mu, q, closed, rotation, method)
    qn_t = qn_t/np.sqrt(cf.innerprod_q2(qn_t,qn_t))

    q1dotq2 = cf.innerprod_q2(mu,qn_t)

    if q1dotq2>1:
        q1dotq2 = 1

    d = np.arccos(q1dotq2)

    u = qn_t - q1dotq2*mu
    normu = np.sqrt(cf.innerprod_q2(u,u))
    if normu>1e-4:
        w = u*np.arccos(q1dotq2)/normu
    else:
        w = np.zeros(qn_t.shape)

    # Project to tangent space of manifold to obtain v_i
    if closed==0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return (v, gamI, d)


## Pre-processing of curves
def curveBatch_preproc(datapointList, cen=True, scale=True, rot=True):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapointList = list of length M, each entry has sample points from observed curves (d x N_i for curves i=1,...,M)
    
    # OPTIONAL:
    # cen = should curves be centered? (default = True)
    # scale = should curves be scaled to unit length? (default = True)
    # rot = should curves be optimally rotated? (default = True)
    
    ## Outputs ##
    # datapointList_pp = list of length m1, each entry has registered sample points from observed curves (d x N_i for curves i=1,...,m1)
    #################################################################################
    # Data dimensions
    if isinstance(datapointList,list):  # multiple curves in list format
        M = len(datapointList)
        d = len(datapointList[0])
        N = []
        for m in range(0,M):
            N.append(len(datapointList[m][0,:]))
        if M==1:
            rot = False  # no optimal rotation for a single curve
    else:  # just a single curve as an array
        M = 1
        d = datapointList.shape[0]
        N = datapointList.shape[1]
        rot = False  # cannot rotate if only one curve
    
    # Center curves
    if cen:
        if M>1 or isinstance(datapointList,list):
            c = [cf.calculatecentroid(datapointList[m]) for m in range(0,M)]
            datapointList_pp = [datapointList[m]-np.tile(c[m],(N[m],1)).T for m in range(0,M)]
        else:
            c = cf.calculatecentroid(datapointList)
            datapointList_pp = datapointList-np.tile(c,(N,1)).T
    else:
        datapointList_pp = datapointList
        c = None

    # Scale curves to unit length
    if scale:
        if M>1 or isinstance(datapointList,list):
            scl = [xy_length(datapointList_pp[m]) for m in range(0,M)]
            datapointList_pp = [datapointList_pp[m]/scl[m] for m in range(0,M)]
        else:
            scl = xy_length(datapointList_pp)
            datapointList_pp = datapointList_pp/scl
    else:
        scl = None
    
    # Rotationally align curves
    if rot:
        if all(n==N[0] for n in N)==False:
            print('ERROR: datapointList curves must be sampled at same number of points for current rotational alignment method')
            return None
        beta_rs = np.zeros((d,N[0],M))
    
        # Convert datapointList to suitable format for fdasrsf module   
        for m in range(0,M):
            beta_rs[:,:,m] = datapointList_pp[m]
                
        # Initialize fdacurve_v2 class
        reg_class = fdacurve_v2(beta_rs, mode='C', N=N[0], scale=False)
        q = reg_class.q        
        beta_shift = []
        O = np.zeros((d,d,M))

        # Find optimal rotation of each curve to first curve
        for m in range(0,M):
            if m==0:
                O[:,:,m] = np.eye(d)
                beta_shift.append(0)
            else:
                qm, Om, ctrm = find_rotation_and_seed_unique_nowarp(q[:,:,0], q[:,:,m], closed=1, rotation=True)
                O[:,:,m] = Om        
                beta_shift.append(ctrm)
                datapointList_pp[m] = O[:,:,m]@datapointList_pp[m]
    else:
        beta_shift = None
        O = None
        
    return datapointList_pp, c, scl, beta_shift, O


## Single-output GP fit for a single curve
def curveSOGP(datapoint, param=None, predpoint=100, truthpoint=None, cen=True, scale=True, CI_mult=1, period_mult=1.0, 
              est_period=False, restrictLS=True, use_Scipy='L-BFGS-B', maxiter=1000, kernel='Matern32', nug=1e-4, 
              autoEnclose=0, showPlot=True, showOutput=True, showEval=True, restrict_nv=True, showCoord=False):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapoint = (d x N)-dimensional array, columns are sample points from a single observed curve
    
    # OPTIONAL:
    # param = N-dimensional array where each entry is parameter value corresponding to each curve point in datapoint (default = None)
    #   (by default, this will be set to an arc-length parameterization, but this allows for manual use of other parameterizations)
    # predpoint = determines points to predict curve using GP at (default = 100 equally-spaced points with respect to param)
    #   this can also take a flat array corresponding to parameter values at which the GP fit should be predicted at
    # truthpoint = (d x Ntrue)-dimensional array, columns are sample points from a single TRUE curve
    #   (if not available, this is set to be the same as the sample points from observed curve, datapoint)
    # cen = should curve be zero-centered as pre-processing? (default = True)
    # scale = should curve be re-scaled to unit length as pre-processing? (default = True)
    # CI_mult = multiplier for prediction intervals (default = 1)
    # period_mult = multiple of estimated period specified for periodic kernel used to model between-point dependence (default = 1.0)
    # est_period = do we want to estimate the periodicity? if True, this overrides PERIOD_mult parameter;
    #   if False, the period parameter is assumed fixed and not estimated (default = False)
    # restrictLS = do we want to restrict the input kernel's length scale parameter to be between 0 and the curve's period? (default = True)
    # use_Scipy = used to specify optimization method; if 'Adam', Adam is used;
    #   otherwise, specify a Scipy optimization method from scipy.optimize.minimize function (default = L-BFGS-B in Scipy)
    # maxiter = maximal number of optimization iterations (default = 1000)
    # kernel = pre-specified kernel object for input covariance, choices below:
    #   'Matern12', 'Matern32' (default), 'Matern52', 'SqExp' (squared exponential), or can input other desired kernel from GPflow package
    # nug = fixed nugget in input covariance specification; if nug <= 0, abs(nug) is used for an additive constant kernel;
    #   if nug > 0, nug is used for an additive white noise kernel (default = 1e-4)
    # autoEnclose = indicator to augment the sparsely sampled points so that the first and the last point in the sample are the same; 
    #   1 means fill the last point with first point; 
    #   -1 means fill the first point with last point; 
    #   0 means do nothing; note that this works for both datapointList and truthpointList (default)
    # showPlot = indicator to show figures (default = True)
    # showOutput = indicator to show convergence and GP fit output (default = True)
    # showEval = indicator to show evaluation metrics on plots (default = True)
    # restrict_nv = indicator to restrict the noise variance hyperparameter estimation to the interval (10e-6, 10e-4)
    #   (this is discussed in the paper linked in the main README file, Supplementary Materials Section "Practical Numerical Issues")
    # showCoord = indicator to show x and y-coordinate GP fits (default = False)
    
    ## Outputs ##
    # my_SOGP_X, my_SOGP_Y = fitted GP objects for univariate x and y-coordinate functions
    # opt_logs_X, opt_logs_Y = optimization details for GP fitting of univariate x and y-coordinate functions
    # mu_Xn, mu_Yn = (1 x Nnew)-dimensional array of predicted mean for univariate x and y-coordinate GP fits
    # sd_Xn, sd_Yn = (1 x Nnew)-dimensional array of predicted standard deviation for univariate x and y-coordinate GP fits
    # perc = estimated arc-length of curve (with respect to truthpoint)
    # period_param = estimated period parameters for x and y-coordinate functions (only if estimatePeriod = True)
    # IMSPE, ESD, IUEA = evaluation metric values for fitting
    #################################################################################   
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpoint, type(None)):
        truth_ind = False
        truthpoint = datapoint
    else:
        truth_ind = True
        
    # Curve dimension (function only works for d=2)
    d, N = datapoint.shape
    if d!=2:  
        print('ERROR: Data format is not 2 by _ array')
        return 
    N_truth = truthpoint.shape[1]
    
    # Pre-process curves (register, re-scale, center if desired)
    if cen or scale:
        if truth_ind:  # true curve exists
            truthpoint, c, scl, beta_shift, O = curveBatch_preproc(truthpoint, cen, scale, rot=False)
            if scale:
                datapoint = datapoint/scl
            if cen:
                datapoint = datapoint-np.tile(c,(N,1)).T
        else:
            datapoint, c, scl, beta_shift, O = curveBatch_preproc(datapoint, cen, scale, rot=False)
            truthpoint = datapoint
    
    # Compute curve arc-lengths (to use as periodicity and for prediction)
    perc = xy_length(truthpoint)   
    if showOutput:
        print('>>length of curve', perc)
        
    # Set default jitter level
    gpflow.config.set_default_jitter(1e-4)
    
    # Specify base kernel on input space, if estimatePeriod==True, leave period parameter blank instead of plug in an estimate.
    if kernel=='Matern12':
        if est_period:
            my_kernX = gpflow.kernels.Periodic(Matern12(active_dims=[0]))
            my_kernY = gpflow.kernels.Periodic(Matern12(active_dims=[0]))
        else:
            my_kernX = gpflow.kernels.Periodic(Matern12(active_dims=[0]), period=period_mult*perc)
            my_kernY = gpflow.kernels.Periodic(Matern12(active_dims=[0]), period=period_mult*perc)
            set_trainable(my_kernX.period, False)
            set_trainable(my_kernY.period, False)
    elif kernel=='Matern32':
        if est_period:
            my_kernX = gpflow.kernels.Periodic(Matern32(active_dims=[0]))
            my_kernY = gpflow.kernels.Periodic(Matern32(active_dims=[0]))
        else:
            my_kernX = gpflow.kernels.Periodic(Matern32(active_dims=[0]), period=period_mult*perc)
            my_kernY = gpflow.kernels.Periodic(Matern32(active_dims=[0]), period=period_mult*perc)
            set_trainable(my_kernX.period, False)
            set_trainable(my_kernY.period, False)
    elif kernel=='Matern52':
        if est_period:
            my_kernX = gpflow.kernels.Periodic(Matern52(active_dims=[0]))
            my_kernY = gpflow.kernels.Periodic(Matern52(active_dims=[0]))
        else:
            my_kernX = gpflow.kernels.Periodic(Matern52(active_dims=[0]), period=period_mult*perc)
            my_kernY = gpflow.kernels.Periodic(Matern52(active_dims=[0]), period=period_mult*perc)
            set_trainable(my_kernX.period, False)
            set_trainable(my_kernY.period, False)
    elif kernel=='SqExp':
        if est_period:
            my_kernX = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]))
            my_kernY = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]))
        else:
            my_kernX = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]), period=period_mult*perc)
            my_kernY = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]), period=period_mult*perc)
            set_trainable(my_kernX.period, False)
            set_trainable(my_kernY.period, False)
    elif kernel=='Arccos':
        est_period = False
        restrictLS = False
        my_kernX = ArcCosine(active_dims=[0])
        my_kernY = ArcCosine(active_dims=[0])
    else:
        my_kernX = kernel
        my_kernY = kernel
    
    # Add white noise or constant kernel if nugget is included
    if nug==None:
        my_kernX = my_kernX
        my_kernY = my_kernY
    elif nug>0:
        my_kernX = my_kernX + gpflow.kernels.Constant(variance=nug)
        my_kernY = my_kernY + gpflow.kernels.Constant(variance=nug)
        set_trainable(my_kernX.kernels[1], False)
        set_trainable(my_kernY.kernels[1], False)
    elif nug<=0:
        my_kernX = my_kernX + gpflow.kernels.White(variance=abs(nug))
        my_kernY = my_kernY + gpflow.kernels.White(variance=abs(nug))
        set_trainable(my_kernX.kernels[1], False)
        set_trainable(my_kernY.kernels[1], False)
    
    # Constrain length scale parameters to be between min_ls and max_ls
    if showOutput:
        print('>>restrictLS=', restrictLS)
        print('>>period=', perc)
    
    if restrictLS:
        min_ls = 0
        max_ls = 0.5*period_mult*perc
        init_ls = 0.5*(min_ls+max_ls)

        if nug==None:
            my_kernX.base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
            my_kernY.base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
        else:
            my_kernX.kernels[0].base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
            my_kernY.kernels[0].base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
        
    # Create data matrix of inputs and outputs
    if autoEnclose>=1:
        # The last point is ignored and filled in by the first point
        if (datapoint[:,0] != datapoint[:,-1]).all():
            datapoint[:,-1] = datapoint[:,0].copy()
        if (truthpoint[:,0] != truthpoint[:,-1]).all():
            truthpoint[:,-1] = truthpoint[:,0].copy()
    elif autoEnclose<=-1: 
        # The first point is ignored and filled in by the last point
        if (datapoint[:,0] != datapoint[:,-1]).all():
            datapoint[:,0] = datapoint[:,-1].copy()
        if (truthpoint[:,0] != truthpoint[:,-1]).all():
            truthpoint[:,0] = truthpoint[:,-1].copy()

    X = datapoint[0,:].reshape(-1,1)
    Y = datapoint[1,:].reshape(-1,1)

    # Convert data to arc-length parameterization
    if isinstance(param, type(None)):
        t = np.array(xy_to_arc_param_new(datapoint,truthpoint)).reshape(-1,1)
        t[-1] = xy_length(truthpoint)
    else:
        t = param
    
    # Optimization for MOGP fitting
    if use_Scipy!='Adam':
        # Optimize using Scipy
        optX = gpflow.optimizers.Scipy()
        optY = gpflow.optimizers.Scipy()
        option_opt = dict(disp=True, maxiter=maxiter)
        
        if restrict_nv:
            my_SOGP_X = gpflow.models.GPR(data=(t,X), kernel=my_kernX, mean_function=None, noise_variance=0.00001)
            my_SOGP_Y = gpflow.models.GPR(data=(t,Y), kernel=my_kernY, mean_function=None, noise_variance=0.00001)
        
            global_nv_lower = 0.0000001
            global_nv_upper = 0.00001
            global_nv_init = 0.000001
            my_SOGP_X.likelihood.variance = bounded_parameter(global_nv_lower, global_nv_upper, global_nv_init)
            my_SOGP_Y.likelihood.variance = bounded_parameter(global_nv_lower, global_nv_upper, global_nv_init)
        else:
            my_SOGP_X = gpflow.models.GPR(data=(t,X), kernel=my_kernX)
            my_SOGP_Y = gpflow.models.GPR(data=(t,Y), kernel=my_kernY)            
        
        opt_logs_X = optX.minimize(my_SOGP_X.training_loss, my_SOGP_X.trainable_variables, method=use_Scipy, options=option_opt)
        opt_logs_Y = optY.minimize(my_SOGP_Y.training_loss, my_SOGP_Y.trainable_variables, method=use_Scipy, options=option_opt)

        # Print output: optimization monitoring and estimated kernel hyperparameters
        if showOutput:
            print(opt_logs_X)
            print(opt_logs_Y)

            print_summary(my_SOGP_X, fmt='notebook')
            print_summary(my_SOGP_Y, fmt='notebook')

            likelihood_X = my_SOGP_X.log_marginal_likelihood()
            likelihood_Y = my_SOGP_Y.log_marginal_likelihood()
            tf.print(f"Optimizer: {use_Scipy} loglik_margX: {likelihood_X: .04f} loglik_margY: {likelihood_Y: .04f}")
    else:
        # Optimize using Adam
        opt_logs_X = []
        opt_logs_Y = []
        
        adam_learning_rate = 0.03  
        optX = tf.optimizers.Adam(adam_learning_rate)
        optY = tf.optimizers.Adam(adam_learning_rate)
        my_SOGP_X = gpflow.models.GPR(data=(t,X), kernel=my_kernX)
        my_SOGP_Y = gpflow.models.GPR(data=(t,Y), kernel=my_kernY)
        
        for i in range(maxiter):
            optX.minimize(my_SOGP_X.training_loss, var_list=my_SOGP_X.trainable_variables)
            optY.minimize(my_SOGP_Y.training_loss, var_list=my_SOGP_Y.trainable_variables)
            
            # Optimization monitoring
            if i%100==0:
                likelihood_X = my_SOGP_X.log_marginal_likelihood().numpy()
                likelihood_Y = my_SOGP_Y.log_marginal_likelihood().numpy()
                
                opt_logs_X.append(np.round(likelihood_X,3))
                opt_logs_Y.append(np.round(likelihood_Y,3))
            
                if showOutput:
                    tf.print(f"Optimizer: Adam   iterations {i} loglik_margX: {likelihood_X: .04f} loglik_margY: {likelihood_Y: .04f}")
        
        # Print output: optimization monitoring and estimated kernel hyperparameters
        if showOutput:
            print_summary(my_SOGP_X, fmt='notebook')
            print_summary(my_SOGP_Y, fmt='notebook')
           
    # Store estimated period parameters if desired
    if est_period:
        if nug==None:
            period_param_X = my_SOGP_X.kernel.period.numpy()
            period_param_Y = my_SOGP_Y.kernel.period.numpy()
        else:
            period_param_X = my_SOGP_X.kernel.kernels[0].period.numpy()
            period_param_Y = my_SOGP_Y.kernel.kernels[0].period.numpy()
        period_param = (period_param_X, period_param_Y)

    # Predict at new input locations, with uncertainty quantified by predictive variance/sd
    if isinstance(predpoint, int):
        predloc = np.linspace(start=0, stop=perc, num=predpoint).reshape(-1,1)
    else:
        predloc = predpoint.reshape(-1,1)
        
    N_pred = predloc.shape[0]
    
    mu_X, var_X = my_SOGP_X.predict_f(predloc)
    mu_Y, var_Y = my_SOGP_Y.predict_f(predloc)
    mu_Xn = mu_X.numpy()
    mu_Yn = mu_Y.numpy()
    var_Xn = var_X.numpy()
    var_Yn = var_Y.numpy()
    sd_Xn = np.sqrt(var_Xn)
    sd_Yn = np.sqrt(var_Yn)
    
    # Compute integrated MSPE, elastic shape distance, and integrated uncertainty ellipse area
    if showEval:
        predloc1 = np.linspace(start=0, stop=perc, num=N_truth).reshape(-1,1)
        mu_X1, var_X1 = my_SOGP_X.predict_f(predloc1)
        mu_Y1, var_Y1 = my_SOGP_Y.predict_f(predloc1)

        mupoint = np.vstack((mu_X1.numpy().reshape(1,-1),mu_Y1.numpy().reshape(1,-1)))
        IMSPE = np.mean((mupoint[0,:]-truthpoint[0,:])**2 + (mupoint[1,:]-truthpoint[1,:])**2)

        try:
            ESD = cf.elastic_distance_curve(truthpoint, mupoint, closed=1, scale=False)
        except ValueError:
            ESD = float("inf")

        sd_X1 = np.sqrt(var_X1.numpy())
        sd_Y1 = np.sqrt(var_Y1.numpy())
        IUEA = np.mean(np.pi*(CI_mult*sd_X1)*(CI_mult*sd_Y1))
    else:
        IMSPE = None
        ESD = None
        IUEA = None
    
    # Plot results
    if showPlot and showCoord:
        plt.rcParams["font.family"] = "serif"
        from matplotlib.ticker import FormatStrFormatter
        fig, (ax0,ax1,ax2) = plt.subplots(1,3, figsize=(20,5))

        # First coordinate fit with prediction interval
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax0.scatter(predloc, mu_Xn, label='mean_x', c='k', s=5, zorder=1)
        ax0.plot(predloc, mu_Xn+CI_mult*sd_Xn, label='uci_x', c='b', linewidth=2, zorder=3)
        ax0.plot(predloc, mu_Xn-CI_mult*sd_Xn, label='lci_x', c='b', linewidth=2, zorder=4)
        ax0.fill_between(np.squeeze(predloc), np.squeeze(mu_Xn-CI_mult*sd_Xn), np.squeeze(mu_Xn+CI_mult*sd_Xn), alpha=0.2, zorder=2)
        ax0.scatter(t, datapoint[0,:], c='r', s=25, zorder=5)
        ax0.tick_params(axis='x', labelsize=15)
        ax0.tick_params(axis='y', labelsize=15)
        ax0.set_xlabel('arc-length', size=17)
        ax0.set_ylabel('x-coordinate', size=17)
        ax0.set_title('#new loc='+str(N_pred), fontsize=20)

        # Second coordinate fit with prediction interval
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.scatter(predloc, mu_Yn, label='mean_y', c='k', s=5, zorder=1)
        ax1.plot(predloc, mu_Yn+CI_mult*sd_Yn, label='uci_y', c='b', linewidth=2, zorder=3)
        ax1.plot(predloc, mu_Yn-CI_mult*sd_Yn, label='lci_y', c='b', linewidth=2, zorder=4)
        ax1.fill_between(np.squeeze(predloc), np.squeeze(mu_Yn-CI_mult*sd_Yn), np.squeeze(mu_Yn+CI_mult*sd_Yn), alpha=0.2, zorder=2)
        ax1.scatter(t, datapoint[1,:], c='r', s=25, zorder=5)
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax1.set_xlabel('arc-length', size=17)
        ax1.set_ylabel('y-coordinate', size=17)
        ax1.set_title('#new loc='+str(N_pred), fontsize=20)

        # Ellipse uncertainty quantification plot
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.scatter(mu_Xn, mu_Yn, label='estimated mean', c='k', s=5, zorder=2)
        if truth_ind:
            ax2.scatter(truthpoint[0,:], truthpoint[1,:], label='truth points', c='y', s=15, zorder=1)
        ax2.scatter(datapoint[0,:], datapoint[1,:], label='observed points', c='r', s=25, zorder=3)
        for i in range(0,N_pred):
            e_cen_x = mu_Xn[i]
            e_cen_y = mu_Yn[i]
            e_rad_x = sd_Xn[i]
            e_rad_y = sd_Yn[i]
            ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6)
            ax2.add_patch(ellipse)
        ax2.tick_params(axis='x', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        if showEval:
            if ESD < float('inf'):
                ax2.set_title(f'IMSPE={IMSPE:.2e}/ESD={ESD:.3f}/IUEA={IUEA:.2e}', fontsize=20)
            else:
                ax2.set_title(f'IMSPE={IMSPE:.2e}/ESD=undef/IUEA={IUEA:.2e}', fontsize=20)
        else:
            ax2.set_title('#observed samples='+str(N), fontsize=20)
        
        plt.tight_layout()
    elif showPlot and not showCoord:
        plt.rcParams["font.family"] = "serif"
        from matplotlib.ticker import FormatStrFormatter
        fig, ax2 = plt.subplots(figsize=(7,5))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.scatter(mu_Xn, mu_Yn, label='estimated mean', c='k', s=5, zorder=2)
        if truth_ind:
            ax2.scatter(truthpoint[0,:], truthpoint[1,:], label='truth points', c='y', s=15, zorder=1)
        ax2.scatter(datapoint[0,:], datapoint[1,:], label='observed points', c='r', s=25, zorder=3)
        for i in range(0,N_pred):
            e_cen_x = mu_Xn[i]
            e_cen_y = mu_Yn[i]
            e_rad_x = sd_Xn[i]
            e_rad_y = sd_Yn[i]
            ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6)
            ax2.add_patch(ellipse)
        ax2.tick_params(axis='x', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        if showEval:
            if ESD < float('inf'):
                ax2.set_title(f'IMSPE={IMSPE:.2e}/ESD={ESD:.3f}/IUEA={IUEA:.2e}', fontsize=20)
            else:
                ax2.set_title(f'IMSPE={IMSPE:.2e}/ESD=undef/IUEA={IUEA:.2e}', fontsize=20)
        else:
            ax2.set_title('#observed samples='+str(N), fontsize=20)
        
    if est_period:            
        return my_SOGP_X, my_SOGP_Y, opt_logs_X, opt_logs_Y, mu_Xn, sd_Xn, mu_Yn, sd_Yn, perc, period_param, IMSPE, ESD, IUEA
    else:
        return my_SOGP_X, my_SOGP_Y, opt_logs_X, opt_logs_Y, mu_Xn, sd_Xn, mu_Yn, sd_Yn, perc, IMSPE, ESD, IUEA
    
## Multiple-output GP fit for multiple curves using two-dimensional encoding
def curveBatchMOGP(datapointList, paramList=None, predList=100, truthpointList=None, cen=True, scale=True, rot=True, 
                   CI_mult=1, period_mult=1.0, est_period=False, restrictLS=True, use_Scipy='L-BFGS-B', maxiter=1000, 
                   kernel='Matern32', nug=1e-4, autoEnclose=0, showPlot=True, showOutput=True, showEval=True, 
                   showElastic=True, restrict_nv=True, showCoord=False, anova_style=False):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapointList = list of length m1, each entry has sample points from observed curves (d x N_i for curves i=1,...,m1)
    
    # OPTIONAL:
    # paramList = list of length m1, each entry is N-dimensional array with parameter value corresponding to each curve point in datapoint (default = None)
    #   (by default, this will be set to an arc-length parameterization with respect to each curve, but this allows for manual use of other parameterizations)
    # predList = determines points to predict each curve using GP at (default = 100 equally-spaced points with respect to paramList)
    #   this can also take a list of flat arrays corresponding to parameter values at which the GP fit should be predicted at
    # truthpointList = list of length m1, each entry has sample points from TRUE curves (d x N_truei for curves i=1,...m1)
    #   (if not available, this is set to be the same as the sample points from observed curves, datapointList)
    # cen = should curves be zero-centered as pre-processing? (default = True)
    # scale = should curves be re-scaled to unit length as pre-processing? (default = True)
    # rot = should curves be rotationally aligned as pre-processing? (default = True)
    # CI_mult = multiplier for prediction intervals (default = 1)
    # period_mult = period specified for periodic covariance kernel used to model between-point dependence (default = 1.0)
    # est_period = do we want to estimate the periodicity, if True, this would overrides the CI_mult and PERIOD_mult parameters (default = False)
    # restrictLS = do we want to restrict the input kernel's length scale parameter to be between 0 and the curve's period? (default = True)
    # use_Scipy = used to specify optimization method; if 'Adam', Adam is used;
    #   otherwise, specify a Scipy optimization method from scipy.optimize.minimize function (default = L-BFGS-B in Scipy)
    # maxiter = maximal number of optimization iterations (default = 1000)
    # kernel = pre-specified kernel object for input covariance, choices below:
    #   'Matern12', 'Matern32' (default), 'Matern52', 'SqExp' (squared exponential), or can input other desired kernel from GPflow package
    # nug = indicator to include nugget in input covariance specification (default = True)
    # autoEnclose = indicator to augment the sparsely sampled points so that the first and the last point in the sample are the same; 
    #   1 means fill the last point with first point; 
    #   -1 means fill the first point with last point; 
    #   0 means do nothing, note that this works for both datapointList and truthpointList (default = 0)
    # showPlot = indicator to show figures (default = True)
    # showOutput = indicator to show GP fit output (default = True)
    # showEval = indicator to show evaluation metrics on plots (default = True)
    # showElastic = indicator to show elastic shape distance as an evaluation metric on plots (default = True)
    # restrict_nv = indicator to restrict the noise variance hyperparameter estimation to the interval (10e-6, 10e-4)
    #   (this is discussed in the paper linked in the main README file, Supplementary Materials Section "Practical Numerical Issues")
    # showCoord = indicator to show x and y-coordinate GP fits (default = False)
    # anova_style = indicator to construct an additive multi-output kernel to fit an additive GP (Duvenaud et al., 2011) (default = False)
    #   source: https://arxiv.org/pdf/1112.4394.pdf
    #   (this is used to compare our separable kernel in the manuscript to the full additive kernel)
    
    ## Outputs ##
    # my_MOGP = fitted GP object
    # opt_logs = optimization details for GP fitting
    # mu_x_List, mu_y_List = lists of length m1, each entry has predicted mean function in each dimension
    # sd_x_List, sd_y_List = lists of length m1, each entry has predicted standard deviation function in each dimension
    # result_arr_X_new = input matrix for prediction in format suitable for coregionalization, useful to feed into other functions
    # perc = estimated arc-lengths of each curve (with respect to truthpointList)
    # IMSPE, ESD, IUEA = lists of evaluation metric values for each curve fitting
    #################################################################################   
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpointList, type(None)):
        truthpointList = datapointList
        truth_ind = False
    else:
        truth_ind = True
        
    # Number of curves
    M = len(datapointList)
    m_truth = len(truthpointList)
    if M!=m_truth:
        print('ERROR: datapointList and truthpointList must be two lists of the same length')
        return None
        
    # Curve dimension (function only works for d=2)
    d = datapointList[0].shape[0]
    if d!=2:  
        print('ERROR: Data format is not 2 by _ array within list')
        return None
    N = [datapointList[m].shape[1] for m in range(0,M)]
    N_total = np.sum(N)
    N_truth = [truthpointList[m].shape[1] for m in range(0,M)]
               
    # Pre-process curves (register, re-scale, center if desired)
    if cen or scale or rot:
        if truth_ind:  # true curves exist
            truthpointList, c, scl, beta_shift, O = curveBatch_preproc(truthpointList, cen, scale, rot)
            if scale:
                datapointList = [datapointList[m]/scl[m] for m in range(0,M)]
            if cen:
                datapointList = [datapointList[m]-np.tile(c[m],(N[m],1)).T for m in range(0,M)]
            if rot:
                datapointList = [O[:,:,m]@datapointList[m] for m in range(0,M)]
        else:
            datapointList, c, scl, beta_shift, O = curveBatch_preproc(datapointList, cen, scale, rot)
            truthpointList = datapointList
    
    # Compute curve arc-lengths (to use as periodicity and for prediction)
    perc = [xy_length(truthpointList[m]) for m in range(0,M)]      
    if showOutput:
        print('>>lengths of curves', perc)
    
    # Set default jitter level
    gpflow.config.set_default_jitter(1e-4)

    # Specify base kernel on input space, if estimatePeriod==True, leave period parameter blank instead of plug in an estimate.
    if kernel=='Matern12':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern32':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern52':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='SqExp':
        if est_period:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Arccos':
        est_period = False
        restrictLS = False
        my_kern = ArcCosine(active_dims=[0])
    else:
        my_kern = kernel
    
    # Add white noise or constant kernel if nugget is included
    if nug==None:
        my_kern = my_kern
    elif nug>0:
        my_kern = my_kern + gpflow.kernels.Constant(variance=nug)
        set_trainable(my_kern.kernels[1], False)
    elif nug<=0:
        my_kern = my_kern + gpflow.kernels.White(variance=abs(nug))
        set_trainable(my_kern.kernels[1], False)
    
    # Constrain length scale parameters to be between min_ls and max_ls
    if showOutput:
        print('>>restrictLS=', restrictLS)
        print('>>period=', perc)
        
    if restrictLS:
        min_ls = 0
        max_ls = 0.5*period_mult*max(perc)
        init_ls = 0.5*(min_ls+max_ls)

        if nug==None:
            my_kern.base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
        else:
            my_kern.kernels[0].base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
    
    # Create augmented data matrix of inputs and outputs (with labels in the second dimension which are separate for each curve 
    # and dimension)
    if isinstance(predList, int):
        predpointList = []
        for m in range(0,M):
            tmp = np.linspace(start=0, stop=perc[m], num=predList).reshape(-1,1)
            predpointList.append(tmp)
    else:
        predpointList = predList
    
    N_total_pred = np.sum([predpointList[m].shape[0] for m in range(0,M)])
    result_arr_X = np.zeros((d*N_total,3))
    result_arr_Y = np.zeros((d*N_total,3)) 
    result_arr_X_new = np.zeros((d*N_total_pred,3))
    
    pointer = int(0)
    pointer_new = int(0)
    for m in range(0,M):
        data_m = datapointList[m]
        truth_m = truthpointList[m]
        new_m = predpointList[m]
        
        if autoEnclose>=1:
            # The last point is ignored and filled in by the first point
            if (data_m[:,0] != data_m[:,-1]).all():
                data_m[:,-1] = data_m[:,0].copy()
            if (truth_m[:,0] != truth_m[:,-1]).all():
                truth_m[:,-1] = truth_m[:,0].copy()
        elif autoEnclose<=-1:
            # The first point is ignored and filled in by the last point
            if (data_m[:,0] != data_m[:,-1]).all():
                data_m[:,0] = data_m[:,-1].copy()
            if (truth_m[:,0] != truth_m[:,-1]).all():
                truth_m[:,0] = truth_m[:,-1].copy()

        N_m = data_m.shape[1]
        X_m = data_m[0,:]
        Y_m = data_m[1,:]
        N_pred_m = new_m.shape[0]
        
        # Convert data to arc-length parameterization
        if isinstance(paramList, type(None)):
            tmp = np.array(xy_to_arc_param_new(data_m,truth_m)).reshape(-1,1)
            tmp[-1] = xy_length(truth_m)
        else:
            tmp = paramList[m]
        
        result_arr_X[pointer:(pointer+N_m),:] = np.hstack((tmp,np.zeros((N_m,1)),m*np.ones((N_m,1))))
        result_arr_Y[pointer:(pointer+N_m),:] = np.hstack((data_m[0,:].reshape(-1,1),np.zeros((N_m,1)),m*np.ones((N_m,1))))
        pointer = pointer + N_m

        result_arr_X[pointer:(pointer+N_m),:] = np.hstack((tmp,np.ones((N_m,1)),m*np.ones((N_m,1))))
        result_arr_Y[pointer:(pointer+N_m),:] = np.hstack((data_m[1,:].reshape(-1,1),np.ones((N_m,1)),m*np.ones((N_m,1))))
        pointer = pointer + N_m
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred_m),:] = np.hstack((new_m,np.zeros((N_pred_m,1)),m*np.ones((N_pred_m,1))))
        pointer_new = pointer_new + N_pred_m
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred_m),:] = np.hstack((new_m,np.ones((N_pred_m,1)),m*np.ones((N_pred_m,1))))
        pointer_new = pointer_new + N_pred_m
    
    # Set up coregionalization kernels across dimensions and curves
    OUTPUT_DIM = len(np.unique(result_arr_Y[:,1]))*len(np.unique(result_arr_Y[:,2]))  # number of unique likelihoods
    #W_NCOL1 = len(np.unique(result_arr_Y[:,1]))
    #coreg1 = Coregion(output_dim=OUTPUT_DIM, rank=W_NCOL1, active_dims=[1])
    coreg1 = Coregion(output_dim=OUTPUT_DIM, rank=1, active_dims=[1])
    if anova_style:
        #var_cont1 = gpflow.kernels.White()
        #var_cont2 = gpflow.kernels.White()
        kern = my_kern + coreg1 + my_kern*coreg1
        #kern = var_cont1*my_kern + var_cont1*coreg1 + var_cont2*my_kern*coreg1 
    else:
        kern = my_kern*coreg1
    
    if M>1:
        W_NCOL2 = len(np.unique(result_arr_Y[:,2]))  # number of curves (for coregionalization)
        coreg2 = Coregion(output_dim=OUTPUT_DIM, rank=W_NCOL2, active_dims=[2])
        if anova_style:
            #var_cont3 = gpflow.kernels.White()
            kern = kern + coreg2 + my_kern*coreg2 + coreg1*coreg2 + my_kern*coreg1*coreg2
            #kern = kern + var_cont1*coreg2 + var_cont2*my_kern*coreg2 + var_cont2*coreg1*coreg2 + var_cont3*my_kern*coreg1*coreg2
        else:
            kern = kern*coreg2
    else:
        # Remove curve-level column from input/output matrices
        result_arr_X = np.delete(result_arr_X, 2, 1)
        result_arr_Y = np.delete(result_arr_Y, 2, 1)
        result_arr_X_new = np.delete(result_arr_X_new, 2, 1)
        
    # Optimization for MOGP fitting
    if use_Scipy!='Adam':
        # Optimize using Scipy
        opt = gpflow.optimizers.Scipy()
        option_opt = dict(disp=True, maxiter=maxiter)
        if restrict_nv:
            my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern, mean_function=None, noise_variance=0.00001)
            global_nv_lower = 0.000001
            global_nv_upper = 0.0001
            global_nv_init = 0.00001
            my_MOGP.likelihood.variance = bounded_parameter(global_nv_lower, global_nv_upper, global_nv_init)
        else:
            my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        opt_logs = opt.minimize(my_MOGP.training_loss, my_MOGP.trainable_variables, method=use_Scipy, options=option_opt)
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print(opt_logs)
            print_summary(my_MOGP, fmt='notebook')

            likelihood = my_MOGP.log_marginal_likelihood()
            tf.print(f"Optimizer: {use_Scipy} loglik_marg: {likelihood: .04f}")
            
            print(coreg1.output_covariance().numpy())
            if M>1:
                print(coreg2.output_covariance().numpy())
    else:
        # Optimize using Adam
        opt_logs = []
        
        adam_learning_rate = 0.03  
        opt = tf.optimizers.Adam(adam_learning_rate)
        my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        
        for i in range(maxiter):
            opt.minimize(my_MOGP.training_loss, var_list=my_MOGP.trainable_variables)
            
            # Optimization monitoring
            if i%100==0:
                likelihood = my_MOGP.log_marginal_likelihood().numpy()      
                opt_logs.append(np.round(likelihood,3))
            
                if showOutput:
                    tf.print(f"Optimizer: Adam   iterations {i} loglik_marg: {likelihood: .04f}")
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print_summary(my_MOGP, fmt='notebook')
            print(coreg1.output_covariance().numpy())
            if M>1:
                print(coreg2.output_covariance().numpy())
    
    # Store estimated period parameters if desired
    if est_period:
        if nug==None:
            period_param = my_MOGP.kernel.kernels[0].period.numpy()
        else:
            period_param = my_MOGP.kernel.kernels[0].kernels[0].period.numpy()

    # Predict at new input locations, with uncertainty quantified by predictive variance/sd
    mu, var = my_MOGP.predict_f(result_arr_X_new, full_cov=True)
    mu = mu.numpy()
    #var = var.numpy()
    sd = np.sqrt(np.diag(var[0]))
    
    if showEval:
        N_total_eval = np.sum([truthpointList[m].shape[1] for m in range(0,M)])
        result_arr_X_eval = np.zeros((d*N_total_eval,3))
        pointer_new = int(0)
        N_eval_m = [truthpointList[m].shape[1] for m in range(0,M)]
        
        for m in range(0,M):
            N_eval = N_eval_m[m]
            new_m = np.linspace(start=0, stop=perc[m], num=N_eval).reshape(-1,1)

            result_arr_X_eval[pointer_new:(pointer_new+N_eval),:] = np.hstack((new_m,np.zeros((N_eval,1)),m*np.ones((N_eval,1))))
            pointer_new = pointer_new + N_eval

            result_arr_X_eval[pointer_new:(pointer_new+N_eval),:] = np.hstack((new_m,np.ones((N_eval,1)),m*np.ones((N_eval,1))))
            pointer_new = pointer_new + N_eval
            
        mu_eval, var_eval = my_MOGP.predict_f(result_arr_X_eval, full_cov=True)
        mu_eval = mu_eval.numpy()
        #sd_eval = np.sqrt(var_eval.numpy())
        sd_eval = np.sqrt(np.diag(var_eval[0]))
        
        IMSPE = []
        if showElastic:
            ESD = []
        else:
            ESD = None
        IUEA = []
    else:
        IMSPE = None
        ESD = None
        IUEA = None
    
    # Subset based on selected curve and parse the prediction result (idx)
    mu_x_List = []
    sd_x_List = []
    lci_x_List = []
    uci_x_List = []  
    mu_y_List = []
    sd_y_List = []
    lci_y_List = []
    uci_y_List = []
    
    pointer_new = int(0)
    pointer_eval = int(0)
    for m in range(0,M):
        data_m = datapointList[m]
        truth_m = truthpointList[m]
        new_m = predpointList[m]
        N_m = data_m.shape[1]
        N_pred_m = new_m.shape[0]
        if showEval:
            N_eval = N_eval_m[m]
        
        mu_x = mu[pointer_new:(pointer_new+N_pred_m),0]
        sd_x = sd[pointer_new:(pointer_new+N_pred_m)]
        mu_x_List.append(mu_x)
        sd_x_List.append(sd_x)
        
        cov_m = var[0][pointer_new:(pointer_new+2*N_pred_m), pointer_new:(pointer_new+2*N_pred_m)].numpy()
        cov_XY = np.array([cov_m[i,i+N_pred_m] for i in range(N_pred_m)])
        
        lci_x_List.append(mu_x-CI_mult*sd_x)
        uci_x_List.append(mu_x+CI_mult*sd_x)
        tmp_x_new = result_arr_X_new[pointer_new:(pointer_new+N_pred_m),0]
        if showEval:
            mu_X1 = mu_eval[pointer_eval:(pointer_eval+N_eval),0]
            sd_X1 = sd_eval[pointer_eval:(pointer_eval+N_eval)]
            
            cov_1 = var_eval[0][pointer_eval:(pointer_eval+2*N_eval), pointer_eval:(pointer_eval+2*N_eval)].numpy()
            cov_XY1 = np.array([cov_1[i,i+N_eval] for i in range(N_eval)])
            
            pointer_eval = pointer_eval + N_eval
        pointer_new = pointer_new + N_pred_m

        mu_y = mu[pointer_new:(pointer_new+N_pred_m),0]
        sd_y = sd[pointer_new:(pointer_new+N_pred_m)]
        mu_y_List.append(mu_y)
        sd_y_List.append(sd_y)
        
        lci_y_List.append(mu_y-CI_mult*sd_y)
        uci_y_List.append(mu_y+CI_mult*sd_y)
        tmp_y_new = result_arr_X_new[pointer_new:(pointer_new+N_pred_m),0]
        if showEval:
            mu_Y1 = mu_eval[pointer_eval:(pointer_eval+N_eval),0]
            sd_Y1 = sd_eval[pointer_eval:(pointer_eval+N_eval)]
            pointer_eval = pointer_eval + N_eval
        pointer_new = pointer_new + N_pred_m
        
        # Compute integrated MSPE, elastic shape distance, integrated uncertainty ellipsoid area
        if showEval:
            mupoint = np.vstack((mu_X1.reshape(1,-1),mu_Y1.reshape(1,-1)))
            IMSPE.append(np.mean((mupoint[0,:]-truth_m[0,:])**2 + (mupoint[1,:]-truth_m[1,:])**2)) 

            if showElastic:
                try:
                    ESD.append(cf.elastic_distance_curve(truth_m, mupoint, closed=1, scale=False))
                except ValueError:
                    ESD.append(float("inf"))

            IUEA.append(np.mean(np.pi*sd_X1*np.sqrt((sd_Y1**2)-((cov_XY1/sd_X1)**2))))
        
        tmp = np.array(xy_to_arc_param_new(data_m,truth_m)).reshape(-1,1)
        tmp[-1] = xy_length(truth_m)
        
        if showPlot and showCoord:
            plt.rcParams["font.family"] = "serif"
            from matplotlib.ticker import FormatStrFormatter
            fig, (ax0,ax1,ax2) = plt.subplots(1,3, figsize=(20,5))

            # First coordinate fit with prediction interval
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax0.scatter(tmp_x_new, mu_x, label='mean_x', c='k', s=5, zorder=1)
            ax0.plot(tmp_x_new, mu_x+CI_mult*sd_x, label='uci_x', c='b', linewidth=2, zorder=3)
            ax0.plot(tmp_x_new, mu_x-CI_mult*sd_x, label='lci_x', c='b', linewidth=2, zorder=4)
            ax0.fill_between(np.squeeze(tmp_x_new), np.squeeze(mu_x-CI_mult*sd_x), np.squeeze(mu_x+CI_mult*sd_x), alpha=0.2, zorder=2)
            ax0.scatter(tmp, data_m[0,:], c='r', s=25, zorder=5)
            ax0.tick_params(axis='x', labelsize=15)
            ax0.tick_params(axis='y', labelsize=15)
            ax0.set_xlabel('arc-length', size=17)
            ax0.set_ylabel('x-coordinate', size=17)
            ax0.set_title('#new loc='+str(N_pred_m), fontsize=20)

            # Second coordinate fit with prediction interval
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.scatter(tmp_y_new, mu_y, label='mean_y', c='k', s=5, zorder=1)
            ax1.plot(tmp_y_new, mu_y+CI_mult*sd_y, label='uci_y', c='b', linewidth=2, zorder=3)
            ax1.plot(tmp_y_new, mu_y-CI_mult*sd_y, label='lci_y', c='b', linewidth=2, zorder=4)
            ax1.fill_between(np.squeeze(tmp_y_new), np.squeeze(mu_y-CI_mult*sd_y), np.squeeze(mu_y+CI_mult*sd_y), alpha=0.2, zorder=2)
            ax1.scatter(tmp, data_m[1,:], c='r', s=25, zorder=5)
            ax1.tick_params(axis='x', labelsize=15)
            ax1.tick_params(axis='y', labelsize=15)
            ax1.set_xlabel('arc-length', size=17)
            ax1.set_ylabel('x-coordinate', size=17)
            ax1.set_title('#new loc='+str(N_pred_m), fontsize=20)
            
            # Ellipse uncertainty quantification plot              
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.scatter(mu_x, mu_y, label='estimated mean', c='k', s=5, zorder=2)
            if truth_ind:
                ax2.scatter(truth_m[0,:], truth_m[1,:], label='truth points', c='y', s=15, zorder=1)
            ax2.scatter(data_m[0,:], data_m[1,:], label='observed points', c='r', s=25, zorder=3)
            #ax2.scatter(data_m[0,0], data_m[1,0], c='g', s=100, zorder=3)
            #ax = plt.gca()
            for i in range(0,N_pred_m):
                e_cen_x = mu_x[i]
                e_cen_y = mu_y[i]
                e_rad_x = sd_x[i]
                e_rad_y = sd_y[i]
                e_ang_xy = cov_XY[i]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6, angle=45*e_ang_xy)
                ax2.add_patch(ellipse)
            ax2.tick_params(axis='x', labelsize=15)
            ax2.tick_params(axis='y', labelsize=15)
            if showEval:
                if showElastic:
                    if ESD[m] < float('inf'):
                        ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/ESD={ESD[m]:.3f}/IUEA={IUEA[m]:.2e}', fontsize=20)
                    else:
                        ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/ESD=undef/IUEA={IUEA[m]:.2e}', fontsize=20)
                else:
                    ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/IUEA={IUEA[m]:.2e}', fontsize=20)
            else:
                ax2.set_title('#observed samples='+str(N_m), fontsize=20)
            
            plt.tight_layout()
            
        elif showPlot and not showCoord:
            plt.rcParams["font.family"] = "serif"
            from matplotlib.ticker import FormatStrFormatter
            fig, ax2 = plt.subplots(figsize=(7,5))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.scatter(mu_x, mu_y, label='estimated mean', c='k', s=5, zorder=2)
            if truth_ind:
                ax2.scatter(truth_m[0,:], truth_m[1,:], label='truth points', c='y', s=15, zorder=1)
            ax2.scatter(data_m[0,:], data_m[1,:], label='observed points', c='r', s=25, zorder=3)
            # ax2.scatter(data_m[0,0], data_m[1,0], c='g', s=100, zorder=3)
            #ax = plt.gca()
            for i in range(0,N_pred_m):
                e_cen_x = mu_x[i]
                e_cen_y = mu_y[i]
                e_rad_x = sd_x[i]
                e_rad_y = sd_y[i]
                e_ang_xy = cov_XY[i]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6, angle=45*e_ang_xy)
                ax2.add_patch(ellipse)
            ax2.tick_params(axis='x', labelsize=15)
            ax2.tick_params(axis='y', labelsize=15)
            if showEval:
                if truth_ind is False:
                    ax2.set_title(f'IUEA={IUEA[m]:.2e}', fontsize=20)
                else:
                    if showElastic:
                        if ESD[m] < float('inf'):
                            ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/ESD={ESD[m]:.3f}/IUEA={IUEA[m]:.2e}', fontsize=20)
                        else:
                            ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/ESD=undef/IUEA={IUEA[m]:.2e}', fontsize=20)
                    else:
                        ax2.set_title(f'IMSPE={IMSPE[m]:.2e}/IUEA={IUEA[m]:.2e}', fontsize=20)
            else:
                ax2.set_title('#observed samples='+str(N_m), fontsize=20)
            
    if est_period:            
            return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, period_param, IMSPE, ESD, IUEA
    else:
            return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, IMSPE, ESD, IUEA
   
    
## Multiple-output GP fit for multiple curves with cluster/class labels
def curveBatchMOGPwithLabel(datapointList, labelList, paramList=None, predList=100, truthpointList=None, cen=True, scale=True,
                            rot=True, CI_mult=1, period_mult=1.0, est_period=False, restrictLS=True, use_Scipy='L-BFGS-B', 
                            maxiter=1000, kernel='Matern32', cl_kernel='Coreg', nug=1e-4, autoEnclose=0, showPlot1=False, 
                            showPlot2=True, showOutput=True, inc_label=True, restrict_nv=True, showEval=True, 
                            showElastic=True):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapointList = list of length m1, each entry has sample points from observed curves (d x N_i for curves i=1,...,m1)
    # labelList = list of length m1 with class labels corresponding to each curve
    
    # OPTIONAL:
    # paramList = list of length m1, each entry is N-dimensional array with parameter value corresponding to each curve point in datapoint (default = None)
    #   (by default, this will be set to an arc-length parameterization with respect to each curve, but this allows for manual use of other parameterizations)
    # predList = determines points to predict each curve using GP at (default = 100 equally-spaced points with respect to paramList)
    #   this can also take a list of flat arrays corresponding to parameter values at which the GP fit should be predicted at
    # truthpointList = list of length m1, each entry has sample points from TRUE curves (d x N_truei for curves i=1,...m1)
    #   (if not available, this is set to be the same as the sample points from observed curves, datapointList)
    # cen = should curves be zero-centered as pre-processing? (default = True)
    # scale = should curves be re-scaled to unit length as pre-processing? (default = True)
    # rot = should curves be rotationally aligned (with respect to each class) as pre-processing? (default = True)
    # CI_mult = multiplier for prediction intervals (default = 1)
    # period_mult = period specified for periodic covariance kernel used to model between-point dependence (default = 1.0)
    # est_period = do we want to estimate the periodicity, if True, this would override the CI_mult and PERIOD_mult parameters (default = False)
    # restrictLS = do we want to restrict the input kernel's length scale parameter to be between 0 and the curve's period? (default = True)
    # use_Scipy = used to specify optimization method; if 'Adam', Adam is used;
    #   otherwise, specify a Scipy optimization method from scipy.optimize.minimize function (default = L-BFGS-B in Scipy)
    # maxiter = maximal number of optimization iterations (default = 1000)
    # kernel = pre-specified kernel object for input covariance, choices below:
    #   'Matern12', 'Matern32' (default), 'Matern52', 'SqExp' (squared exponential), or can input other desired kernel from GPflow package
    # cl_kernel = pre-specified kernel object for class label covariance, choices below:
    #   'Matern12', 'Matern32', 'Matern52', 'SqExp' (squared exponential), 'Coreg' (coregionalization, default), or can input other desired kernel from GPflow package
    # nug = indicator to include nugget in input covariance specification (default = True)
    # autoEnclose = indicator to augment the sparsely sampled points so that the first and the last point in the sample are the same; 
    #   1 means fill the last point with first point; 
    #   -1 means fill the first point with last point; 
    #   0 means do nothing, note that this works for both datapointList and truthpointList (default = 0)
    # showPlot1 = indicator to show individual curve fit figures (default = False)
    # showPlot2 = indicator to show all curve fits grouped by class simultaneously (default = True)
    # inc_label = allows GP fitting which ignores the curve label, this should yield similar results to curveBatchMOGP function (default = False)
    # restrict_nv = indicator to restrict the noise variance hyperparameter estimation to the interval (10e-6, 10e-4)
    #   (this is discussed in the paper linked in the main README file, Supplementary Materials Section "Practical Numerical Issues")
    # showEval = indicator to show evaluation metrics on plots (default = True)
    # showElastic = indicator to show elastic shape distance as an evaluation metric on plots (default = True)
    
    ## Outputs ##
    # my_MOGP = fitted GP object
    # opt_logs = optimization details for GP fitting
    # mu_x_List, mu_y_List = lists of length m1, each entry has predicted mean function in each dimension
    # sd_x_List, sd_y_List = lists of length m1, each entry has predicted standard deviation function in each dimension
    # result_arr_X_new = input matrix for prediction in format suitable for coregionalization, useful to feed into other functions
    # perc = estimated arc-lengths of each curve (with respect to truthpointList)
    # IMSPE, ESD, IUEA = lists of evaluation metric values for each curve fitting
    #################################################################################   
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpointList, type(None)):
        truthpointList = datapointList
        truth_ind = False
    else:
        truth_ind = True
        
    # Number of curves
    M = len(datapointList)
    m_truth = len(truthpointList)
    if M!=m_truth:
        print('ERROR: datapointList and truthpointList must be two lists of the same length')
        return None
    
    if M==1:
        print('ERROR: must input more than one curve for this function')
        return None
        
    # Curve dimension (function only works for d=2)
    d = datapointList[0].shape[0]
    if d!=2:  
        print('ERROR: Data format is not 2 by _ array within list')
        return None
    N = [datapointList[m].shape[1] for m in range(0,M)]
    N_total = np.sum(N)
    N_truth = [truthpointList[m].shape[1] for m in range(0,M)]
    
    # Label list
    if labelList==None:
        print('ERROR: Must provide labelList or use curveBatchMOGP function')
        return None        
    
    if len(labelList)!=M:
        print('ERROR: datapointList and labelList must be two lists of the same length')
        return None
               
    # Pre-process curves (register, re-scale, center if desired)
    if cen or scale or rot:
        if truth_ind:  # true curves exist
            truthpointList, c, scl, beta_shift, O = curveBatch_preproc(truthpointList, cen, scale, rot=False)
            if scale:
                datapointList = [datapointList[m]/scl[m] for m in range(0,M)]
            if cen:
                datapointList = [datapointList[m]-np.tile(c[m],(N[m],1)).T for m in range(0,M)]
            if rot:
                unique_lab = set(labelList)
                for l in unique_lab:
                    idx_class = [labelList[idx]==l for idx in range(len(labelList))]
                    idx_class = [i for i, x in enumerate(idx_class) if x]
                    data_class = [truthpointList[idx] for idx in idx_class]
                    data_class, c, scl, beta_shift, O = curveBatch_preproc(data_class, cen=False, scale=False, rot=True)
                    ctr = 0
                    for idx in idx_class:
                        truthpointList[idx] = data_class[ctr]
                        datapointList[idx] = O[:,:,ctr]@datapointList[idx]
                        ctr += 1    
        else:
            datapointList, c, scl, beta_shift, O = curveBatch_preproc(datapointList, cen, scale, rot=False)
            if rot:
                # Rotate curves within class to match class' first curve
                unique_lab = set(labelList)
                for l in unique_lab:
                    idx_class = [labelList[idx]==l for idx in range(len(labelList))]
                    idx_class = [i for i, x in enumerate(idx_class) if x]
                    data_class = [datapointList[idx] for idx in idx_class]
                    data_class, c, scl, beta_shift, O = curveBatch_preproc(data_class, cen=False, scale=False, rot=True)
                    ctr = 0
                    for idx in idx_class:
                        datapointList[idx] = data_class[ctr]
                        ctr += 1     
            truthpointList = datapointList
    
    # Compute curve arc-lengths (to use as periodicity and for prediction)
    perc = [xy_length(truthpointList[m]) for m in range(0,M)]      
    if showOutput:
        print('>>lengths of curves', perc)
    
    # Set default jitter level
    gpflow.config.set_default_jitter(1e-4)

    # Specify base kernel on input space, if estimatePeriod==True, leave period parameter blank instead of plug in an estimate.
    if kernel=='Matern12':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern32':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern52':
        if est_period:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='SqExp':
        if est_period:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]), period=period_mult*max(perc))
            set_trainable(my_kern.period, False)  
    else:
        my_kern = kernel
    
    # Add white noise or constant kernel if nugget is included
    if nug==None:
        my_kern = my_kern
    elif nug>0:
        my_kern = my_kern + gpflow.kernels.Constant(variance=nug)
        set_trainable(my_kern.kernels[1], False)
    elif nug<=0:
        my_kern = my_kern + gpflow.kernels.White(variance=abs(nug))
        set_trainable(my_kern.kernels[1], False)
    
    # Constrain length scale parameters to be between min_ls and max_ls
    if showOutput:
        print('>>restrictLS=', restrictLS)
        print('>>period=', perc)
        
    if restrictLS:
        min_ls = 0
        max_ls = 0.5*period_mult*max(perc)
        init_ls = 0.5*(min_ls+max_ls)

        if nug==None:
            my_kern.base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
        else:
            my_kern.kernels[0].base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
    
    # Create augmented data matrix of inputs and outputs (with labels in the second dimension which are separate for each curve 
    # and dimension)
    if isinstance(predList, int):
        predpointList = []
        for m in range(0,M):
            tmp = np.linspace(start=0, stop=perc[m], num=predList).reshape(-1,1)
            predpointList.append(tmp)
    else:
        predpointList = predList
    
    N_total_pred = np.sum([predpointList[m].shape[0] for m in range(0,M)])
    result_arr_X = np.zeros((d*N_total,4))
    result_arr_Y = np.zeros((d*N_total,4)) 
    result_arr_X_new = np.zeros((d*N_total_pred,4))
    
    pointer = int(0)
    pointer_new = int(0)
    for m in range(0,M):
        data_m = datapointList[m]
        truth_m = truthpointList[m]
        new_m = predpointList[m]
        
        if autoEnclose>=1:
            # The last point is ignored and filled in by the first point
            if (data_m[:,0] != data_m[:,-1]).all():
                data_m[:,-1] = data_m[:,0].copy()
            if (truth_m[:,0] != truth_m[:,-1]).all():
                truth_m[:,-1] = truth_m[:,0].copy()
        elif autoEnclose<=-1:
            # The first point is ignored and filled in by the last point
            if (data_m[:,0] != data_m[:,-1]).all():
                data_m[:,0] = data_m[:,-1].copy()
            if (truth_m[:,0] != truth_m[:,-1]).all():
                truth_m[:,0] = truth_m[:,-1].copy()

        N_m = data_m.shape[1]
        X_m = data_m[0,:]
        Y_m = data_m[1,:]
        N_pred_m = new_m.shape[0]
        
        # Convert data to arc-length parameterization
        if isinstance(paramList, type(None)):
            tmp = np.array(xy_to_arc_param_new(data_m,truth_m)).reshape(-1,1)
            tmp[-1] = xy_length(truth_m)
        else:
            tmp = paramList[m]
        
        result_arr_X[pointer:(pointer+N_m),:] = np.hstack((tmp,np.zeros((N_m,1)),m*np.ones((N_m,1)),labelList[m]*np.ones((N_m,1))))
        result_arr_Y[pointer:(pointer+N_m),:] = np.hstack((data_m[0,:].reshape(-1,1),np.zeros((N_m,1)),m*np.ones((N_m,1)),
                                                           labelList[m]*np.ones((N_m,1))))
        pointer = pointer + N_m

        result_arr_X[pointer:(pointer+N_m),:] = np.hstack((tmp,np.ones((N_m,1)),m*np.ones((N_m,1)),labelList[m]*np.ones((N_m,1))))
        result_arr_Y[pointer:(pointer+N_m),:] = np.hstack((data_m[1,:].reshape(-1,1),np.ones((N_m,1)),m*np.ones((N_m,1)),
                                                           labelList[m]*np.ones((N_m,1))))
        pointer = pointer + N_m
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred_m),:] = np.hstack((new_m,np.zeros((N_pred_m,1)),m*np.ones((N_pred_m,1)),
                                                                            labelList[m]*np.ones((N_pred_m,1))))
        pointer_new = pointer_new + N_pred_m
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred_m),:] = np.hstack((new_m,np.ones((N_pred_m,1)),m*np.ones((N_pred_m,1)),
                                                                            labelList[m]*np.ones((N_pred_m,1))))
        pointer_new = pointer_new + N_pred_m
    
    # Set up coregionalization kernels across dimensions and curves
    OUTPUT_DIM = len(np.unique(result_arr_Y[:,1]))*len(np.unique(result_arr_Y[:,2]))  # number of unique likelihoods
    coreg1 = Coregion(output_dim=OUTPUT_DIM, rank=1, active_dims=[1])
    kern = my_kern*coreg1
    
    W_NCOL2 = len(np.unique(result_arr_Y[:,2]))  # number of curves (for coregionalization)
    coreg2 = Coregion(output_dim=OUTPUT_DIM, rank=W_NCOL2, active_dims=[2])
    kern = kern*coreg2 
        
    # Add kernel for cluster label column
    if inc_label: 
        if cl_kernel=='Matern12':
            kern = kern + Matern12(active_dims=[3])
        elif cl_kernel=='Matern32':
            kern = kern + Matern32(active_dims=[3])
        elif cl_kernel=='Matern52':
            kern = kern + Matern52(active_dims=[3])
        elif cl_kernel=='SqExp':
            kern = kern + SquaredExponential(active_dims=[3])
        elif cl_kernel=='Coreg':
            coreg3 = Coregion(output_dim=OUTPUT_DIM, rank=len(np.unique(result_arr_Y[:,3])), active_dims=[3])
            kern = kern*coreg3
        else:
            kern = kern + cl_kernel
    else: 
        # Remove label-level column from input/output matrices
        result_arr_X = np.delete(result_arr_X, 3, 1)
        result_arr_Y = np.delete(result_arr_Y, 3, 1)
        result_arr_X_new = np.delete(result_arr_X_new, 3, 1)
    
    # Optimization for MOGP fitting
    if use_Scipy!='Adam':
        # Optimize using Scipy
        opt = gpflow.optimizers.Scipy()
        option_opt = dict(disp=True, maxiter=maxiter)
        if restrict_nv:
            my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern, mean_function=None, noise_variance=0.00001)
            global_nv_lower = 0.000001
            global_nv_upper = 0.0001
            global_nv_init = 0.00001
            my_MOGP.likelihood.variance = bounded_parameter(global_nv_lower, global_nv_upper, global_nv_init)
        else:
            my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        opt_logs = opt.minimize(my_MOGP.training_loss, my_MOGP.trainable_variables, method=use_Scipy, options=option_opt)
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print(opt_logs)
            print_summary(my_MOGP, fmt='notebook')

            likelihood = my_MOGP.log_marginal_likelihood()
            tf.print(f"Optimizer: {use_Scipy} loglik_marg: {likelihood: .04f}")
            
            print(coreg1.output_covariance().numpy())
            if M>1:
                print(coreg2.output_covariance().numpy())
    else:
        # Optimize using Adam
        opt_logs = []
        
        adam_learning_rate = 0.03  
        opt = tf.optimizers.Adam(adam_learning_rate)
        my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        
        for i in range(maxiter):
            opt.minimize(my_MOGP.training_loss, var_list=my_MOGP.trainable_variables)
            
            # Optimization monitoring
            if i%100==0:
                likelihood = my_MOGP.log_marginal_likelihood().numpy()      
                opt_logs.append(np.round(likelihood,3))
            
                if showOutput:
                    tf.print(f"Optimizer: Adam   iterations {i} loglik_marg: {likelihood: .04f}")
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print_summary(my_MOGP, fmt='notebook')
            print(coreg1.output_covariance().numpy())
            if M>1:
                print(coreg2.output_covariance().numpy())
    
    # Store estimated period parameters if desired
    if est_period:
        if nug==None:
            period_param = my_MOGP.kernel.kernels[0].period.numpy()
        else:
            period_param = my_MOGP.kernel.kernels[0].kernels[0].period.numpy()

    # Predict at new input locations, with uncertainty quantified by predictive variance/sd
    mu, var = my_MOGP.predict_f(result_arr_X_new, full_cov=True)
    mu = mu.numpy()
    #var = var.numpy()
    sd = np.sqrt(np.diag(var[0]))
    
    if showEval:
        N_total_eval = np.sum([truthpointList[m].shape[1] for m in range(0,M)])
        result_arr_X_eval = np.zeros((d*N_total_eval,4))
        pointer_new = int(0)
        N_eval_m = [truthpointList[m].shape[1] for m in range(0,M)]
        
        for m in range(0,M):
            N_eval = N_eval_m[m]
            new_m = np.linspace(start=0, stop=perc[m], num=N_eval).reshape(-1,1)

            result_arr_X_eval[pointer_new:(pointer_new+N_eval),:] = np.hstack((new_m,np.zeros((N_eval,1)),m*np.ones((N_eval,1)),labelList[m]*np.ones((N_eval,1))))
            pointer_new = pointer_new + N_eval

            result_arr_X_eval[pointer_new:(pointer_new+N_eval),:] = np.hstack((new_m,np.ones((N_eval,1)),m*np.ones((N_eval,1)),labelList[m]*np.ones((N_eval,1))))
            pointer_new = pointer_new + N_eval
            
        mu_eval, var_eval = my_MOGP.predict_f(result_arr_X_eval, full_cov=True)
        mu_eval = mu_eval.numpy()
        #sd_eval = np.sqrt(var_eval.numpy())
        sd_eval = np.sqrt(np.diag(var_eval[0]))
        
        IMSPE = []
        if showElastic:
            ESD = []
        else:
            ESD = None
        IUEA = []
    else:
        IMSPE = None
        ESD = None
        IUEA = None
    
    # Subset based on selected curve and parse the prediction result (idx)
    mu_x_List = []
    sd_x_List = []
    lci_x_List = []
    uci_x_List = []  
    mu_y_List = []
    sd_y_List = []
    lci_y_List = []
    uci_y_List = []
    
    pointer_new = int(0)
    pointer_eval = int(0)
    for m in range(0,M):
        data_m = datapointList[m]
        truth_m = truthpointList[m]
        new_m = predpointList[m]
        N_m = data_m.shape[1]
        N_pred_m = new_m.shape[0]
        if showEval:
            N_eval = N_eval_m[m]
        
        mu_x = mu[pointer_new:(pointer_new+N_pred_m),0]
        sd_x = sd[pointer_new:(pointer_new+N_pred_m)]
        mu_x_List.append(mu_x)
        sd_x_List.append(sd_x)
        
        cov_m = var[0][pointer_new:(pointer_new+2*N_pred_m), pointer_new:(pointer_new+2*N_pred_m)].numpy()
        cov_XY = np.array([cov_m[i,i+N_pred_m] for i in range(N_pred_m)])
        
        lci_x_List.append(mu_x-CI_mult*sd_x)
        uci_x_List.append(mu_x+CI_mult*sd_x)
        tmp_x_new = result_arr_X_new[pointer_new:(pointer_new+N_pred_m),0]
        if showEval:
            mu_X1 = mu_eval[pointer_eval:(pointer_eval+N_eval),0]
            sd_X1 = sd_eval[pointer_eval:(pointer_eval+N_eval)]
            
            cov_1 = var_eval[0][pointer_eval:(pointer_eval+2*N_eval), pointer_eval:(pointer_eval+2*N_eval)].numpy()
            cov_XY1 = np.array([cov_1[i,i+N_eval] for i in range(N_eval)])
            
            pointer_eval = pointer_eval + N_eval
        pointer_new = pointer_new + N_pred_m

        mu_y = mu[pointer_new:(pointer_new+N_pred_m),0]
        sd_y = sd[pointer_new:(pointer_new+N_pred_m)]
        mu_y_List.append(mu_y)
        sd_y_List.append(sd_y)
        
        lci_y_List.append(mu_y-CI_mult*sd_y)
        uci_y_List.append(mu_y+CI_mult*sd_y)
        tmp_y_new = result_arr_X_new[pointer_new:(pointer_new+N_pred_m),0]
        if showEval:
            mu_Y1 = mu_eval[pointer_eval:(pointer_eval+N_eval),0]
            sd_Y1 = sd_eval[pointer_eval:(pointer_eval+N_eval)]
            pointer_eval = pointer_eval + N_eval
        pointer_new = pointer_new + N_pred_m
        
        # Compute integrated MSPE, elastic shape distance, integrated uncertainty ellipsoid area
        if showEval:
            mupoint = np.vstack((mu_X1.reshape(1,-1),mu_Y1.reshape(1,-1)))
            IMSPE.append(np.mean((mupoint[0,:]-truth_m[0,:])**2 + (mupoint[1,:]-truth_m[1,:])**2)) 

            if showElastic:
                try:
                    ESD.append(cf.elastic_distance_curve(truth_m, mupoint, closed=1, scale=False))
                except ValueError:
                    ESD.append(float("inf"))

            IUEA.append(np.mean(np.pi*sd_X1*np.sqrt((sd_Y1**2)-((cov_XY1/sd_X1)**2))))
        
        if showPlot1:
            plt.figure(figsize=(20,5))

            # First coordinate fit with prediction interval
            plt.subplot(1,3,1)
            plt.scatter(tmp_x_new, mu_x, label='mean_x', c='k', s=15)
            plt.scatter(tmp_x_new, mu_x+CI_mult*sd_x, label='uci_x', c='r', s=15)
            plt.scatter(tmp_x_new, mu_x-CI_mult*sd_x, label='lci_x', c='b', s=15)
            plt.xlabel('arc-length')
            plt.ylabel('x-coordinate')
            plt.title('#new loc='+str(N_pred_m))

            # Second coordinate fit with prediction interval
            plt.subplot(1,3,2)
            plt.scatter(tmp_y_new, mu_y, label='mean_y', c='k', s=15)
            plt.scatter(tmp_y_new, mu_y+CI_mult*sd_y, label='uci_y', c='r', s=15)
            plt.scatter(tmp_y_new, mu_y-CI_mult*sd_y, label='lci_y', c='b', s=15)
            plt.xlabel('arc-length')
            plt.ylabel('y-coordinate')
            plt.title('#new loc='+str(N_pred_m))
            
            # Ellipse uncertainty quantification plot
            plt.subplot(1,3,3)
            plt.scatter(mu_x, mu_y, label='estimated mean', c='k', s=5)
            if truth_ind:
                plt.scatter(truth_m[0,:], truth_m[1,:], label='truth points', c='y', s=15)
            plt.scatter(data_m[0,:], data_m[1,:], label='observed points', c='r', s=15)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            for i in range(0,N_pred_m):
                e_cen_x = mu_x[i]
                e_cen_y = mu_y[i]
                e_rad_x = sd_x[i]
                e_rad_y = sd_y[i]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6)
                ax.add_patch(ellipse)
            plt.title('#observed samples='+str(N_m))
            
    if showPlot2:
        # Plot of curves colored by cluster label
        for label in set(labelList):
            n_col = 4
            idx_class = [labelList[idx]==label for idx in range(len(labelList))]
            idx_class = [i for i, x in enumerate(idx_class) if x]
            M = len(idx_class)
            n_row = int(np.ceil(M/n_col))
        
            plt.figure(figsize=(20,20))
            ax = plt.subplot(111)
            x_offset = 0.32+0.08*label   # horizontal spacing between curves (may need to adjust)
            y_offset = 0.35    # vertical spacing between curves (may need to adjust)
            plt.setp(ax, 'frame_on', False)
            ax.set_xlim([-0.53,n_col*x_offset])
            ax.set_ylim([-(n_row)*y_offset,y_offset])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid('off')

        #color_choices = ['red','green','blue','cyan','magenta','black']
        #colors = [color_choices[l-1] for l in labelList]
        
            idx = 0
            for k in np.arange(0,n_row):
                for l in np.arange(0,n_col):
                    ax.plot(mu_x_List[idx_class[idx]]+l*x_offset, mu_y_List[idx_class[idx]]-k*y_offset, c='k', linewidth=2, zorder=2)
                    if truth_ind:
                        ax.scatter(truthpointList[idx_class[idx]][0,:]+l*x_offset, truthpointList[idx_class[idx]][1,:]-k*y_offset, c='y', s=15, alpha=0.7, zorder=1)
                    ax.scatter(datapointList[idx_class[idx]][0,:]+l*x_offset, datapointList[idx_class[idx]][1,:]-k*y_offset, c='r', s=15, zorder=3)
                    ax = plt.gca()
                    ax.set_aspect('equal', adjustable='box')
                    for i in range(0,N_pred_m):
                        e_cen_x = mu_x_List[idx_class[idx]][i]
                        e_cen_y = mu_y_List[idx_class[idx]][i]
                        e_rad_x = sd_x_List[idx_class[idx]][i]
                        e_rad_y = sd_y_List[idx_class[idx]][i]
                        ellipse = Ellipse(xy=(e_cen_x+l*x_offset,e_cen_y-k*y_offset), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.3)
                        ax.add_patch(ellipse)
                    idx = idx+1
                    if idx==M:
                        break
            
    if est_period:            
        return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, period_param, IMSPE, ESD, IUEA
    else:
        return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, IMSPE, ESD, IUEA