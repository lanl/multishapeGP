__author__ = "Hengrui Luo, Justin Strait"
__copyright__ = "© 2023. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so."
__license__ = "BSD-3"
__version__ = "1"
__maintainer__ = "Justin Strait (CCS-6)"
__email__ = "jstrait@lanl.gov"

##############################
## Landmarking from GP fits ##
##############################
from utils import *
from curveGP import *
from scipy.optimize import basinhopping

print('landmarkGP VERSION: last updated 2023-11-27')

## Landmarking based on single-output GP fit for a single curve
# Function to evaluate MSPE for single-output GP fits (used internally in curveSOGP_LM)
# lambda_param controls the weight given to the MSPE for the x-coordinate fit
def MSPE_SOGP(param, m_x, m_y, lambda_param):
    if np.isscalar(param):
        param = np.array([param])
        take_first = True
    else:
        param = param.reshape(-1,1)
        take_first = False
    if param.shape[0]<=1:
        param = np.vstack((param,param))

    mu_x, var_x = m_x.predict_f(param)
    mu_y, var_y = m_y.predict_f(param)

    MSPE = lambda_param*var_x.numpy() + (1-lambda_param)*var_y.numpy()
    MSPE = MSPE.ravel().reshape(-1,1)

    if take_first:
        MSPE = MSPE[0]
    return MSPE

# Primary function for sequential landmarking using single-output GP fit of a single curve
def curveSOGP_LM(datapoint, param=None, predpoint=100, truthpoint=None, cen=True, scale=True,  CI_mult=1, period_mult=1.0, 
                 est_period=False, restrictLS=True, use_Scipy='L-BFGS-B', maxiter=1000, kernel='Matern32', nug=1e-4, 
                 autoEnclose=0, lambda_param=0.5, lmk_base=1, showPlot=True, showMSPE=False, showOutput=False, showEval=True, 
                 restrict_nv=False):
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
    # lambda_param = weight attributed to x-coordinate function for computing weighted MSPE (default = 0.5)
    # lmk_base = map estimated landmark to specific curve;
    #   0 means with respect to estimated mean curve (after GP fit);
    #   1 means with respect to truthpoint (default)
    #   2 means with respect to both 0 and 1
    # showPlot = indicator to show figures (default = True)
    # showMSPE = indicator to show plot of MSPE as a function of the parameter value based on observed sample points, used in choosing the next landmark (default = False)
    # showOutput = indicator to show convergence and GP fit output (default = False)
    # showEval = indicator to show evaluation metrics on plots (default = True)
    # restrict_nv = indicator to restrict the noise variance hyperparameter estimation to the interval (10e-6, 10e-4) (default = False)
    #   (this is discussed in the paper linked in the main README file, Supplementary Materials Section "Practical Numerical Issues")
      
    ## Outputs ##
    # lmk_param = estimated next landmark on curve
    # lmk = corresponding (x,y)-coordinates for lmk_param on curve
    # mspe_sogp = weighted MSPE computed at all parameter values in predpoint
    #################################################################################
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpoint, type(None)):
        truthpoint = datapoint
        truth_ind = False
    else:
        truth_ind = True

    # Pre-process curves (register, re-scale, center if desired)
    if cen or scale:
        if truth_ind:  # true curve exists
            truthpoint, c, scl, beta_shift, O = curveBatch_preproc(truthpoint, cen, scale, rot=False)
            if scale:
                datapoint = datapoint/scl
            if cen:
                N = datapoint.shape[1]
                datapoint = datapoint-np.tile(c,(N,1)).T
        else:
            datapoint, c, scl, beta_shift, O = curveBatch_preproc(datapoint, cen, scale, rot=False)
            truthpoint = datapoint
    
    # Fit SOGP model (suppresses plots and outputs by default)
    #cen = False
    #scale = False
    if est_period:
        my_SOGP_X, my_SOGP_Y, opt_logs_X, opt_logs_Y, mu_Xn, sd_Xn, mu_Yn, sd_Yn, perc, period_param, IMSPE, ESD, IUEA = curveSOGP(datapoint, param, predpoint, truthpoint, cen, scale, CI_mult, period_mult, est_period, restrictLS, use_Scipy, maxiter, kernel, nug, autoEnclose, showPlot=False, showOutput=False, showEval=showEval, restrict_nv=restrict_nv, showCoord=False)
    else:
        my_SOGP_X, my_SOGP_Y, opt_logs_X, opt_logs_Y, mu_Xn, sd_Xn, mu_Yn, sd_Yn, perc, IMSPE, ESD, IUEA = curveSOGP(datapoint, param, predpoint, truthpoint, cen, scale, CI_mult, period_mult, est_period, restrictLS, use_Scipy, maxiter, kernel, nug, autoEnclose, showPlot=False, showOutput=False, showEval=showEval, restrict_nv=restrict_nv, showCoord=False)
    
    # Optimize MSPE over each marginal curve for landmarking
    # basinhopping class for optimization
    class MyBounds(object):
        def __init__(self, xmax=perc, xmin=0):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin 
                      
    # Bounds and initialization for optimization
    mybounds = MyBounds(xmax=perc)
    x0 = perc*0.5
    
    # Predict at new input locations, with uncertainty quantified by predictive variance/sd
    if isinstance(predpoint, int):
        predloc = np.linspace(start=0, stop=perc, num=predpoint).reshape(-1,1)
    else:
        predloc = predpoint.reshape(-1,1)
    
    # Negative MSPE function to be minimized
    mspe_sogp = MSPE_SOGP(predloc, my_SOGP_X, my_SOGP_Y, lambda_param)
    
    def nMSPE_SOGP(param):
        return -MSPE_SOGP(param, my_SOGP_X, my_SOGP_Y, lambda_param)[0]
    
    # Optimization using basinhopping
    resb = basinhopping(nMSPE_SOGP, x0, niter=100, T=1.0, stepsize=1, accept_test=mybounds)
    lmk_param = resb.x
        
    # Convert to (x,y)-coordinates
    if lmk_base==0:
        # With respect to estimated mean curve
        mu_stack = np.vstack((mu_Xn.reshape(1,-1),mu_Yn.reshape(1,-1)))
        lmk = arc_to_xy_param_new(lmk_param,mu_stack)
    elif lmk_base==1:
        # With respect to true curve
        lmk = arc_to_xy_param_new(lmk_param,truthpoint)
    else:
        # With respect to both estimated mean curve and true curve
        mu_stack = np.vstack((mu_Xn.reshape(1,-1),mu_Yn.reshape(1,-1)))
        lmk1 = arc_to_xy_param_new(lmk_param,mu_stack)   
        lmk2 = arc_to_xy_param_new(lmk_param,truthpoint)
        
    if showPlot:
        N = datapoint.shape[1]
        N_new = predloc.shape[0]
        
        plt.rcParams["font.family"] = "serif"
        from matplotlib.ticker import FormatStrFormatter
        
        if showMSPE:
            fig, (ax0,ax1) = plt.subplots(1,2, figsize=(15,5))
            
            # Energy landscape plot
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
            ax0.plot(predloc.reshape(-1,1), mspe_sogp,c='b')
            ax0.vlines(x=lmk_param, ymin=0, ymax=MSPE_SOGP(lmk_param,my_SOGP_X,my_SOGP_Y,lambda_param)[0], colors='r')
            ax0.set_xlabel("t", size=17)
            ax0.set_ylabel("MSPE", size=17)
            ax0.tick_params(axis='x', labelsize=15)
            ax0.tick_params(axis='y', labelsize=15)
            
            # Ellipse uncertainty quantification plot with next landmark placed
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.scatter(mu_Xn,mu_Yn,c='k',s=5, zorder=2)
            if truth_ind:
                ax1.scatter(truthpoint[0,:],truthpoint[1,:],c='y', s=15, zorder=1)
            ax1.scatter(datapoint[0,:],datapoint[1,:],c='r',s=15, zorder=3)
            for j in range(0,N_new):
                e_cen_x = mu_Xn[j]
                e_cen_y = mu_Yn[j]
                e_rad_x = sd_Xn[j]
                e_rad_y = sd_Yn[j]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', 
                                  lw=2, alpha=0.4)
                ax1.add_patch(ellipse)
            ax1.tick_params(axis='x', labelsize=15)
            ax1.tick_params(axis='y', labelsize=15)
            if lmk_base!=0 and lmk_base!=1:
                ax1.scatter(lmk1[0],lmk1[1],label='next lmk wrt mean',c='g',s=100)
                ax1.scatter(lmk2[0],lmk2[1],label='next lmk wrt truth',c='c',s=100)
                #ax1.legend(loc="upper right")
            else:
                ax1.scatter(lmk[0],lmk[1],c='g',s=100)
                #plt.legend(loc="upper right")
            
            if showEval:
                ax1.set_title(f'IMSPE={IMSPE:.2e}/ESD={ESD:.3f}/IUEA={IUEA:.2e}', fontsize=20)
            else:
                ax1.set_title('#observed samples='+str(N), fontsize=20)
        else:
            fig, ax = plt.subplots(figsize=(7,5))
            
            # Ellipse uncertainty quantification plot with next landmark placed
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.scatter(mu_Xn,mu_Yn,c='k',s=5, zorder=2)
            if truth_ind:
                ax.scatter(truthpoint[0,:],truthpoint[1,:],c='y',s=15, zorder=1)
            ax.scatter(datapoint[0,:],datapoint[1,:],c='r',s=15, zorder=3)
            #ax = plt.gca()
            for j in range(0,N_new):
                e_cen_x = mu_Xn[j]
                e_cen_y = mu_Yn[j]
                e_rad_x = sd_Xn[j]
                e_rad_y = sd_Yn[j]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', 
                                  lw=2, alpha=0.4)
                ax.add_patch(ellipse)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            if lmk_base!=0 and lmk_base!=1:
                ax.scatter(lmk1[0],lmk1[1],label='next lmk wrt mean',c='g',s=100)
                ax.scatter(lmk2[0],lmk2[1],label='next lmk wrt truth',c='c',s=100)
                #ax.legend(loc="upper right")
            else:
                ax.scatter(lmk[0],lmk[1],c='g',s=100)
                #plt.legend(loc="upper right")
            
            if showEval:
                ax.set_title(f'IMSPE={IMSPE:.2e}/ESD={ESD:.3f}/IUEA={IUEA:.2e}', fontsize=20)
            else:
                ax.set_title('#observed samples='+str(N), fontsize=20)            
       
    if lmk_base!=0 and lmk_base!=1:
        lmk = []
        lmk.append(lmk1)
        lmk.append(lmk2)
    
    if showOutput:
        print(lmk_param)
        print(lmk)
    
    return lmk_param, lmk, mspe_sogp

## Landmarking based on multiple-output GP fit using two-dimensional encoding
# Function to evaluate MSPE for multiple-output GP fits (used internally in curveBatchMOGP_LM)
# lambda_param controls the weight given to the MSPE for the x-coordinate fit
def MSPE_MOGP(param, m, curve_idx, lambda_param):
    # Input either a single number, or array of form np.array([a,b,c])
    if np.isscalar(param):
        param = np.array([param])
        take_first = True
    else:
        param = param.reshape(-1,1)
        take_first = False
    if param.shape[0]<=1:
        param = np.vstack((param,param))

    # Form matching matrix for selected curve in order to predict    
    N_new = param.shape[0]
    param_mat = np.zeros((2*N_new,3))
    
    param_mat[0:N_new,:] = np.hstack((param,np.zeros((N_new,1)),curve_idx*np.ones((N_new,1))))
    param_mat[N_new:(2*N_new),:] = np.hstack((param,np.ones((N_new,1)),curve_idx*np.ones((N_new,1))))
    mu, var = m.predict_f(param_mat)
    mu = mu.numpy()
    var = var.numpy()

    MSPE = lambda_param*var[0:N_new,0] + (1-lambda_param)*var[N_new:(2*N_new),0]
    MSPE = MSPE.ravel().reshape(-1,1)

    if take_first:
        MSPE = MSPE[0]
    return MSPE, mu

# Primary function for sequential landmarking using multiple-output GP fit for a single or multiple curves
def curveBatchMOGP_LM(datapointList, paramList=None, predList=100, truthpointList=None, cen=True, scale=True, rot=True, 
                      CI_mult=1, period_mult=1.0, est_period=False, restrictLS=True, use_Scipy='L-BFGS-B', maxiter=1000, 
                      kernel='Matern32', nug=1e-4, autoEnclose=0, lambda_param=0.5, lmk_base=1, showPlot=True, showMSPE=False,
                      showOutput=False, restrict_nv=False):
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
    # lambda_param = weight attributed to x-coordinate functions for computing weighted MSPE (default = 0.5)
    # lmk_base = map estimated landmark to specific curve;
    #   0 means with respect to estimated mean curve (after GP fit);
    #   1 means with respect to truthpoint (default)
    #   2 means with respect to both 0 and 1
    # showPlot = indicator to show figures (default = True)
    # showMSPE = indicator to show plot of MSPE as a function of the parameter value based on observed sample points, used in choosing the next landmark (default = False)
    # showOutput = indicator to show convergence and GP fit output (default = False)
    # restrict_nv = indicator to restrict the noise variance hyperparameter estimation to the interval (10e-6, 10e-4) (default=False)
    #   (this is discussed in the paper linked in the main README file, Supplementary Materials Section "Practical Numerical Issues")
    
    ## Outputs ##
    # lmk_param = estimated next landmark on curves
    # lmk = corresponding (x,y)-coordinates for lmk_param on curves
    # mspe_mogp = weighted MSPE computed at all parameter values in predList
    #################################################################################
    # Initialize
    lmk_param = []
    lmk = []
    if lmk_base!=0 and lmk_base!=1:
        lmk1 = []
        lmk2 = []
    mspe_mogp = []
    
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpointList, type(None)):
        truthpointList = datapointList
        truth_ind = False
    else:
        truth_ind = True
        
    # datapointList is a list of length m1, each entry in this list represents sample points from curves
    # truthpointList should be a list of length m1 as well
    # predList should be a list of length m1 as well, represented as one-dimensional cols (e.g., shape=(n_pred,))
    m1 = len(datapointList)  # number of curves
    m2 = len(truthpointList)
    if m1!=m2:
        print('ERROR: datapointList and truthpointList must be two lists of the same length')
        return None

    # Pre-process curves (register, re-scale, center if desired)
    if cen or scale or rot:
        if truth_ind:  # true curves exist
            truthpointList, c, scl, beta_shift, O = curveBatch_preproc(truthpointList, cen, scale, rot)
            if scale:
                datapointList = [datapointList[i]/scl[i] for i in range(0,m1)]
            if cen:
                N = [datapointList[i].shape[1] for i in range(0,m1)]
                datapointList = [datapointList[i]-np.tile(c[i],(N[i],1)).T for i in range(0,m1)]
            if rot:
                datapointList = [O[:,:,i]@datapointList[i] for i in range(0,m1)]
        else:
            datapointList, c, scl, beta_shift, O = curveBatch_preproc(datapointList, cen, scale, rot)
            truthpointList = datapointList
            
    # Fit MOGP model (suppresses plots and outputs by default)
    #cen = False
    #scale = False
    #rot = False
    if est_period:
        my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, period_param, IMSPE, ESD, IUEA = \
            curveBatchMOGP(datapointList, paramList, predList, truthpointList, cen, scale, rot, CI_mult, period_mult, 
                           est_period, restrictLS, use_Scipy, maxiter, kernel, nug, autoEnclose, showPlot=False, 
                           showOutput=False, restrict_nv=restrict_nv, showCoord=False)        
    else:
        my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, IMSPE, ESD, IUEA = \
            curveBatchMOGP(datapointList, paramList, predList, truthpointList, cen, scale, rot, CI_mult, period_mult, 
                           est_period, restrictLS, use_Scipy, maxiter, kernel, nug, autoEnclose, showPlot=False, 
                           showOutput=False, restrict_nv=restrict_nv, showCoord=False)
    
    # basinhopping class for optimization
    class MyBounds(object):
        def __init__(self, xmax=1, xmin=0):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin 
    
    # Create augmented data matrix of inputs and outputs (with labels in the second dimension which are separate for each curve 
    # and dimension)
    if isinstance(predList, int):
        predpointList = []
        for i in range(0,m1):
            tmp = np.linspace(start=0, stop=perc[i], num=predList).reshape(-1,1)
            predpointList.append(tmp)
    else:
        predpointList = predList
    
    # Optimize MSPE over each marginal curve for landmarking
    for i in range(0,m1):
        # Bounds and initialization for optimization
        mybounds = MyBounds(xmax=perc[i])
        x0 = perc[i]*0.5
    
        # Negative MSPE function to be minimized
        tmp = MSPE_MOGP(predpointList[i], my_MOGP, i, lambda_param)
        mspe_mogp.append(tmp[0])
    
        def nMSPE_MOGP(param):
            return -MSPE_MOGP(param, my_MOGP, i, lambda_param)[0][0]
    
        # Optimization using basinhopping
        resb = basinhopping(nMSPE_MOGP, x0, niter=100, T=1.0, stepsize=1, accept_test=mybounds)
        lmk_param.append(resb.x)
        
        # Convert to (x,y)-coordinates
        if lmk_base==0:
            # With respect to estimated mean curve
            mu_stack = np.vstack((mu_x_List[i].reshape(1,-1),mu_y_List[i].reshape(1,-1)))
            tmp = arc_to_xy_param_new(lmk_param[i],mu_stack)
            lmk.append(tmp)
        elif lmk_base==1:
            # With respect to true curve
            tmp = arc_to_xy_param_new(lmk_param[i],truthpointList[i])
            lmk.append(tmp)
        else:
            # With respect to both estimated mean curve and true curve
            mu_stack = np.vstack((mu_x_List[i].reshape(1,-1),mu_y_List[i].reshape(1,-1)))
            tmp = arc_to_xy_param_new(lmk_param[i],mu_stack)
            lmk1.append(tmp)
            tmp = arc_to_xy_param_new(lmk_param[i],truthpointList[i])
            lmk2.append(tmp)

    if showPlot:
        plt.rcParams["font.family"] = "serif"
        from matplotlib.ticker import FormatStrFormatter
        
        ctr = int(1)
        for i in range(0,m1):
            data_i = datapointList[i]
            truth_i = truthpointList[i]
            new_i = predpointList[i]
            if lmk_base!=0 and lmk_base!=1:
                lmk1_i = lmk1[i]
                lmk2_i = lmk2[i]
            else:
                lmk_i = lmk[i]
            mu_x = mu_x_List[i]
            mu_y = mu_y_List[i]
            sd_x = sd_x_List[i]
            sd_y = sd_y_List[i]
            N_i = data_i.shape[1]
            Ni_new = new_i.shape[0]
            
            if showMSPE:              
                fig, (ax0,ax1) = plt.subplots(1,2, figsize=(15,5))
                
                # Energy landscape plot
                ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                ax0.plot(predpointList[i].reshape(-1,1),mspe_mogp[i],c='b')
                ax0.vlines(x=lmk_param[i], ymin=0, ymax=MSPE_MOGP(lmk_param[i],my_MOGP,i,lambda_param)[0][0], colors='r')
                ax0.set_xlabel("t", size=17)
                ax0.set_ylabel("MSPE", size=17)
                ax0.tick_params(axis='x', labelsize=15)
                ax0.tick_params(axis='y', labelsize=15)

                # Ellipse uncertainty quantification plot with next landmark placed
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax1.scatter(mu_x,mu_y,c='k',s=5, zorder=2)
                if truth_ind:
                    ax1.scatter(truth_i[0,:],truth_i[1,:],c='y',s=15, zorder=1)
                ax1.scatter(data_i[0,:],data_i[1,:],c='r',s=15, zorder=3)
                for j in range(0,Ni_new):
                    e_cen_x = mu_x[j]
                    e_cen_y = mu_y[j]
                    e_rad_x = sd_x[j]
                    e_rad_y = sd_y[j]
                    ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', 
                                      fc='None', lw=2, alpha=0.4)
                    ax1.add_patch(ellipse)
                ax1.tick_params(axis='x', labelsize=15)
                ax1.tick_params(axis='y', labelsize=15)
                if lmk_base!=0 and lmk_base!=1:
                    ax1.scatter(lmk1_i[0],lmk1_i[1],label='next lmk wrt mean',c='g',s=100)
                    ax1.scatter(lmk2_i[0],lmk2_i[1],label='next lmk wrt truth',c='c',s=100)
                    #plt.legend(loc="upper right")
                else:
                    ax1.scatter(lmk_i[0],lmk_i[1],c='g',s=100)
                    ax1.set_title(f'IMSPE={IMSPE[i]:.2e}/ESD={ESD[i]:.3f}/IUEA={IUEA[i]:.2e}', fontsize=20)
                    #plt.title('#observed samples='+str(N_i))
                    #plt.legend(loc="upper right")
            else:
                # Ellipse uncertainty quantification plot with next landmark placed
                fig, ax = plt.subplots(figsize=(7,5))
                ax.scatter(mu_x,mu_y,c='k',s=5, zorder=2)
                if truth_ind:
                    ax.scatter(truth_i[0,:],truth_i[1,:],c='y',s=15, zorder=1)
                ax.scatter(data_i[0,:],data_i[1,:],c='r',s=15, zorder=3)
                for j in range(0,Ni_new):
                    e_cen_x = mu_x[j]
                    e_cen_y = mu_y[j]
                    e_rad_x = sd_x[j]
                    e_rad_y = sd_y[j]
                    ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', 
                                      fc='None', lw=2, alpha=0.4)
                    ax.add_patch(ellipse)
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                if lmk_base!=0 and lmk_base!=1:
                    ax.scatter(lmk1_i[0],lmk1_i[1],label='next lmk wrt mean',c='g',s=100)
                    ax.scatter(lmk2_i[0],lmk2_i[1],label='next lmk wrt truth',c='c',s=100)
                    #plt.legend(loc="upper right")
                else:
                    ax.scatter(lmk_i[0],lmk_i[1],c='g',s=100)
                    ax.set_title(f'IMSPE={IMSPE[i]:.2e}/ESD={ESD[i]:.3f}/IUEA={IUEA[i]:.2e}', fontsize=20)
                    #plt.title('#observed samples='+str(N_i))
                    #plt.legend(loc="upper right")

            ctr = ctr + 1
            
    if lmk_base!=0 and lmk_base!=1:
        lmk.append(lmk1)
        lmk.append(lmk2)
    
    if showOutput:
        print(lmk_param)
        print(lmk)
    
    return lmk_param, lmk, mspe_mogp