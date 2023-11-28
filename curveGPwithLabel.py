__author__ = "Hengrui Luo, Justin Strait"
__copyright__ = "© 2023. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so."
__license__ = "BSD-3"
__version__ = "1"
__maintainer__ = "Justin Strait (CCS-6)"
__email__ = "jstrait@lanl.gov"

######################################
## Multiple-Output GP Model Fitting ##
######################################
from utils import *
print('curveGPwithLabel VERSION: 2022-01-15')

import gpflow
import tensorflow as tf
print(tf.__version__)

import tensorflow_probability as tfp
print(tfp.__version__)

from gpflow.kernels import Coregion, SquaredExponential, Matern12, Matern32, Matern52
from gpflow.utilities import print_summary

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gpflow import set_trainable
from tensorflow_probability import bijectors as tfb

import fdasrsf as fs
print(fs.__version__)


## Function for bounding length scale hyperparameter when optimizing
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

## Single-output GP fit for a single curve
# Justin updated 12/6/21
## Multiple-output GP fit for multiple curves using two-dimensional encoding with option for registration
# Justin updated 12/6/21
def curveBatchMOGPwithLabel(datapointList,labelList, N_pred=100, truthpointList=None, preproc=True, warp=True, scale=True, cen=True, resamp=False, N_reg=100,
                   CI_mult=1.96, PERIOD_mult=1.0, estimatePeriod=False, restrictLS=True, use_Scipy='L-BFGS-B', n_maxiter=1000,
                   kernel='Matern32', nug=1e-4, autoEnclose=0, showPlot=True, showOutput=True):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapointList = list of length m1, each entry has sample points from observed curves (d x N_i for curves i=1,...,m1)
    # labelList = list of length m1, each entry has 1 clustering label information corresponding to the sample points from observed curves (a scalar for different curves i=1,...,m1), but there are at most m1 distinct values.
    # (deprecated)predList = list of length m1, each entry has parameter values to predict GP fit at (1 x N_newi for curves i=1,...,m1)
    
    # OPTIONAL:
    # truthpointList = list of length m1, each entry has sample points from TRUE curves (d x N_truei for curves i=1,...m1)
    #   (if not available, this is set to be the same as the sample points from observed curves, datapointList)
    # CI_mult = multiplier for prediction intervals (default = 1.96)
    # PERIOD_mult = period specified for periodic covariance kernel used to model between-point dependence (default = 1.0)
    # estimatePeriod = do we want to estimate the periodicity, if True, this would overrides the CI_mult and PERIOD_mult parameters (default = False)
    # use_Scipy = used to specify optimization method; if 'Adam', Adam is used;
    #   otherwise, specify a Scipy optimization method from scipy.optimize.minimize function (default = L-BFGS-B in Scipy)
    # kernel = pre-specified kernel object for input covariance, choices below:
    #   'Matern12', 'Matern32' (default), 'Matern52', 'SqExp' (squared exponential), or can input other desired kernel from GPflow package
    # nug = indicator to include nugget in input covariance specification (default = True)
    # autoEnclose = indicator to augment the sparsely sampled points so that the first and the last point in the sample are the same; 
    #   1 means fill the last point with first point; 
    #   -1 means fill the first point with last point; 
    #   0 means do nothing, note that this works for both datapointList and truthpointList (default = 0)
    # showPlot = indicator to show figures (default = True)
    # showOutput = indicator to show GP fit output (default = True)
    
    ## Outputs ##
    # my_MOGP = fitted GP object
    # opt_logs = optimization details for GP fitting
    # mu_x_List, mu_y_List = lists of length m1, each entry has predicted mean function in each dimension
    # sd_x_List, sd_y_List = lists of length m1, each entry has predicted standard deviation function in each dimension
    # result_arr_X_new = input matrix for prediction in format suitable for coregionalization, useful to feed into other functions
    # perc = estimated arc-lengths of each curve (with respect to truthpointList)
    #################################################################################   
    # Use true curve as reference (if available, e.g., for simulations)
    if isinstance(truthpointList, type(None)) or warp:
        # Should not use true curve information if observed curves are registered
        truthpointList = datapointList
        truth_ind = False
    else:
        truth_ind = True
        
    # Number of curves
    m1 = len(datapointList)
    m2 = len(truthpointList)
    if m1!=m2:
        print('ERROR: datapointList and truthpointList must be two lists of the same length')
        return None
        
    # Curve dimension (function only works for d=2)
    d = datapointList[0].shape[0]
    if d!=2:  
        print('ERROR: Data format is not 2 by _ array within list')
        return None
    
    # Pre-process curves (register, re-scale, center if desired)
    if preproc:
        if warp:
            datapointList, gam = curveBatch_preproc(datapointList, warp, scale, cen, resamp, N_reg)
            truthpointList = datapointList  # don't use true curve info if performing registration
        else:
            datapointList = curveBatch_preproc(datapointList, warp, scale, cen, resamp, N_reg)
            truthpointList = curveBatch_preproc(truthpointList, warp, scale, cen, resamp, N_reg)
    
    # Compute curve arc-lengths (to use as periodicity and for prediction)
    Ntotal = 0
    perc = []
    
    for i in range(0,m1):
        data_i = datapointList[i]
        truth_i = truthpointList[i]
        Ntotal = Ntotal + data_i.shape[1]  # total number of sampling points across all curves
        perc.append(xy_length(truth_i))
        
    if showOutput:
        print('lengths of curves', perc)
    
    # Set default jitter level
    gpflow.config.set_default_jitter(1e-4)

    # Specify base kernel on input space, if estimatePeriod==True, leave period parameter blank instead of plug in an estimate.
    if kernel=='Matern12':
        if estimatePeriod:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern12(active_dims=[0]), period=PERIOD_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern32':
        if estimatePeriod:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern32(active_dims=[0]), period=PERIOD_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='Matern52':
        if estimatePeriod:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(Matern52(active_dims=[0]), period=PERIOD_mult*max(perc))
            set_trainable(my_kern.period, False)
    elif kernel=='SqExp':
        if estimatePeriod:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]))
        else:
            my_kern = gpflow.kernels.Periodic(SquaredExponential(active_dims=[0]), period=PERIOD_mult*max(perc))
            set_trainable(my_kern.period, False)
    else:
        my_kern = kernel
    
    # Add white noise or constant kernel if nugget is included
    if nug==None:
        my_kern = my_kern
    elif nug>0:
        my_kern = my_kern + gpflow.kernels.Constant(variance=nug)
        set_trainable(my_kern.kernels[1], False)
    elif nug <= 0:
        my_kern = my_kern + gpflow.kernels.White(variance=abs(nug))
        set_trainable(my_kern.kernels[1], False)
    
    # Constrain length scale parameters to be between min_ls and max_ls
    print('>>restrictLS=', restrictLS)
    print('>>periodic=', perc)
    if restrictLS:
        min_ls = 0
        max_ls = 0.5*PERIOD_mult*max(perc)
        init_ls = 0.5*(min_ls+max_ls)

        if nug==None:
            my_kern.base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
        else:
            my_kern.kernels[0].base_kernel.lengthscales = bounded_lengthscale(min_ls, max_ls, init_ls)
    
    # Create augmented data matrix of inputs and outputs (with labels in the second dimension which are separate for each curve 
    # and dimension)
    result_arr_X = np.zeros((d*Ntotal,3))
    result_arr_Y = np.zeros((d*Ntotal,3)) 
    result_arr_X_new = np.zeros((d*N_pred*m1,3))
    
    pointer = int(0)
    pointer_new = int(0)
    
    predList = []
    for i in range(0,m1):
        data_i = datapointList[i]
        truth_i = truthpointList[i]
        new_i = np.linspace(start=0, stop=perc[i], num=N_pred).reshape(-1,1)
        
        if autoEnclose>=1:
            # The last point is ignored and filled in by the first point
            if (data_i[:,0] != data_i[:,-1]).all():
                data_i[:,-1] = data_i[:,0].copy()
            if (truth_i[:,0] != truth_i[:,-1]).all():
                truth_i[:,-1] = truth_i[:,0].copy()
        elif autoEnclose<=-1:
            # The first point is ignored and filled in by the last point
            if (data_i[:,0] != data_i[:,-1]).all():
                data_i[:,0] = data_i[:,-1].copy()
            if (truth_i[:,0] != truth_i[:,-1]).all():
                truth_i[:,0] = truth_i[:,-1].copy()

        N_i = data_i.shape[1]
        X_i = data_i[0,:]
        Y_i = data_i[1,:]
        
        # Convert data to arc-length parameterization
        tmp = np.array(xy_to_arc_param_new(data_i,truth_i)).reshape(-1,1)
        tmp[-1] = xy_length(truth_i)
        
        # Map with respect to warping functions (if applicable)
        if warp:
            from scipy.interpolate import interp1d
            grid = np.linspace(start=0, stop=perc[i], num=len(gam[i]))        
            int_reg = interp1d(grid, perc[i]*gam[i], kind='cubic')
            tmp = int_reg(np.squeeze(tmp.reshape(1,-1))).reshape(-1,1)
            tmp[0] = 0
            tmp[-1] = perc[i]
            new_i = int_reg(new_i.reshape(1,-1)).reshape(-1,1)
            new_i[0] = 0
            new_i[-1] = perc[i]
            
        predList.append(new_i)
        
        result_arr_X[pointer:(pointer+N_i),:] = np.hstack((tmp,np.zeros((N_i,1)),i*np.ones((N_i,1))))
        result_arr_Y[pointer:(pointer+N_i),:] = np.hstack((data_i[0,:].reshape(-1,1),np.zeros((N_i,1)),i*np.ones((N_i,1))))
        pointer = pointer + N_i

        result_arr_X[pointer:(pointer+N_i),:] = np.hstack((tmp,np.ones((N_i,1)),i*np.ones((N_i,1))))
        result_arr_Y[pointer:(pointer+N_i),:] = np.hstack((data_i[1,:].reshape(-1,1),np.ones((N_i,1)),i*np.ones((N_i,1))))
        pointer = pointer + N_i
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred),:] = np.hstack((new_i,np.zeros((N_pred,1)),i*np.ones((N_pred,1))))
        pointer_new = pointer_new + N_pred
        
        result_arr_X_new[pointer_new:(pointer_new+N_pred),:] = np.hstack((new_i,np.ones((N_pred,1)),i*np.ones((N_pred,1))))
        pointer_new = pointer_new + N_pred
    
    # Set up coregionalization kernels across dimensions and curves
    OUTPUT_DIM = len(np.unique(result_arr_Y[:,1]))*len(np.unique(result_arr_Y[:,2]))  # number of unique likelihoods
    # W_NCOL1 = len(np.unique(result_arr_Y[:,1]))  # number of dimensions within curve (for coregionalization)
    coreg1 = Coregion(output_dim=OUTPUT_DIM, rank=1, active_dims=[1])
    kern = my_kern*coreg1
    
    if m1>1:
        W_NCOL2 = len(np.unique(result_arr_Y[:,2]))  # number of curves (for coregionalization)
        coreg2 = Coregion(output_dim=OUTPUT_DIM, rank=W_NCOL2, active_dims=[2])
        kern = kern*coreg2
    else:
        # Remove curve-level column from input/output matrices
        result_arr_X = np.delete(result_arr_X, 2, 1)
        result_arr_Y = np.delete(result_arr_Y, 2, 1)
        result_arr_X_new = np.delete(result_arr_X_new, 2, 1)
    
    #If clustering label information is available, then we append an additive penalizing term into the kernel object kern. 
    if len(labelList) > 0:
        assert len(labelList) == len(datapointList) #We must match the number of curves and number of labels
        assert m1>1 #There is no point to use cluster label if there is only one curve.
        #Must match the datapoint list
        #Augment the fourth column as the label column.
        #The first column is the coordinate of the points from the curve, either X or Y, active_dims = [0]
        #The second column is the first coregionalization, accounting the dependency between X and Y for the same curve,  active_dims = [1]
        #The third column is the second coregionalization, accounting the dependency between different curves,  active_dims = [2]
        #The (newly augmented) fourth column is an independent additive component, which enhances the correlation within one group.  active_dims = [3]
        #Prepare the label column
        result_arr_label = np.zeros((d*Ntotal,1))
        result_arr_label_new = np.zeros((d*N_pred*m1,1))
        pointer = 0
        for i in range(m1):
            #print('brewing label:',pointer,(pointer+N_i))
            result_arr_label[pointer:(pointer+d*N_i),0] = labelList[i]#.reshape(-1,1)
            pointer = pointer + d*N_i
            
        pointer_new = 0
        for j in range(m1):
            #print('brewing label new:',pointer_new,(pointer_new+d*N_pred))
            result_arr_label_new[pointer_new:(pointer_new+d*N_pred),0] = labelList[j]
            pointer_new = pointer_new + d*N_pred
            
        result_arr_X = np.hstack((result_arr_X,result_arr_label))
        result_arr_Y = np.hstack((result_arr_Y,result_arr_label))
        
        result_arr_X_new = np.hstack((result_arr_X_new,result_arr_label_new))
        #Put an additive component into the kernel that only works on the label column
        ##print(result_arr_X.shape,result_arr_X)
        ##print(result_arr_X_new.shape,result_arr_X_new)
        #import matplotlib.pyplot as plt
        #plt.matshow(result_arr_X)
        #plt.matshow(result_arr_X_new)
        
        print('----------')
        
        assert result_arr_X.shape[1]==4
        #By default we use Matern12 kernel.
        kern = kern + Matern12(active_dims=[3])
        print('\n Using kernel on label column:\n',kern)
        
    # Optimization for MOGP fitting
    if use_Scipy!='Adam':
        # Optimize using Scipy
        opt = gpflow.optimizers.Scipy()
        option_opt = dict(disp=True, maxiter=n_maxiter)
        my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        opt_logs = opt.minimize(my_MOGP.training_loss, my_MOGP.trainable_variables, method=use_Scipy, options=option_opt)
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print(opt_logs)
            print_summary(my_MOGP,fmt='notebook')

            likelihood = my_MOGP.log_marginal_likelihood()
            tf.print(f"Optimizer: {use_Scipy} loglik_marg: {likelihood: .04f}")
            
            print(coreg1.output_covariance().numpy())
            if m1>1:
                print(coreg2.output_covariance().numpy())
    else:
        # Optimize using Adam
        opt_logs = []
        
        adam_learning_rate = 0.03  
        opt = tf.optimizers.Adam(adam_learning_rate)
        my_MOGP = gpflow.models.GPR(data=(result_arr_X,result_arr_Y), kernel=kern)
        
        for i in range(n_maxiter):
            opt.minimize(my_MOGP.training_loss, var_list=my_MOGP.trainable_variables)
            
            # Optimization monitoring
            if i % 100==0:
                likelihood = my_MOGP.log_marginal_likelihood().numpy()      
                opt_logs.append(np.round(likelihood, 3))
            
                if showOutput:
                    tf.print(f"Optimizer: Adam   iterations {i} loglik_marg: {likelihood: .04f}")
        
        # Print output: optimization monitoring, estimated kernel hyperparameters, coregionalization matrix
        if showOutput:
            print_summary(my_MOGP,fmt='notebook')
            print(coreg1.output_covariance().numpy())
            if m1>1:
                print(coreg2.output_covariance().numpy())
    
    # Store estimated period parameters if desired
    if estimatePeriod:
        if nug==None:
            period_param = my_MOGP.kernel.kernels[0].period.numpy()
        else:
            period_param = my_MOGP.kernel.kernels[0].kernels[0].period.numpy()

    # Predict at new input locations, with uncertainty quantified by predictive variance/sd
    mu, var = my_MOGP.predict_f(result_arr_X_new)
    mu = mu.numpy()
    var = var.numpy()
    sd = np.sqrt(var)
    
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
    
    for i in range(0,m1):
        data_i = datapointList[i]
        truth_i = truthpointList[i]
        new_i = predList[i]
        N_i = data_i.shape[1]
        
        mu_x = mu[pointer_new:(pointer_new+N_pred),0]
        sd_x = sd[pointer_new:(pointer_new+N_pred),0]
        mu_x_List.append(mu_x)
        sd_x_List.append(sd_x)
        lci_x_List.append(mu_x-CI_mult*sd_x)
        uci_x_List.append(mu_x+CI_mult*sd_x)
        tmp_x_new = result_arr_X_new[pointer_new:(pointer_new+N_pred),0]
        pointer_new = pointer_new + N_pred

        mu_y = mu[pointer_new:(pointer_new+N_pred),0]
        sd_y = sd[pointer_new:(pointer_new+N_pred),0]
        mu_y_List.append(mu_y)
        sd_y_List.append(sd_y)
        lci_y_List.append(mu_y-CI_mult*sd_y)
        uci_y_List.append(mu_y+CI_mult*sd_y)
        tmp_y_new = result_arr_X_new[pointer_new:(pointer_new+N_pred),0]
        pointer_new = pointer_new + N_pred
        
        if showPlot:
            plt.figure(figsize=(20,5))

            # First coordinate fit with prediction interval
            plt.subplot(1,3,1)
            plt.scatter(tmp_x_new,mu_x,label='mean_x',c='k',s=15)
            plt.scatter(tmp_x_new,mu_x+CI_mult*sd_x,label='uci_x',c='r',s=15)
            plt.scatter(tmp_x_new,mu_x-CI_mult*sd_x,label='lci_x',c='b',s=15)
            plt.xlabel('arc-length')
            plt.ylabel('x-coordinate')
            plt.title('#new loc='+str(N_pred))

            # Second coordinate fit with prediction interval
            plt.subplot(1,3,2)
            plt.scatter(tmp_y_new,mu_y,label='mean_y',c='k',s=15)
            plt.scatter(tmp_y_new,mu_y+CI_mult*sd_y,label='uci_y',c='r',s=15)
            plt.scatter(tmp_y_new,mu_y-CI_mult*sd_y,label='lci_y',c='b',s=15)
            plt.xlabel('arc-length')
            plt.ylabel('y-coordinate')
            plt.title('#new loc='+str(N_pred))
            
            # Ellipse uncertainty quantification plot
            plt.subplot(1,3,3)
            plt.scatter(mu_x,mu_y,label='estimated mean',c='k',s=5)
            if truth_ind:
                plt.scatter(truth_i[0,:],truth_i[1,:],label='truth points',c='y',s=15)
            plt.scatter(data_i[0,:],data_i[1,:],label='observed points',c='r',s=15)
            ax = plt.gca()
            for i in range(0,N_pred):
                e_cen_x = mu_x[i]
                e_cen_y = mu_y[i]
                e_rad_x = sd_x[i]
                e_rad_y = sd_y[i]
                ellipse = Ellipse(xy=(e_cen_x,e_cen_y), width=CI_mult*e_rad_x, height=CI_mult*e_rad_y, edgecolor='b', fc='None', lw=2, alpha=0.6)
                ax.add_patch(ellipse)
            plt.title('#observed samples='+str(N_i))
            
            # Coregionalization matrices
            #plt.subplot(1,2,1)      
            
            #plt.subplot(1,2,2)
            
    if estimatePeriod:            
        return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc, period_param
    else:
        return my_MOGP, opt_logs, mu_x_List, sd_x_List, mu_y_List, sd_y_List, result_arr_X_new, perc
    
## Elastic registration of multiple curves
# Justin updated 12/6/21
def curveBatch_preproc(datapointList, warp=True, scale=True, cen=True, resamp=False, N_reg=100):
    #################################################################################
    ## Inputs ##
    # REQUIRED:
    # datapointList = list of length m1, each entry has sample points from observed curves (d x N_i for curves i=1,...,m1)
    
    # OPTIONAL:
    # scale = should curves be scaled to unit length? (default = True)
    # cen = should curves be centered? (default = True)
    # resamp = should curves be re-sampled for registration?
    #    if True: curves are aligned using SRVF registration
    # N_rs = number of points to re-sample curves to (only used if resamp=True)
    
    ## Outputs ##
    # datapointList_reg = list of length m1, each entry has registered sample points from observed curves (d x N_i for curves i=1,...,m1)
    #################################################################################
    from fdasrsf.curve_stats import fdacurve
    import fdasrsf.curve_functions as cf
    
    # Data dimensions
    M = len(datapointList)
    d = len(datapointList[0])
    N = []
    for m in range(0,M):
        N.append(len(datapointList[m][0]))

    if warp==True and resamp==False:
        if all(n==N[0] for n in N)==False:
            print('ERROR: datapointList curves must be sampled at same number of points if resamp=False')
            return None
        N_reg = N[0]
        
    # Scale curves to unit length
    if scale:
        datapointList_pp = [datapointList[m]/xy_length(datapointList[m]) for m in range(0,M)]
    else:
        datapointList_pp = [datapointList[m] for m in range(0,M)]
        
    # Estimate optimal rotation and warping functions
    if warp:
        beta_rs = np.zeros((d,N_reg,M))
    
        # Convert datapointList to suitable format for fdasrsf module   
        if resamp==False:
            for m in range(0,M):
                beta_rs[:,:,m] = datapointList_pp[m]
        else:
            for m in range(0,M):
                beta_rs[:,:,m] = cf.resamplecurve(datapointList_pp[m], N=N_reg, time=None, mode='C')
                
        # Register curves to estimated Karcher mean
        curve_obj = fdacurve(beta_rs, mode='C', N=N_reg, scale=scale)
        curve_obj.srvf_align()
        
        # Extract estimated quantities from multiple registration
        q = curve_obj.q
        qbar = curve_obj.q_mean
        beta_reg = curve_obj.betan
        gam_reg = curve_obj.gams
        gam = []
        
        for m in range(0,M):
            # Find optimal rotation and rotate original curve accordingly
            tmp, O = cf.find_best_rotation(qbar, q[:,:,m])
            datapointList_pp[m] = O@datapointList_pp[m]

            # Estimated warping function -- re-sample to same number of points as original curve
            if resamp:
                from scipy.interpolate import interp1d
                grid = np.linspace(start=0, stop=xy_length(beta_reg[:,:,m]), num=N_reg)
                int_reg = interp1d(grid, gam_reg[:,m], kind='cubic')
                tmp_gam = int_reg(np.linspace(start=0, stop=xy_length(beta_reg[:,:,m]), num=N[m]))
            else:
                tmp_gam = gam_reg[:,m]
            gam.append(tmp_gam)
    
    # Center curves, if desired
    if cen:
        for m in range(0,M):
            c = cf.calculatecentroid(datapointList_pp[m])
            datapointList_pp[m][0,:] = datapointList_pp[m][0,:]-c[0]
            datapointList_pp[m][1,:] = datapointList_pp[m][1,:]-c[1]
    
    if warp:
        return datapointList_pp, gam
    else:
        return datapointList_pp
