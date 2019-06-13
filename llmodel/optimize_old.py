# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:33:30 2018

@author: Patrick LaChance
"""

#import important modules
import numpy as np
#import cupy as cp
import numba as nb
from scipy.sparse import kron,spdiags,eye
import warnings

#how many bins should we use for spatial parameters? (e.g. 8 gives 8**2 = 64 bins)
gr=20
#how many bins should we use for head direction parameters?
hd_bins = 30
#ego bins
ego_gr = 8
framerate = 1./30.

smooth = True

def compute_diags():
    ''' create diagonal matrices for grouped penalization -- implementation 
    modified from Hardcastle 2017 '''
    
    'diagonal matrix for computing differences between adjacent circular 1D bins'

    #make a list of ones length of number of hd_bins
    pos_ones = np.ones(hd_bins)
    #make a (29,30) matrix with negative ones on the main diagonal and positive
    #ones on the first diagonal up
    circ1 = spdiags([-pos_ones,pos_ones],[0,1],hd_bins-1,hd_bins)
    #multiply the matrix by its transpose to get a (30,30) matrix with twos on
    #the main diagonal and negative ones on either side (except for the corners, which 
    #is bad for circular data)
    circ_diag = circ1.T * circ1
    
    #make the matrix dense so we can fix it with numpy
    circ_diag=np.asarray(circ_diag.todense())
    
    #fix the top row so it has a two in the corner with -1 on either side
    circ_diag[0] = np.roll(circ_diag[1],-1)
    #fix the bottom row so it has a two in the corner with a -1 on either side
    circ_diag[hd_bins-1] = np.roll(circ_diag[hd_bins-2],1)
        
    #make it sparse again
#    circ_diag=csr_matrix(circ_diag)
    
    'also one for noncircular'

    #make a list of ones length of number of bins
    pos_ones = np.ones(20)
    #make a (19,20) matrix with negative ones on the main diagonal and positive
    #ones on the first diagonal up
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],20-1,20)
    #multiply the matrix by its transpose to get a (20,20) matrix with twos on
    #the main diagonal and negative ones on either side (except for the corners, which 
    #is bad for circular data)
    noncirc_diag = noncirc1.T * noncirc1
    noncirc_diag = np.asarray(noncirc_diag.todense())
    
    'diagonal matrix for computing differences between adjacent noncircular bins in 2D'
    
    #make a list of ones the length of our grid_res (20)
    pos_ones = np.ones(gr)
    #make a (19,20) matrix with negative ones on the main diagonal and positive
    #ones on the first diagonal up
    D1 = spdiags([-pos_ones,pos_ones],[0,1],gr-1,gr)
    #multiply the matrix by its transpose to get a (20,20) matrix with twos on the main diagonal
    #and negative ones on either side (except for the corners - these have positive ones)
    DD1 = D1.T * D1

    #make a (400,400) matrix with the same diagonal as DD1 and zeros elsewhere
    M1 = kron(eye(gr),DD1)
    #make a (400,400) matrix with ones on the main diagonal and zeros elsewhere
    M2 = kron(DD1,eye(gr))
    #add them together to get a (400,400) matrix with threes down the main diagonal
    #and negative ones on either side (except for the corners - these have positive twos)
    spatial_diag = M1+M2
    
    'same for egocentric'
    
    #make a list of ones the length of our grid_res (20)
    pos_ones = np.ones(ego_gr)
    #make a (19,20) matrix with negative ones on the main diagonal and positive
    #ones on the first diagonal up
    eD1 = spdiags([-pos_ones,pos_ones],[0,1],ego_gr-1,ego_gr)
    #multiply the matrix by its transpose to get a (20,20) matrix with twos on the main diagonal
    #and negative ones on either side (except for the corners - these have positive ones)
    eDD1 = eD1.T * eD1

    #make a (400,400) matrix with the same diagonal as DD1 and zeros elsewhere
    eM1 = kron(eye(ego_gr),eDD1)
    #make a (400,400) matrix with ones on the main diagonal and zeros elsewhere
    eM2 = kron(eDD1,eye(ego_gr))
    #add them together to get a (400,400) matrix with threes down the main diagonal
    #and negative ones on either side (except for the corners - these have positive twos)
    ego_diag = eM1+eM2
    
        
    return circ_diag, noncirc_diag, np.array(spatial_diag.todense()), np.array(ego_diag.todense())

def objective(params,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas):
    ''' computes negative log-likelihood and returns it '''
    
    #start lambda as array of ones
    lamb = np.ones(len(spike_train))
    
    #clip parameters so we don't get an overflow error -- this is only useful at
    #the very beginning f optimization
    for t in params.keys():
        if params[t] is not None:
            params[t][params[t] > 20] = 20.
    
    #for each active variable, multiply lambda by the prediction for that variable,
    #calculated by multiplying the relevant occupancy matrix (X) by the exponentiated parameters
    if params['allo'] is not None:
        lamb *= (Xa * np.exp(params['allo']))
    if params['ego'] is not None:
        lamb *= (Xe * np.exp(params['ego']))
    if params['spatial'] is not None:
        lamb *= (Xp * np.exp(params['spatial']))
    if params['speed'] is not None:
        lamb *= (Xs * np.exp(params['speed']))
    if params['ahv'] is not None:
        lamb *= (Xahv * np.exp(params['ahv']))
    if params['theta'] is not None:
        lamb *= (Xt * np.exp(params['theta']))
    if params['novelty'] is not None:
        lamb *= (Xn * np.exp(params['novelty']))

    #calculate the objective function
    f = np.nansum(lamb - spike_train * np.log(lamb))
        
    #if we're smoothing the parameters...
    if smooth:
        #add penalty terms from grouped regularization
        f = penalize_objective(f,params,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)

    #return the objective
    return f

@nb.njit()
def penalize_ego(ego_params,circ_diag,ego_diag,ebeta):
    ''' imposes objective grouped penalties on egocentric params '''
    
    #start with 0 penalty
    f = 0
    #reshape the egocentric parameters to a 3D grid
    ego_params = ego_params.reshape((ego_gr,ego_gr,hd_bins))

    #calculate penalties for squared differences between adjacent locations
    #at the same egocentric angle bin (e.g. the egocentric curve for position
    #X1Y1 should be similar to that for position X0Y1, X1Y0, X2Y1, and X1Y2)
    for i in range(hd_bins):     
        ego_grid = np.copy(ego_params[:,:,i]).reshape(ego_gr**2)
        f += np.sum(ebeta * .5 * (np.dot(ego_grid.T, ego_diag) * ego_grid).reshape(ego_gr,ego_gr))
 
    #calculate penalties for squared differences between adjacent angle bins in
    #each tuning curve
    for i in range(ego_gr):
        for j in range(ego_gr):
            f += np.sum(ebeta * .5 * ( np.dot(ego_params[i][j].T, circ_diag) * ego_params[i][j]) )
                        
    #return the objective penalty
    return f
    
def penalize_objective(f,params,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas=None):
    ''' add L2 penalties for differences between grouped neighboring parameters '''

    #grab our smoothing hyperparameters
    pbeta = betas[0]
    ebeta = betas[1]
    abeta = betas[2]
    sbeta = betas[3]
    nbeta = betas[3]
    ahvbeta = betas[4]
    tbeta = betas[5]
    
    #for each variable, calculate the objective penalty based on the sum of squared differences
    #between adjacent parameter bins, multiplied by the relevant smoothing hyperaparameter, and
    #add it to the value of the objective function
    if params['spatial'] is not None:
        f += np.sum(pbeta * .5 * np.dot(params['spatial'].T, spatial_diag) * params['spatial'])
        
    if params['allo'] is not None:
        f += np.sum(abeta * .5 * np.dot(params['allo'].T, circ_diag) * params['allo'] )
        
    if params['ego'] is not None:
        f += penalize_ego(params['ego'],circ_diag,ego_diag,ebeta)
        
    if params['speed'] is not None:
        f += np.sum(sbeta * .5 * np.dot(params['speed'].T, noncirc_diag) * params['speed'])
        
    if params['ahv'] is not None:
        f += np.sum(ahvbeta * .5 * np.dot(params['ahv'].T, noncirc_diag) * params['ahv'])
        
    if params['novelty'] is not None:
        f += np.sum(nbeta * .5 * np.dot(params['novelty'].T, noncirc_diag) * params['novelty'])
        
    if params['theta'] is not None:
        f += np.sum(tbeta * .5 * ( np.dot(params['theta'].T, circ_diag) * params['theta'] ))
        
    #return the penalized objective
    return f    

def grad(params,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas):
    ''' computes the gradient of the function and returns it '''
    
    #start lambda as sequence of ones
    lamb = np.ones(len(spike_train))
    
    #clip parameters so we don't get an overflow error -- this is only useful at the
    #start of optimization and shouldn't affect subsequent optimization
    for t in params.keys():
        if params[t] is not None:
            params[t][params[t] > 20] = 20.
    
    #for each active variable, multiply lambda by the prediction for that variable,
    #calculated by multiplying the relevant occupancy matrix (X) by the exponentiated parameters
    if params['spatial'] is not None:
        lamb *= (Xp * np.exp(params['spatial']))
    if params['allo'] is not None:
        lamb *= (Xa * np.exp(params['allo']))
    if params['ego'] is not None:
        lamb *= (Xe * np.exp(params['ego']))
    if params['speed'] is not None:
        lamb *= (Xs * np.exp(params['speed']))
    if params['ahv'] is not None:
        lamb *= (Xahv * np.exp(params['ahv']))
    if params['theta'] is not None:
        lamb *= (Xt * np.exp(params['theta']))
    if params['novelty'] is not None:
        lamb *= (Xn * np.exp(params['novelty']))
        
    #calculate the difference between lambda and the actual spike train (errors)
    diff = lamb - spike_train
    
    #start the gradient as array of length 0
    df = np.zeros(0)
    
    #for each active variable, append the individual parameter gradients, defined
    #as the errors summed over the times points each parameter bin was active
    if params['spatial'] is not None:
        df = np.concatenate([df,Xp.T * diff])
    if params['allo'] is not None:
        df = np.concatenate([df,Xa.T * diff])
    if params['ego'] is not None:
        df = np.concatenate([df,Xe.T * diff])
    if params['speed'] is not None:
        df = np.concatenate([df,Xs.T * diff])
    if params['ahv'] is not None:
        df = np.concatenate([df,Xahv.T * diff])
    if params['theta'] is not None:
        df = np.concatenate([df,Xt.T * diff])
    if params['novelty'] is not None:
        df = np.concatenate([df,Xn.T * diff])

    #if we're smoothing the parameters...
    if smooth:
        #add penalty terms from grouped regularization
        df = penalize_grad(df,params,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)
        
    #return the gradient
    return df

@nb.njit()
def ego_grad(ego_params,circ_diag,ego_diag,ebeta):
    ''' calculate the gradient penalty for egocentric parameters '''
    
    #reshape ego param vector into 3D grid
    ego_params = ego_params.reshape((ego_gr,ego_gr,hd_bins))
    #create an identically-sized grid for holding gradient values
    ego_grads = np.zeros((ego_gr,ego_gr,hd_bins))

    #calculate gradient penalties based on differences between 
    #the same angle bin for adjacent position bins
    for l in range(hd_bins):
        ego_grid = np.copy(ego_params[:,:,l]).reshape(ego_gr**2)
        ego_grads[:,:,l] = ebeta * ( np.dot( ego_diag , ego_grid ).reshape((ego_gr,ego_gr)))
        
    #calculate gradient penalties based on differences between
    #adjacent angle bins in each position bin
    for i in range(ego_gr):
        for j in range(ego_gr):
            ego_grads[i][j] += ebeta * np.dot( circ_diag, ego_params[i][j])

    #return the gradient as a vector
    return ego_grads.reshape(hd_bins*ego_gr**2)

def penalize_grad(df,params,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas):
    ''' add L2 gradient penalties for differences between grouped neighboring parameters '''
    
    #grab the appropriate smoothing hyperparameters
    pbeta = betas[0]
    ebeta = betas[1]
    abeta = betas[2]
    sbeta = betas[3]
    nbeta = betas[3]
    ahvbeta = betas[4]
    tbeta = betas[5]
        
    #start the gradient as an empty array
    grad = np.zeros(0)
        
    #for each active variable, append the penalty gradients for differences between
    #adjacent parameter bins
    if params['spatial'] is not None:
        spatial_grad = pbeta * np.dot(spatial_diag, params['spatial'])
        grad = np.concatenate([grad,spatial_grad])
        
    if params['allo'] is not None:
        allo_grad = abeta * np.dot(circ_diag , params['allo'])
        grad = np.concatenate([grad,allo_grad])
        
    if params['ego'] is not None:
        e_grad = ego_grad(params['ego'],circ_diag,ego_diag,ebeta)
        grad = np.concatenate([grad,e_grad])
        
    if params['speed'] is not None:
        speed_grad = sbeta * np.dot(noncirc_diag, params['speed'])
        grad = np.concatenate([grad,speed_grad])
        
    if params['ahv'] is not None:
        ahv_grad = ahvbeta * np.dot(noncirc_diag, params['ahv'])
        grad = np.concatenate([grad,ahv_grad])
            
    if params['theta'] is not None:
        theta_grad = tbeta * np.dot(circ_diag , params['theta'])
        grad = np.concatenate([grad,theta_grad])
        
    if params['novelty'] is not None:
        novelty_grad = nbeta * np.dot(noncirc_diag, params['novelty'])
        grad = np.concatenate([grad,novelty_grad])
            
    #add penalties to the gradient
    df += grad
    
    #return the gradient
    return df

def grad_hess(params,args,vector):
    ''' calculate the hessian-vector product by approximation '''

    #unpack arguments
    Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas = args

    #the difference value we will use to approximate the hessian-vector product --
    #smaller values provide a closer approximation
    r = np.float(1e-10)
    
    #make our parameter dict into a single vector
    param_vector = make_vector(params)
    
    #calculate the gradient for the current parameter values plus or minus r times
    #the current negative gradient (search direction)
    df1 = line_search_grad(param_vector+r*vector,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)
    df2 = line_search_grad(param_vector-r*vector,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)
    
    #the hessian-vector product is the difference between these two values divided by
    #2r
    hvp = (df1 - df2)/(2*r)

    #return the hessian-vector product
    return hvp

def make_vector(param_dict):
    ''' make the parameters into a vector for easy computation '''
    
    #start with empty array
    params = np.zeros(0)
    
    #append each set of active parameters
    if param_dict['spatial'] is not None:
        params = np.concatenate([params,param_dict['spatial']])
    if param_dict['allo'] is not None:
        params = np.concatenate([params,param_dict['allo']])
    if param_dict['ego'] is not None:
        params = np.concatenate([params,param_dict['ego']])
    if param_dict['speed'] is not None:
        params = np.concatenate([params,param_dict['speed']])
    if param_dict['ahv'] is not None:
        params = np.concatenate([params,param_dict['ahv']])
    if param_dict['theta'] is not None:
        params = np.concatenate([params,param_dict['theta']])
    if param_dict['novelty'] is not None:
        params = np.concatenate([params,param_dict['novelty']])
        
    #return the vector
    return params

def make_dict(params,modeltype):
    ''' make param vector into a dict for easy reading '''
    
    #start a count so we know our position in the vector
    param_count = 0
    
    #variables we'll be looking for
    variables = ['spatial','allo','ego','speed','ahv','theta','novelty']
    
    #start a dict with the variables as entries and set all to NONE
    param_dict = {}
    for v in variables:
        param_dict[v] = None   
        
    #for each variable in the current model, assign the appropriate param
    #vector elements to the appropriate dict entry
    if 'spatial' in modeltype:
        param_dict['spatial'] = params[:gr**2]
        param_count += gr**2
    if 'allo' in modeltype:
        param_dict['allo'] = params[param_count:param_count+hd_bins]
        param_count += hd_bins
    if 'ego' in modeltype:
        param_dict['ego'] = params[param_count:param_count+hd_bins*ego_gr**2]
        param_count += hd_bins*ego_gr**2
    if 'speed' in modeltype:
        param_dict['speed'] = params[param_count:param_count+20]
        param_count += 20
    if 'ahv' in modeltype:
        param_dict['ahv'] = params[param_count:param_count+20]
        param_count += 20
    if 'theta' in modeltype:
        param_dict['theta'] = params[param_count:param_count+hd_bins]
        param_count += hd_bins
    if 'novelty' in modeltype:
        param_dict['novelty'] = params[param_count:]
        
    #return the dict
    return param_dict

def line_search_grad(params,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas):
    ''' convenience function for calculating gradient from param vector '''

    param_dict = make_dict(params,modeltype)
    df = grad(param_dict,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)
    
    return df

def line_search_objective(params,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas):
    ''' convenience function for calculating objective from param vector '''
    
    param_dict = make_dict(params,modeltype)
    f = objective(param_dict,Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)
    
    return f


def newton_cg(data, modeltype, betas, tol=1e-4, maxiter=200, maxinner=30, x0=None):
    """
    modified (heavily) from scipy.optimize.minimize Newton-CG algorithm
    """
    
    #the variables we'll (potentially) be working with
    variables = ['allo','ego','spatial','ahv','speed','theta','novelty']
    #make param dict and set each variable entry to NONE
    params = {}
    for v in variables:
        params[v] = None
    
    #compute diagonal matrices for computing penalties
    circ_diag, noncirc_diag, spatial_diag, ego_diag = compute_diags()

    #grab spike train and variable occupancy matrices    
    spike_train=data['spikes']
    Xp=data['Xp']
    Xa=data['Xa']
    Xe=data['Xe']
    Xs=data['Xs']
    Xahv=data['Xahv']
    Xt=data['Xt']
    Xn = data['Xn']
            
    #either start with initial parameter guess or zeros for each variable
    if x0 is not None:
        params = x0
    else:
        if 'allo' in modeltype:
            params['allo'] = np.zeros(hd_bins)
        if 'ego' in modeltype:
            params['ego'] = np.zeros(hd_bins*ego_gr**2)
        if 'spatial' in modeltype:
            params['spatial'] = np.zeros(gr**2)
        if 'speed' in modeltype:
            params['speed'] = np.zeros(20)
        if 'ahv' in modeltype:
            params['ahv'] = np.zeros(20)
        if 'theta' in modeltype:
            params['theta'] = np.zeros(hd_bins)
        if 'novelty' in modeltype:
            params['novelty'] = np.zeros(20)
        
    #throw everything into an 'args' tuple
    args = (Xp,Xe,Xa,Xs,Xahv,Xt,Xn,spike_train,framerate,modeltype,circ_diag,noncirc_diag,spatial_diag,ego_diag,betas)   

    #calc objective for starting parameters
    old_fval = objective(params,*args)
    old_old_fval = None

    #start counter at 1
    k = 1
    # Outer loop: our Newton iteration
    while k <= maxiter:
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - fgrad f(xk) starting from 0.
        
        #calc current gradient
        fgrad = grad(params, *args)
        #find absolute value of gradient
        absgrad = np.abs(fgrad)
        #if maximum gradient value less than tolerance, we're done!
        if np.max(absgrad) < tol:
            break
        #calc sum of absolute values of gradient
        maggrad = np.sum(absgrad)
        #find smallest of 0.5 or square root of sum of absolute gradient values
        eta = np.min([0.5, np.sqrt(maggrad)])
        #termination condition = eta * sum of absolute gradient values
        termcond = eta * maggrad

        # Inner loop: solve the Newton update by conjugate gradient, to
        # avoid inverting the Hessian
        updates = cg_update(params,grad_hess, fgrad, maxiter=maxinner, tol=termcond, args=args)
        #make parameters a vector for use in line search
        param_vector = make_vector(params)
        #perform line search to find best step size
        alphak = line_search(param_vector, updates, args, max_iter=200)
                
        old_old_fval = objective(params,*args)
        #update our parameters
        param_vector = param_vector + alphak * updates
        params = make_dict(param_vector, modeltype)
        old_fval = objective(params,*args)
            
        #figure out if we've reached some threshold stopping
        #criterion for improvement in the objective (doesn't have to be perfect)
        if np.abs(old_fval - old_old_fval) < .05:
            break

        #increment the counter
        k += 1
    
    #if we've exceeded our allowed iterations, it didn't converge :(
    if k > maxiter:
        warnings.warn("newton-cg failed to converge. Increase the "
                      "number of iterations.")
                
    #return our (hopefully) optimized parameters
    return params

#@nb.jit()
def cg_update(params, fhess_p, fgrad, maxiter, tol, args):
    #make a vector for updates to add to the parameters
    updates = np.zeros(len(fgrad))
    #negative gradient is definitely a descent direction
    search_dir = -fgrad
    #start a counter
    i = 0
    #dot product of the gradient vector
    dot_grad0 = np.dot(fgrad, fgrad)

    #start a loop (will run up to maxiter)
    for it in xrange(maxiter):
        #if the sum of the gradient is less than our tolerance,
        #we're done -- exit
        if np.sum(np.abs(fgrad)) <= tol:
            break

        #hessian vector product with the search direction
        hvp = fhess_p(params,args,search_dir)
        
        #check curvature -- if it's zero or negative then we're probably done,
        #otherwise stick with our steepest descent direction
        curv = np.dot(search_dir, hvp)
        if 0 <= curv <= 3 * np.finfo(np.float64).eps:
            break
        elif curv < 0:
            if i > 0:
                break
            else:
                updates += dot_grad0 / curv * search_dir
                break
            
        #calculate alpha term by dividing gradient dot product by curvature
        alphai = dot_grad0 / curv
        #add alpha * direction to our param update vector
        updates += alphai * search_dir
        #add alpha multiplied by hessian vector product to gradient
        fgrad = fgrad + alphai * hvp
        #calc new gradient dot product
        dot_grad1 = np.dot(fgrad, fgrad)
        #calc beta term by dividing new grad dot product by old dot product
        betai = dot_grad1 / dot_grad0
        #calc new search direction by adding beta term multiplied by old direction
        search_dir = -fgrad + betai * search_dir
        #increment the counter
        i += 1
        #update gradient dot product
        dot_grad0 = dot_grad1  

    return updates

@nb.jit()
def line_search(params,updates,args,max_iter):
    ''' modified (heavily) from https://nlperic.github.io/line-search/ '''
    
    alpha = 0
    beta = 1000
    step = 1.
    c2 = .3
    i = 0
    
    #calc starting objective and gradient
    start_f = line_search_objective(params,*args)
    start_grad = line_search_grad(params,*args)
    
    old_f = 0
    
    while i <= max_iter:
        #calc objective for current step
        new_f = line_search_objective(params + step*updates,*args)
        #first test for convergence -- we're done if last objective is pretty close
        #to current objective
        if np.abs(new_f - old_f) < .05:
            break
        #if objective is larger than our starting objective...
        if new_f > start_f:
            #set beta equal to step size
            beta = step
            #new step size is average of alpha and beta
            step = (alpha + beta)/2.
        #otherwise, if we're on the wrong side of the curvature...
        elif line_search_grad(params + step*updates,*args).dot(updates) < c2*start_grad.dot(updates):
            #set alpha equal to step size
            alpha = step
            #if beta is large, set step equal to 2*alpha
            if beta > 100:
                step = 2*alpha
            #otherwise, set step equal to average of alpha and beta
            else:
                step = (alpha + beta)/2.
        #otherwise, looks like we have a good step size
        else:
            break
        #increment iteration counter            
        i += 1
        #collect last objective
        old_f = new_f
    
    return step
