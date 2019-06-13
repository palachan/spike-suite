# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:13:23 2017

log-likelihood model with cross-validation option

@author: Patrick
"""

#import important modules
import pickle
import numpy as np
import numba as nb
import bisect
import tkFileDialog
import math
from scipy.stats import pearsonr
#from scipy.ndimage import gaussian_filter
from astropy.convolution.kernels import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve
import copy
from scipy.optimize import minimize
from scipy.sparse import csr_matrix,kron,spdiags,eye

#import scripts from spike-analysis
import main
from spatial import rayleigh_r


import matplotlib.pyplot as plt
"""""""""""""""""""""""""""""
Decisions
"""""""""""""""""""""""""""""
#maximize the log-likelihood of the data (perform corrections for multiple
#independent correlates)
optimize = True
#run 10-fold cross-validation (False means train and test on whole session)
cross_val = True
#just correct egocentric and allocentric HD for each other, leave spatial the way it is
just_allo = False
#just look at the first half of the session (meant for inactivation data)
first_half = False
second_half = False
#optimize each model separately (otherwise just optimize the 3-component model)
opt_all = False
#
local_ego = False


#how many bins should we use for spatial parameters? (e.g. 8 gives 8**2 = 64 bins)
gr=20
#how many bins should we use for head direction parameters?
hd_bins = 30


def load_data(fname):
    ''' load pickled numpy arrays '''

    try:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    except:
        print('couldn\'t open data file! try again!!')
    
    return data

@nb.jit(nopython=True)
def start_log_likelihood(lam,ani_spikes,framerate):
    ''' the important part -- the log-likelihood model 
    
    log-likelihood = sum(n_spikes(t) * log(lambda * dt) - lambda * dt - log(n_spikes(t)!))
    
    summed over every time point in the session 
    
    This function is for preconditioning the parameters through iteration before
    passing them to the scipy minimize function
    
    '''
    
    #start array for log-factorials
    lgammas = np.zeros(len(ani_spikes))
        
    #for every time point...
    for h in range(len(ani_spikes)):
        #calculate the log-factorial of the number of spikes during that frame,
        #using base 2 log and gamma function on (n_spikes + 1) so numba can work
        lgammas[h] = np.log2(math.gamma(ani_spikes[h]+1))
                
    #calculate the log-likelihood
    ll = np.nansum(ani_spikes * np.log2(lam * 1./framerate) - lam * 1./framerate)
    
    ll -= np.sum(lgammas)
    
    return ll

@nb.jit(nopython=True)
def calc_ego_params(ego_params,a_vals,ani_spikes,ego_bins):
    ''' calculate egocentric parameters 
    
    for a given spatial bin [k,l] and a given egocentric head direction bin [m]
    over a whole session of time points [t]:
        
    ego_param(klm) = n_spikes(klm) / sum( p(t) * a(t) * dt )
    
    or the portion of the firing rate attributable to that egocentric component
    and neither spatial nor allocentric components
    
    '''
    dwells = np.full(len(ani_spikes),1./framerate)
    spikes = ani_spikes
    
    #start an array for holding denominator elements
    edenom_sums = np.zeros((gr,gr,hd_bins))
    e_spikes = np.zeros((gr,gr,hd_bins))
    occupancies = np.zeros((gr,gr,hd_bins))

    for k in range(gr):
        for l in range(gr):

            spikes = ani_spikes
            dwells = np.full(len(ani_spikes),1./framerate)
            
            
            for h in np.arange(len(ego_bins[0][0])):
                #figure out the location's current direction from the animal's heading
                m=ego_bins[k][l][h]
                
                if not np.isnan(m):
                    e_spikes[k][l][np.int(m)] += spikes[h]
                    #increment the denominator of the equation for that bin
                    edenom_sums[k][l][np.int(m)] += np.float(a_vals[h]) * np.float(dwells[h])
                    #increment the occupancy counter
                    occupancies[k][l][np.int(m)] += 1

    #for every reference point and egocentric hd bin...
    for k in range(gr):
        for l in range(gr):  

            for n in range(hd_bins):
                #if the animal has sampled this bin...
                if edenom_sums[k][l][n] > 0:
                    #divide the number of spikes in that bin by the appropriate denominator
                    ego_params[k][l][n] = np.float(e_spikes[k][l][n])/edenom_sums[k][l][n]

                #if we have sampling but for some reason the denominator is zero, there
                #must not be a firing component for that bin -- set it to zero
                elif occupancies[k][l][n] > 0:
                    ego_params[k][l][n] = 0.
                    
                #otherwise set to nan
                else:
                    ego_params[k][l][n] = np.nan

    #return the parameters
    return ego_params
    
@nb.jit(nopython=True)
def calc_e_vals(ego_params,ego_bins):
    ''' estimate a firing rate for every time point in the session based on
    our egocentric parameters -- this is averaged out over all of the reference
    points (assuming that the holistic map is important) so we're only returning 
    one set of firing rate estimates for the egocentric component '''
        
    #create array for firing rate estimates
    e_vals = np.zeros(len(ego_bins[0][0]))
    num_contributors = np.zeros(len(e_vals))

    #for each time point in the session...
    for h in range(len(ego_bins[0][0])):
        #start with a firing rate of zero
        e_sum=0
        #for every reference location...
        for k in range(gr):
            for l in range(gr):
                #if this is a top 10% bin...
                if not np.isnan(ego_bins[k][l][h]):
                    #add the associated firing rate estimate from the current
                    #egocentric directional bin for that location
                    
#                    if local_ego_weights is not None:
                    e_sum += ego_params[k][l][np.int(ego_bins[k][l][h])]
                    num_contributors[h] += 1.
                    
#                    else:
#                    e_sum += ego_params[k][l][np.int(ego_bins[k][l][h])]
#                    num_contributors[h] += 1.

        #assign the mean to the current time point as our firing rate estimate
        e_vals[h] = np.float(e_sum)/np.float(num_contributors[h])

    #return the estimates
    return e_vals


@nb.jit(nopython=True)
def calc_allo_params(allo_params,e_vals,ani_spikes,allo_bins):
    ''' calculate allocentric head direction parameters 
    
    for a given allocentric head direction bin [n] over a whole session of time points [t]:
        
    allo_param(n) = n_spikes(n) / sum( p(t) * e(t) * dt )
    
    or the portion of the firing rate attributable to that allocentric component
    and neither spatial nor egocentric components
    
    '''
    
    #start an array for holding denominator elements
    adenom_sums = np.zeros(hd_bins)
    occupancies = np.zeros(hd_bins)
    allo_spikes = np.zeros(hd_bins)
    #for every time point in the session...
    for h in np.arange(len(allo_bins)):
        #grab the current allocentric head direction
        n=np.int(allo_bins[h])
        
        allo_spikes[n] += ani_spikes[h]
        #increment the denominator of the equation for that bin
        adenom_sums[n] += np.float(e_vals[h]) * np.float(1./framerate)
        #increment the occupancy for that bin
        occupancies[n] += 1
 
    #for every head direction bin...
    for n in range(hd_bins):
        #if we have sampling in that bin and the denominator is greater than zero...
        if adenom_sums[n] > 0:
            #divide the number of spikes in that bin by the denominator
            allo_params[n] = np.float(allo_spikes[n])/adenom_sums[n]
        #if we have sampling but for some reason the denominator is zero, there
        #must not be a firing component for that bin -- set it to zero
        elif occupancies[n] > 0:
            allo_params[n] = 0.
        #otherwise set to nan - this bin hasn't been sampled
        else:
            allo_params[n] = np.nan
            
    #return the params
    return allo_params

@nb.jit(nopython=True)
def calc_a_vals(allo_params,allo_bins):
    ''' use our allocentric head direction parameters to estimate a firing
    rate for every time point based on the tracking data '''
    
    #create array for firing rate estimates
    a_vals = np.zeros(len(allo_bins))
                
    #for every time point...
    for h in range(len(allo_bins)):
        #grab the current head direction
        n=np.int(allo_bins[h])
        #assign the associated firing rate
        a_vals[h] = allo_params[n]

    #return the estimates
    return a_vals


def assign_data(center_x, center_y, angles, spike_train):
    
    #make arrays for assigning bins to spatial and directional
    #tracking and spike data
    allo_bins = np.ones((len(angles)),dtype=np.int)
    pos_bins = np.ones((2,len(center_x)),dtype=np.int)

    #for every video frame in the session...
    for i in range(len(angles)):
        #calculate the head direction bin for this frame
        hd_bin = int(angles[i]/(360/hd_bins))
        #assign it the appropriate array
        allo_bins[i] = hd_bin

    #calculate our egocentric bins and dwell times
    ego_bins = ego_loop(center_y,center_x,angles,pos_bins,allo_bins)

    data = {}
    data['ego_bins'] = copy.deepcopy(ego_bins)
    data['allo_bins'] = copy.deepcopy(allo_bins)
    data['spikes'] = np.asarray(spike_train,dtype=np.float)
            
    return data

@nb.jit(nopython=True)
def ego_loop(center_y,center_x,angles,pos_bins,allo_bins):
    """ transform allocentric to egocentric angles and assign dwell times """

    #create arrays for x and y coords of spatial bins
    xcoords = np.zeros(gr)
    ycoords = np.zeros(gr)
        
    ego_bins = np.zeros((gr,gr,len(center_x)))
    
    #assign coordinates for x and y axes for each bin
    #(x counts up, y counts down)
    for x in range(gr):
        xcoords[x] = (np.float(x)/np.float(gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
        ycoords[x] = np.float(np.max(center_y)) - (np.float(x)/np.float(gr))*np.float((np.max(center_y)-np.min(center_y)))

    #for each y position...
    for i in range(gr):

        #fill an array with the current y coord
        cue_y = ycoords[i]

        #for each x position...
        for j in range(gr):
                                                
            #fill an array with the current x coord
            cue_x = xcoords[j]

            #calc array of egocentric angles of this bin from pos x axis centered 
            #on animal using arctan
            new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
            #calculate ecd angles by subtracting allocentric
            #angles from egocentric angles
            ecd_angles = (new_angles-angles)%360
            #assign to bin
            ecd_bins = ecd_angles/(360/hd_bins)

            ego_bins[i][j] = ecd_bins
       
    #return appropriate arrays
    return ego_bins

def allocentrize_ecd(ego_params,center_x,center_y):

    thresh_dist = 100
    
    #create arrays for x and y coords of spatial bins
    xcoords = np.zeros(gr)
    ycoords = np.zeros(gr)
    #assign coordinates for x and y axes for each bin
    #(x counts up, y counts down)
    for x in range(gr):
        xcoords[x] = (np.float(x)/np.float(gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
        ycoords[x] = np.float(np.max(center_y)) - (np.float(x)/np.float(gr))*np.float((np.max(center_y)-np.min(center_y)))

    xgrid,ygrid = np.meshgrid(xcoords,ycoords)

    @nb.jit(nopython=True)
    def loop(ego_params,xcoords,ycoords,xgrid,ygrid,thresh_dist):

        allocentrized_curves = np.zeros((gr,gr,hd_bins))
        
        for i in range(gr):
            current_y = ycoords[i]
            for j in range(gr):
                current_x = xcoords[j]
                
                num_responsible = np.zeros(hd_bins)
                weights = np.ones((gr,gr))
                
                    
                dists = np.sqrt((xgrid - current_x)**2 + (ygrid - current_y)**2)
                for v in range(gr):
                    for w in range(gr):
                        
                        if local_ego:
                            if dists[v][w] > thresh_dist or dists[v][w] < 10:
                                dists[v][w] = np.nan
                                
                        else:
                            if dists[v][w] < 10:
                                dists[v][w] = np.nan
                                
                if local_ego:
                    weights = np.sin(np.deg2rad(90+((dists - (np.nanmin(dists)))/(np.nanmax(dists)-np.nanmin(dists)))*90))
                
                for v in range(gr):
                    for w in range(gr):
                        
                        if np.isnan(dists[v][w]):
                            weights[v][w] = 0

                
                for l in range(gr):
                    other_y = ycoords[l]
                    for m in range(gr):
      
                        other_x = xcoords[m] 
                        new_angle = np.rad2deg(np.arctan2((other_y-current_y),(other_x-current_x)))
    
                        for k in range(hd_bins):
                            hd_angle = k*360/float(hd_bins)
                            ecd_bin = np.int(((new_angle - hd_angle)%360)/(360/hd_bins))
                            allocentrized_curves[i][j][k] += ego_params[l][m][ecd_bin] * weights[l][m]
                            num_responsible[k] += weights[l][m]
                            
                allocentrized_curves[i][j] /= num_responsible
                
        return allocentrized_curves

    allocentrized_curves = loop(ego_params,xcoords,ycoords,xgrid,ygrid,thresh_dist)

#    import matplotlib.pyplot as plt
#    rayleighs = np.zeros((gr,gr))
#    hd_angles = np.arange(0,360,12)
#    plt.figure()
#    for i in range(gr):
#        for j in range(gr):
#            rayleighs[i][j],_ = rayleigh_r(hd_angles,allocentrized_curves[i][j])
#            
#    plt.imshow(rayleighs)
#    plt.show()
#

    return allocentrized_curves


def optimize_params(data, modeltype = ['ego','allo']):
    ''' run the log-likelihood optimization for each tuning component based
    on the training data '''
    
    #make arrays for holding the parameters (essentially tuning curves)
    allo_params = np.ones(hd_bins)
    ego_params = np.ones((gr,gr,hd_bins))

    #initialize allo, ego, and spatial estimate vectors
    a_vals = np.ones((len(data['spikes'])))
    e_vals = np.ones((len(data['spikes'])))
        
    #start with an impossibly low log-likelihood for optimization comparisons
    old_ll = -1*np.inf
    
    #count iterations - we could get stuck in an endless loop
    iter_count = 0
    
    #start the optimization loop
    while True:

        if 'ego' in modeltype:
        
            #calculate egocentric tuning
            ego_params = calc_ego_params(ego_params,a_vals,data['spikes'],data['ego_bins'])
            
            #calculate firing rate estimates for each time point based
            #on the egocentric parameters to be used for optimization
            e_vals = calc_e_vals(ego_params,data['ego_bins'])

        if 'allo' in modeltype:
            
            #use optimized params for calculating allocentric tuning
            allo_params = calc_allo_params(allo_params,e_vals,data['spikes'],data['allo_bins'])
            
            #calculate firing rate estimates for each time point based on
            #allo params
            a_vals = calc_a_vals(allo_params,data['allo_bins'])
            
            
        #multiply the firing rate estimates to get our lambda
        lamb = e_vals*a_vals
        
        #replace zero values with nans -- zeros don't matter during optimization
        #and break the ll algorithm
        lamb[lamb == 0] = np.nan

        #calculate the log-likelihood of the data and print
        ll = start_log_likelihood(lamb,data['spikes'],framerate)

        #if it's within some threshold of the last log-likelihood,
        #the parameters are fully optimized
        if (ll - old_ll < .005 and ll - old_ll > -.005) or iter_count > 20:    

            break
        
        #otherwise, save this log-likelihood and start the loop over
        else:
            old_ll = ll
            iter_count += 1

    return allo_params, ego_params


def interp_vals(spatial_params,allo_params,ego_params):
    ''' linearly interpolate values in our parameter estimates that are NaNs '''
    
    def nan_helper(y_vals):
        ''' returns where NaNs are for use by np.interp function '''
        return np.isnan(y_vals), lambda z: z.nonzero()[0]

    #copy spatial parameters twice for interpolating along on x and y axes
    x_spatial_params = copy.deepcopy(spatial_params)
    y_spatial_params = copy.deepcopy(spatial_params)
    
    #interpolate spatial params along x-axis
    for i in range(gr):
        nans, x = nan_helper(spatial_params[i])
        x_spatial_params[i][nans] = np.interp(x(nans), x(~nans), x_spatial_params[i][~nans])
        
    #interpolate spatial params along y-axis
    for j in range(gr):
        nans, x = nan_helper(spatial_params[:,j])
        y_spatial_params[:,j][nans] = np.interp(x(nans), x(~nans), y_spatial_params[:,j][~nans])
        
    #average the two interpolated spatial maps
    spatial_params = (x_spatial_params + y_spatial_params)/2.
        
    #interpolate empty allocentric HD spots
    nans, x = nan_helper(allo_params)
    allo_params[nans] = np.interp(x(nans), x(~nans), allo_params[~nans], period = hd_bins)
    
    #interpolate empty egocentric spots 
    for i in range(gr):
        for j in range(gr):
            nans, x = nan_helper(ego_params[i][j])
            ego_params[i][j][nans] = np.interp(x(nans), x(~nans), ego_params[i][j][~nans], period=hd_bins)

    #return our interpolated parameters
    return spatial_params, allo_params, ego_params

def run_raw_model(allo_params, ego_params, spatial_params,test_data,train_data,framerate,modeltype=['ego','allo','spatial']):
    
    cdict={}
    
    
    
    #initialize allo, ego, and spatial estimate vectors
    a_vals = np.ones((len(test_data['spikes'])))
    e_vals = np.ones((len(test_data['spikes'])))
    p_vals = np.ones((len(test_data['spikes'])))

    if 'ego' in modeltype:
    
#        e_vals = calc_e_vals(ego_params,test_data['ego_bins'],test_data['ego_weights'])
    
        e_vals = calc_allocentrized_e_vals(ego_params,test_data['spatial_bins'],test_data['allo_bins'])
        
    if 'allo' in modeltype:
        
        allo_params = convolve(allo_params,Gaussian1DKernel(stddev=1,x_size=5))
        
        a_vals = calc_a_vals(allo_params,test_data['allo_bins'])
        
    if 'spatial' in modeltype:
        
        spatial_params = convolve(spatial_params,Gaussian2DKernel(stddev=1,x_size=3,y_size=3))

        p_vals = calc_p_vals(spatial_params,test_data['spatial_bins'])
        

    if modeltype == ['uniform']:
        
        p_vals = np.full(len(p_vals),np.mean(train_data['spikes'])) / (1./framerate)
        scale_factor = 1.
        
    if modeltype != ['uniform'] and not opt_all:
        
        train_a_vals = np.ones((len(train_data['spikes'])))
        train_e_vals = np.ones((len(train_data['spikes'])))
        train_p_vals = np.ones((len(train_data['spikes'])))
        
        if 'ego' in modeltype:
            
#            train_e_vals = calc_e_vals(ego_params,train_data['ego_bins'],train_data['ego_weights'])
            
            train_e_vals = calc_allocentrized_e_vals(ego_params,train_data['spatial_bins'],train_data['allo_bins'])
    
            
        if 'allo' in modeltype:
            
            train_a_vals = calc_a_vals(allo_params,train_data['allo_bins'])
            
        if 'spatial' in modeltype:
            
            train_p_vals = calc_p_vals(spatial_params,train_data['spatial_bins'])
            
        mean_train_fr = np.mean(train_data['spikes']) / (1./framerate)
        
        train_lamb = train_e_vals * train_a_vals * train_p_vals
        test_lamb = e_vals * a_vals * p_vals
        
        train_pred = np.nanmean(train_lamb)
        test_pred = np.nanmean(test_lamb)
        
#        scale_factor = np.mean(test_data['spikes']) / test_pred
        
        scale_factor = (mean_train_fr * test_pred / train_pred) / test_pred
        
    elif opt_all:
        
        scale_factor = 1.
        
    #multiply the firing rate estimates to get our lambda
    lamb = e_vals*a_vals*p_vals*scale_factor
    
    #replace zero values with nans -- zeros are difficult to interpret so we won't
    #bother with them for now
    lamb[lamb < 0.1] = np.nan
    lamb[np.isnan(lamb)] = np.nanmin(lamb)
    raw_lamb = lamb * (1./framerate)

    #calculate the log-likelihood of the data and print
    ll = start_log_likelihood(lamb,test_data['spikes'],framerate)

#    print ll
    
    #divide by number of spikes in the session for log-likelihood per spike
    llps = ll/np.sum(test_data['spikes'])

    #calculate pearson r between the estimated spike train and
    #actual spike train, first smoothing with a gaussian filter
    smoothed_spikes = convolve(test_data['spikes'], Gaussian1DKernel(stddev=2,x_size=11))

    r,p = pearsonr(smoothed_spikes,raw_lamb)
    if np.isnan(r):
        r = 0
        
    #now calculate the percent of variance explained by the model estimates
    mean_fr = np.mean(smoothed_spikes)
    explained_var = 1 - np.nansum((smoothed_spikes - raw_lamb)**2)/np.sum((smoothed_spikes - mean_fr)**2)
    
    print('-----------------------')
    print(modeltype)
    print(' ')
    print('log-likelihood: %f' % ll)
    print('llps: %f' % llps)
    print('correlation: %f' % r)
    print('explained_var: %f' % explained_var)
    print(' ')
    print('-----------------------')

    #add relevant variables to the appropriate dictionary
    cdict['ll'] = ll
    if np.sum(test_data['spikes']) > 0:
        cdict['llps'] = float(ll/np.sum(test_data['spikes']))
    else:
        cdict['llps'] = ll
    cdict['lambda'] = lamb
    cdict['corr_r'] = r
    cdict['explained_var'] = explained_var
    cdict['allo_params'] = allo_params
    cdict['ego_params'] = ego_params
    cdict['spatial_params'] = spatial_params
    cdict['test_spikes'] = test_data['spikes']
    cdict['train_spikes'] = train_data['spikes']
    cdict['tot_spikes'] = np.sum(test_data['spikes'])
#    cdict['scale_factor'] = scale_factor
    
    
    return cdict

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
if __name__ == '__main__':
    
    #the order that we'll create our models in
#    order = [['uniform'],['allo'],['globalego'],['localego'],['spatial'],['globalego','allo'],['localego','allo'],['globalego','spatial'],['localego','spatial'],['globalego','localego'],['allo','spatial'],['localego','allo','spatial'],['globalego','allo','spatial'],['localego','globalego','allo'],['localego','globalego','spatial'],['localego','globalego','spatial','allo']]
    
    order = [['uniform'],['allo'],['ego'],['spatial'],['ego','allo'],['ego','spatial'],['allo','spatial'],['ego','allo','spatial']]
    

    #in number form, so numba can understand
    jitnums = [np.asarray([0],dtype=np.int64),np.asarray([1],dtype=np.int64),np.asarray([2],dtype=np.int64),np.asarray([3],dtype=np.int64),np.asarray([1,2],dtype=np.int64),np.asarray([2,3],dtype=np.int64),np.asarray([1,3],dtype=np.int64),np.asarray([1,2,3],dtype=np.int64)]

    #ask for the directory we'll be operating in
#    fdir = tkFileDialog.askdirectory(initialdir='G:\mm44\PoS')
#    fdir = '//ROOM3TAUBE/Users/Jeffrey Taube/Desktop/Patrick/TT03/2018-02-16_14-49-24 right'
#    fdir = 'H:/mm44/MEC'
#    fdir = 'H:/shawn_inactivation/SSW74/Baseline'
#    fdir = 'H:/PL37 grids/multi-level 10-13-17'
#    fdir = 'C:/Users/Jeffrey_Taube/Desktop/adn hdc'
    fdir = 'H:/Patrick/egocentric/PL61'

    #figure out what sessions (trials) are in the directory
    ops,trials = main.find_trials({'multi_session':False,'acq':'neuralynx'},fdir)
    #workaround
    ops['labview'] = False
    ops['single_cluster'] = False
#    ops['acq'] = 'neuralynx'
            
    count = 0
    #for every session...
    for trial in trials:
        count += 1
        if count > 1:
            continue
                
        #grab the name of the data file we need
        fname = trial+'/all_trial_data.pickle'
        #load the data file
        all_trial_data = load_data(fname)
        #grab the advanced options and collect the framerate
        adv = all_trial_data['adv']
        framerate=adv['framerate']

        #collect names of the clusters we'll be analyzing
        trial_data = main.read_files(ops,fdir,trial,metadata=True)
        cluster_names = trial_data['filenames']

        #for each cluster...
        for name in cluster_names:
            
            if cross_val:
                #start a list of lists to hold items for each segment of the session
                cdict = [[]]*10  
            else:
                cdict = {}
            #report which cluster we're working on                      
            print(name)
            
            #grab appropriate data from the data file
            trial_data = all_trial_data[name][0]['trial_data']
            cluster_data = all_trial_data[name][0]['cluster_data']
            spike_data = all_trial_data[name][0]['spike_data']

            #grab relevant tracking and spike data
            center_x = np.asarray(trial_data['center_x'])
            center_y = np.asarray(trial_data['center_y'])
            angles = np.asarray(trial_data['angles'])
            spike_train = spike_data['ani_spikes']

            #grab our data for the session
            data = assign_data(center_x, center_y, angles, spike_train)

            #estimate our parameters according to options set above
            allo_params, ego_params = optimize_params(data)
                
            rayleighs = np.zeros((len(ego_params),len(ego_params)))
            for i in range(len(ego_params)):
                for j in range(len(ego_params)):
                    mr = np.nansum(ego_params[i][j]*np.exp(1j*np.linspace(0,2*np.pi-(1/hd_bins)*2*np.pi,hd_bins)))/np.nansum(ego_params[i][j])
                    rayleighs[i][j] = np.abs(mr)
            
            top_spot = np.where(rayleighs==np.max(rayleighs))[0]

