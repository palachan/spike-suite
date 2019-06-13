# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:13:23 2017

log-likelihood model with cross-validation option

@author: Patrick
"""

#import important modules
import numpy as np
import numba as nb
import bisect
import math
from scipy.stats import pearsonr
from astropy.convolution.kernels import Gaussian1DKernel
from astropy.convolution import convolve
import copy
from scipy.sparse import csr_matrix


"""""""""""""""""""""""""""""
Decisions
"""""""""""""""""""""""""""""

local_ego = False

#how many bins should we use for spatial parameters? (e.g. 8 gives 8**2 = 64 bins)
gr=20
#how many bins should we use for head direction parameters?
hd_bins = 30
#how many bins should we use for ego parameters? (similar to spatial bins)
ego_gr = 8


def assign_data(center_x, center_y, angles, speeds, ahvs, novelties, break_points, cross_val=True, ego_loc = None):
    
    #figure out spatial bin edges for the arena
    h,yedges,xedges = np.histogram2d(center_x,center_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])
    
    if cross_val:
        
        train_dict = [[]]*10
        test_dict = [[]]*10
        
        Xp, Xa, Xe, Xs, Xahv, Xn = make_design_matrix(center_x,center_y,angles,speeds,ahvs,novelties,xedges,yedges,ego_loc)
        
        for fold in range(10):
            print('processing fold %d data' % fold)
            
            train_dict[fold] = {}
            test_dict[fold] = {}
            
            slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                                  break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
            
            test_dict[fold]['Xp'] = csr_matrix(Xp[slices])
            test_dict[fold]['Xa'] = csr_matrix(Xa[slices])
            test_dict[fold]['Xe'] = csr_matrix(Xe[slices])
            test_dict[fold]['Xs'] = csr_matrix(Xs[slices])
            test_dict[fold]['Xahv'] = csr_matrix(Xahv[slices])
            test_dict[fold]['Xn'] = csr_matrix(Xn[slices])

            train_dict[fold]['Xp'] = csr_matrix(np.delete(Xp,slices,axis=0))
            train_dict[fold]['Xa'] = csr_matrix(np.delete(Xa,slices,axis=0))
            train_dict[fold]['Xe'] = csr_matrix(np.delete(Xe,slices,axis=0))
            train_dict[fold]['Xs'] = csr_matrix(np.delete(Xs,slices,axis=0))
            train_dict[fold]['Xahv'] = csr_matrix(np.delete(Xahv,slices,axis=0))
            train_dict[fold]['Xn'] = csr_matrix(np.delete(Xn,slices,axis=0))
            
        return train_dict,test_dict
    
    else:
        
        data = {}
        
        Xp, Xa, Xe, Xs, Xahv, Xn = make_design_matrix(center_x,center_y,angles,speeds,ahvs,novelties,xedges,yedges,ego_loc)
        
        data['Xe'] = csr_matrix(Xe)
        data['Xa'] = csr_matrix(Xa)
        data['Xp'] = csr_matrix(Xp)  
        data['Xs'] = csr_matrix(Xs)
        data['Xahv'] = csr_matrix(Xahv)
        data['Xn'] = csr_matrix(Xn)
        
        return data
    
def assign_phases(phases,break_points,fold,cross_val=True):
    
    if cross_val:
        
        slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                              break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
        
        test_phases = copy.deepcopy(phases)[slices]
        test_Xt = make_phase_matrix(test_phases)
        test_Xt = csr_matrix(test_Xt)
        
        train_phases = np.delete(copy.deepcopy(phases),slices)
        train_Xt = make_phase_matrix(train_phases)
        train_Xt = csr_matrix(train_Xt)
        
        return train_Xt,test_Xt
    
    else:
        
        Xt = make_phase_matrix(phases)
        return csr_matrix(Xt)
    
def make_phase_matrix(phases):
    
    Xt = np.zeros((len(phases),hd_bins))
    
    for j in range(len(phases)):
        Xt[j][phases[j]] = 1
        
    return Xt
    
def assign_spikes(spike_train, break_points, fold, cross_val=True):

    if cross_val:

        slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                              break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
        
        test_spikes = copy.deepcopy(spike_train)[slices]
        test_spikes = np.asarray(test_spikes,dtype=np.float)
        
        train_spikes = np.delete(copy.deepcopy(spike_train),slices)
        train_spikes = np.asarray(train_spikes,dtype=np.float)
        
        return train_spikes, test_spikes
    
    else:
        
        spikes = np.asarray(spike_train,dtype=np.float)
        
        return spikes
    

def make_design_matrix(center_y,center_x,angles,speeds,ahvs,novelties,xedges,yedges,ego_loc = None):
    """ transform allocentric to egocentric angles and assign dwell times """

    Xp = np.zeros((len(center_x),gr,gr))
    Xa = np.zeros((len(center_x),hd_bins))
    Xs = np.zeros((len(center_x),20))
    Xahv = np.zeros((len(center_x),20))
    Xn = np.zeros((len(center_x),20))
    
    #make arrays for assigning bins to spatial and directional
    #tracking and spike data
    allo_bins = np.ones((len(angles)),dtype=np.int)
    pos_bins = np.ones((2,len(center_x)),dtype=np.int)
    ahv_bins = np.ones((len(ahvs)),dtype=np.int)
    speed_bins = np.ones((len(speeds)),dtype=np.int)
    novelty_bins = np.ones((len(novelties)),dtype=np.int)
    
    speed_cutoffs = np.arange(0,101,5)
    ahv_cutoffs = np.arange(-250,251,25)
    novelty_cutoffs = np.arange(np.min(novelties),np.max(novelties),np.ptp(novelties)/20.)

    #for every video frame in the session...
    for i in range(len(angles)):
        #find the closest x and y spatial bins for this frame
        x_bin = bisect.bisect_left(xedges,center_x[i]) - 1
        y_bin = bisect.bisect_left(yedges,center_y[i]) - 1
        #assign spatial bins to appropriate array
        if x_bin == -1:
            x_bin = 0
        elif x_bin == gr:
            x_bin = gr-1
        if y_bin == -1:
            y_bin = 0
        elif y_bin == gr:
            y_bin = gr - 1
        pos_bins[0][i] = x_bin
        pos_bins[1][i] = y_bin
        
        #calculate the head direction bin for this frame
        hd_bin = int(angles[i]/(360/hd_bins))
        #assign it the appropriate array
        allo_bins[i] = hd_bin
        
        speed_bin = bisect.bisect_left(speed_cutoffs,speeds[i]) - 1
        if speed_bin == -1:
            speed_bin = 0
        if speed_bin == 20:
            speed_bin = 20-1
        speed_bins[i] = speed_bin
        
        ahv_bin = bisect.bisect_left(ahv_cutoffs,ahvs[i]) - 1
        if ahv_bin == -1:
            ahv_bin = 0
        if ahv_bin == 20:
            ahv_bin = 20-1
        ahv_bins[i] = ahv_bin
        
        novelty_bin = bisect.bisect_left(novelty_cutoffs,novelties[i]) - 1
        if novelty_bin == -1:
            novelty_bin = 0
        if novelty_bin == 20:
            novelty_bin = 20-1
        novelty_bins[i] = novelty_bin
        
    for j in range(len(center_x)):
        Xp[j][np.int(pos_bins[0][j])][np.int(pos_bins[1][j])] = 1
        Xa[j][np.int(allo_bins[j])] = 1
        Xs[j][np.int(speed_bins[j])] = 1
        Xahv[j][np.int(ahv_bins[j])] = 1
        Xn[j][np.int(novelty_bins[j])] = 1
        
    def single_ego_loop(center_x,center_y,angles,ego_loc):
        
        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(ego_gr)
        ycoords = np.zeros(ego_gr)

        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(ego_gr):
            xcoords[x] = (np.float(x+.5)/np.float(ego_gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
            ycoords[x] = np.float(np.max(center_y)) - (np.float(x+.5)/np.float(ego_gr))*np.float((np.max(center_y)-np.min(center_y)))

        Xe = np.zeros((len(center_x),hd_bins))
        
        cue_y = ycoords[ego_loc[0]]
        cue_x = xcoords[ego_loc[1]]
        
        #calc array of egocentric angles of this bin from pos x axis centered 
        #on animal using arctan
        new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
        #calculate ecd angles by subtracting allocentric
        #angles from egocentric angles
        ecd_angles = (new_angles-angles)%360
        #assign to bin
        ego_bins = ecd_angles/(360/hd_bins)
        
#        dists = np.sqrt((cue_y - center_y)**2 + (cue_x - center_x)**2)
#        dists[dists < 10] = np.nan
#        
#        ego_bins[np.isnan(dists)] = np.nan

        for k in range(len(center_x)):
            if not np.isnan(ego_bins[k]):
                Xe[k][np.int(ego_bins[k])] = 1.
            
        return Xe
        

    @nb.jit(nopython=True)
    def ego_loop(center_x,center_y,angles):
        
        ego_radius = 100

        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(ego_gr)
        ycoords = np.zeros(ego_gr)
        
        Xe = np.zeros((len(center_x),ego_gr,ego_gr,hd_bins))
        ego_bins = np.zeros((ego_gr,ego_gr,len(center_x)))
    
        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(ego_gr):
            xcoords[x] = (np.float(x+.5)/np.float(ego_gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
            ycoords[x] = np.float(np.max(center_y)) - (np.float(x+.5)/np.float(ego_gr))*np.float((np.max(center_y)-np.min(center_y)))
        
        #for each y position...
        for i in range(ego_gr):
    
            #fill an array with the current y coord
            cue_y = ycoords[i]
    
            #for each x position...
            for j in range(ego_gr):
                                                    
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
                
#                dists = np.sqrt((cue_y - center_y)**2 + (cue_x - center_x)**2)
#                dists[dists < 10] = np.nan
#                
#                if local_ego:
#                    dists[dists > ego_radius] = np.nan
#                
#                ecd_bins[np.isnan(dists)] = np.nan
                ego_bins[i][j] = ecd_bins

        for k in range(len(center_x)):
            for i in range(ego_gr):
                for j in range(ego_gr):
                    if not np.isnan(ego_bins[i][j][k]):
                        Xe[k][i][j][np.int(ego_bins[i][j][k])] = 1.
                        
        return Xe

    
    if ego_loc is not None:
        Xe = single_ego_loop(center_y,center_x,angles,ego_loc)
        print Xe
        print np.sum(Xe)
    else:
        Xe = ego_loop(center_y,center_x,angles)
        Xe = Xe.reshape((len(center_x),hd_bins*ego_gr**2))
    print np.shape(Xe)
    Xp = Xp.reshape((len(center_x),gr**2))

    #return appropriate arrays
    return Xp, Xa, Xe, Xs, Xahv, Xn

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
    for i in range(ego_gr):
        for j in range(ego_gr):
            nans, x = nan_helper(ego_params[i][j])
            ego_params[i][j][nans] = np.interp(x(nans), x(~nans), ego_params[i][j][~nans], period=hd_bins)
            
    #return our interpolated parameters
    return spatial_params, allo_params, ego_params

def calc_scale_factor(params,train_data,modeltype):
    ''' calculate a scaling factor that optimizes the fit to the training data '''
    
    #grab training spike train and covariate matrix
    spike_train=train_data['spikes']
    Xp=train_data['Xp']
    Xe=train_data['Xe']
    Xa=train_data['Xa']
    Xs=train_data['Xs']
    Xahv=train_data['Xahv']
    Xt=train_data['Xt']
    Xn=train_data['Xn']
    
    lamb = np.ones(len(spike_train))
    
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

    #make a scaling factor such that the rate vector adds up to the correct
    #number of spikes from the training data to satisfy the assumption that
    # sum( y(t) - rate(t) ) = 0 (Cameron and Windmeijer, 1995)
    scale_factor = np.mean(spike_train)/np.mean(lamb)

    #note what we've done
    print('scale_factor')
    print(scale_factor)
    
    #return it
    return scale_factor

def run_model(scale_factor, train_data, test_data, params, modeltype = None, single_ego=False):
    ''' run all versions of the model using our optimized parameters '''

    #create dictionary for holding values
    cdict = {}

    #collect test spike train
    spike_train=test_data['spikes']
    
    #if not a uniform firing rate model...
    if 'uniform' not in modeltype:

        Xp=test_data['Xp']
        Xe=test_data['Xe']
        Xa=test_data['Xa']
        Xs=test_data['Xs']
        Xahv=test_data['Xahv']
        Xt=test_data['Xt']
        Xn=test_data['Xn']
        
        lamb = np.ones(len(spike_train))
        
        if params['allo'] is not None:
            lamb *= (Xa * np.exp(params['allo']))
            cdict['allo_params'] = np.exp(params['allo'])
        if params['ego'] is not None:
            lamb *= (Xe * np.exp(params['ego']))
            if single_ego:
                cdict['ego_params'] = np.exp(params['ego'])
            else:
                cdict['ego_params'] = np.reshape(np.exp(params['ego']),(ego_gr,ego_gr,hd_bins))
        if params['spatial'] is not None:
            lamb *= (Xp * np.exp(params['spatial']))
            cdict['spatial_params'] = np.reshape(np.exp(params['spatial']),(gr,gr))
        if params['speed'] is not None:
            lamb *= (Xs * np.exp(params['speed']))
            cdict['speed_params'] = np.exp(params['speed'])
        if params['ahv'] is not None:
            lamb *= (Xahv * np.exp(params['ahv']))
            cdict['ahv_params'] = np.exp(params['ahv'])
        if params['theta'] is not None:
            lamb *= (Xt * np.exp(params['theta']))
            cdict['theta_params'] = np.exp(params['theta'])
        if params['novelty'] is not None:
            lamb *= (Xn * np.exp(params['novelty']))
            cdict['novelty_params'] = np.exp(params['novelty'])
    
        f = -np.nansum(lamb * scale_factor - spike_train*np.log(lamb * scale_factor))

    #if this is a uniform model...
    else:
        #the rate for every frame is just the average spikes/frame of the train data
        lamb = np.full(len(spike_train),np.mean(train_data['spikes']))
        #calculate the log-likelihood old-school
        f = -np.sum(lamb - spike_train*np.log(lamb))
        
    #start array for log-factorials
    lgammas = np.zeros(len(spike_train))
        
    #for every time point...
    for h in range(len(spike_train)):
        #calculate the log-factorial of the number of spikes during that frame,
        #using base 2 log and gamma function on (n_spikes + 1) so numba can work
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    #subtract the log-factorials
    f -= np.sum(lgammas)
    
    #change from nats to bits
    f = f/np.log(2)

    #divide by number of spikes in the session for log-likelihood per spike
    llps = f/np.sum(test_data['spikes'])

    #calculate pearson r between the estimated spike train and
    #actual spike train, first smoothing with a gaussian filter
    smoothed_spikes = convolve(test_data['spikes'], Gaussian1DKernel(stddev=2,x_size=11))

    r,p = pearsonr(smoothed_spikes,lamb*scale_factor)
    if np.isnan(r):
        r = 0
        
    #now calculate the percent of variance explained by the model estimates
    mean_fr = np.mean(smoothed_spikes)
    explained_var = 1 - np.nansum((smoothed_spikes - lamb*scale_factor)**2)/np.sum((smoothed_spikes - mean_fr)**2)
    
    pseudo_r2 = 1 - np.nansum(test_data['spikes'] * np.log(test_data['spikes'] / (lamb*scale_factor)) - (test_data['spikes'] - lamb*scale_factor)) / np.nansum(test_data['spikes'] * np.log(test_data['spikes'] / np.mean(test_data['spikes'])))
    uniform_r2 = 0
#    1 - np.nansum(test_data['spikes'] * np.log(test_data['spikes'] / (np.mean(lamb*scale_factor))) - (test_data['spikes'] - np.mean(lamb*scale_factor))) / np.nansum(test_data['spikes'] * np.log(test_data['spikes'] / np.mean(test_data['spikes'])))
    
    pseudo_r2 = pseudo_r2 - uniform_r2
    
    print('-----------------------')
    print(modeltype)
    print(' ')
    print('log-likelihood: %f' % f)
    print('llps: %f' % llps)
    print('correlation: %f' % r)
    print('explained_var: %f' % explained_var)
    print('pseudo_r2: %f' % pseudo_r2)
    print(' ')
    print('-----------------------')

    #add relevant variables to the appropriate dictionary
    cdict['ll'] = f
    if np.sum(test_data['spikes']) > 0:
        cdict['llps'] = float(f/np.sum(test_data['spikes']))
    else:
        cdict['llps'] = f
    cdict['lambda'] = lamb * scale_factor
    cdict['corr_r'] = r
    cdict['pseudo_r2'] = pseudo_r2
    cdict['explained_var'] = explained_var
    cdict['test_spikes'] = test_data['spikes']
    cdict['train_spikes'] = train_data['spikes']
    cdict['tot_spikes'] = np.sum(test_data['spikes'])
    cdict['scale_factor'] = scale_factor

    return cdict
