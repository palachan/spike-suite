# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:17:55 2018


@author: Patrick
"""
import os
import numpy as np
import math
from scipy.optimize import minimize
from scipy.sparse import kron, spdiags, eye, csr_matrix, hstack
from scipy.stats import wilcoxon
from itertools import chain, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
import copy
import csv

def compute_diags(bin_nums):
    ''' create diagonal matrices for grouped penalization -- implementation 
    modified from Hardcastle 2017 '''
        
    xbin_num = bin_nums['pos_x']
    ybin_num = bin_nums['pos_y']
    hd_bin_num = bin_nums['hd']
    speed_bin_num = bin_nums['speed']
    ahv_bin_num = bin_nums['ahv']
    bearing_bin_num = bin_nums['bearing']
    dist_bin_num = bin_nums['dist']
    
    smooth_dict = {}

    ''' for position variable '''
    pos_ones = np.ones(xbin_num)
    D1 = spdiags([-pos_ones,pos_ones],[0,1],xbin_num-1,xbin_num)
    DD1 = D1.T * D1
    
    pos_ones = np.ones(ybin_num)
    D2 = spdiags([-pos_ones,pos_ones],[0,1],ybin_num-1,ybin_num)
    DD2 = D2.T * D2
    
    M1 = kron(eye(xbin_num),DD2)
    M2 = kron(DD1,eye(ybin_num))
    spatial_diag = M1+M2
    spatial_diag = np.asarray(spatial_diag.todense())
    
    smooth_dict['pos'] = spatial_diag


    ''' for HD variable '''
    pos_ones = np.ones(hd_bin_num)
    circ1 = spdiags([-pos_ones,pos_ones],[0,1],hd_bin_num-1,hd_bin_num)
    hd_diag = circ1.T * circ1
    hd_diag=np.asarray(hd_diag.todense())
    hd_diag[0] = np.roll(hd_diag[1],-1)
    hd_diag[hd_bin_num-1] = np.roll(hd_diag[hd_bin_num-2],1)
    
    smooth_dict['hd'] = hd_diag
        

    ''' for speed variable '''
    pos_ones = np.ones(speed_bin_num)
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],speed_bin_num-1,speed_bin_num)
    speed_diag = noncirc1.T * noncirc1
    speed_diag = np.asarray(speed_diag.todense())
    
    smooth_dict['speed'] = speed_diag
    
    
    ''' for ahv variable '''
    pos_ones = np.ones(ahv_bin_num)
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],ahv_bin_num-1,ahv_bin_num)
    ahv_diag = noncirc1.T * noncirc1
    ahv_diag = np.asarray(ahv_diag.todense())
    
    smooth_dict['ahv'] = ahv_diag

    
    ''' for bearing variable '''
    pos_ones = np.ones(bearing_bin_num)
    circ1 = spdiags([-pos_ones,pos_ones],[0,1],bearing_bin_num-1,bearing_bin_num)
    bearing_diag = circ1.T * circ1
    bearing_diag=np.asarray(bearing_diag.todense())
    bearing_diag[0] = np.roll(bearing_diag[1],-1)
    bearing_diag[bearing_bin_num-1] = np.roll(bearing_diag[bearing_bin_num-2],1)
    
    smooth_dict['bearing'] = bearing_diag
    
    
    ''' for dist variable '''
    pos_ones = np.ones(dist_bin_num)
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],dist_bin_num-1,dist_bin_num)
    dist_diag = noncirc1.T * noncirc1
    dist_diag = np.asarray(dist_diag.todense())
    
    smooth_dict['dist'] = dist_diag
    
    return smooth_dict


def objective(params,X,spike_train,param_counts,penalties,smoothers,smooth=True):
        
    u = X * params
    rate = np.exp(u)
    
    f = np.sum(rate - spike_train * u)
    grad = X.T * (rate - spike_train)
    
    if smooth:
        fpen,fgrad = penalize(params,X,spike_train,param_counts,penalties,smoothers)
        f += fpen
        grad += fgrad
    
    return f,grad

def penalize(params,X,spike_train,param_counts,penalties,smoothers):
    
    counter = 0
    for i in range(len(param_counts)):
        
        params0 = params[counter:counter+param_counts[i]]

        if i == 0:
            params0 = params[counter:counter+param_counts[i]]
            f = np.sum(penalties[i] * .5 * np.dot(params0.T, smoothers[i]) * params0)
            grad = penalties[i] * np.dot(smoothers[i], params0)
            
        else:
            f += np.sum(penalties[i] * .5 * np.dot(params0.T, smoothers[i]) * params0)
            grad = np.concatenate((grad, penalties[i] * np.dot(smoothers[i], params0)))
            
        counter += param_counts[i]
        
    return f, grad

    
def make_X(trial_data, variables, bin_nums):

    xbin_num = bin_nums['pos_x']
    ybin_num = bin_nums['pos_y']
    hd_bin_num = bin_nums['hd']
    speed_bin_num = bin_nums['speed']
    ahv_bin_num = bin_nums['ahv']
    bearing_bin_num = bin_nums['bearing']
    dist_bin_num = bin_nums['dist']

    behavior_matrices = {}
    X_list = []
    
    if 'pos' in variables:
        center_x = np.asarray(trial_data['center_x'])
        center_y = np.asarray(trial_data['center_y'])
        xbins = np.digitize(center_x,np.arange(np.min(center_x),np.max(center_x),6.)) - 1
        ybins = np.digitize(center_y,np.arange(np.min(center_y),np.max(center_y),6.)) - 1
        xbin_num = len(np.arange(np.min(center_x),np.max(center_x),6.))
        ybin_num = len(np.arange(np.min(center_y),np.max(center_y),6.))
        Xp = np.zeros((len(center_x),xbin_num,ybin_num))
        for i in range(len(xbins)):
            Xp[i][xbins[i]][ybins[i]] = 1.
        Xp = np.reshape(Xp,(len(center_x),xbin_num * ybin_num))
        behavior_matrices['pos'] = csr_matrix(Xp)
        X_list.append(csr_matrix(Xp))
        
    if 'hd' in variables:
        angles = np.asarray(trial_data['angles'])
        angle_bins = np.digitize(angles,np.linspace(0,360,hd_bin_num,endpoint=False)) - 1
        Xa = np.zeros((len(angles),hd_bin_num))
        for i in range(len(angle_bins)):
            Xa[i][angle_bins[i]] = 1.
        behavior_matrices['hd'] = csr_matrix(Xa)
        X_list.append(csr_matrix(Xa))

    if 'speed' in variables:
        speeds = np.asarray(trial_data['speeds'])
        speed_bins = np.digitize(speeds,np.linspace(0,40,speed_bin_num,endpoint=False)) - 1
        Xs = np.zeros((len(speeds),speed_bin_num))
        for i in range(len(speed_bins)):
            Xs[i][speed_bins[i]] = 1.
        behavior_matrices['speed'] = csr_matrix(Xs)
        X_list.append(csr_matrix(Xs))
        
    if 'ahv' in variables:
        ahvs = np.asarray(trial_data['ahvs'])
        ahv_bins = np.digitize(ahvs,np.linspace(-200,200,ahv_bin_num,endpoint=False)) - 1
        Xahv = np.zeros((len(ahvs),ahv_bin_num))
        for i in range(len(ahv_bins)):
            Xahv[i][ahv_bins[i]] = 1.   
        behavior_matrices['ahv'] = csr_matrix(Xahv)
        X_list.append(csr_matrix(Xahv))
        
    if 'bearing' in variables:
        bearings = np.asarray(trial_data['center_bearings'])
        bearing_bins = np.digitize(bearings,np.linspace(0,360,bearing_bin_num,endpoint=False)) - 1
        Xe = np.zeros((len(bearings),bearing_bin_num))
        for i in range(len(bearing_bins)):
            Xe[i][bearing_bins[i]] = 1.    
        behavior_matrices['bearing'] = csr_matrix(Xe)
        X_list.append(csr_matrix(Xe))
        
    if 'dist' in variables:
        dists = np.asarray(trial_data['center_dists'])
        dist_bins = np.digitize(dists,np.linspace(0,np.max(dists),dist_bin_num,endpoint=False))-1
        Xd = np.zeros((len(dists),dist_bin_num))
        for i in range(len(bearing_bins)):
            Xd[i][dist_bins[i]] = 1.    
        behavior_matrices['dist'] = csr_matrix(Xd)
        X_list.append(csr_matrix(Xd))
        
    X = hstack(X_list)
    
    return X, behavior_matrices


def split_data(X,behavior_matrices,variables,spike_train,fold):

    test_matrices = copy.deepcopy(behavior_matrices)
    train_matrices = copy.deepcopy(behavior_matrices)
    
    break_points = np.linspace(0,len(spike_train),51).astype(np.int)


    slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                          break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
    
    test_spikes = spike_train[slices]
    test_X = csr_matrix(X.todense()[slices])
    for var in variables:
        test_matrices[var] = csr_matrix(test_matrices[var].todense()[slices])
    
    train_spikes = np.delete(spike_train,slices,axis=0)
    train_X = csr_matrix(np.delete(X.todense(),slices,axis=0))
    for var in variables:
        train_matrices[var] = csr_matrix(np.delete(train_matrices[var].todense(),slices,axis=0))
    
    return test_spikes,test_X,test_matrices,train_spikes,train_X,train_matrices
    

def calc_scale_factor(model,param_dict,train_matrices,train_spikes):
    
    u = np.zeros(len(train_spikes))
    
    for var in model:
        u += train_matrices[var] * param_dict[var]
    
    rate = np.exp(u)
    
    scale_factor = np.sum(train_spikes)/np.sum(rate)
    
    return scale_factor
    

def run_final(model,scale_factor,param_dict,behavior_matrices,spike_train):
    
    ''' run the model and collect the results '''
    
    if model != 'uniform':
    
        u = np.zeros(len(spike_train))
        
        for var in model:
            u += behavior_matrices[var] * param_dict[var]
        
        rate = scale_factor * np.exp(u)
        
    else:
        
        rate = np.full(len(spike_train),np.mean(spike_train))
    
    f = -np.sum(rate - spike_train*np.log(rate))
    
    lgammas = np.zeros(len(spike_train))
    for h in range(len(spike_train)):
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    f -= np.sum(lgammas)
    
    #change from nats to bits
    f = f/np.log(2)

    if np.sum(spike_train) > 0:
        llps = f/np.sum(spike_train)
    else:
        llps = f
    
    return {'llps':llps}
    

def get_all_models(variables):
    ''' convenience function for calculating all possible
    combinations of nagivational variables '''
    
    def powerset(variables):
        return list(chain.from_iterable(combinations(variables, r) for r in range(1,len(variables)+1)))
    
    all_models = powerset(variables)
    
    for i in range(len(all_models)):
        all_models[i] = frozenset(all_models[i])
    
    return all_models


def select_model(cdict,variables):
    ''' perform forward search procedure to choose the best model based on
    cross-validation results '''
    
    #models we'll start with are single-variable models
    models = copy.deepcopy(variables)
    #we haven't found a best model yet so set to NONE
    best_model = None
        
    #while we haven't found a winning model...
    while best_model == None:
        
        #make models frozensets to make things easier
        for i in range(len(models)):
            models[i] = frozenset(models[i])
        
        #start dict for llps measures
        ll_dict = {}
        
        #for each fold in the cross-val...
        for fold in range(10):
            #for each model...
            for modeltype in models:
                #start an entry for that model if we're just starting
                if fold == 0:
                    ll_dict[modeltype] = []
                #collect the llps increase compared to the uniform model
                ll_dict[modeltype].append(cdict[fold][modeltype]['llps']-cdict[fold]['uniform']['llps'])
        
        #make a dict that contains the median llps value for each model
        median_dict = {}
        for modeltype in models:
            median_dict[modeltype] = np.median(ll_dict[modeltype])
        #let's say the potential best new model is the one with the highest median score
        top_model = max(median_dict.keys(), key=(lambda key: median_dict[key]))
        
        #if the top model is a single variable...
        if len(top_model) == 1:
            #set the top model llps data as 'last_model' data
            last_model = ll_dict[top_model]
            #set the top model the 'last_modeltype'
            last_modeltype = top_model
            #create the next set of models -- the current best model plus each new variable
            #-- then start over
            models = []
            for var in variables:
                if var[0] not in top_model:
                    models.append(frozenset(chain(list(top_model),list(var))))
        #otherwise...
        else:
            #use wilcoxon signed ranks to see if the new model is better than the last model
            w,p = wilcoxon(last_model,ll_dict[top_model])
            #if the new model is better...
            if np.median(last_model) < np.median(median_dict[top_model]) and p < .1:
                #if we can't add any more variables...
                if len(top_model) == len(variables):
                    #test to see if the top model is better than the null model
                    w,p = wilcoxon(ll_dict[top_model],np.zeros(10))
                    #if it is, then this is the best model!
                    if np.median(ll_dict[top_model]) > 0 and p < .1:
                        best_model = top_model
                    #otherwise, the cell is unclassifiable
                    else:
                        best_model = 'uniform'
                    
                #otherwise, set this model's llps data as 'last_model' data
                last_model = ll_dict[top_model]
                #set this modeltype as 'last_modeltype'
                last_modeltype = top_model
                #make new set of models -- current best model plus each new variable
                #-- then start over
                models = []
                for var in variables:
                    if var[0] not in top_model:
                        models.append(frozenset(chain(list(top_model),list(var))))
            #otherwise, the best model is probably the last model
            else:
                #check if the last model is better than the null  model
                w,p = wilcoxon(last_model,np.zeros(10))
                #if it is, then this is the best model!
                if np.median(last_model) > 0 and p < .1:
                    best_model = last_modeltype
                #otherwise, the cell is unclassifiable
                else:
                    best_model = 'uniform'
                    
    #return the best model
    return best_model


def run_classifier(class_ops,adv,trial_data,cluster_data,spike_train):

    #all possible behavioral correlates we'll be analyzing
    #variable_order = position, HD, speed, ahv, bearing, distance
    possible_variables = [('pos'),('hd'),('speed'),('ahv'),('bearing'),('dist')]
    
    #variables we'll actually be using
    variables = []
    for var in possible_variables:
        if class_ops[var]:
            variables.append((var))

    if len(variables) == 0:
        return

    print('classifying...')

    #set number of bins for each variable
    bin_nums = {}
    bin_nums['pos_x'] = int(np.ceil((np.max(trial_data['center_x']) - np.min(trial_data['center_x']))/6.))
    bin_nums['pos_y'] = int(np.ceil((np.max(trial_data['center_y']) - np.min(trial_data['center_y']))/6.))
    bin_nums['hd'] = 30
    bin_nums['speed'] = 10
    bin_nums['ahv'] = 21
    bin_nums['bearing'] = 30
    bin_nums['dist'] = 10
    bin_nums['dist'] = int(np.ceil((np.max(trial_data['center_dists']) - np.min(trial_data['center_dists']))/6.))

    
    penalty_dict = {}
    #set smoothing penalty for each variable
    penalty_dict['pos'] = 2.
    penalty_dict['hd'] = 20.
    penalty_dict['speed'] = 20.
    penalty_dict['ahv'] = 20.
    penalty_dict['bearing'] = 20.
    penalty_dict['dist'] = 20.
    
    smooth_dict = compute_diags(bin_nums)
    
    #make a list of all possible models
    all_models = get_all_models(variables)
    
    X, behavior_matrices = make_X(trial_data, variables, bin_nums)
    
    param_counts = []
    penalties = []
    smoothers = []
    for var in variables:
        
        if var == 'pos':
            param_counts.append(bin_nums['pos_x'] * bin_nums['pos_y'])
        else:
            param_counts.append(int(bin_nums[var]))
            
        penalties.append(float(penalty_dict[var]))
        smoothers.append(smooth_dict[var])
    
    cdict = {}

    for fold in range(10):
        cdict[fold] = {}

        test_spikes,test_X,test_matrices,train_spikes,train_X,train_matrices = split_data(X,behavior_matrices,variables,spike_train,fold)

        params = np.zeros(np.shape(train_X)[1])

        result = minimize(objective,params,args=(train_X,train_spikes,param_counts,penalties,smoothers),jac=True,method='L-BFGS-B')

        params = result.x
        
        param_dict = {}
        
        counter = 0
        for i in range(len(variables)):
            param_dict[variables[i]] = params[counter:counter + param_counts[i]]
            counter += param_counts[i]
        
        for model in all_models:
            
            scale_factor = calc_scale_factor(model,param_dict,train_matrices,train_spikes)
            cdict[fold][model] = run_final(model,scale_factor,param_dict,test_matrices,test_spikes)
        
        cdict[fold]['uniform'] = run_final('uniform',1.,param_dict,test_matrices,test_spikes)

    best_model = select_model(cdict, [(v,) for v in variables])
    print('best model:')
    if best_model == 'uniform':
        print('uniform')
    else:
        print(list(best_model))
    print('')
    
    if class_ops['save_profiles']:
        
        savedir = cluster_data['new_folder'] + '/glm_profiles'
        if 'dist' in variables:
            max_dist = np.max(trial_data['center_dists'])
            plot_params(adv,variables,bin_nums,behavior_matrices,param_dict,spike_train,savedir,max_dist=max_dist)
        else:
            plot_params(adv,variables,bin_nums,behavior_matrices,param_dict,spike_train,savedir)
            
        if best_model == 'uniform':
            best_model_list = ['uniform']
        else:
            best_model_list = list(best_model)
            
        with open(savedir + '/best_model.txt','w') as f:
            writer = csv.writer(f)
            writer.writerow(best_model_list)
    
    return best_model


def plot_params(adv,variables,bin_nums,behavior_matrices,param_dict,spike_train,savedir,max_dist=None):
    
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
        
    os.makedirs(savedir)
    
    framerate = adv['framerate']
    
    if 'pos' in variables:
        
        xbin_num = bin_nums['pos_x']
        ybin_num = bin_nums['pos_y']
        
        pos_params = param_dict['pos'].reshape((xbin_num,ybin_num))
        
        scale_factor = calc_scale_factor(frozenset(('pos',)),param_dict,behavior_matrices,spike_train)
    
        heatmap = np.reshape(np.exp(pos_params) * scale_factor, (xbin_num,ybin_num)) * framerate
        fig = plt.figure()
        ax = fig.add_subplot(111)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('left', size='5%', pad=0.05)
        im = ax.imshow(heatmap.T,vmin=0,vmax = np.ceil(np.nanmax(heatmap)),cmap='jet',origin='lower') 
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cbar.set_ticks([0,np.ceil(np.max(heatmap))])
        cbar.set_ticklabels(['0 Hz','%i Hz' % np.ceil(np.nanmax(heatmap))])
        ax.axis('off')
        ax.axis('equal')
    
        fig.savefig(savedir + '/pos_params.png',dpi=adv['pic_resolution'])
        plt.close()
    
    if 'hd' in variables:
        
        bin_num = bin_nums['hd']
        
        hd_params = param_dict['hd']
    
        scale_factor = calc_scale_factor(frozenset(('hd',)),param_dict,behavior_matrices,spike_train)
    
        hd_curve = np.exp(hd_params) * scale_factor * framerate
        hd_curve = np.concatenate((hd_curve,hd_curve[0,np.newaxis]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
        ax.plot(hd_curve,'k-',linewidth=3.0)
        ax.set_xlim([0,bin_nums['hd']])
        ax.set_xticks([0,bin_num/4.,bin_num/2.,3.*bin_num/4.,bin_num])
        ax.set_xticklabels([0,90,180,270,360])
        ax.set_xlabel('Head direction (deg)')
        ax.set_ylabel('Firing rate (spikes/s)')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1.2 * np.max(hd_curve)])
        plt.tight_layout()
        fig.savefig(savedir + '/hd_params.png',dpi=adv['pic_resolution'])
        
        plt.close()

    if 'speed' in variables:
        
        bin_num = bin_nums['speed']
        
        speed_params = param_dict['speed']
        
        bin_size = 40./bin_num
        xvals = np.linspace(0,40,bin_num,endpoint=False) + bin_size/2.
    
        scale_factor = calc_scale_factor(frozenset(('speed',)),param_dict,behavior_matrices,spike_train)
    
        speed_curve = np.exp(speed_params) * scale_factor * framerate
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xvals,speed_curve,'k-',linewidth=3.0)
        ax.set_xlim([0,40])
        ax.set_xticks([0,10,20,30,40])
        ax.set_xlabel('Speed (cm/s)')
        ax.set_ylabel('Firing rate (spikes/s)')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1.2 * np.max(speed_curve)])
        plt.tight_layout()
        fig.savefig(savedir + '/speed_params.png',dpi=adv['pic_resolution'])
        plt.close()
    
    if 'ahv' in variables:
        
        bin_num = bin_nums['ahv']
        
        ahv_params = param_dict['ahv']
    
        scale_factor = calc_scale_factor(frozenset(('ahv',)),param_dict,behavior_matrices,spike_train)
    
        ahv_curve = np.exp(ahv_params) * scale_factor * framerate
        ymax = 1.2 * np.max(ahv_curve)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ahv_curve,'k-',linewidth=3.0)
        ax.vlines((bin_num-1)/2.,ymin=0,ymax=ymax,alpha=0.5)
        ax.set_xlim([0,bin_num-1])
        ax.set_xticks([0,(bin_num-1)/4.,(bin_num-1)/2.,3.*(bin_num-1)/4.,bin_num-1])
        ax.set_xticklabels([-200,-100,0,100,200])
        ax.set_xlabel('Angular head velocity (deg/s)')
        ax.set_ylabel('Firing rate (spikes/s)')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
        ax.set_ylim([0,ymax])
        plt.tight_layout()
        fig.savefig(savedir + '/ahv_params.png',dpi=adv['pic_resolution'])
        plt.close()
        
    if 'bearing' in variables:
        
        bin_num = bin_nums['bearing']
        
        bearing_params = param_dict['bearing']
    
        scale_factor = calc_scale_factor(frozenset(('bearing',)),param_dict,behavior_matrices,spike_train)
    
        bearing_curve = np.exp(bearing_params) * scale_factor * framerate
        bearing_curve = np.concatenate((bearing_curve,bearing_curve[0,np.newaxis]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bearing_curve,'k-',linewidth=3.0)
        ax.set_xlim([0,bin_num])
        ax.set_xticks([0,bin_num/4.,bin_num/2.,3.*bin_num/4.,bin_num])
        ax.set_xticklabels([0,90,180,270,360])
        ax.set_xlabel('Center bearing (deg)')
        ax.set_ylabel('Firing rate (spikes/s)')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1.2 * np.max(bearing_curve)])
        plt.tight_layout()
        fig.savefig(savedir + '/bearing_params.png',dpi=adv['pic_resolution'])
        
        plt.close()
        
    if 'dist' in variables:
        
        bin_num = bin_nums['dist']
        
        dist_params = param_dict['dist']
        
        bin_size = max_dist/bin_num
        xvals = np.linspace(0,max_dist,bin_num,endpoint=False) + bin_size/2.
    
        scale_factor = calc_scale_factor(frozenset(('dist',)),param_dict,behavior_matrices,spike_train)
            
        dist_curve = np.exp(dist_params) * scale_factor * framerate
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xvals,dist_curve,'k-',linewidth=3.0)
        ax.set_xlim([0,max_dist])
        ax.set_xticks(np.arange(0,max_dist,20).astype(int))
        ax.set_xlabel('Center dist (cm)')
        ax.set_ylabel('Firing rate (spikes/s)')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1.2 * np.max(dist_curve)])
        plt.tight_layout()
        fig.savefig(savedir + '/dist_params.png',dpi=adv['pic_resolution'])
        
        plt.close()
