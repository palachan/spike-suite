# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:53:23 2018

classify cells functionally using poisson log-likelihood maximization with 10-fold
cross-validation, create the winning model, calculate variable contributions to
goodness-of-fit, and save the results

@author: Patrick
"""

#import important modules
import pickle
import os
import numpy as np
import copy
from itertools import chain, combinations

#import scripts from spike-analysis
import llmodel
import optimize
import collect_data
import phase_analysis
import model_select
import interpret

#set options
first_half = False
second_half = False
opt_all = False

#which variables will we use?
variables = [('allo'),('spatial'),('speed'),('ahv'),('theta'),('ego')]
variables = [('allo'),('spatial'),('theta')]

#smoothing hyperparameters
betas = [1.5,20.,20.,150.,150.,150.]


''''''''''''''''''''''''''''''''''''''''''''''''


def load_data(fname):
    ''' load pickled numpy arrays '''

    try:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    except:
        print('couldn\'t open data file! try again!!')
    
    return data

def get_all_models(variables):
    ''' convenience function for calculating all possible
    combinations of nagivational variables '''
    
    def powerset(variables):
        return list(chain.from_iterable(combinations(variables, r) for r in range(1,len(variables)+1)))
    
    all_models = powerset(variables)
    all_models.append(('uniform',))
    
    for i in range(len(all_models)):
        all_models[i] = frozenset(all_models[i])
    
    return all_models
    
def classify_trials(fdir):
    ''' run everything for a directory of recording sessions '''
    
    #figure out what sessions (trials) are in the directory
    trials = collect_data.find_trials(fdir)
        
    #for every session...
    for trial in trials:
                                    
        #calculate tracking data
        trial_data = collect_data.tracking_stuff(fdir,trial)
        trial_data = collect_data.speed_stuff(trial_data)
        trial_data = collect_data.calc_novelty(trial_data)
        
        #extract relevant correlates
        center_x = np.asarray(trial_data['center_x'])
        center_y = np.asarray(trial_data['center_y'])
        angles = np.asarray(trial_data['angles'])
        speeds = np.asarray(trial_data['speeds'])
        ahvs = np.asarray(trial_data['ahvs'])
        novelties = np.asarray(trial_data['novelties'])
        
        #if we're just analyzing the first half of the session (like for an
        #inactivation study) just take the first half of the data
        if first_half:
            center_x = center_x[0:len(center_x)/2]
            center_y = center_y[0:len(center_y)/2]
            angles = angles[0:len(angles)/2]
            speeds = speeds[0:len(speeds)/2]
            ahvs = ahvs[0:len(ahvs)/2]
            novelties = novelties[0:len(novelties)/2]
            
        if second_half:
            center_x = center_x[len(center_x)/2:]
            center_y = center_y[len(center_y)/2:]
            angles = angles[len(angles)/2:]
            speeds = speeds[len(speeds)/2:]
            ahvs = ahvs[len(ahvs)/2:]
            novelties = novelties[len(novelties)/2:]
    
        #calculate the points where we'll divide our session into 10 segments
        break_points = np.arange(0,(len(center_x) + len(center_x)/50),len(center_x)/50)
        
        #calculate train and test covariate matrices for each fold f the cross-validation
        train_dict, test_dict = llmodel.assign_data(center_x, center_y, angles, speeds, ahvs, novelties, break_points, cross_val=True)
        
        #grab the names of the clusters we'll be analyzing
        cluster_names = trial_data['filenames']
    
        #for each cluster...
        for name in cluster_names:
            
            
            #report which cluster we're working on                      
            print(name)
            
            #calculate spike train
            spike_train = collect_data.get_spikes(trial,name,trial_data)
            #calculate theta phase information
            phases = phase_analysis.run_phase_analysis(trial,name,trial_data)
    
            #split in half if need be
            if first_half:
                spike_train = spike_train[0:len(spike_train)/2]
                phases = phases[0:len(phases)/2]
            if second_half:
                spike_train = spike_train[len(spike_train)/2:]
                phases = phases[len(phases)/2:]
                
            #for each fold in the cross-validation...
            for fold in range(10):
                #split the spike train into train and test data
                train_dict[fold]['spikes'], test_dict[fold]['spikes'] = llmodel.assign_spikes(spike_train, break_points, fold, cross_val=True)
                #split the theta phases into train and test data
                train_dict[fold]['Xt'], test_dict[fold]['Xt'] = llmodel.assign_phases(phases, break_points, fold, cross_val=True)

            #run the cross-validation and collect the result
            cdict = independent_cval(trial, name, train_dict, test_dict)
            
            #run the forward search procedure to find the best model
            best_model = model_select.select_model(cdict)
            print(tuple(best_model))
                
            #calculate relevant data for training and testing on the full session
            data = llmodel.assign_data(center_x, center_y, angles, speeds, ahvs, novelties, [], cross_val=False)
            data['spikes'] = llmodel.assign_spikes(spike_train,[],[],cross_val=False)
            data['Xt'] = llmodel.assign_phases(phases,[],[],cross_val=False)
            
            #run the best model and collect the result
            model_dict,best_params = independent_best_model(best_model,data)
            
            #calculate the change in each goodness-of-fit measure from adding or
            #subtracting each variable from the best model
            model_dict = interpret.calc_contribs(best_model,model_dict,data,best_params)
            
            #save the results for later
            with open((trial+'/theta_best_model_%s.pickle' % name),'wb') as f:
                pickle.dump(model_dict,f,protocol=2)
                
def classify_trials_single_ego(fdir):
    ''' run everything for a directory of recording sessions -- 
    for each session, first make a full model using all egocentric spatial
    bins. then extract the location of the bin with the highest rayleigh r, and
    subsequently only use that location for all analyses '''
    
    #figure out what sessions (trials) are in the directory
    trials = collect_data.find_trials(fdir)
        
    #for every session...
    for trial in trials:
                                                
        #calculate tracking data
        trial_data = collect_data.tracking_stuff(fdir,trial)
        trial_data = collect_data.speed_stuff(trial_data)
        trial_data = collect_data.calc_novelty(trial_data)
        
        #extract relevant correlates
        center_x = np.asarray(trial_data['center_x'])
        center_y = np.asarray(trial_data['center_y'])
        angles = np.asarray(trial_data['angles'])
        speeds = np.asarray(trial_data['speeds'])
        ahvs = np.asarray(trial_data['ahvs'])
        novelties = np.asarray(trial_data['novelties'])
        
        #if we're just analyzing the first half of the session (like for an
        #inactivation study) just take the first half of the data
        if first_half:
            center_x = center_x[0:len(center_x)/2]
            center_y = center_y[0:len(center_y)/2]
            angles = angles[0:len(angles)/2]
            speeds = speeds[0:len(speeds)/2]
            ahvs = ahvs[0:len(ahvs)/2]
            novelties = novelties[0:len(novelties)/2]
            
        if second_half:
            center_x = center_x[len(center_x)/2:]
            center_y = center_y[len(center_y)/2:]
            angles = angles[len(angles)/2:]
            speeds = speeds[len(speeds)/2:]
            ahvs = ahvs[len(ahvs)/2:]
            novelties = novelties[len(novelties)/2:]
            
        #calculate the points where we'll divide our session into 10 segments
        break_points = np.arange(0,(len(center_x) + len(center_x)/50),len(center_x)/50)
        
        #grab the names of the clusters we'll be analyzing
        cluster_names = trial_data['filenames']
    
        #for each cluster...
        for name in cluster_names:
                                
            #report which cluster we're working on                      
            print(name)
            
            #calculate spike train
            spike_train = collect_data.get_spikes(trial,name,trial_data)
            #calculate theta phase information
            phases = phase_analysis.run_phase_analysis(trial,name,trial_data)
    
            #split in half if need be
            if first_half:
                spike_train = spike_train[0:len(spike_train)/2]
                phases = phases[0:len(phases)/2]
            if second_half:
                spike_train = spike_train[len(spike_train)/2:]
                phases = phases[len(phases)/2:]
                
            #calculate relevant data for training and testing on the full session
            start_data = llmodel.assign_data(center_x, center_y, angles, speeds, ahvs, novelties, [], cross_val=False)
            start_data['spikes'] = llmodel.assign_spikes(spike_train,[],[],cross_val=False)
            start_data['Xt'] = llmodel.assign_phases(phases,[],[],cross_val=False)

            Xe = copy.deepcopy(np.array(start_data['Xe'].todense()))
            
            ego_gr = 8
            llps_dict = np.zeros((ego_gr,ego_gr))
            
            from scipy.sparse import csr_matrix
            
            for i in range(ego_gr):
                for j in range(ego_gr):
                    
#                    bin_data = copy.deepcopy(start_data)
                    start_data['Xe'] = csr_matrix(Xe[:,(i*8+j)*30:((i*8+j)*30 + 30)])
                    #run the best model and collect the result
                    model_dict,best_params = independent_best_model(frozenset(variables),start_data,single_ego=True)
            
                    llps_dict[i][j] = model_dict['llps']

            ego_loc = np.where(llps_dict == np.min(llps_dict))
            
            #calculate train and test covariate matrices for each fold f the cross-validation
            train_dict, test_dict = llmodel.assign_data(center_x, center_y, angles, speeds, ahvs, novelties, break_points, cross_val=True, ego_loc=ego_loc)
                
            #for each fold in the cross-validation...
            for fold in range(10):
                #split the spike train into train and test data
                train_dict[fold]['spikes'], test_dict[fold]['spikes'] = llmodel.assign_spikes(spike_train, break_points, fold, cross_val=True)
                #split the theta phases into train and test data
                train_dict[fold]['Xt'], test_dict[fold]['Xt'] = llmodel.assign_phases(phases, break_points, fold, cross_val=True)

            #run the cross-validation and collect the result
            cdict = independent_cval(trial, name, train_dict, test_dict, single_ego=True)
            
            #run the forward search procedure to find the best model
            best_model = model_select.select_model(cdict)
            print(tuple(best_model))
                
            #calculate relevant data for training and testing on the full session
            data = llmodel.assign_data(center_x, center_y, angles, speeds, ahvs, novelties, [], cross_val=False, ego_loc=ego_loc)
            data['spikes'] = llmodel.assign_spikes(spike_train,[],[],cross_val=False)
            data['Xt'] = llmodel.assign_phases(phases,[],[],cross_val=False)
            
            #run the best model and collect the result
            model_dict,best_params = independent_best_model(best_model,data, single_ego=True)
            
            #calculate the change in each goodness-of-fit measure from adding or
            #subtracting each variable from the best model
            model_dict = interpret.calc_contribs(best_model,model_dict,data,best_params,single_ego=True)
            
            #save the results for later
            with open((trial+'/all_single_ego_corrected_best_model_%s.pickle' % name),'wb') as f:
                pickle.dump(model_dict,f,protocol=2)


def independent_cval(trial,cluster,train_dict,test_dict,single_ego=False):
    ''' run 10-fold cross-validation for independent variables '''
    
    #here is the full model we will optimize
    full_model = frozenset(variables)
    #and the powerset of all variables
    all_models = get_all_models(variables)
    
    #make a list for holding cross-val results
    cdict = [[]]*10
    
    #for each fold of the procedure...
    for fold in range(10):
        
        #assign a dictionary for the fold
        cdict[fold] = {}
                                        
        #split the data into train and test data
        train_data = train_dict[fold]
        test_data = test_dict[fold]

        #optimize parameters for the full model
        params = optimize.newton_cg(train_data, full_model, betas, single_ego)
        
        #for every model...
        for modeltype in all_models:
            
            #if this model has variables...
            if 'uniform' not in modeltype:
                #change the param dict to contain only the variables included
                #in the current model
                model_params = copy.deepcopy(params)
                for key in model_params:
                    if key not in modeltype:
                        model_params[key] = None
                #calculate a scaling factor such that the predicted spike train
                #has the same mean firing rate as the training spike train
                scale_factor = llmodel.calc_scale_factor(model_params,train_data,modeltype)
            
            else:
                #otherwise, set scale factor to 1 with empty params
                scale_factor = 1.
                model_params = {}

            #run the model on the test data and collect the results
            cdict[fold][modeltype] = llmodel.run_model(scale_factor, train_data, test_data, model_params, modeltype, single_ego)
            
#    save the results for later
    with open((trial+'/new_cval_%s.pickle' % cluster),'wb') as f:
        pickle.dump(cdict,f,protocol=2)
        
    #return the results
    return cdict

def independent_best_model(best_model,data,single_ego=False):
    ''' create and run the best model using data from the whole session '''
    
    #here is the full model we will optimize
    full_model = frozenset(variables)
    
    #calculate the independent parameters by optimizing the full model
    best_params = optimize.newton_cg(data, full_model, betas, single_ego)
    
    #if the model has variables...
    if 'uniform' not in best_model:
        
        #remove the extraneous variables from the optimized parameters
        top_params = copy.deepcopy(best_params)
        for key in top_params:
            if key not in best_model:
                print(key)
                top_params[key] = None
        #calculate a scale factor such that the model predicts the right number of spikes
        scale_factor = llmodel.calc_scale_factor(top_params,data,best_model)
                
    else:
        #otherwise, set scale factor to 1 with empty params
        top_params = {}
        scale_factor = 1.

    #run the model using our optimized parameters
    model_dict = llmodel.run_model(scale_factor, data, data, top_params, best_model, single_ego)
    
    #return the result and the optimized parameters
    return model_dict, best_params
        
                    
def correlated_cval(trial,cluster,train_dict,test_dict):
    ''' run cross-validation using parameters that have not been made independent
    (not currently used) '''
    
    cdict = [[]]*10
    all_models = get_all_models(variables)
    
    #for each fold of the procedure...
    for fold in range(10):
        #assign a dictionary for the fold
        cdict[fold] = {}

        train_data = train_dict[fold]
        test_data = test_dict[fold]

        scale_factor = 1.
        
        #for every model...
        for modeltype in all_models:

            if 'uniform' not in modeltype:
                params = optimize.newton_cg(train_data, modeltype, betas)

            cdict[fold][modeltype] = llmodel.run_model(scale_factor, train_data, test_data, params, modeltype)

    with open((trial+'/new_cval_%s.pickle' % cluster),'wb') as f:
        pickle.dump(cdict,f,protocol=2)
        
###############################################
###############################################

if __name__ == '__main__':
    classify_trials(fdir)
    
#    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC','V1B','PoR']
#    areas = ['deep POR']
#    overdir = 'H:/Patrick/egocentric'
##    overdir = 'C:/Users/Jeffrey_Taube/desktop/adn hdc'
#    
#    for animal in os.listdir(overdir):
#        animaldir = overdir + '/' + animal
#        print animaldir
#
#        for area in os.listdir(animaldir):
#            if area in areas:
#                areadir = animaldir + '/' + area
#                print areadir
#                                
#                classify_trials(areadir)


                            
