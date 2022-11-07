# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:57:26 2017

-main script for running analyses on spike data
-most functions here are administrative or call other plotting/calculation functions

@author: Patrick
"""

import os
import pickle
import shutil
from spike_analysis import read_nlx, read_oe, read_taube, read_plexon, classify, spatial, animation
import copy
import csv
import numpy as np

#######################

def Run(fname,self):
    ''' Run the main program! '''
        
    #make dictionaries for options, advanced options, and all data for plotting purposes
    ops = {}
    adv = {}
    class_ops = {}
    all_data = {}
    
    #open pickled options and assign options to ops dict
    with open('options.pickle','rb') as f:
        [ops['run_hd'],ops['run_spatial'],ops['hd_map'],ops['run_grid'],ops['run_speed'],ops['run_ahv'],ops['run_autocorr'],
        ops['run_ego'],ops['run_ebc'],ops['run_center_ego'],ops['run_wall_ego'],ops['run_view'],ops['animated_path_spike'],ops['animated_hd_map'],ops['heatmap_animation'],
        ops['save_all'],ops['single_cluster'],ops['speedmode'],ops['save_data'],ops['acq']] = pickle.load(f)
    #open pickled advanced options and assign to adv dict
    with open('advanced.pickle','rb') as f:
        [adv['hd_calc'],adv['arena_x'],adv['arena_y'],adv['pic_resolution'],
         adv['spatial_bin_size'],adv['speed_bin_size'],adv['ahv_bin_size'],adv['ego_dist_bin_size'],
         adv['ebc_dist_bin_size'],adv['ebc_bearing_bin_size'],adv['ego_ref_bin_size'],
         adv['bin_size'],adv['autocorr_width'],adv['framerate'],adv['hd_bin_size'],adv['sample_cutoff'],
         adv['speed_cutoff'],adv['ani_speed']]= pickle.load(f)
    #open classify options and assign to class_ops dict
    with open('classify_options.pickle','rb') as f:
        [class_ops['pos'],class_ops['hd'],class_ops['speed'],class_ops['ahv'],class_ops['bearing'],class_ops['dist'],class_ops['save_profiles']] = pickle.load(f)
    #if single cluster mode, set cluster_file equal to selected file and figure out 
    #the base directory
    if ops['single_cluster']:
        ops['single_cluster_file'] = fname[0]
        fdir = os.path.dirname(fname[0])
    #otherwise, the filename is the base directory
    else:
        fdir = fname
        
    adv['hd_bins'] = int(360./adv['hd_bin_size'])
    
    #add it to the data dict
    all_data['fdir'] = fdir
    #set up a list for folders ('trials') in case of multiple sessions
    #(and add multi-session bool to ops dict so we can reference it later)
    ops,trials = find_trials(ops,fdir)
    
    #add trials to data dict
    all_data['trials'] = trials
    #figure out which plots we have to make and assign to metadata dict
    metadata = collect_metadata(ops,fdir,trials,[])
    
    #signal the all_data and metadata dicts to the main gui
    self.worker.data_dict.emit(all_data)
    self.worker.init_data.emit(metadata)

    #for each trial...
    for trial in trials:
                
        if self.worker.isrunning:
            #add an entry for the trial to the all_data dict
            all_data[trial] = {}
    
            if ops['acq'] == 'taube':
                trial_data = read_files(ops,fdir,trial,metadata=True)
                
            else:
                #get tracking information (positions, angles) for trial
                trial_data, adv = tracking_stuff(ops,adv,fdir,trial,self)
                
            #scale the tracking data to cm, if possible
            adv, trial_data = spatial.scale_tracking_data(adv,trial_data,trial)

            #calculate speed and ahv data if specified in ops
            trial_data = speed_stuff(ops,class_ops,adv,trial_data,self)
            
            #calculate center and wall bearing data if specified in ops
            trial_data = ego_stuff(ops,class_ops,adv,trial_data,self)
            
            # #bin edges for plotting stuff
            x_gr = len(np.arange(0,np.max(trial_data['center_x']),adv['spatial_bin_size']))
            y_gr = len(np.arange(0,np.max(trial_data['center_y']),adv['spatial_bin_size']))
            h,xedges,yedges = np.histogram2d(np.array(trial_data['center_x']),np.array(trial_data['center_y']),[x_gr,y_gr],[[min(trial_data['center_x']),max(trial_data['center_x'])],[min(trial_data['center_y']),max(trial_data['center_y'])]])    
            trial_data['heat_xedges'] = xedges
            trial_data['heat_yedges'] = yedges
            
            #for each cluster in the trial...
            for cluster in range(len(trial_data['cluster_files'])):
                if self.worker.isrunning:
                    #create cluster entry for all_data dicts
                    all_data[trial][trial_data['filenames'][cluster]] = {}
                    
                    #collect administrative data for cluster
                    cluster_data = cell_admin(ops,trial_data,cluster)
                        
                    if self.worker.isrunning:
                        
                        #store in trial_data dict what our current trial and cluster are
                        trial_data['current_trial'] = trial
                        trial_data['current_cluster'] = trial_data['filenames'][cluster]
                        
                        #make dictionary in cluster_data to keep track of which plots have already
                        #been saved - set them all to False to start out with
                        cluster_data['saved'] = {}
                        for plot in metadata['all_plots']:
                            
                            cluster_data['saved'][plot] = False
                        
                        if not ops['acq'] == 'taube':
                            #process the ts file
                            cluster_data['spike_list'] = ts_file_reader(cluster_data['ts_file'],ops)
                        
                            #create lists of spike data (for HD, spatial, time plots, animations, etc)
                            spike_data, cluster_data = spatial.create_spike_lists(ops,adv,trial_data,cluster_data)
                            
                        else:
                            print(trial_data['filenames'][cluster])
                            trial_data, spike_data = taube_tracking_stuff(ops,adv,trial + '/' + trial_data['filenames'][cluster],trial_data,self)
                            trial_data = speed_stuff(ops,adv,trial_data,self)
                            spike_data, cluster_data = taube_spike_stuff(ops, adv, spike_data, trial_data, cluster_data)
                            
                        if self.worker.isrunning:
                            
                            best_model = classify.run_classifier(class_ops,adv,trial_data,cluster_data,spike_data['ani_spikes'])
                        
                            if best_model == 'uniform':
                                best_model = frozenset(('couldn\'t classify!',))
                        
                            cluster_data['best_model'] = best_model
                            all_data[trial_data['current_trial']][trial_data['current_cluster']]['cluster_data'] = copy.deepcopy(cluster_data)
                            #send the dict to the main gui
                            self.worker.data_dict.emit(all_data)
                            
                            #run all of the analysis functions specified by ops
                            trial_data,cluster_data,spike_data = analysis_stuff(ops,adv,trial_data,cluster_data,spike_data,self)
                            
                            #assign dicts for this session to the appropriate key in all_data dict
                            all_data[trial_data['current_trial']][trial_data['current_cluster']]['cluster_data'] = copy.deepcopy(cluster_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']]['spike_data'] = copy.deepcopy(spike_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']]['trial_data'] = copy.deepcopy(trial_data)
        
                        #send the dict to the main gui
                        self.worker.data_dict.emit(all_data)
                            
            if ops['save_data']:
                #assign data for current trial to new dict for saving
                all_trial_data = all_data[trial_data['current_trial']]
                #also save ops, adv, and metadata
                all_trial_data['ops'] = ops
                all_trial_data['adv'] = adv
                all_trial_data['metadata'] = metadata
                #pickle the dictionary for future unpickling!
                print('saving data...')
                with open(trial +'/all_trial_data.pickle','wb') as f:
                    pickle.dump(all_trial_data,f,protocol=2)
                

    #we're all done
    print('Done!')
    #terminate the process running this script
    self.worker.isrunning = False
                
def find_trials(ops,fdir,llmodel=False):
    ''' search the chosen directory for trial folders '''
    #start with an empty list
    trials = []
    #for every file in the directory...
    for file in os.listdir(fdir):
        #if there are any folders, check if they have timestamp files in them
        if os.path.isdir(fdir + '/' + file):
            #start by assuming we have no 
            count = 0
            video_file = False
            if llmodel:
                video_file = True
            #for every file in folder...
            for nfile in os.listdir(fdir + '/' + file):
                #if there's a tetrode timestamp file...
                if nfile.endswith(".txt") and nfile.startswith('TT'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilofiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                elif ops['acq'] == 'taube' and not nfile.endswith('.ts') and not nfile.endswith('Icon'):
                    count += 1
                    video_file = True
                #if there's a stereotrode timestamp file...
                elif nfile.endswith(".txt") and nfile.startswith('ST'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilofiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                #if there's a video file, make a note of it!
                elif nfile.endswith('.nvt'):
                    video_file = True
                elif ops['acq'] == 'openephys' and nfile == 'vt1.txt':
                    video_file = True
            #if the folder contains timestamp files and a video file, add it to the "trials" list
            if count > 0 and video_file:
                trials.append(fdir + '/' + file)

    #if there are trials in the list, fdir is a multi-session folder
    if len(trials) > 0:
       ops[' multi_session'] = True
    #otherwise the only "trial" is your main folder, so add that to the trial list
    else:
        trials.append(fdir)
        ops['multi_session'] = False

    #return our options (with multi-session entry) and trials list
    return ops,trials

def collect_metadata(ops,fdir,trials,all_trial_data):
    ''' collect info on what we need to plot '''
    #collect both the user-specified plots as well as a list of all possible plots
    #for future use
    plots,all_plots = set_trial_plots(ops)
    #start data structures for metadata collection
    plot_list = []
    metadata = {}
    #for each trial...
    for trial in trials:
        #add entry for trial in metadata dict
        metadata[trial] = {}
        #if we have offline sorter files, translate them
        translate_plexon_files(ops,trial)
        #read all the files and collect names for cluster (timestamp) files
        trial_data = read_files(ops,{},fdir,trial,metadata=True)
        #for each cluster...
        for cluster in trial_data['filenames']:
            #add entry for cluster in metadata dict
            metadata[trial][cluster] = {}
            #also add dict to keep track of whether data is ready
            #for each plot
            metadata[trial][cluster]['dataready'] = {}
            #for every possible plot...
            for plot in all_plots:
                metadata[trial][cluster]['dataready'][plot] = False
            #make a list containing entries for every plot to be made for this trial,
            #including trial id, cluster id, and plot id
            #this will be used by the gui to know what order to plot things in
            for plot in plots:
                plot_list.append([trial,cluster,plot])
                    
    #add everything to the metadata dict
    metadata['plot_list'] = plot_list
    metadata['plots'] = plots
    metadata['all_plots'] = all_plots
        
    #return it!
    return metadata


def translate_plexon_files(ops,trial):
    ''' look for plexon offline sorter output files and transform them into compatible files '''

    #do stuff with relevant files in the trial folder
    for file in os.listdir(trial):

        #for plexon offline sorter files
        if file.endswith(".txt") and file.startswith('TT') and '_SS_' not in file and os.stat(trial+'/'+file).st_size != 0:
            read_plexon.translate_plexon_cluster_file(trial,file)
            
        #for plexon offline sorter files
        elif file.endswith(".txt") and file.startswith('ST') and '_SS_' not in file and os.stat(trial+'/'+file).st_size != 0:
            read_plexon.translate_plexon_cluster_file(trial,file)
                
   
        
def read_files(ops,adv,fdir,trial,metadata=False):
    ''' read important files in the trial folder and extract relevant info '''
    
    #make a dict for trial_data
    trial_data = {}
        
    #make note of the current trial
    trial_data['trial'] = trial

    #make lists for cluster ts files and their names
    trial_data['cluster_files'] = []
    trial_data['plexon_cluster_files'] = []
    trial_data['filenames'] = []
    trial_data['bounds'] = None
    
    #if single cluster mode, assign the cluster path and filename to appropriate
    #entries in trial_data dict
    if ops['single_cluster']:
        trial_data['cluster_files'].append(ops['single_cluster_file'])
        trial_data['filenames'].append(os.path.basename(ops['single_cluster_file'])[:(len(os.path.basename(ops['single_cluster_file']))-4)])

    #do stuff with relevant files in the trial folder
    for file in os.listdir(trial):
        #if event file, grab the path
        if file.endswith('.nev'):
            event_file = trial + '/' + file
            if metadata:
                print('  ')
                print('Reading event file!!')
                print('  ')
            #read the relevant info with read_nev function
            trial_data = read_nlx.read_nev(event_file,trial_data)

        #if we're not using single-file mode...
        if not ops['single_cluster']:
            #add tetrode ts file paths and filenames to appropriate entries in trial_data dict
            if file.endswith(".txt") and file.startswith('TT') and '_SS_' in file and os.stat(trial+'/'+file).st_size != 0:
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file)-4)])

            #same as tetrodes but with stereotrodes
            if file.endswith(".txt") and file.startswith('ST') and '_SS_' in file and os.stat(trial+'/'+file).st_size != 0:
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file)-4)])

            #for taube data
            if ops['acq'] == 'taube' and os.path.getsize(trial+'/'+file) > 1000 and 'b' in file and not file.endswith('.ts') and 'export' not in file and not file.endswith('Icon') and not file.endswith('.o') and not file.endswith('.pic') and '.s' not in file and not os.path.isdir(trial + '/' + file):
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file))])
                
        if file == 'bounds.txt':
            bounds_file = trial + '/' + file
            with open(bounds_file,'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    trial_data['bounds'] = np.array(row).astype(np.float)
                    
        #if video file, grab the path and filename
        if file.endswith('.nvt'):
            if not metadata:
                video_file = trial + '/' + file
                #extract raw tracking data from the video file using read_nvt function
                trial_data, raw_vdata, adv = read_nlx.read_nvt(adv,video_file,trial_data)
               
        if file.startswith('vt1') and file.endswith('.txt') and ops['acq'] == 'openephys':
            if not metadata:
                video_file = trial + '/' + file
                filename = file
                trial_data, raw_vdata = read_oe.read_video_file(video_file,filename,trial_data)

    #if we're not collecting metadata, return everything
    if not metadata:
        return trial_data,raw_vdata,ops,adv
    #if we are collecting metadata, just return the trial_data dict
    else:
        return trial_data
        
def tracking_stuff(ops,adv,fdir,trial,self):
    ''' collect video tracking data and interpolate '''
    
    #if we're not loading previous data...
    if self.worker.isrunning:
        #read relevant files to extract raw video data
        trial_data,raw_vdata,ops,adv = read_files(ops,adv,fdir,trial)
        
        print('framerate: %s' % str(adv['framerate']))
    
        print('processing tracking data for this session...')
        #interpolate nondetects using extracted angles and coordinates
        trial_data = spatial.interp_points(raw_vdata,trial_data,adv)
        #calculate centerpoints and timestamp vectors
        trial_data = spatial.center_points(ops,adv,trial_data)
        
    return trial_data,adv

def taube_tracking_stuff(ops,adv,fname,trial_data,self):
    ''' collect video tracking data and interpolate '''
    
    #if we're not loading previous data...
    if self.worker.isrunning:
        #read relevant files to extract raw video data
        raw_vdata,spike_data = read_taube.read_taube(fname)
    
        print('processing tracking data for this session...')
        #interpolate nondetects using extracted angles and coordinates
        trial_data = spatial.interp_points(raw_vdata,trial_data)
        #calculate centerpoints and timestamp vectors
        trial_data = spatial.center_points(ops,adv,trial_data)
        
    return trial_data,spike_data

def taube_spike_stuff(ops, adv, spike_data, trial_data, cluster_data):

    spike_x = []
    spike_y = []
    spike_angles = []
    spike_speeds = []
    spike_ahvs = []
    
    for i in range(len(spike_data['ani_spikes'])):
        for j in range(int(spike_data['ani_spikes'][i])):
            spike_x.append(trial_data['center_x'][i])
            spike_y.append(trial_data['center_y'][i])
            spike_angles.append(trial_data['angles'][i])
            if ops['run_speed']:
                spike_speeds.append(trial_data['speeds'][i])
            if ops['run_ahv']:
                spike_ahvs.append(trial_data['ahvs'][i])
            
    #returns spatial and temporal information for each spike
    spike_data['spike_x'] = spike_x
    spike_data['spike_y'] = spike_y
    spike_data['spike_angles'] = spike_angles
    spike_data['spike_speeds'] = spike_speeds
    spike_data['spike_ahvs'] = spike_ahvs
    
    return spike_data, cluster_data

        
def speed_stuff(ops,class_ops,adv,trial_data,self):
    ''' calculate speed and ahv info from tracking data '''
    
    if (ops['run_speed'] or class_ops['speed'] or adv['speed_cutoff'] > 0) and self.worker.isrunning:
        print('processing speed data...')
        #calculate running speeds for each frame
        trial_data = spatial.calc_speed(adv,trial_data)
        
    if ops['run_hd'] and self.worker.isrunning:
        trial_data = spatial.calc_movement_direction(adv,trial_data)
        
    if (ops['run_ahv'] or class_ops['ahv']) and self.worker.isrunning:
        print('processing AHV data...')
        #calculate ahvs for each frame
        trial_data = spatial.calc_ahv(adv,trial_data)
        
    #apply speed cutoff
    speed_cutoff = adv['speed_cutoff']
    
    if speed_cutoff > 0:
        trial_data['og_speeds'] = copy.deepcopy(np.array(trial_data['speeds']))
        trial_data['center_x'] = trial_data['center_x'][trial_data['og_speeds'] > speed_cutoff]
        trial_data['center_y'] = trial_data['center_y'][trial_data['og_speeds'] > speed_cutoff]
        trial_data['angles'] = trial_data['angles'][trial_data['og_speeds'] > speed_cutoff]
        trial_data['speeds'] = trial_data['speeds'][trial_data['og_speeds'] > speed_cutoff]
        
        if ops['run_hd']:
            trial_data['movement_directions'] = trial_data['movement_directions'][trial_data['og_speeds'] > speed_cutoff]

        if ops['run_ahv'] or class_ops['ahv']:
            trial_data['ahvs'] = trial_data['ahvs'][trial_data['og_speeds'] > speed_cutoff]
        
    return trial_data

def ego_stuff(ops,class_ops,adv,trial_data,self):
    ''' calculate center and wall egocentric variables '''
    
    if (ops['run_center_ego'] or class_ops['bearing'] or class_ops['dist']) and self.worker.isrunning:
        trial_data = spatial.calc_center_ego(adv,trial_data)
                
    if ops['run_wall_ego'] and self.worker.isrunning:
        trial_data = spatial.calc_wall_ego(adv,trial_data)
        
    return trial_data

def ts_file_reader(ts_file,ops):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'))
    
    if ops['acq'] == 'openephys':
        for row in reader:
            spike_list.append(int(float(row[0])))

    elif ops['acq'] == 'neuralynx':
        for row in reader:
            spike_list.append(int(row[0]))
                
    #return it!
    return spike_list
        
def cell_admin(ops,trial_data,cluster):
    ''' administrative tasks for each cluster '''
    
    #start a data dict for the cluster
    cluster_data = {}
    #grab ts file locations from trial_data dict
    cluster_files = trial_data['cluster_files']
    #choose the appropriate ts file
    ts_file = cluster_files[cluster]
    #make a new folder if one doesn't already exist, named after the ts file
    if ops['acq'] != 'taube':
        new_folder = ts_file[:(len(ts_file)-4)]
    else:
        new_folder = ts_file + '_figures'
    if ops['save_all']:
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
    #which cluster are we working on?
    print('')
    print('processing %s (cluster # %s of %d)' % (os.path.basename(ts_file),(cluster+1),len(cluster_files)))
    print('')

    #add 'important' variables to cluster_data dict
    cluster_data['ts_file'] = ts_file
    cluster_data['new_folder'] = new_folder
    cluster_data['cluster'] = cluster
    
    #return it!
    return cluster_data
        
def analysis_stuff(ops,adv,trial_data,cluster_data,spike_data,self):
    ''' the bulk of the program - contains all of our analyses '''
        
    #spatial analyses
    if ops['run_spatial'] and self.worker.isrunning:   
        print('making spatial plots...')
        spatial.plot_path(ops,adv,trial_data,cluster_data,spike_data,self) #plot path
        cluster_data = spatial.plot_heat(ops,adv,trial_data,cluster_data,spike_data,self) #plot heatmaps
        
    #grid analyses (check to see if spatial analyses have already been done -- if not, skip these)
    if ops['run_grid'] and self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_smoothed_heat'] and self.worker.isrunning:
        print('making grid plots...')
        cluster_data = spatial.spatial_autocorrelation(ops,adv,trial_data,cluster_data,spike_data,self) #plot spatial autocorrelation
        cluster_data = spatial.grid_score(ops,adv,trial_data,cluster_data,spike_data,self) #plot gridness score
        
    #head direction analyses
    if ops['run_hd'] and self.worker.isrunning:
        print('making head direction plots...')
        cluster_data,spike_data = spatial.plot_hd(ops,adv,trial_data,cluster_data,spike_data,self) #plot HD data
        cluster_data = spatial.plot_half_hds(ops,adv,trial_data,cluster_data,spike_data,self) #plot HD data for each half of session
        
    #HD x spatial analyses
    if ops['hd_map'] and self.worker.isrunning:
        print('making spatial x HD plots...')
        cluster_data = spatial.plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self) #plot spatial by HD 
        cluster_data = spatial.plot_hd_vectors(ops,adv,trial_data,cluster_data,spike_data,self) #plot spatial by HD vectors
        
    #linear speed analyses
    if ops['run_speed'] and self.worker.isrunning:
        print('making speed plot...')
        cluster_data = spatial.plot_speed(ops,adv,trial_data,cluster_data,spike_data,self) #plot speed vs firing rate
        
    #AHV analyses
    if ops['run_ahv'] and self.worker.isrunning:
        print('making ahv plot...')
        cluster_data = spatial.plot_ahv(ops,adv,trial_data,cluster_data,spike_data,self) #plot AHV vs firing rate

    #ego analyses
    if ops['run_ego'] and self.worker.isrunning:
        print('making ego bearing and view plots...')
        cluster_data = spatial.plot_ego(ops,adv,trial_data,cluster_data,spike_data,self) #plot ego rayleighs and mean angles

    if ops['run_ebc'] and self.worker.isrunning:
        print('making ebc plots...')
        cluster_data = spatial.plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self,direction_variable='hd') #plot ebc using HD
#        cluster_data = spatial.plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self,direction_variable='md') #plot ebc using MD

    if ops['run_center_ego'] and self.worker.isrunning:
        print('making center ego plots...')
        cluster_data = spatial.plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self)
        
    if ops['run_wall_ego'] and self.worker.isrunning:
        print('making wall ego plots...')
        cluster_data = spatial.plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self)
        
    #animations
    if ops['animated_path_spike'] and self.worker.isrunning:
        animation.animated_path_spike(ops,adv,trial_data,cluster_data,spike_data,spike_hd=False) #animated path/spike plot
    if ops['animated_hd_map'] and self.worker.isrunning:
        animation.animated_path_spike(ops,adv,trial_data,cluster_data,spike_data,spike_hd=True) #animated HD x spatial plot
    if ops['heatmap_animation'] and self.worker.isrunning:
        animation.animated_heatmap(ops,adv,trial_data,cluster_data,spike_data) #animated heatmap
        
    #time analyses
    if ops['run_autocorr'] and self.worker.isrunning:
        print('making time plots...')
        cluster_data = spatial.spike_autocorr(ops,adv,trial_data,cluster_data,spike_data,self) #plot ISI histogram and spike autocorrelation

    #return data dicts for saving and relaying to main gui
    return trial_data,cluster_data,spike_data

def set_trial_plots(ops):
    ''' make a list of which plots we need to make '''
    
    #start the list
    plots = []
    #add appropriate plots to list based on user-chosen options
    if ops['run_spatial']:
        plots.append('plot_path')
        plots.append('plot_raw_heat')
#        plots.append('plot_interpd_heat')
        plots.append('plot_smoothed_heat')
    if ops['run_grid']:
        plots.append('plot_spatial_autocorr')
        plots.append('plot_grid_score')
    if ops['run_hd']:
        plots.append('plot_hd')
        plots.append('plot_half_hds')
    if ops['hd_map']:
        plots.append('plot_hd_map')
        plots.append('plot_hd_vectors')
    if ops['run_speed']:
        plots.append('plot_speed')
    if ops['run_ahv']:
        plots.append('plot_ahv')
    if ops['run_ego']:
        plots.append('plot_ego')
        plots.append('plot_ego_angle')
    if ops['run_ebc']:
        plots.append('plot_ebc')
    if ops['run_center_ego']:
        plots.append('plot_center_bearing')
        plots.append('plot_center_dist')
        plots.append('plot_center_ego_map')
    if ops['run_wall_ego']:
        plots.append('plot_wall_bearing')
        plots.append('plot_wall_dist')
        plots.append('plot_wall_ego_map')
    if ops['run_autocorr']:
        plots.append('plot_isi')
        plots.append('plot_spike_autocorr')

        
    #also make a list of every possible plot for future use    
    all_plots = ['plot_path','plot_raw_heat','plot_smoothed_heat',
                 #plot_interpd_heat,
                 'plot_spatial_autocorr','plot_grid_score','plot_hd','plot_half_hds',
                 'plot_hd_map','plot_hd_vectors','plot_speed','plot_ahv','plot_ego','plot_ego_angle','plot_ebc',
                 'plot_center_bearing','plot_center_dist','plot_center_ego_map','plot_wall_bearing',
                 'plot_wall_dist','plot_wall_ego_map','plot_isi','plot_spike_autocorr']
    
    #return the lists
    return plots,all_plots