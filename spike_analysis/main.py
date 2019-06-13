# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:57:26 2017

-main script for running analyses on spike data
-most functions here are administrative or call other plotting/calculation functions

@author: Patrick
"""

import spatial
import multi
import animation
import os
import subprocess
import pickle
import shutil
import read_nlx
import read_oe
import read_taube
import copy
import csv
import numpy as np

#######################

def Run(fname,self):
    ''' Run the main program! '''
        
    #make dictionaries for options, advanced options, and all data for plotting purposes
    ops = {}
    adv = {}
    all_data = {}
    
    #open pickled options and assign options to ops dict
    with open('options.pickle','rb') as f:
        [ops['run_hd'],ops['run_spatial'],ops['hd_map'],ops['run_grid'],ops['run_speed'],ops['run_ahv'],ops['run_autocorr'],
        ops['spike_timed'],ops['run_ego'],ops['run_view'],ops['run_center_ego'],ops['run_wall_ego'],ops['animated_hd_map'],ops['heatmap_animation'],ops['labview'],
        ops['save_all'],ops['sumplots'],ops['single_cluster'],ops['speedmode'],ops['save_data'],ops['load_data'],ops['acq']] = pickle.load(f)
    #open pickled advanced options and assign to adv dict
    with open('advanced.pickle','rb') as f:
        [adv['pic_resolution'],adv['grid_res'],adv['bin_size'],adv['autocorr_width'],adv['framerate'],adv['hd_bins'],adv['sample_cutoff']]= pickle.load(f)

    #if single cluster mode, set cluster_file equal to selected file and figure out 
    #the base directory
    if ops['single_cluster']:
        ops['single_cluster_file'] = fname[0]
        fdir = os.path.dirname(fname[0])
    #otherwise, the filename is the base directory
    else:
        fdir = fname
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
                trial_data['cam_ids'] = [1]
            else:
                #get tracking information (positions, angles) for trial
                trial_data = tracking_stuff(ops,adv,fdir,trial,self)
                if '1.2m' in trial:
                    center_x = np.array(trial_data['center_x'])
                    center_y = np.array(trial_data['center_y'])
                    center_x = center_x - np.min(center_x)
                    center_x = center_x * 120./np.max(center_x)
                    center_y = center_y - np.min(center_y)
                    center_y = center_y * 120./np.max(center_y)
                    trial_data['center_x'] = center_x.tolist()
                    trial_data['center_y'] = center_y.tolist()
                    
                    
                #calculate speed and ahv data if specified in ops
                trial_data = speed_stuff(ops,adv,trial_data,self)
                #if multi-camera (or multi-event) trial, split up the data we just calc'd
                #for each camera and store in multi_data dict
                trial_data,multi_data = multi_stuff(ops,trial_data,self)
            
            #for each cluster in the trial...
            for cluster in range(len(trial_data['cluster_files'])):
                if self.worker.isrunning:
                    #create cluster entry for all_data dicts
                    all_data[trial][trial_data['filenames'][cluster]] = []
                    
                    #collect administrative data for cluster
                    cluster_data = cell_admin(ops,trial_data,cluster)
                    
                    #run labview if we need to
                    if ops['labview']:
                        labview(ops,trial_data,cluster_data)
        
                    #for each section with a different camera (set to 1 if not multi-cam)
                    for cam in range(len(trial_data['cam_ids'])):
                        if self.worker.isrunning:
                            #TODO: is this necessary?
                            cluster_data['cam'] = cam
                            
                            #store in trial_data dict what our current trial,cluster,and cam are
                            trial_data['current_trial'] = trial
                            trial_data['current_cluster'] = trial_data['filenames'][cluster]
                            trial_data['current_cam_ind'] = cam
                            
                            #make entries for this cam in all_data dicts
                            all_data[trial_data['current_trial']][trial_data['current_cluster']].append({})
                            
                            #make dictionary in cluster_data to keep track of which plots have already
                            #been saved - set them all to False to start out with
                            cluster_data['saved'] = {}
                            for plot in metadata['all_plots']:
                                cluster_data['saved'][plot] = False
                            
                            if not ops['acq'] == 'taube':
                                #if a multi-cam (or multi-event) trial...
                                if ops['multi_cam']:
                                    #change the trial_data dict to only contain tracking info for current cam
                                    trial_data = multi.assign_multi_data(ops,cam,multi_data,trial_data)
                                    #add spike timestamps to cluster_data dict for current cam
                                    cluster_data['spike_list'] = multi_data['spike_data'][trial_data['filenames'][cluster]][cam]
                                else:
                                    #otherwise, we just need to process the ts file
                                    cluster_data['spike_list'] = ts_file_reader(cluster_data['ts_file'],ops)
                            
                            
                                #create lists of spike data (for HD, spatial, time plots, animations, etc)
                                spike_data, cluster_data = spatial.create_spike_lists(ops,trial_data,cluster_data)
                                
                            else:
                                print trial_data['filenames'][cluster]
                                trial_data, spike_data = taube_tracking_stuff(ops,adv,trial + '/' + trial_data['filenames'][cluster],trial_data,self)
                                trial_data = speed_stuff(ops,adv,trial_data,self)
                                spike_data, cluster_data = taube_spike_stuff(ops, adv, spike_data, trial_data, cluster_data)
                            
                            #run all of the analysis functions specified by ops
                            trial_data,cluster_data,spike_data = analysis_stuff(ops,adv,trial_data,cluster_data,spike_data,self)
                            
                            #assign dicts for this cam session to the appropriate key in all_data dict
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['cluster_data'] = copy.deepcopy(cluster_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['spike_data'] = copy.deepcopy(spike_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['trial_data'] = copy.deepcopy(trial_data)
            
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
                
                
#            with open('H:/mm44/allo_hyper/ego_beta_dict.pickle','wb') as f:
#                pickle.dump(beta_dict,f,protocol=pickle.HIGHEST_PROTOCOL)

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
        #read the files and collect names for cluster (timestamp) files
        trial_data = read_files(ops,fdir,trial,metadata=True)
        #for each cluster...
        for cluster in trial_data['filenames']:
            #add entry for cluster in metadata dict
            metadata[trial][cluster] = []
            #for each cam...
            for cam in range(len(trial_data['cam_id'])):
                #add entry for cam in metadata dict
                metadata[trial][cluster].append({})
                #also add dict to cam entry to keep track of whether data is ready
                #for each plot
                metadata[trial][cluster][cam]['dataready'] = {}
                #for every possible plot...
                for plot in all_plots:
                    #if we're loading data, make note of which data has already been
                    #processed so we don't needlessly recompute anything
                    if ops['load_data']:
                        if all_trial_data['metadata'][trial][cluster][cam]['dataready'][plot]:
                            metadata[trial][cluster][cam]['dataready'][plot] = True
                        else:
                            metadata[trial][cluster][cam]['dataready'][plot] = False
                    #otherwise, set everything to False
                    else:
                        metadata[trial][cluster][cam]['dataready'][plot] = False
                #make a list containing entries for every plot to be made for this trial,
                #including trial id, cluster id, cam id, and plot id
                #this will be used by the gui to know what order to plot things in
                for plot in plots:
                    plot_list.append([trial,cluster,cam,plot])
                    
    #add everything to the metadata dict
    metadata['plot_list'] = plot_list
    metadata['plots'] = plots
    metadata['all_plots'] = all_plots
        
    #return it!
    return metadata
        
def read_files(ops,fdir,trial,metadata=False):
    ''' read important files in the trial folder and extract relevant info '''
    
    #make a dict for trial_data
    trial_data = {}
    #if we're running labview, set up a folder to save output files in
    if ops['labview']:
        labview_folder = fdir + "/labview_files"
        if not os.path.exists(labview_folder):
            os.makedirs(labview_folder)
        trial_data['labview_folder'] = labview_folder
        
    #make note of the current trial
    trial_data['trial'] = trial

    #make lists for cluster ts files and their names
    trial_data['cluster_files'] = []
    trial_data['filenames'] = []
    trial_data['bounds'] = None
    
    #start by assuming only one camera was used
    ops['multi_cam'] = False
    
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
            print('  ')
            print('Figuring out camera stuff!!')
            print('  ')
            #read the relevant info with read_nev function
            trial_data = read_nlx.read_nev(event_file,trial_data)
            #if more than one camera, then it's a multi-cam session
            if len(trial_data['cam_id']) > 1:
                ops['multi_cam'] = True

        #if we're not using single-file mode...
        if not ops['single_cluster']:
            #add tetrode ts file paths and filenames to appropriate entries in trial_data dict
            if file.endswith(".txt") and file.startswith('TT') and os.stat(trial+'/'+file).st_size != 0:
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file)-4)])
            #same as tetrodes but with stereotrodes
            if file.endswith(".txt") and file.startswith('ST') and os.stat(trial+'/'+file).st_size != 0:
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file)-4)])     
            #for taube data
            if ops['acq'] == 'taube' and os.path.getsize(trial+'/'+file) > 1000 and 'b' in file and not file.endswith('.ts') and 'export' not in file and not file.endswith('Icon') and not file.endswith('.o') and not file.endswith('.pic') and '.s' not in file and not os.path.isdir(trial + '/' + file):
                trial_data['cluster_files'].append((trial+'/'+file))
                trial_data['filenames'].append(file[:(len(file))])
        if file == 'bounds.txt':
            bounds_file = trial + '/' + file
            with open(bounds_file,'rb') as f:
                reader = csv.reader(f,dialect='excel-tab')
                for row in reader:
                    trial_data['bounds'] = np.array(row).astype(np.float)
        #if video file, grab the path and filename
        if file.endswith('.nvt'):
            if not metadata:
                video_file = trial + '/' + file
                filename = file
                #extract raw tracking data from the video file using read_nvt function
                trial_data, raw_vdata = read_nlx.read_nvt(video_file,filename,trial_data)
               
        if file.startswith('vt1') and file.endswith('.txt') and ops['acq'] == 'openephys':
            if not metadata:
                video_file = trial + '/' + file
                filename = file
                trial_data, raw_vdata = read_oe.read_video_file(video_file,filename,trial_data)
                
    #workaround for trials with only one camera
    try:
        trial_data['cam_id']
    except KeyError:
        trial_data['cam_id'] = [1]
        trial_data['cam_timestamp'] = []
        
    #if we're not collecting metadata, return everything
    if not metadata:
        return trial_data,raw_vdata,ops
    #if we are collecting metadata, just return the trial_data dict
    else:
        return trial_data
        
def tracking_stuff(ops,adv,fdir,trial,self):
    ''' collect video tracking data and interpolate '''
    
    #if we're not loading previous data...
    if not ops['load_data'] and self.worker.isrunning:
        #read relevant files to extract raw video data/multi-cam info
        trial_data,raw_vdata,ops = read_files(ops,fdir,trial)
    
        print('processing tracking data for this session...')
        #interpolate nondetects using extracted angles and coordinates
        trial_data = spatial.interp_points(raw_vdata,trial_data)
        #calculate centerpoints and timestamp vectors
        trial_data = spatial.center_points(ops,adv,trial_data)
        
    return trial_data

def taube_tracking_stuff(ops,adv,fname,trial_data,self):
    ''' collect video tracking data and interpolate '''
    
    #if we're not loading previous data...
    if not ops['load_data'] and self.worker.isrunning:
        #read relevant files to extract raw video data/multi-cam info
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
    cluster_data['halfway_ind'] = int(len(spike_x)/2)
    
    return spike_data, cluster_data

        
def speed_stuff(ops,adv,trial_data,self):
    ''' calculate speed and ahv info from tracking data '''
    
    trial_data['speeds'] = []
    trial_data['ahvs'] = []
    
    if ops['run_speed'] and self.worker.isrunning:
        print('processing speed data...')
        #calculate running speeds for each frame
        trial_data = spatial.calc_speed(adv,trial_data)
        
    trial_data = spatial.calc_movement_direction(adv,trial_data)        
        
    if ops['run_ahv'] and self.worker.isrunning:
        print('processing AHV data...')
        #calculate ahvs for each frame
        trial_data = spatial.calc_ahv(adv,trial_data)
        
    return trial_data
        
def multi_stuff(ops,trial_data,self):
    ''' split tracking and spike data for each cam if multiple used '''
    
    #if multi-cam session
    if ops['multi_cam'] and self.worker.isrunning:
        #split up the data into segments by camera (tracking then spike data)
        multi_data,trial_data = multi.file_creator(ops,trial_data)
        multi_data = multi.spikefile_cutter(trial_data,multi_data)
    #otherwise 
    else:
        trial_data['cam_ids'] = [1]
        multi_data = []
    
    #return relevant dicts
    return trial_data,multi_data

def ts_file_reader(ts_file,ops):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'),dialect='excel-tab')
    
    if ops['acq'] == 'openephys':
        for row in reader:
            spike_list.append(int(float(row[0])))# * 1000000./30000.))

    elif ops['acq'] == 'neuralynx':
        for row in reader:
            spike_list.append(int(row[0]))
                
    #return it!
    return spike_list
    
def labview(ops,trial_data,cluster_data):
    ''' runs relevant files through labview '''
    
    pass

    #figure out a name for the new file (based on trial and cluster ts file names)
    clust_name = trial_data['labview_folder'] + '\\' + os.path.basename(trial_data['trial']) + " " + os.path.basename(cluster_data['new_folder'])
    clust_name=clust_name.replace('/', '\\')
    #grab the timestamp file
    ts_file=cluster_data['ts_file'].replace('/', '\\')
    #grab the video txt file we made
    video_txt_file=trial_data['video_txt_file'].replace('/', '\\')
    #grab the event txt file we made
    event_txt_file = trial_data['event_txt_file'].replace('/', '\\')
    #start the process!
    print('labview conversion in progress...')
    #if multi_cam...
    if ops['multi_cam']:
        #run the multi-cam labview stitch program
        p=subprocess.Popen(r'"C:\Program Files\National Instruments\LabVIEW 2011\LabVIEW.exe" "C:\Users\Jeffrey Taube\Desktop\Patrick\python\labview\Neuralynx --- LabView.1 Cell.Multi Cam.5 Cameras.Patrick.vi" -- "%s" "%s" "%s" "%s"' % (clust_name,video_txt_file,ts_file,event_txt_file))
        p.communicate()
        cam_file = clust_name + '.cam'
        #run the cuboidal sort labview program on the output files
        p=subprocess.Popen(r'"C:\Program Files\National Instruments\LabVIEW 2011\LabVIEW.exe" "C:\Users\Jeffrey Taube\Desktop\Patrick\python\labview\Cuboidal Sort.Patrick.vi" -- "%s" "%s"' % (cam_file,clust_name))
        p.communicate()
    #otherwise...
    else:
        #just run the regular labview stitch program
        p=subprocess.Popen(r'"C:\Program Files\National Instruments\LabVIEW 2011\LabVIEW.exe" "C:\Users\Jeffrey Taube\Desktop\Patrick\python\labview\Neuralynx_labview_stitch.vi" -- "%s" "%s" "%s"' % (clust_name,video_txt_file,ts_file))
        p.communicate()
        
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
    print('processing cluster # %s of %d' % ((cluster+1),len(cluster_files)))

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
    if ops['run_grid'] and self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_smoothed_heat'] and self.worker.isrunning:
        print('making grid plots...')
        cluster_data = spatial.spatial_autocorrelation(ops,adv,trial_data,cluster_data,spike_data,self) #plot spatial autocorrelation
        cluster_data = spatial.grid_score(ops,adv,trial_data,cluster_data,spike_data,self) #plot gridness score
        
    #head direction analyses
    if ops['run_hd'] and self.worker.isrunning:
        print('making head direction plots...')
        cluster_data,spike_data = spatial.plot_hd(ops,adv,trial_data,cluster_data,spike_data,self) #plot HD data
        cluster_data = spatial.plot_half_hds(ops,adv,trial_data,cluster_data,spike_data,self) #plot HD data for each half of session
        
    #linear speed analyses
    if ops['run_speed'] and self.worker.isrunning:
        print('making speed plot...')
        cluster_data = spatial.plot_speed(ops,adv,trial_data,cluster_data,spike_data,self) #plot speed vs firing rate
        
    #AHV analyses
    if ops['run_ahv'] and self.worker.isrunning:
        print('making ahv plot...')
        cluster_data = spatial.plot_ahv(ops,adv,trial_data,cluster_data,spike_data,self) #plot AHV vs firing rate
        
#    if ops['run_novelty'] and self.worker.isrunning:
#        print('making novelty plot...')
#        cluster_data = spatial.plot_novelty(ops,adv,trial_data,cluster_data,spike_data,self)
        
    if ops['run_center_ego'] and self.worker.isrunning:
        print('making center ego plots...')
        trial_data = spatial.calc_center_ego_ahv(adv,trial_data)
        cluster_data = spatial.plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self)
        
    if ops['run_wall_ego'] and self.worker.isrunning:
        print('making wall ego plots...')
        cluster_data = spatial.plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self)
        
        
    #HD x spatial analyses
    if ops['hd_map'] and self.worker.isrunning:
        print('making spatial x HD plot...')
        cluster_data = spatial.plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self) #plot spatial by HD 
        
    #animations
    if ops['animated_hd_map'] and self.worker.isrunning:
        animation.animated_hd_map(ops,adv,trial_data,cluster_data,spike_data,self) #animated HD x spatial plot
    if ops['heatmap_animation'] and self.worker.isrunning:
        animation.animated_heatmap(ops,adv,trial_data,cluster_data,spike_data,self) #animated heatmap
        
    #time analyses
    if ops['run_autocorr'] and self.worker.isrunning:
        print('making time plots...')
        cluster_data = spatial.spike_autocorr(ops,adv,trial_data,cluster_data,spike_data,self) #plot ISI histogram and spike autocorrelation

    #spike-triggered analyses
    if ops['spike_timed'] and self.worker.isrunning:
        print('making spike-triggered plots')
        cluster_data = spatial.plot_st_heat(ops,adv,trial_data,cluster_data,spike_data,self) #plot spike-triggered path, heatmap, grid stuff if specified
 
    #ego analyses
    if ops['run_ego'] and ops['run_view'] and self.worker.isrunning:
        print('making ego and view plots')
        cluster_data = spatial.plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,ego=True,view=True) #plot ego rayleighs and mean angles
    elif ops['run_ego'] and self.worker.isrunning:
        print('making ego plots')
        cluster_data = spatial.plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,ego=True) #plot ego rayleighs and mean angles
    elif ops['run_view'] and self.worker.isrunning:
        print('making view plots')
        cluster_data = spatial.plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,view=True) 
        
    #make summary plots
    if ops['sumplots'] and self.worker.isrunning:
        print('making summary plots...')
        spatial.subplot_designer(ops,adv,trial_data,cluster_data,spike_data,self) #make summary plots
    
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
        plots.append('plot_interpd_heat')
        plots.append('plot_smoothed_heat')
    if ops['run_grid']:
        plots.append('plot_spatial_autocorr')
        plots.append('plot_grid_score')
    if ops['run_hd']:
        plots.append('plot_hd')
        plots.append('plot_half_hds')
    if ops['run_speed']:
        plots.append('plot_speed')
    if ops['run_ahv']:
        plots.append('plot_ahv')
#    if ops['run_novelty']:
#        plots.append('plot_novelty')
    if ops['run_center_ego']:
        plots.append('plot_center_ego')
        plots.append('plot_center_dist')
        plots.append('plot_center_ego_hd_map')
        plots.append('plot_center_ego_map')
    if ops['run_wall_ego']:
        plots.append('plot_wall_ego')
        plots.append('plot_wall_dist')
        plots.append('plot_wall_ego_hd_map')
        plots.append('plot_wall_ego_map')
    if ops['hd_map']:
        plots.append('plot_hd_map')
    if ops['run_autocorr']:
        plots.append('plot_isi')
        plots.append('plot_spike_autocorr')
    if ops['spike_timed']:
        plots.append('plot_spike_timed')
        plots.append('plot_spike_timed_heat')
        if ops['run_grid']:
            plots.append('plot_spike_timed_autocorr')
            plots.append('plot_spike_timed_gridness')
    if ops['run_ego']:
        plots.append('plot_ego')
        plots.append('plot_ego_angle')
    if ops['run_view']:
        plots.append('plot_view')
    if ops['sumplots']:
        plots.append('subplot_designer')
        
    #also make a list of every possible plot for future use    
    all_plots = ['plot_path','plot_raw_heat','plot_interpd_heat','plot_smoothed_heat',
                 'plot_spatial_autocorr','plot_grid_score','plot_hd','plot_half_hds',
                 'plot_speed','plot_ahv','plot_center_ego','plot_center_dist','plot_center_ego_hd_map',
                 'plot_center_ego_map','plot_wall_ego',
                 'plot_wall_dist','plot_hd_map','plot_wall_ego_hd_map','plot_wall_ego_map','plot_isi','plot_spike_autocorr',
                 'plot_spike_timed','plot_spike_timed_heat','plot_spike_timed_autocorr',
                 'plot_spike_timed_gridness','plot_ego','plot_ego_angle','plot_view','subplot_designer']
    
    #return the lists
    return plots,all_plots

def load_data(self,fname):
    ''' similar to "Run" function but loads existing data and skips some extraneous steps '''
    
    #make dicts for options, advanced options, and all_data
    ops = {}
    adv = {}
    all_data = {}

    #load advanced options
    with open('advanced.pickle','rb') as f:
        [adv['pic_resolution'],adv['grid_res'],adv['bin_size'],adv['autocorr_width'],adv['framerate'],adv['hd_bins'],adv['sample_cutoff']]= pickle.load(f)
    #load regular options
    with open('options.pickle','rb') as f:
        [ops['run_hd'],ops['run_spatial'],ops['hd_map'],ops['run_grid'],ops['run_speed'],ops['run_ahv'],ops['run_autocorr'],
        ops['spike_timed'],ops['run_ego'],ops['run_view'],ops['run_novelty'],ops['animated_hd_map'],ops['heatmap_animation'],ops['labview'],
        ops['save_all'],ops['sumplots'],ops['single_cluster'],ops['speedmode'],ops['save_data'],ops['load_data'],ops['acq']] = pickle.load(f)
    
    #if single_cluster mode, collect cluster file path and filename
    if ops['single_cluster']:
        ops['single_cluster_file'] = fname[0]
        ops['single_cluster_filename'] = os.path.basename(ops['single_cluster_file'])[:(len(os.path.basename(ops['single_cluster_file']))-4)]
        fname = os.path.dirname(fname[0])

    #gather how many trials we'll be looking at
    ops,trials = find_trials(ops,fname)
        
    #start a counter for plotting
    self.plot_counter = 0
    
    #for each trial...
    for trial in trials:
        if self.worker.isrunning:
            
            #make an entry in the all_data dict
            all_data[trial] = {}
            
            #try to open the pickled all_trial_data dict 
            while True:
                print('trying to open data file...')
                try:
                    with open(trial+'/all_trial_data.pickle','rb') as f:
                        all_trial_data = pickle.load(f) 
                    print('success!')
                except:
                    print('couldn\'t open data file! try again!!')
                    break
                break
                      
            #if this is the first trial...          
            if trial == trials[0]:
                
                #collect metadata
                metadata = collect_metadata(ops,fname,trials,all_trial_data)
    
                #figure out what the first cluster and cam are
                first_cluster = metadata['plot_list'][0][1]
                first_cam = metadata['plot_list'][0][2]
                
                #select data for the first cluster (/cam)
                first_trial_data = all_trial_data[first_cluster][first_cam]['trial_data']
                first_cluster_data = all_trial_data[first_cluster][first_cam]['cluster_data']
                first_spike_data = all_trial_data[first_cluster][first_cam]['spike_data']
                
                #check to see which data we already have and which we need to compute
                metadata = check_new_plots(all_trial_data,metadata,trial,first_cluster,first_cam)
                #change the plot options accordingly
                new_ops = change_ops(ops,metadata)
                
                #send metadata to the main gui
                self.worker.init_data.emit(metadata)
                
                #if we already have data for the first plot, plot it!
                if all_trial_data['metadata'][trial][first_cluster][first_cam]['dataready'][metadata['plots'][0]]:
                    self.worker.plotting_data.emit(ops,adv,first_trial_data,first_cluster_data,first_spike_data)
            
            #if we don't need to split up the tracking data for multiple cameras...
            if not ops['multi_cam']: 
                #and if we need to calculate speed or ahv data...
                if 'run_speed' in new_ops or 'run_ahv' in new_ops:
                    #calculate speed and ahv data for whole trial
                    speed_trial_data = speed_stuff(new_ops,adv,all_trial_data[metadata['plot_list'][self.plot_counter][1]][0]['trial_data'],self)
                    speeds = speed_trial_data['speeds']
                    ahvs = speed_trial_data['ahvs']
            
            #if single cluster mode, grab the cluster filenmae
            if ops['single_cluster']:
                cluster_filenames = [ops['single_cluster_filename']]
            #otherwise, grab filenames from loaded data
            else:
                cluster_filenames = all_trial_data[metadata['plot_list'][self.plot_counter][1]][0]['trial_data']['filenames']
            
            #for each cluster...
            for cluster in range(len(cluster_filenames)):
                if self.worker.isrunning:
                    
                    #make all_data dict entry for cluster
                    all_data[trial][cluster_filenames[cluster]] = []
                    
                    #grab the cameras used during the trial
                    cams = all_trial_data[metadata['plot_list'][self.plot_counter][1]][0]['trial_data']['cam_ids']
                        
                    #for each camera used...
                    for cam in range(len(cams)):
                        if self.worker.isrunning:
                        
                            #assign appropriate data dicts
                            trial_data = all_trial_data[cluster_filenames[cluster]][cam]['trial_data']
                            cluster_data = all_trial_data[cluster_filenames[cluster]][cam]['cluster_data']
                            spike_data = all_trial_data[cluster_filenames[cluster]][cam]['spike_data']    
                            
                            #add all_data dict entry for cam
                            all_data[trial_data['current_trial']][trial_data['current_cluster']].append({})
                            
                            #if just one camera and we just calculated speed/ahv data, add it to
                            #trial_data and spike_data dicts
                            if not ops['multi_cam']:
                                if 'run_speed' in new_ops:
                                    trial_data['speeds'] = speeds
                                    for i in range(len(speeds)):
                                        if spike_data['ani_spikes'][i]>0:
                                            for n in range(spike_data['ani_spikes'][i]):
                                                spike_data['spike_speeds'].append(speeds[i])                   
                                if 'run_ahv' in new_ops:
                                    trial_data['ahvs'] = ahvs
                                    for i in range(len(ahvs)):
                                        if spike_data['ani_spikes'][i]>0:
                                            for n in range(spike_data['ani_spikes'][i]):
                                                spike_data['spike_ahvs'].append(ahvs[i])
                            #otherwise, get speed and ahv data for this cam if we need to
                            elif 'run_speed' in new_ops or 'run_ahv' in new_ops:
                                trial_data = speed_stuff(new_ops,adv,trial_data,self)
                                             
                            #run the main analysis program using our new ops dict
                            trial_data,cluster_data,spike_data = analysis_stuff(new_ops,adv,trial_data,cluster_data,spike_data,self)
                            
                            #assign dicts for this cam session to the appropriate key in all_data dict
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['cluster_data'] = copy.deepcopy(cluster_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['spike_data'] = copy.deepcopy(spike_data)
                            all_data[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['trial_data'] = copy.deepcopy(trial_data)
                            
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
        
    #all done!
    print('Done!')
    #terminate the process running this script
    self.worker.isrunning = False
    

def check_new_plots(all_trial_data,metadata,first_trial,first_cluster,first_cam):
    ''' compare previously calculated data to requested plots to see what we need
    to calculate '''
    
    #make a 'new_plots' entry in metadata dict
    metadata['new_plots'] = []
    #for each plot we've been requested to make...
    for p in metadata['plots']:
        #if the data hasn't been computed before...
        if not all_trial_data['metadata'][first_trial][first_cluster][first_cam]['dataready'][p]:
            #add that plot to the new_plots list
            metadata['new_plots'].append(p)
            
    #return the metadata dict
    return metadata
            
def change_ops(ops,metadata):
    ''' make a new options dict so we don't recalculate any data '''
    
    #first make a copy of the OG ops dict
    new_ops = copy.deepcopy(ops)
    
    #then check our recently created new_plots list and set any options not in list
    #to False so we don't recompute old data
    if 'plot_path' not in metadata['new_plots'] and 'plot_raw_heat' not in metadata['new_plots'] and 'plot_interpd_heat' not in metadata['new_plots'] and 'plot_smoothed_heat' not in metadata['new_plots']:
        new_ops['run_spatial'] = False
    if 'plot_spatial_autocorr' not in metadata['new_plots']  and 'plot_grid_score' not in metadata['new_plots']:
        new_ops['run_grid'] = False
    if 'plot_hd'  not in metadata['new_plots'] and 'plot_half_hds' not in metadata['new_plots']:
        new_ops['run_hd'] = False
    if 'plot_speed' not in metadata['new_plots']:
        new_ops['run_speed'] = False
    if 'plot_ahv' not in metadata['new_plots']:
        new_ops['run_ahv'] = False
    if 'plot_hd_map' not in metadata['new_plots']:
        new_ops['hd_map'] = False
    if 'plot_isi' not in metadata['new_plots'] and 'plot_spike_autocorr' not in metadata['new_plots']:
        new_ops['run_autocorr'] = False
    if 'plot_spike_timed' not in metadata['new_plots'] and 'plot_spike_timed_heat' not in metadata['new_plots'] and 'plot_spike_timed_autocorr' not in metadata['new_plots'] and 'plot_spike_timed_gridness' not in metadata['new_plots']:
        new_ops['spike_timed'] = False
    if 'plot_ego' not in metadata['new_plots'] and 'plot_ego_angle' not in metadata['new_plots']:
        new_ops['run_ego'] = False
    if 'plot_view' not in metadata['new_plots']:
        new_ops['run_view'] = False
#    if 'plot_novelty' not in metadata['new_plots']:
#        new_ops['run_novelty'] = False
    if 'plot_center_ego' not in metadata['new_plots'] and 'plot_center_dist' not in metadata['new_plots'] and 'plot_center_ego_hd_map' not in metadata['new_plots'] and 'plot_center_ego_map' not in metadata['new_plots']:
        new_ops['run_center_ego'] = False
    if 'plot_wall_ego' not in metadata['new_plots'] and 'plot_wall_dist' not in metadata['new_plots'] and 'plot_wall_ego_hd_map' not in metadata['new_plots'] and 'plot_wall_ego_map' not in metadata['new_plots']:
        new_ops['run_wall_ego'] = False
        
    #return the new options dict
    return new_ops
    



    
#if ops['debug_mode']:       
#    trial_data,cluster_data=Run('fname')  
#####for visualizing variations in actual data capture frequency (i.e. 30 fps should give straight line at ~33000 us)
#timestamp_gaps = []
#for i in range(len(timestamps)-1):
#    timestamp_gaps.append(timestamps[i+1]-timestamps[i])
#samples = np.arange(len(timestamps)-1)
#axes=plt.gca()
#axes.set_ylim([0,60000])
#plt.plot(samples,timestamp_gaps)
#print(np.mean(timestamp_gaps))