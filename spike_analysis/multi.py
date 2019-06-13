# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:37:24 2017

@author: Patrick
"""

import bisect
import csv

def file_creator(ops,trial_data):
    ''' creates multi_data dict splitting trial data by camera (event) '''
        
    #create the multi_data dict with relevant fields
    multi_data = {'center_x':[],'center_y':[],'timestamps':[],'spike_timestamps':[],'angles':[],'speeds':[],'ahvs':[],'cam_id':[],'folders':[]}
    #assign relevant administrative info to multi_data dict
    multi_data['filenames'] = trial_data['filenames']
    multi_data['cluster_files'] = trial_data['cluster_files']
    multi_data['trial'] = trial_data['trial']
    #find camera timestamps and spike timestamps (high precision) closest to camera switches
    ts_inds = []
    sts_inds = []
    for i in range(len(trial_data['cam_timestamp'])):
        ts_inds.append(bisect.bisect_left(trial_data['timestamps'],trial_data['cam_timestamp'][i]))
        sts_inds.append(bisect.bisect_left(trial_data['spike_timestamps'],trial_data['cam_timestamp'][i]))
    #for each camera switch (coded by a camera timestamp index)
    for j in range(len(ts_inds)):
        #if this is the first camera switch...
        if j == 0:
            #assign data from first timestamp (index 0) to this timestamp
            multi_data['center_x'].append(trial_data['center_x'][0:ts_inds[j]])
            multi_data['center_y'].append(trial_data['center_y'][0:ts_inds[j]])
            multi_data['timestamps'].append(trial_data['timestamps'][0:ts_inds[j]])
            multi_data['spike_timestamps'].append(trial_data['spike_timestamps'][0:sts_inds[j]])
            multi_data['angles'].append(trial_data['angles'][0:ts_inds[j]])
            if ops['run_speed']:
                multi_data['speeds'].append(trial_data['speeds'][0:ts_inds[j]])
            if ops['run_speed']:
                multi_data['ahvs'].append(trial_data['ahvs'][0:ts_inds[j]])
            multi_data['cam_id'].append(trial_data['cam_id'][j])
        else:
            #assign data from last switch timestamp to current switch timestamp
            multi_data['center_x'].append(trial_data['center_x'][ts_inds[j-1]:ts_inds[j]])
            multi_data['center_y'].append(trial_data['center_y'][ts_inds[j-1]:ts_inds[j]])
            multi_data['timestamps'].append(trial_data['timestamps'][ts_inds[j-1]:ts_inds[j]])
            multi_data['spike_timestamps'].append(trial_data['spike_timestamps'][sts_inds[j-1]:sts_inds[j]])
            multi_data['angles'].append(trial_data['angles'][ts_inds[j-1]:ts_inds[j]])
            if ops['run_speed']:
                multi_data['speeds'].append(trial_data['speeds'][ts_inds[j-1]:ts_inds[j]])
            if ops['run_ahv']:
                multi_data['ahvs'].append(trial_data['ahvs'][ts_inds[j-1]:ts_inds[j]])
            multi_data['cam_id'].append(trial_data['cam_id'][j])
            
    #return relevant data
    multi_data['ts_inds'] = ts_inds
    multi_data['spike_ts_inds'] = sts_inds
    trial_data['cam_ids'] = multi_data['cam_id']

    return multi_data,trial_data

def spikefile_cutter(trial_data,multi_data):
    ''' cuts the spike timestamp file along event boundaries '''
    
    #grab appropriate data
    cluster_files = trial_data['cluster_files']
    filenames = trial_data['filenames']
    cam_timestamp = trial_data['cam_timestamp']
    
    #start dictionary for spike data
    spike_data = {}
    #for each cluster timestamp file
    for x in range(len(cluster_files)):
        ts_file = cluster_files[x]
        #make a spike_data dict entry for this file
        spike_data[filenames[x]] = []
        #start a list for spike timestamps
        spike_list = []
        #read the timestamp file and assign timestamps to spike_list
        reader = csv.reader(open(ts_file,'rb'))
        for row in reader:
            spike_list.append(int(row[0]))
        
        #make a list for spike timestamp indices
        spike_inds = []
        #assign indices for spike timestamps closest to camera switch timestamps
        for i in range(len(cam_timestamp)):
            spike_inds.append(bisect.bisect_left(spike_list,cam_timestamp[i]))
        
        #for each cam switch index...
        for j in range(len(spike_inds)):
            #assign relevant spike data to relevant list in spike_data
            if j == 0:
                spike_data[filenames[x]].append(spike_list[0:spike_inds[j]])        
            else:
                spike_data[filenames[x]].append(spike_list[spike_inds[j-1]:spike_inds[j]])
    
    #return data
    multi_data['spike_data'] = spike_data
    
    return multi_data

def assign_multi_data(ops,w,multi_data,trial_data):
    #set all trial_data entries equal to the data from this camera session
    trial_data['center_x'] = multi_data['center_x'][w]
    trial_data['center_y'] = multi_data['center_y'][w]
    trial_data['timestamps'] = multi_data['timestamps'][w]
    trial_data['spike_timestamps'] = multi_data['spike_timestamps'][w]
    trial_data['angles'] = multi_data['angles'][w]
    if ops['run_speed']:
        trial_data['speeds'] = multi_data['speeds'][w]
    if ops['run_ahv']:
        trial_data['ahvs'] = multi_data['ahvs'][w]
    trial_data['cam_id'] = multi_data['cam_id'][w]

    return trial_data
        