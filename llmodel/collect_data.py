# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:39:35 2018

spatial/admin stuff for llmodel (so we don't have to directly call stuff 
from spike-analysis)

@author: Patrick
"""

import os
import shutil
import csv
import bisect
import numpy as np

import read_nlx
import read_oe

adv = {}
adv['framerate'] = 30.
adv['grid_res'] = 20
adv['bin_size'] = .05
adv['hd_bins'] = 30

ops = {}
ops['acq'] = 'neuralynx'

def find_trials(fdir):
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
            #for every file in folder...
            for nfile in os.listdir(fdir + '/' + file):
                #if there's a tetrode timestamp file...
                if nfile.endswith(".txt") and nfile.startswith('TT'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilosorted_spikefiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                #if there's a stereotrode timestamp file...
                elif nfile.endswith(".txt") and nfile.startswith('ST'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilosorted_spikefiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                #if there's a video file, make a note of it!
                elif nfile.endswith('.nvt'):
                    video_file = True
                elif ops['acq']=='openephys' and nfile.startswith('vt1') and nfile.endswith('.txt'):
                    video_file = True
            #if the folder contains timestamp files and a video file, add it to the "trials" list
            if count > 0 and video_file:
                trials.append(fdir + '/' + file)

    #return our options (with multi-session entry) and trials list
    return trials

def read_files(fdir,trial,video=True):
    ''' read important files in the trial folder and extract relevant info '''
    
    #make a dict for trial_data
    trial_data = {}
        
    #make note of the current trial
    trial_data['trial'] = trial

    #make lists for cluster ts files and their names
    trial_data['cluster_files'] = []
    trial_data['filenames'] = []

    #do stuff with relevant files in the trial folder
    for file in os.listdir(trial):

        #add tetrode ts file paths and filenames to appropriate entries in trial_data dict
        if file.endswith(".txt") and file.startswith('TT') and os.stat(trial+'/'+file).st_size != 0:
            trial_data['cluster_files'].append((trial+'/'+file))
            trial_data['filenames'].append(file[:(len(file)-4)])
        #same as tetrodes but with stereotrodes
        if file.endswith(".txt") and file.startswith('ST') and os.stat(trial+'/'+file).st_size != 0:
            trial_data['cluster_files'].append((trial+'/'+file))
            trial_data['filenames'].append(file[:(len(file)-4)])            
        #if video file, grab the path and filename
        if video:
            if file.endswith('.nvt'):
                video_file = trial + '/' + file
                filename = file
                #extract raw tracking data from the video file using read_nvt function
                trial_data, raw_vdata = read_nlx.read_nvt(video_file,filename,trial_data)
                                   
            if file.startswith('vt1') and file.endswith('.txt') and ops['acq'] == 'openephys':
                video_file = trial + '/' + file
                filename = file
                trial_data, raw_vdata = read_oe.read_video_file(video_file,filename,trial_data)
    if video:
        return trial_data,raw_vdata
    else:
        return trial_data

def tracking_stuff(fdir,trial):
    ''' collect video tracking data and interpolate '''

    #read relevant files to extract raw video data/multi-cam info
    trial_data,raw_vdata = read_files(fdir,trial)

    print('processing tracking data for this session...')
    #interpolate nondetects using extracted angles and coordinates
    trial_data = interp_points(raw_vdata,trial_data)
    #calculate centerpoints and timestamp vectors
    trial_data = center_points(trial_data)
        
    return trial_data
        
def speed_stuff(trial_data):
    ''' calculate speed and ahv info from tracking data '''
    
    trial_data['speeds'] = []
    trial_data['ahvs'] = []
    
    print('processing speed data...')
    #calculate running speeds for each frame
    trial_data = calc_speed(trial_data)
    
    print('processing AHV data...')
    #calculate ahvs for each frame
    trial_data = calc_ahv(trial_data)
        
    return trial_data

def interp_points(raw_vdata,trial_data):
    """interpolates nondetects in raw neuralynx data"""
    
    #grab data from video conversion
    interpd_pos =  raw_vdata['positions']
    interpd_angles = raw_vdata['angles']
    #interpolate nondetects in LED location data
    for i in range(len(interpd_pos)):
        for j in range(1,5):
            if interpd_pos[i][j] == 0:
                x = 0
                count = 1
                while x == 0:
                    if i+count < len(interpd_pos):
                        if interpd_pos[i+count][j] == 0:
                            count +=1
                        elif interpd_pos[i+count][j] > 0:
                            interpd_pos[i][j] = interpd_pos[i-1][j] + (interpd_pos[i+count][j] - interpd_pos[i-1][j])/count
                            x=1
                    else:
                        interpd_pos[i][j] = 250
                        x=1
        #interpolate nondetects in head angle data
        if interpd_angles[i] == 450 or interpd_angles[i] == 0:
            x = 0
            count = 1
            while x == 0:
                if i+count < len(interpd_pos):
                    if interpd_angles[i+count] == 450 or interpd_angles[i+count] == 0:
                        count +=1
                    else:
                        interpd_angles[i] = interpd_angles[i-1] + (interpd_angles[i+count] - interpd_angles[i-1])/count
                        x=1
                else:
                    interpd_angles[i] = interpd_angles[i-1]
                    x=1
        #adjust angles so move counterclockwise starting at positive x axis
        interpd_angles[i] = 450 - interpd_angles[i]
        if interpd_angles[i] >= 360:
            interpd_angles[i] -= 360
            
    #return interpolated LED location and head angle data
    trial_data['positions'] = interpd_pos
    trial_data['angles'] = interpd_angles
    
    return trial_data
    
def center_points(trial_data):
    """finds center point between LEDs for each timestamp"""
    
    #grab interpolated position data
    interpd_pos = trial_data['positions']
    #make lists for timestamps and center coordinates
    timestamps = [] 
    center_x = []
    center_y = []

    #for every frame...
    for i in range(len(interpd_pos)):
        #add timestamp to timestamp list
        timestamps.append(interpd_pos[i][0])
        #find center x and y coordinates
        x = (interpd_pos[i][1]+interpd_pos[i][3])/2
        y = (interpd_pos[i][2]+interpd_pos[i][4])/2
        #add them to appropriate lists
        
        if ops['acq'] == 'neuralynx':
            center_x.append(x)
            center_y.append(y)
        elif ops['acq'] == 'openephys':
            center_y.append(x)
            center_x.append(y)
        
        
    if ops['acq'] == 'neuralynx':
        #now we have to flip neuralynx's inverted y-axis --
        #first find the midpoint of the y axis so we can flip over it
        mid_y = (max(center_y)+min(center_y))/2
        #now flip all the y-coordinates!
        for i in range(len(center_y)):
            center_y[i] = 2 * mid_y - center_y[i]
            
    elif ops['acq'] == 'openephys':
        #now we have to flip the x axis (oops)
        mid_x = (max(center_x)+min(center_x))/2
        #now flip all the y-coordinates!
        for i in range(len(center_x)):
            center_x[i] = 2 * mid_x - center_x[i]
    
    #collect histogram of spatial occupancy so we can use the x and y edges for plotting      
    h,xedges,yedges = np.histogram2d(center_x,center_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    

    #interpolates new timestamps between existing timestamps for spike time     
    #analyses to reach temporal precision given by bin_size variable 
    spike_timestamps = []
    for i in range(len(timestamps)):
        if i < len(timestamps)-1:
            increment = (timestamps[i+1]-timestamps[i])/(1000./(adv['framerate']*adv['bin_size']))
            for j in range(int(1000./((adv['framerate']*adv['bin_size'])))):
                spike_timestamps.append(timestamps[i]+j*increment)
                
    #return coordinates of head center and timestamps for spatial and spike
    #analysis (and x and y edges of heatmap for plotting purposes)
    trial_data['center_x'] = center_x
    trial_data['center_y'] = center_y
    trial_data['timestamps'] = timestamps
    trial_data['spike_timestamps'] = spike_timestamps
    trial_data['heat_xedges'] = xedges
    trial_data['heat_yedges'] = yedges
    
    return trial_data


def calc_speed(trial_data):
    """calculates 'instantaneous' linear speeds for each video frame"""
    
    #grab appropriate tracking data
    center_x=trial_data['center_x']
    center_y=trial_data['center_y']
    
    #make an array of zeros to assign speeds to
    speeds = np.zeros(len(center_x),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(center_x)-2):
        #grab 5 x and y positions centered on that frame
        x_list = center_x[(i-2):(i+3)]
        y_list = center_y[(i-2):(i+3)]
        #find the best fit line for those 5 points (slopes are x and y components
        #of velocity)
        xfitline = np.polyfit(range(0,5),x_list,1)
        yfitline = np.polyfit(range(0,5),y_list,1)
        #total velocity = framerate * sqrt(x component squared plus y component squared)
        speeds[i] = adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
    #set unassigned speeds equal to closest assigned speed
    speeds[0] = speeds[2]
    speeds[1] = speeds[2]
    speeds[len(speeds)-1] = speeds[len(speeds)-3]
    speeds[len(speeds)-2] = speeds[len(speeds)-3]
    
    #return calculated speeds
    trial_data['speeds'] = speeds
    return trial_data

def calc_ahv(trial_data):
    """calculates 'instantaneous' angular head velocity for each video frame"""
    
    #grab appropriate tracking data
    angles=trial_data['angles']
    #TODO: smooth angles?
    #make an array of zeros to assign ahvs to
    ahvs = np.zeros(len(angles),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(angles)-2):
        #start a list of head directions with the direction 2 frames ago
        angle_list = [angles[i-2]]
        #for each frame until angle_list is 5 angles long...
        for j in range(-1,3):
            #check to see if the difference between the current angle and the preceding angle is
            #greater or less than 180 degrees. if it's greater, it's probably an artifact of circular
            #data expressed in non-circular numbers (i.e. if a turn from 2 degrees to 358 degrees, probably a 
            #clockwise 4 degree turn instead of a counterclockwise 356 degree turn), so adjust the current
            #angle accordingly
            qangle = angles[i+j] + 360*(int(angle_list[len(angle_list)-1])/360)
            if qangle - angle_list[len(angle_list)-1] < 180 and qangle - angle_list[len(angle_list)-1] > -180:
                angle_list.append(qangle)
            elif qangle - angle_list[len(angle_list)-1] >= 180:
                angle_list.append(qangle-360)
            elif qangle - angle_list[len(angle_list)-1] <= -180:
                angle_list.append(qangle+360)
        #find a line of best fit for the 5 directions in angle_list (slope is AHV)
        fitline = np.polyfit(range(0,5),angle_list,1)
        #AHV = framerate X slope of fit line
        ahvs[i] = adv['framerate']*fitline[0]

    #set empty spots equal to closest value
    ahvs[0] = ahvs[2]
    ahvs[1] = ahvs[2]
    ahvs[len(ahvs)-1] = ahvs[len(ahvs)-3]
    ahvs[len(ahvs)-2] = ahvs[len(ahvs)-3]

    #return our calculated ahvs
    trial_data['ahvs'] = ahvs
    return trial_data

def calc_novelty(trial_data):
    
    occ_map = np.zeros((adv['grid_res'],adv['grid_res']))
    
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    
    novelties = np.zeros(len(center_x))
    
    xedges = np.arange(np.min(center_x),np.max(center_x),np.ptp(center_x)/np.float(adv['grid_res']-1))
    yedges = np.arange(np.min(center_y),np.max(center_y),np.ptp(center_y)/np.float(adv['grid_res']-1))
    
    x_bins = np.digitize(center_x,bins=xedges,right=True)
    y_bins = np.digitize(center_y,bins=yedges,right=True)
    
    x_bins[x_bins >= 20] = 19
    y_bins[y_bins >= 20] = 19
    
    print np.max(x_bins)
    print np.min(x_bins)

    for i in range(len(novelties)):

        occ_map[x_bins[i],y_bins[i]] += 1
        novelties[i] = occ_map[x_bins[i],y_bins[i]]
        
    cutoff = np.sort(novelties)[int(len(novelties)*.90)]
    novelties = np.clip(novelties,0,cutoff)
    
    trial_data['novelties'] = novelties
    
    return trial_data

def get_spikes(trial,name,trial_data):
    
    ts_file = trial + '/' + name + '.txt'
    
    spike_list = ts_file_reader(ts_file)
    spike_train = create_spike_train(trial_data,spike_list)
    
    return spike_train

def ts_file_reader(ts_file):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'),dialect='excel-tab')

    for row in reader:
        spike_list.append(int(row[0]))
                
    #return it!
    return spike_list

def create_spike_train(trial_data,spike_list):
    """makes lists of spike data"""

    #array of zeros length of video timestamps for plotting/animation purposes
    spike_train = np.zeros(len(trial_data['timestamps']),dtype=np.int)
    #for each spike timestamp...
    for i in spike_list:
        #find closest video frame 
        ind = bisect.bisect_left(trial_data['timestamps'],i)
        
        if ind < len(trial_data['timestamps']):
            #increments the # of spikes in corresponding video frame
            spike_train[ind] += 1
    
    return spike_train