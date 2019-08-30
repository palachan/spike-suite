# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:30:09 2016

calculates data for plots

@author: Patrick
"""
import bisect
import numpy as np
from scipy.stats import pearsonr,zscore
from scipy.stats.mstats import linregress
from scipy.stats.mstats import pearsonr as mapearsonr
from scipy.optimize import curve_fit,leastsq
from skimage.transform import rotate
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
import copy
import warnings
import numba as nb

#####################

def interp_points(raw_vdata,trial_data):
    """interpolates nondetects in raw neuralynx data"""
    
    #grab data from video conversion
    interpd_pos =  np.array(raw_vdata['positions'])
    interpd_angles = np.array(raw_vdata['angles'])
    
    bounds = trial_data['bounds']
    
    if bounds is not None:
        zeros = np.zeros(len(interpd_pos),dtype=np.bool)

        red_x = np.array(interpd_pos[:,1]).flatten()
        green_x = np.array(interpd_pos[:,3]).flatten()
        zeros[(red_x<bounds[0])|(red_x>bounds[1])|(green_x<bounds[0])|(green_x>bounds[1])] = 1
        
        red_y = np.array(interpd_pos[:,2]).flatten()
        green_y = np.array(interpd_pos[:,4]).flatten()
        zeros[(red_y<bounds[2])|(red_y>bounds[3])|(green_y<bounds[2])|(green_y>bounds[3])] = 1
        
        interpd_pos[zeros,1:] = 0
        interpd_angles[zeros] = 450
    
    else:
        for i in range(1,5):
            zscores = np.abs(zscore(interpd_pos[:,i]))
            for j in np.where(zscores>2)[0]:
                interpd_pos[j,i] = 0 

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
                            if i>0:
                                interpd_pos[i][j] = interpd_pos[i-1][j] + (interpd_pos[i+count][j] - interpd_pos[i-1][j])/count
                                x=1
                            else:
                                interpd_pos[i][j] = interpd_pos[i+count][j]
                                x=1
                    else:
                        interpd_pos[i][j] = interpd_pos[i-1][j]
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
    
def center_points(ops,adv,trial_data):
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
        
        if ops['acq'] == 'neuralynx' or ops['acq']=='taube':
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
            
#    for i in range(len(center_x)):
#        if center_x[i] > 750:
#            center_x[i] = 750
#    center_x[center_x>590] = 590
#    print np.max(center_x)
            
#            if center_x[i] > 500:
#                center_x[i] = 500
            
#    center_x = np.array(center_x) - np.mean(center_x)
#    center_y = np.array(center_y) - np.mean(center_y)
#    
#    polar_dirs = np.arctan2(center_y,center_x)
#    polar_dists = np.sqrt(np.array(center_x)**2 + np.array(center_y)**2)
#    
#    polar_dirs -= np.pi/4.
#    
#    center_x = polar_dists * np.cos(polar_dirs)
#    center_y = polar_dists * np.sin(polar_dirs)
#    angles = np.array(trial_data['angles']) - 45.
#    angles = angles%360
#    
#    center_x = center_x.tolist()
#    center_y = center_y.tolist()
#    angles = angles.tolist()
#    
#    trial_data['angles'] = angles
    
    #collect histogram of spatial occupancy so we can use the x and y edges for plotting      
    h,xedges,yedges = np.histogram2d(center_x,center_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    

    #interpolates new timestamps between existing timestamps for spike time     
    #analyses to reach temporal precision given by bin_size variable 
    spike_timestamps = np.arange(np.min(timestamps),np.max(timestamps),adv['bin_size']*1000.)

#    spike_timestamps = []
#    for i in range(len(timestamps)):
#        if i < len(timestamps)-1:
#            increment = (timestamps[i+1]-timestamps[i])/(1000./(adv['framerate']*adv['bin_size']))
#            for j in range(int(1000./((adv['framerate']*adv['bin_size'])))):
#                spike_timestamps.append(timestamps[i]+j*increment)
    
                
    #return coordinates of head center and timestamps for spatial and spike
    #analysis (and x and y edges of heatmap for plotting purposes)
    
    trial_data['center_x'] = center_x
    trial_data['center_y'] = center_y
    trial_data['timestamps'] = timestamps
    trial_data['spike_timestamps'] = spike_timestamps
    trial_data['heat_xedges'] = xedges
    trial_data['heat_yedges'] = yedges
    
    return trial_data
    
def create_spike_lists(ops,trial_data,cluster_data):
    """makes lists of spike data"""

    #dictionary for spike data
    spike_data = {}
    #lists containing info for each spike (or isi)
    spike_x = []
    spike_y = []
    spike_angles = []
    spike_speeds = []
    spike_ahvs = []
    isi_list = []

    #creates array of zeros length of spike_timestamps to create spike train
    spike_train = np.zeros(len(trial_data['spike_timestamps']))
    #array of zeros length of video timestamps for plotting/animation purposes
    ani_spikes = np.zeros(len(trial_data['timestamps']),dtype=np.int)
    #for each spike timestamp...
    for i in cluster_data['spike_list']:
        #find closest video frame 
        ind = bisect.bisect_left(trial_data['timestamps'],i)
        #find closest entry in high precision 'spike timestamps' list
        spike_ind = bisect.bisect_left(trial_data['spike_timestamps'],i)
        
        if ind < len(trial_data['timestamps']):
            #increments the # of spikes in corresponding video frame
            ani_spikes[ind] += 1
            #creates lists of center position and head angle for each spike
            spike_x.append(trial_data['center_x'][ind])
            spike_y.append(trial_data['center_y'][ind])
            #list of head directions for each spike
            spike_angles.append(trial_data['angles'][ind])
            if ops['run_speed']:
                #list of running speeds for each spike
                spike_speeds.append(trial_data['speeds'][ind])
            if ops['run_ahv']:
                #list of angular head velocities for each spike
                spike_ahvs.append(trial_data['ahvs'][ind])
            if spike_ind < len(spike_train):
                #add 1 to spike train at appropriate spot
                spike_train[spike_ind] = 1
    #find the video timestamp at the halfway point
    halfway_ind = bisect.bisect_left(cluster_data['spike_list'],trial_data['timestamps'][np.int(len(trial_data['timestamps'])/2)]) - 1

    #creates list of interspike intervals in microseconds
    for i in range(len(cluster_data['spike_list'])-1):
        isi_list.append(cluster_data['spike_list'][i+1]-cluster_data['spike_list'][i])
            
    
    #returns spatial and temporal information for each spike
    spike_data['spike_x'] = spike_x
    spike_data['spike_y'] = spike_y
    spike_data['spike_angles'] = spike_angles
    spike_data['spike_speeds'] = spike_speeds
    spike_data['spike_ahvs'] = spike_ahvs
    spike_data['ani_spikes'] = ani_spikes
    spike_data['spike_train'] = spike_train
    spike_data['isi_list'] = isi_list
    cluster_data['halfway_ind'] = halfway_ind
    
    return spike_data, cluster_data
  
def plot_path(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots rat's running path and spike locations"""
    
    #note that path plot data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_path'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

def plot_heat(ops,adv,trial_data,cluster_data,spike_data,self,test=False):
    """plots raw, interpolated, and smoothed spatial heatmaps"""
    
    #grab appropriate tracking data
    if not test:
        if not self.spike_timed:
            center_x = trial_data['center_x']
            center_y = trial_data['center_y']
            spike_x = spike_data['spike_x']
            spike_y = spike_data['spike_y']
        else:
            center_x = cluster_data['st_long_xs']
            center_y = cluster_data['st_long_ys']
            spike_x = cluster_data['st_long_sxs']
            spike_y = cluster_data['st_long_sys']
    elif test:
        center_x = trial_data['center_x']
        center_y = trial_data['center_y']
        spike_x = spike_data['spike_x']
        spike_y = spike_data['spike_y']
        
#    ahvs = np.array(trial_data['ahvs'])
#    spike_ahvs = np.array(spike_data['spike_ahvs'])
#    print np.shape(center_x)
#    print np.shape(ahvs)
#    center_x[ahvs<0] = np.nan
#    center_y[ahvs<0] = np.nan
#    spike_x[spike_ahvs<0] = np.nan
#    spike_y[spike_ahvs<0] = np.nan
    
    #calculate 2D histogram of spatial occupancies for the rat's path, break 
    #arena into bins assigned by grid_res parameter
    h,xedges,yedges = np.histogram2d(center_x,center_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    #calculate 2D histogram of spikes per bin
    spikeh,spikexedges,spikeyedges = np.histogram2d(spike_x,spike_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]]) 
    #only include bins with sampling greater than set by sample_cutoff parameter
    for i in range(len(h)):
        for j in range(len(h[i])):
            if float(h[i][j])/adv['framerate'] < adv['sample_cutoff']:
                h[i][j] = 0
                spikeh[i][j] = 0
                
    info,sparsity = spatial_info(adv,center_x,center_y,spike_x,spike_y)
    print('spatial info and sparsity:')
    print(info)
    print(sparsity)

    #divide spikes by occupancy time per bin
    with np.errstate(invalid='ignore'):
        raw_heatmap = spikeh/h
    #correct axes of heatmap (backwards from histogram function)
    raw_heatmap=np.swapaxes(raw_heatmap,0,1).tolist()[::-1]
    #make it an array
    raw_heatmap = np.asarray(raw_heatmap)
    
    #change from spikes/frame to spikes/sec
    raw_heatmap *= adv['framerate']

    #send the raw heatmap to interpolation function
    interpd_heatmap = interp_raw_heatmap(raw_heatmap)
        
    #create a smoothed heatmap using convolution with a Gaussian kernel
    smoothed_heatmap = convolve(interpd_heatmap, Gaussian2DKernel(stddev=2))

    if test:
        return smoothed_heatmap
        
    #if we're not making spike-triggered heatmaps...
    if not self.spike_timed:
        #add heatmaps to cluster data
        cluster_data['heat_xedges'] = xedges
        cluster_data['heat_yedges'] = yedges
        cluster_data['raw_heatmap'] = raw_heatmap
        cluster_data['interpd_heatmap'] = interpd_heatmap
        cluster_data['smoothed_heatmap'] = smoothed_heatmap
        
        #note that raw_heatmap data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_raw_heat'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
        #note that interpd_heatmap data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_interpd_heat'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
            
        #note that smoothed_heatmap data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_smoothed_heat'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    #if we are making spike-triggered heatmaps...
    else:
        #just add spike-triggered heatmap data to cluster_data dict
        cluster_data['st_xedges'] = xedges
        cluster_data['st_yedges'] = yedges
        cluster_data['st_smoothed_heatmap'] = smoothed_heatmap
        
    #return data
    return cluster_data
    
#@nb.njit()
def interp_raw_heatmap(raw_heatmap):
    """interpolates empty bins in the raw heatmap"""
    
    #store the raw heatmap in a new variable for interpolation
    interpd_heatmap = copy.deepcopy(raw_heatmap)
    #find the empty spots in the raw heatmap
    where_are_NaNs = np.isnan(interpd_heatmap)
    #assign dummy firing rate (300) to unvisited spots
    interpd_heatmap[where_are_NaNs] = 300
    
    #march through data, if you find a dummy bin (fr = 300) interp according to
    #closest real values along x, y, and diagonal directions
    for i in range(len(interpd_heatmap)):
        for j in range(len(interpd_heatmap[i])):
            if interpd_heatmap[i][j] == 300:
                x = y = z = w = 0
                countx = county = countz = countw = 1
                while x == 0 or y == 0 or z == 0 or w == 0:
                    if j+countx < len(interpd_heatmap[i]):
                        if interpd_heatmap[i][j+countx] == 300:
                            countx +=1
                        elif interpd_heatmap[i][j+countx] != 300:
                            if j > 0:
                                x_val = interpd_heatmap[i][j-1] + (interpd_heatmap[i][j+countx] - interpd_heatmap[i][j-1])/countx
                                x = 1
                            else:
                                x_val = 0.
                                x = 1
                    elif j > 0:
                        x_val = interpd_heatmap[i][j-1]
                        x = 1
                    else:
                        x_val = 0.
                        x=1
                    if i+county < len(interpd_heatmap):
                        if interpd_heatmap[i+county][j] == 300:
                            county +=1
                        elif interpd_heatmap[i+county][j] != 300:
                            if i > 0:
                                y_val = interpd_heatmap[i-1][j] + (interpd_heatmap[i+county][j] - interpd_heatmap[i-1][j])/county
                                y = 1
                            else:
                                y_val = 0.
                                y = 1
                    elif i > 0:
                        y_val = interpd_heatmap[i-1][j]
                        y = 1
                    else:
                        y_val = 0.
                        y = 1
                    if i+countz < len(interpd_heatmap)-1 and j+countz < len(interpd_heatmap[i])-1:
                        if interpd_heatmap[i+countz][j+countz] == 300:
                            countz +=1
                        elif interpd_heatmap[i+countz][j+countz] != 300:
                            if i > 0 and j > 0:
                                z_val = interpd_heatmap[i-1][j-1] + (interpd_heatmap[i+countz][j+countz] - interpd_heatmap[i-1][j-1])/countz
                                z = 1
                            else:
                                z_val = 0.
                                z = 1
                    elif i > 0 and j > 0:
                        z_val = interpd_heatmap[i-1][j-1]
                        z = 1
                    else:
                        z_val = 0.
                        z = 1
                    if i+countw < len(interpd_heatmap) and j-countw > 0:
                        if interpd_heatmap[i+countw][j-countw] == 300:
                            countw +=1
                        elif interpd_heatmap[i+countw][j-countw] != 300:
                            if i > 0 and j > 0 and (j+1) < len(interpd_heatmap[i]):
                                w_val = interpd_heatmap[i-1][j+1] + (interpd_heatmap[i+countw][j-countw] - interpd_heatmap[i-1][j+1])/countw
                                w = 1
                            else:
                                w_val = 0.
                                w = 1
                    elif i > 0 and j > 0:
                        w_val = interpd_heatmap[i-1][j-1]
                        w = 1
                    else:
                        w_val = 0.
                        w = 1
                interpd_heatmap[i][j] = np.mean([x_val,y_val,z_val,w_val])
                
    #return the interpolated heatmap
    return interpd_heatmap

def plot_st_heat(ops,adv,trial_data,cluster_data,spike_data,self):
    """makes path, smoothed heatmap, and grid plots (if specified)
    with trajectories/spikes plotted for a set window_length after each spike 
    and position of triggering spike moved to center of plot"""
    
    #grab appropriate tracking and spike data
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    ani_spikes = spike_data['ani_spikes']

    #make lists for x and y values for path and spikes
    long_xs=[]
    long_ys=[]
    long_sxs=[]
    long_sys=[]
    
    #window length in seconds
    window_length = 20.

    #for each timestamp...
    for i in range(len(ani_spikes)):
        #if there are spikes associated with it...
        if ani_spikes[i] > 0:
            #find the index of the video frame one window length in the future
            stop_ind = i + int(window_length*adv['framerate'])
            if i + stop_ind < len(ani_spikes):
                #for each spike in the start frame...
                for k in range(ani_spikes[i]):
                    #for every frame in the window...
                    for j in range(i,stop_ind):
                        #add spike locations if they exist
                        if ani_spikes[j] > 0:
                            for k in range(ani_spikes[j]):
                                long_sxs.append(center_x[j]-center_x[i])
                                long_sys.append(center_y[j]-center_y[i])
                        #add animal location
                        long_xs.append(center_x[j]-center_x[i])
                        long_ys.append(center_y[j]-center_y[i])
                        
    #let the calculation functions know we're making spike-triggered maps so they
    #don't call the regular plotting functions (we'll call them in a few lines)
    self.spike_timed = True

    #assign data to cluster_data dict for use by calc/plot functions
    cluster_data['st_long_xs'] = long_xs
    cluster_data['st_long_ys'] = long_ys
    cluster_data['st_long_sxs'] = long_sxs
    cluster_data['st_long_sys'] = long_sys
    
    #note that path plot data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spike_timed'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #run the heatmap function, collect the data
    cluster_data = plot_heat(ops,adv,trial_data,cluster_data,spike_data,self)
    #note that heatmap data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spike_timed_heat'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    if ops['run_grid']:
        #run the autocorrelation function, collect the data
        cluster_data = spatial_autocorrelation(ops,adv,trial_data,cluster_data,spike_data,self)
        #note that autocorrelation data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spike_timed_autocorr'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
        #run the grid score function, collect the data
        cluster_data = grid_score(ops,adv,trial_data,cluster_data,spike_data,self)
        #note that grid score data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spike_timed_gridness'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
    #let the functions know we're done making spike-triggered plots
    self.spike_timed = False
    
    #return the data
    return cluster_data

def plot_novelty(ops,adv,trial_data,cluster_data,spike_data,self):
    
    occ_map = np.zeros((adv['grid_res'],adv['grid_res']))
    
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    spike_train = spike_data['ani_spikes']
    
    novelty_list = np.zeros(len(spike_train))
    novelty_spikes = []
    
    xedges = np.arange(np.min(center_x),np.max(center_x),np.ptp(center_x)/np.float(adv['grid_res']-1))
    yedges = np.arange(np.min(center_y),np.max(center_y),np.ptp(center_y)/np.float(adv['grid_res']-1))
    
    x_bins = np.digitize(center_x,bins=xedges,right=True)
    y_bins = np.digitize(center_y,bins=yedges,right=True)
    
    print np.max(x_bins)
    print np.min(x_bins)

    for i in range(len(spike_train)):

        occ_map[x_bins[i],y_bins[i]] += 1
        novelty_list[i] = occ_map[x_bins[i],y_bins[i]]
        
    cutoff = np.sort(novelty_list)[int(len(novelty_list)*.90)]
    novelty_list = np.clip(novelty_list,0,cutoff)
            
    for i in range(len(spike_train)):
        for j in range(spike_train[i]):
            novelty_spikes.append(novelty_list[i])
        
    novelty_counts = np.histogram(novelty_list,bins=20,range=(0,np.max(novelty_list)))[0]
    novelty_spikes = np.histogram(novelty_spikes,bins=20,range=(0,np.max(novelty_list)))[0] 
    firing_rates = novelty_spikes.astype(np.float) * 30./novelty_counts.astype(np.float)
    
    cluster_data['novelty_rates'] = firing_rates
    cluster_data['novelty_vals'] = np.arange(np.min(novelty_list),np.max(novelty_list),np.ptp(novelty_list)/20.)/30.
    trial_data['novelties'] = novelty_list
    
    
    #note that novelty data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_novelty'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data
            

def plot_hd(ops,adv,trial_data,cluster_data,spike_data,self,test=False):
    """makes a cartesian head direction plot"""
    
    #grab appropriate tracking and spike data
    angles = np.asarray(trial_data['angles'])
    spike_angles = np.asarray(spike_data['spike_angles'])
    
    #if multi-cam and camera 1 (tilted) tilt it!
    if ops['multi_cam'] and trial_data['cam_id'] == 1: # and cluster_data['cam'] == 0:
        for angle in range(len(angles)):
            angles[angle] += 45
            if angles[angle] >= 360:
                angles[angle] -= 360
        for spike_angle in range(len(spike_angles)):
            spike_angles[spike_angle] += 45
            if spike_angles[spike_angle] >= 360:
                spike_angles[spike_angle] -= 360
#    elif ops['multi_cam'] and trial_data['cam_id'] == 1 and cluster_data['cam'] != 0:
#        for angle in range(len(angles)):
#            angles[angle] -= 50
#            if angles[angle] < 0:
#                angles[angle] += 360  
#        for spike_angle in range(len(spike_angles)):
#            spike_angles[spike_angle] -= 50
#            if spike_angles[spike_angle] < 0:
#                spike_angles[spike_angle] += 360 
                
    


#    ahvs = np.asarray(trial_data['ahvs'])
#    spike_ahvs = np.asarray(spike_data['spike_ahvs'])
#    angles[np.where(np.logical_or(ahvs<-150,ahvs>150))[0]] = np.nan
#    spike_angles[np.where(np.logical_or(spike_ahvs<-150,spike_ahvs>150))[0]] = np.nan


#    speeds = np.asarray(trial_data['speeds'])
#    spike_speeds = np.asarray(spike_data['spike_speeds'])
#    angles[np.where(speeds<20)[0]] = np.nan
#    spike_angles[np.where(spike_speeds<20)[0]] = np.nan
                

    #create histogram of spikes per direction bin specified by hd_bins parameter
    spike_hd_hist = np.histogram(spike_angles,bins=adv['hd_bins'],range=(0,360))
    #create histogram of frames spent in each bin
    hd_hist = np.histogram(angles,bins=adv['hd_bins'],range=(0,360))
    
    info,sparsity = hd_info(adv,angles,spike_angles)
    print('hd info and sparsity:')
    print(info)
    print(sparsity)
    
    #make results into lists and copy the first element to the end (bc circular data)
    spike_hd_vals = spike_hd_hist[0].tolist()
    spike_hd_vals.append(spike_hd_vals[0])
    hd_vals = hd_hist[0].tolist()
    hd_vals.append(hd_vals[0])
    #transform occupancy times into seconds by dividing by framerate
    for i in range(len(hd_vals)):
        hd_vals[i] = hd_vals[i]*1/adv['framerate']
    #divide spike counts by occupancy times to get firing rates
    rates = []
    for val in range(len(spike_hd_vals)):
        #set unsampled bin rates equal to zero (#TODO: mask these values)
        if hd_vals[val] == 0:
            rates.append(0)
        else:
            rates.append(spike_hd_vals[val]/hd_vals[val])
            
    #grab angles from head direction histogram
    hd_angles = hd_hist[1].tolist()
    
    #fit a curve to the plot, collect fit pfd and firing rates
    pfd, gauss_rates = hd_fit_curve(hd_angles,rates)

    #calculate rayleigh r and rayleigh angle
    r, mean_angle = rayleigh_r(hd_angles,rates)
    
    if test:
        return rates

    #assign new data to cluster/spike data dicts
    cluster_data['hd_angles'] = hd_angles
    cluster_data['hd_rates'] = rates
    cluster_data['gauss_rates'] = gauss_rates
    cluster_data['pfd'] = pfd
    cluster_data['angles'] = angles
    spike_data['spike_angles'] = spike_angles
    cluster_data['rayleigh'] = r
    cluster_data['rayleigh_angle'] = mean_angle
    
    #note that HD data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_hd'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #return data
    return cluster_data,spike_data

def hd_fit_curve(hd_angles,rates):
    """creates a gaussian curve to fit head direction data"""
    
    #calculate initial PFD using highest rate in rates list
    pfd_i = hd_angles[rates.index(max(rates))]
    #calculate initial approximations for each parameter in pseudo-Gaussian
    #curve fitting equation
    baseline_i = min(rates) #baseline firing rate
    peak_i = max(rates) #peak firing rate
    mean = np.mean(rates) #mean firing rate
    K_i = (1/(2*np.pi))*((peak_i-baseline_i)/(mean-baseline_i))**2 #constant defined by this equation
    B_i = peak_i/(np.e**K_i) #constant defined by B*e**K = peak firing rate
    #define model equation as a function    
    def func(x,baseline,peak,K,B,pfd):
        return baseline + B*np.e**(K*np.cos(np.deg2rad(x)-np.deg2rad(pfd)))
    #if the fitting doesn't work we get a runtime error and the whole program stops,
    #so we have to safeguard against that
    #also if it can't estimate the covariance of the parameters it will give us an annoying
    #warning, so we'll ignore that as well
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            #if the fitting works, take those values
            [baseline,peak,K,B,pfd],pcov = curve_fit(func,hd_angles,rates,p0=[baseline_i,peak_i,K_i,B_i,pfd_i])
        except RuntimeError:
            #if it doesn't, just use the initial values - it's probably not an HD cell anyway
            [baseline,K,B,pfd] = [baseline_i,K_i,B_i,pfd_i]
    #calculate new rates for fit curve using new parameters    
    gauss_rates = []
    for i in hd_angles:
        y = baseline + B*np.e**(K*np.cos(np.deg2rad(i)-np.deg2rad(pfd)))
        gauss_rates.append(y)
    #set pfd equal to value with highest rate in fit curve
    pfd = hd_angles[gauss_rates.index(max(gauss_rates))]
    
    return pfd, gauss_rates

def rayleigh_r(spike_angles,rates=None,ego=False):
    """finds rayleigh mean vector length for head direction curve"""
    
    #start vars for x and y rayleigh components
    rx = 0
    ry = 0
    
    #convert spike angles into x and y coordinates, sum up the results -- 
    #if firing rates are provided along with HD plot edges instead of spike angles,
    #do the same thing but with those
    if rates is None:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))
            ry += np.sin(np.deg2rad(spike_angles[i]))
    else:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))*rates[i]
            ry += np.sin(np.deg2rad(spike_angles[i]))*rates[i]

    #calculate average x and y values for vector coordinates
    if rates is None:
        if len(spike_angles) == 0:
            spike_angles.append(1)
        rx = rx/len(spike_angles)
        ry = ry/len(spike_angles)
    
    else:
        rx = rx/sum(rates)
        ry = ry/sum(rates)

    #calculate vector length
    r = np.sqrt(rx**2 + ry**2)
    
    #calculate the angle the vector points (rayleigh pfd)
    #piecewise because of trig limitations
    if rx == 0:
#        rx = 1
        mean_angle = 0
    elif rx > 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx))
    elif rx < 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
    try:
        if mean_angle < 0:
            mean_angle = mean_angle + 360
    except:
        mean_angle = 0
        
    if ego:
        return r,rx,ry, mean_angle
    else:
        return r, mean_angle
    
def plot_half_hds(ops,adv,trial_data,cluster_data,spike_data,self):
    """creates head direction plots for each half of session"""
    
    #grab appropriate tracking and spike data
    angles = trial_data['angles']
    spike_train = spike_data['ani_spikes']
    
    midpoint = int(len(angles)/2)
    
    hd_angles = []
    rates = []
    
    for pair in [[0,midpoint],[midpoint,len(angles)]]:
        
        spikes = np.zeros(adv['hd_bins'])
        occ = np.zeros(adv['hd_bins'])
        
        half_angles = angles[pair[0]:pair[1]]
        half_spikes = spike_train[pair[0]:pair[1]]
        
        ref_angles = np.linspace(0,360,adv['hd_bins'],endpoint=False)
        
        angle_bins = np.digitize(half_angles,ref_angles)-1
        
        for i in range(len(half_angles)):
            spikes[angle_bins[i]] += half_spikes[i]
            occ[angle_bins[i]] += 1./adv['framerate']
            
        curve = spikes/occ
        
        curve = np.append(curve,curve[0])
        ref_angles = np.append(ref_angles,360)
        
        rates.append(list(curve))
        hd_angles.append(list(ref_angles))
        
    
    #send new data to cluster data dict
    cluster_data['half_hd_angles'] = hd_angles
    cluster_data['half_hd_rates'] = rates
    
    #note that half HD data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_half_hds'] = True
    #send updated data to gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #return data
    return cluster_data


def plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots each spike colored by head direction"""
        
    #note that HD map data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_hd_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #return data
    return cluster_data
    

def spatial_autocorrelation(ops,adv,trial_data,cluster_data,spike_data,self):
    """makes a spatial autocorrelation plot for grid cells"""
    
    #grab the smoothed heatmap and number of bins (grid_res parameter)
    if not self.spike_timed:
        smoothed_heatmap = cluster_data['smoothed_heatmap']
    else:
        smoothed_heatmap = cluster_data['st_smoothed_heatmap']
    gr = adv['grid_res']
    #make a matrix of zeros 2x the length and width of the smoothed heatmap (in bins)
    corr_matrix = np.zeros((2*gr,2*gr))
    
    #for every possible overlap between the smoothed heatmap and its copy, 
    #correlate those overlapping bins and assign them to the corresponding index
    #in the corr_matrix
    for i in range(-len(smoothed_heatmap),len(smoothed_heatmap)):
        for j in range(-len(smoothed_heatmap[0]),len(smoothed_heatmap)):
            if i < 0:
                if j < 0:
                    array1 = smoothed_heatmap[(-i):(gr),(-j):(gr)]
                    array2 = smoothed_heatmap[0:(gr+i),0:(gr+j)]
                elif j >= 0:
                    array1 = smoothed_heatmap[(-i):(gr),0:(gr-j)]
                    array2 = smoothed_heatmap[0:(gr+i),(j):gr]
            elif i >= 0:
                if j < 0:
                    array1 = smoothed_heatmap[0:(gr-i),(-j):(gr)]
                    array2 = smoothed_heatmap[(i):gr,0:(gr+j)]
                elif j >= 0:
                    array1 = smoothed_heatmap[0:(gr-i),0:(gr-j)]
                    array2 = smoothed_heatmap[(i):gr,(j):gr]
            
            #this will give us annoying warnings for issues that don't matter --
            #we'll just ignore them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #get the pearson r for the overlapping arrays
                corr,p = pearsonr(np.ndarray.flatten(array1),np.ndarray.flatten(array2))
                #assign the value to the appropriate spot in the autocorr matrix
                corr_matrix[gr+i][gr+j] = corr

    #if we're not making a spike-triggered plot...
    if not self.spike_timed:
        #return the spatial autocorrelation matrix for later use
        cluster_data['spatial_autocorr'] = corr_matrix
        #note that spatial autocorrelation data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spatial_autocorr'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    else:
        #return spike-triggered autocorrelation
        cluster_data['st_spatial_autocorr'] = corr_matrix

    return cluster_data
    
def grid_score(ops,adv,trial_data,cluster_data,spike_data,self):
    """computes gridness score for the cluster"""
    
    #grab the autocorrelation matrix and grid resolution
    if not self.spike_timed:
        corr_matrix = cluster_data['spatial_autocorr']
    else:
        corr_matrix = cluster_data['st_spatial_autocorr']
        
    #assign grid resolution to shorter variable for easier readability
    gr = adv['grid_res']
    
    #make x and y grids size of corr_matrix for defining rings
    x,y = np.meshgrid(range(-gr,gr),range(-gr,gr))
    #set default indices for donut rings in case NOT a grid cell
    outer_radius = int(gr*.9)
    inner_radius = int(gr*.3)

    #start counters for keeping track of negative and positive average correlation values
    neg_count = 0
    pos_count = 0
    #for just about every possible radius from the center of the corr_matrix
    for i in range(1,gr-2):
        #if the average correlation for the ring specified between radius i and i+2 is negative...
        #(always starts out positive)
        if sum(corr_matrix[(x**2 + y**2 > i**2) & (x**2 + y**2 < (i+2)**2)])/len(corr_matrix[(x**2 + y**2 > i**2) & (x**2 + y**2 < (i+2)**2)]) < 0:
            #if it's the first time, increment neg_count to 1
            if neg_count == 0 and pos_count == 0:
                neg_count += 1
            #if it's not the first time and we've already assigned an inner ring,
            #assign this as the radius of the outer ring and break the loop 
            if neg_count > 0 and pos_count > 0 and sum(corr_matrix[(x**2 + y**2 > (i-1)**2) & (x**2 + y**2 < (i+1)**2)])/len(corr_matrix[(x**2 + y**2 > (i-1)**2) & (x**2 + y**2 < (i+1)**2)]) > 0:
                outer_radius = i
                break
        #if the average correlation is positive and we've already encountered a negative
        #average correlation, set the inner radius equal to i
        if sum(corr_matrix[(x**2 + y**2 > i**2) & (x**2 + y**2 < (i+2)**2)])/len(corr_matrix[(x**2 + y**2 > i**2) & (x**2 + y**2 < (i+2)**2)]) > 0 and sum(corr_matrix[(x**2 + y**2 > (i-1)**2) & (x**2 + y**2 < (i+1)**2)])/len(corr_matrix[(x**2 + y**2 > (i-1)**2) & (x**2 + y**2 < (i+1)**2)]) < 0 and pos_count == 0:
            inner_radius = i
            pos_count += 1
            
    #if we failed to find appropriate radii (probably not a grid cell then!) set the
    #outer radius equal to the edge of the plot
    if outer_radius < inner_radius:
        outer_radius = gr - 1
            
    #copy the corr matrix to make the donut
    donut = copy.deepcopy(corr_matrix)
    #mask every bin inside the inner radius and outside the outer radius
    donut[(x**2 + y**2<= inner_radius**2)|(x**2 + y**2 >= outer_radius**2)] = np.nan
    donut = np.ma.masked_invalid(donut)

    #make lists for rotation angles and correlation values
    rot_angles = range(0,181,3)
    rot_values = []
    
    #from 0 to 180 in 3 degree increments, rotate the donut (becomes donut2)
    #and correlate it with its unrotated self
    for i in range(0,181,3):
        donut2 = copy.deepcopy(corr_matrix)
        donut2 = rotate(donut2,i,cval=np.nan,preserve_range=True)
        donut2[(x**2 + y**2<= inner_radius**2)|(x**2 + y**2 >= outer_radius**2)] = np.nan
        donut2 = np.ma.masked_invalid(donut2)
        corr,p = mapearsonr(donut2.flatten(),donut.flatten())
        rot_values.append(corr)
        
    #grab the supposed peaks and valleys of the gridness plot
    peaks = [rot_values[20],rot_values[40]]
    valleys = [rot_values[10],rot_values[30],rot_values[50]]
    #set gridness score equal to lowest peak minus highest valley
    gridness = min(peaks) - max(valleys)

    #return appropriate data for later use
    if not self.spike_timed:
        cluster_data['rot_angles'] = rot_angles
        cluster_data['rot_values'] = rot_values
        cluster_data['gridness'] = gridness
        #note that gridness data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_grid_score'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    else:
        cluster_data['st_rot_angles'] = rot_angles
        cluster_data['st_rot_values'] = rot_values
        cluster_data['st_gridness'] = gridness

    return cluster_data

#def kalman_xy(x, P, measurement, R,
#              motion = np.matrix('0. 0. 0. 0.').T,
#              Q = np.matrix(np.eye(4))):
#    """
#    Parameters:    
#    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
#    P: initial uncertainty convariance matrix
#    measurement: observed position
#    R: measurement noise 
#    motion: external motion added to state vector x
#    Q: motion noise (same shape as P)
#    """
#    return kalman(x, P, measurement, R, motion, Q,
#                  F = np.matrix('''
#                      1. 0. 1. 0.;
#                      0. 1. 0. 1.;
#                      0. 0. 1. 0.;
#                      0. 0. 0. 1.
#                      '''),
#                  H = np.matrix('''
#                      1. 0. 0. 0.;
#                      0. 1. 0. 0.'''))
#
#def kalman(x, P, measurement, R, motion, Q, F, H):
#    '''
#    Parameters:
#    x: initial state
#    P: initial uncertainty convariance matrix
#    measurement: observed position (same shape as H*x)
#    R: measurement noise (same shape as H)
#    motion: external motion added to state vector x
#    Q: motion noise (same shape as P)
#    F: next state function: x_prime = F*x
#    H: measurement function: position = H*x
#
#    Return: the updated and predicted new values for (x, P)
#
#    See also http://en.wikipedia.org/wiki/Kalman_filter
#
#    This version of kalman can be applied to many different situations by
#    appropriately defining F and H 
#    '''
#    # UPDATE x, P based on measurement m    
#    # distance between measured and current position-belief
#    y = np.matrix(measurement).T - H * x
#    S = H * P * H.T + R  # residual convariance
#    K = P * H.T * S.I    # Kalman gain
#    x = x + K*y
#    I = np.matrix(np.eye(F.shape[0])) # identity matrix
#    P = (I - K*H)*P
#
#    # PREDICT x, P based on motion
#    x = F*x + motion
#    P = F*P*F.T + Q
#
#    return x, P
#
#def demo_kalman_xy(trial_data):
#    x = np.matrix('0. 0. 0. 0.').T 
#    P = np.matrix(np.eye(4))*1000 # initial uncertainty
#    
#    import matplotlib.pyplot as plt
#
##    N = 20
##    true_x = np.linspace(0.0, 10.0, N)
##    true_y = true_x**2
#    observed_x = np.array(trial_data['center_x'])
#    observed_y = np.array(trial_data['center_y'])
##    plt.plot(observed_x, observed_y, 'ro')
#    result = []
#    R = 0.01**2
#    for meas in zip(observed_x, observed_y):
#        x, P = kalman_xy(x, P, meas, R)
#        result.append((x[:2]).tolist())
#    kalman_x, kalman_y = zip(*result)
#    plt.figure
#    plt.plot(kalman_x)
#    plt.plot(observed_x)
#    plt.show()
#
#demo_kalman_xy()

def calc_movement_direction(adv,trial_data):
    
    #grab appropriate tracking data
    center_x=trial_data['center_x']
    center_y=trial_data['center_y']
    
    from astropy.convolution import convolve
    from astropy.convolution.kernels import Gaussian1DKernel, Box1DKernel
    
    smooth_x = convolve(center_x,kernel=Gaussian1DKernel(stddev=5))
    smooth_y = convolve(center_y,kernel=Gaussian1DKernel(stddev=5))
    
    dx = smooth_x - np.roll(smooth_x,1)
    dy = smooth_y - np.roll(smooth_y,1)
    
    mds = np.rad2deg(np.arctan2(dy,dx))%360
    
    #make an array of zeros to assign speeds to
#    mds = np.zeros(len(center_x),dtype=np.float)
#    xvals = np.zeros(len(center_x),dtype=np.float)
#    yvals = np.zeros(len(center_x),dtype=np.float)
#    #for every frame from 2 to total - 2...
#    for i in range(2,len(center_x)-2):
#        #grab 5 x and y positions centered on that frame
#        x_list = center_x[(i-2):(i+3)]
#        y_list = center_y[(i-2):(i+3)]
#        #find the best fit line for those 5 points (slopes are x and y components
#        #of velocity)
#        xfitline = np.polyfit(range(0,5),x_list,1)
#        yfitline = np.polyfit(range(0,5),y_list,1)
#        #total velocity = framerate * sqrt(x component squared plus y component squared)
#        xvals[i] = xfitline[0]
#        yvals[i] = yfitline[0]
#        
#    from astropy.convolution.kernels import Gaussian1DKernel
#    xvals = convolve(xvals,kernel=Gaussian1DKernel(stddev=2))
#    yvals = convolve(yvals,kernel=Gaussian1DKernel(stddev=2))
#    mds = np.rad2deg(np.arctan2(yvals,xvals))%360
##        adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
#    #set unassigned speeds equal to closest assigned speed
#    mds[0] = mds[2]
#    mds[1] = mds[2]
#    mds[len(mds)-1] = mds[len(mds)-3]
#    mds[len(mds)-2] = mds[len(mds)-3]
    
    #return calculated speeds
    trial_data['movement_directions'] = mds
    return trial_data

def calc_speed(adv,trial_data):
    """calculates 'instantaneous' linear speeds for each video frame"""
    
    #grab appropriate tracking data
#    center_x=trial_data['center_x']
#    center_y=trial_data['center_y']
    
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    
    from astropy.convolution import convolve
    from astropy.convolution.kernels import Gaussian1DKernel, Box1DKernel
    
    smooth_x = convolve(center_x,kernel=Gaussian1DKernel(stddev=5))
    smooth_y = convolve(center_y,kernel=Gaussian1DKernel(stddev=5))
    
    dx = smooth_x - np.roll(smooth_x,1)
    dy = smooth_y - np.roll(smooth_y,1)
    
    speeds = np.sqrt(dx**2 + dy**2)*adv['framerate']
    speeds[speeds>100] = 100.
    
#    plt.plot(speeds)

    
#    #make an array of zeros to assign speeds to
#    speeds = np.zeros(len(center_x),dtype=np.float)
#    #for every frame from 2 to total - 2...
#    for i in range(2,len(center_x)-2):
#        #grab 5 x and y positions centered on that frame
#        x_list = center_x[(i-2):(i+3)]
#        y_list = center_y[(i-2):(i+3)]
#        #find the best fit line for those 5 points (slopes are x and y components
#        #of velocity)
#        xfitline = np.polyfit(range(0,5),x_list,1)
#        yfitline = np.polyfit(range(0,5),y_list,1)
#        #total velocity = framerate * sqrt(x component squared plus y component squared)
#        speeds[i] = adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
#    #set unassigned speeds equal to closest assigned speed
#    speeds[0] = speeds[2]
#    speeds[1] = speeds[2]
#    speeds[len(speeds)-1] = speeds[len(speeds)-3]
#    speeds[len(speeds)-2] = speeds[len(speeds)-3]
    
    #return calculated speeds
    trial_data['speeds'] = speeds
    return trial_data
    
def plot_speed(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots firing rate vs speed"""
    
    #grab our calculated speed data
    speeds = trial_data['speeds']
    
    #we'll limit ourselves to maximum 150 pixels/sec with 2 pixels/sec bins, so 75 bins
    max_bin = 150
    bin_size = 2
    speed_bins = int(max_bin/bin_size)
    
    #make a histogram of speeds that spikes happened at
    spike_speed_hist = np.histogram(spike_data['spike_speeds'],bins=speed_bins,range=(0,max_bin))
    #create histogram of frames spent in each bin
    speed_hist = np.histogram(speeds,bins=speed_bins,range=(0,max_bin))
    
    #make results into arrays
    spike_speed_vals = np.asarray(spike_speed_hist[0].tolist(),dtype=np.float)
    speed_vals = np.asarray(speed_hist[0].tolist(),dtype=np.float)
    #mask speeds that were sampled for less than a second
    speed_vals = np.ma.masked_less(speed_vals,adv['framerate'])
    spike_speed_vals = np.ma.masked_array(spike_speed_vals,speed_vals.mask)
    spike_speed_vals.unshare_mask()
    
    #transform occupancy times into seconds by dividing by framerate
    for i in range(len(speed_vals)):
        speed_vals[i] = speed_vals[i]*1/adv['framerate']
    #create a masked array to assign firing rates to
    speed_rates = np.ma.masked_array(np.zeros(np.shape(speed_vals)),speed_vals.mask,dtype=np.float)
    speed_rates.unshare_mask()
    #divide spike counts by occupancy times to get firing rates
    for val in range(len(spike_speed_vals)):
        speed_rates[val] = (spike_speed_vals[val]/speed_vals[val])
    #grab our edges for plotting speed rates, mask appropriately
    speed_edges = speed_hist[1].tolist()[:len(speed_hist[1].tolist())-1]
    speed_edges = np.ma.masked_array(speed_edges,speed_vals.mask,dtype=np.float)
    speed_edges.unshare_mask()
    
    #plot a linear regression and grab the y values for plotting, along with r and p values
    slope, intercept, r_value,p_value,std_err = linregress(speed_edges,speed_rates)
    fit_y = []
    for i in range(len(speed_edges)):
        fit_y.append(slope*speed_edges[i] + intercept)

    #return data for plotting later
    cluster_data['speed_edges'] = speed_edges
    cluster_data['speed_rates'] = speed_rates
    cluster_data['max_speed'] = max_bin
    cluster_data['speed_fit_y'] = fit_y
    cluster_data['speed_r'] = r_value
    cluster_data['speed_p'] = p_value
    
    #note that speed data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_speed'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data

def calc_center_ego_ahv(adv,trial_data):
    """calculates 'instantaneous' angular head velocity for each video frame"""
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2

    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    angles = (center_ego_angles-angles)%360
    
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
    trial_data['center_ego_ahvs'] = ahvs
    return trial_data

def plot_center_ego_ahv(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots firing rate vs AHV"""
    
    #grab our calculated ahvs
    ahvs = trial_data['center_ego_ahvs']
    
    spike_ahvs = []
    for i in range(len(spike_data['ani_spikes'])):
        for j in range(spike_data['ani_spikes'][i]):
            spike_ahvs.append(ahvs[i])
    spike_ahvs = np.array(spike_ahvs)
    
    #we'll limit ourselves to maximum 402 deg/sec with bin size 6 deg/sec
    min_bin = -402
    max_bin = 402
    bin_size = 6
    ahv_bins = (max_bin-min_bin)/bin_size
    
    #create histogram of AHVs spikes occurred at
    spike_ahv_hist = np.histogram(spike_ahvs,bins=ahv_bins,range=(min_bin,max_bin))
    #create histogram of frames spent in each 6 degree bin
    ahv_hist = np.histogram(ahvs,bins=ahv_bins,range=(min_bin,max_bin))
    #make results into arrays
    spike_ahv_vals = np.asarray(spike_ahv_hist[0].tolist(),dtype=np.float)
    ahv_vals = np.asarray(ahv_hist[0].tolist(),dtype=np.float)
    #mask any AHVs sampled for less than a second
    ahv_vals = np.ma.masked_less(ahv_vals,adv['framerate'])
    spike_ahv_vals = np.ma.masked_array(spike_ahv_vals,ahv_vals.mask)
    spike_ahv_vals.unshare_mask()

    #transform occupancy times into seconds by dividing by framerate
    for i in range(len(ahv_vals)):
        ahv_vals[i] = ahv_vals[i]*1/adv['framerate']
    #create a masked array for assigning firing rates
    ahv_rates = np.ma.masked_array(np.zeros(np.shape(ahv_vals)),ahv_vals.mask,dtype=np.float)
    ahv_rates.unshare_mask()

    #divide spike counts by occupancy times to get firing rates
    for val in range(len(spike_ahv_vals)):
        ahv_rates[val]=spike_ahv_vals[val]/ahv_vals[val]
    #grab edges for plotting
    ahv_angles = ahv_hist[1].tolist()[:len(ahv_hist[1].tolist())-1]
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ahv_angles,ahv_rates)
    ax.set_xlim([-400,400])
    ax.set_ylim([0,1.2*np.max(ahv_rates)+10])
    egopath = cluster_data['new_folder'] + '/egocentric'
    import os
    if not os.path.isdir(egopath):
        os.makedirs(egopath)
    fig.savefig(egopath + '/center_ahv',dpi=600)

#    #return data for plotting later
#    cluster_data['center_ego_ahv_angles'] = ahv_angles
#    cluster_data['center_ego_ahv_rates'] = ahv_rates
#    cluster_data['center_ego_max_ahv'] = max_bin
#    
#    #note that AHV data is ready
#    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_ahv'] = True
#    #send updated data to the gui
#    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
#        
#    return cluster_data
    
def calc_ahv(adv,trial_data):
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
    
def plot_ahv(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots firing rate vs AHV"""
    
    #grab our calculated ahvs
    ahvs = trial_data['ahvs']
    
    #we'll limit ourselves to maximum 402 deg/sec with bin size 6 deg/sec
    min_bin = -402
    max_bin = 402
    bin_size = 6
    ahv_bins = (max_bin-min_bin)/bin_size
    
    #create histogram of AHVs spikes occurred at
    spike_ahv_hist = np.histogram(spike_data['spike_ahvs'],bins=ahv_bins,range=(min_bin,max_bin))
    #create histogram of frames spent in each 6 degree bin
    ahv_hist = np.histogram(ahvs,bins=ahv_bins,range=(min_bin,max_bin))
    #make results into arrays
    spike_ahv_vals = np.asarray(spike_ahv_hist[0].tolist(),dtype=np.float)
    ahv_vals = np.asarray(ahv_hist[0].tolist(),dtype=np.float)
    #mask any AHVs sampled for less than a second
    ahv_vals = np.ma.masked_less(ahv_vals,adv['framerate'])
    spike_ahv_vals = np.ma.masked_array(spike_ahv_vals,ahv_vals.mask)
    spike_ahv_vals.unshare_mask()

    #transform occupancy times into seconds by dividing by framerate
    for i in range(len(ahv_vals)):
        ahv_vals[i] = ahv_vals[i]*1/adv['framerate']
    #create a masked array for assigning firing rates
    ahv_rates = np.ma.masked_array(np.zeros(np.shape(ahv_vals)),ahv_vals.mask,dtype=np.float)
    ahv_rates.unshare_mask()

    #divide spike counts by occupancy times to get firing rates
    for val in range(len(spike_ahv_vals)):
        ahv_rates[val]=spike_ahv_vals[val]/ahv_vals[val]
    #grab edges for plotting
    ahv_angles = ahv_hist[1].tolist()[:len(ahv_hist[1].tolist())-1]

    #return data for plotting later
    cluster_data['ahv_angles'] = ahv_angles
    cluster_data['ahv_rates'] = ahv_rates
    cluster_data['max_ahv'] = max_bin
    
    #note that AHV data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_ahv'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
    return cluster_data
    
def spike_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    """calculates and plots ISI histogram and temporal autocorrelation"""
    
    #grab appropriate spike data
    isi_list = spike_data['isi_list']
    spike_train = spike_data['spike_train']
    
    #convert ISI times from microseconds to seconds
    for i in range(len(isi_list)):
        isi_list[i] = float(isi_list[i])/1000000.
    #remove ISIs longer than 1 second from the list    
    def remove_values_from_list(the_list, val):
        return [value for value in the_list if value < val]
    isi_list = remove_values_from_list(isi_list, 1)
    
    #make a histogram of the isi's
    isi_hist = np.histogram(isi_list,bins=1000,range=[0,1])
    
    isi_xvals = np.arange(0,1,1./1000.)
        
    #add the ISI hist to cluster data dict
    cluster_data['isi_hist'] = isi_hist
    cluster_data['isi_xvals'] = isi_xvals
    
    #note that ISI data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_isi'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #figure out how many bins we want per window in our spike autocorrelation
    #default comes out to 500
    bins_per_window = int(adv['autocorr_width']*1000/adv['bin_size'])
    
    #grab where the spikes are in our spike train
    spike_inds = np.where(spike_train == 1)
    
    #start a vector for holding our autocorrelation values
    ac_matrix = np.zeros(2*bins_per_window + 1)
    
    #for each spike...
    for i in spike_inds[0]:
        #if we're at least one half window width from the beginning and end of the session...
        if (i-bins_per_window)>=0 and (i+bins_per_window)<len(spike_train):
            #for each bin in the window
            for j in range(-bins_per_window,bins_per_window+1):
                #add the corresponding value from the spike train
                ac_matrix[j+bins_per_window] += spike_train[i+j]
                
    ac_matrix[bins_per_window] = 0
                
    #choose our x-ticks
    x_vals=np.arange(-adv['autocorr_width'],adv['autocorr_width']+float(adv['bin_size'])/1000.,float(adv['bin_size'])/1000.)

    #return data for plotting
    cluster_data['ac_xvals'] = copy.deepcopy(x_vals)
    cluster_data['ac_vals'] = copy.deepcopy(ac_matrix)
    
    ''' bonus! '''
    
    timestamps = trial_data['timestamps']
    bin_size = .2
    spike_timestamps = []
    for i in range(len(timestamps)):
        if i < len(timestamps)-1:
            increment = (timestamps[i+1]-timestamps[i])/(1000./(adv['framerate']*bin_size))
            for j in range(int(1000./((adv['framerate']*bin_size)))):
                spike_timestamps.append(timestamps[i]+j*increment)
                
    #creates array of zeros length of spike_timestamps to create spike train
    st = np.zeros(len(spike_timestamps))

    #for each spike timestamp...
    for i in cluster_data['spike_list']:
        #find closest entry in high precision 'spike timestamps' list
        spike_ind = bisect.bisect_left(spike_timestamps,i)
        if spike_ind < len(st):
            #add 1 to spike train at appropriate spot
            st[spike_ind] = 1

    #figure out how many bins we want per window in our spike autocorrelation
    #default comes out to 500
    bins_per_window = 50
    
    #grab where the spikes are in our spike train
    spike_inds = np.where(st == 1)
    
    #start a vector for holding our autocorrelation values
    ac_matrix = np.zeros(2*bins_per_window + 1)
    
    #for each spike...
    for i in spike_inds[0]:
        #if we're at least one half window width from the beginning and end of the session...
        if (i-bins_per_window)>=0 and (i+bins_per_window)<len(st):
            #for each bin in the window
            for j in range(-bins_per_window,bins_per_window+1):
                #add the corresponding value from the spike train
                ac_matrix[j+bins_per_window] += st[i+j]
                
    ac_matrix[bins_per_window] = 0
                
    #choose our x-ticks
    x_vals=np.arange(-.1,.1+float(.2)/100.,float(.2)/100.)

    cluster_data['small_ac_xvals'] = copy.deepcopy(x_vals)
    cluster_data['small_ac_vals'] = copy.deepcopy(ac_matrix)
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_spike_autocorr'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data


def spatial_info(adv,center_x,center_y,spike_x,spike_y):
    
    gr = adv['grid_res']
    
    #calculate 2D histogram of spatial occupancies for the rat's path, break 
    #arena into bins assigned by grid_res parameter
    occ_hist,xedges,yedges = np.histogram2d(center_x,center_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    #calculate 2D histogram of spikes per bin
    spike_hist,spikexedges,spikeyedges = np.histogram2d(spike_x,spike_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]]) 

    #make sure we're using floats bc integer counts won't work
    occ_hist = np.asarray(occ_hist,dtype=np.float)
    spike_hist = np.asarray(spike_hist,dtype=np.float)

    #remove bins where animal never sampled
    occ_hist[occ_hist==0] = np.nan
    spike_hist[np.isnan(occ_hist)] = np.nan
    
    #calculate a matrix of firing rates
    fr_mat = spike_hist / occ_hist
    #calculate the mean firing rate
    mean_fr = np.nansum(spike_hist) / np.nansum(occ_hist)
    
    #total occupancy time
    tot_occ = np.nansum(occ_hist)
    #calc a matrix of occupancy probabilities
    prob_mat = occ_hist / tot_occ

    #calculate the information content
    info = np.nansum( prob_mat * (fr_mat / mean_fr) * np.log2(fr_mat / mean_fr))
    
    #calculate sparsity of the signal
    sparsity = ( np.nansum(prob_mat * fr_mat)**2 ) / np.nansum(prob_mat * fr_mat**2)

    return info, sparsity

def hd_info(adv,angles,spike_angles):
    
    hd_bins = adv['hd_bins']
    
    angles = np.asarray(angles,dtype=np.float)
    spike_angles = np.asarray(spike_angles,dtype=np.float)
    
    #create histogram of spikes per direction bin specified by hd_bins parameter
    spike_hist, _ = np.histogram(spike_angles,bins=hd_bins,range=(0,360))
    #create histogram of frames spent in each bin
    occ_hist, _ = np.histogram(angles,bins=hd_bins,range=(0,360))
        
    #make sure we're using floats because integer counts won't work
    occ_hist = np.asarray(occ_hist,dtype=np.float)
    spike_hist = np.asarray(spike_hist,dtype=np.float)

    #take away bins where the rat never sampled
    occ_hist[occ_hist==0] = np.nan
    spike_hist[np.isnan(occ_hist)] = np.nan
    
    #calculate a matrix of firing rates
    fr_mat = spike_hist / occ_hist
    #calculate the mean firing rate
    mean_fr = np.nansum(spike_hist) / np.nansum(occ_hist)
    
    #total occupancy time
    tot_occ = np.nansum(occ_hist)
    #calc a matrix of occupancy probabilities
    prob_mat = occ_hist / tot_occ

    #calculate the information content
    info = np.nansum( prob_mat * (fr_mat / mean_fr) * np.log2(fr_mat / mean_fr))
    
    #calculate sparsity of the signal
    sparsity = ( np.nansum(prob_mat * fr_mat)**2 ) / np.nansum(prob_mat * fr_mat**2)

    return info, sparsity

def plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    #assign grid_res and hd_bins to shorter var for easier reading
    gr = adv['grid_res']
    hd_bins = adv['hd_bins']
    framerate = adv['framerate']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2

    radial_dists = np.sqrt(center_x**2 + center_y**2)
    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_ego_angles = (center_ego_angles-angles)%360
#    center_ego_mds = (center_ego_angles - np.asarray(trial_data['movement_directions']))
    
    trial_data['center_ego_angles'] = center_ego_angles

    ego_bins = np.digitize(center_ego_angles,np.linspace(0,360,hd_bins,endpoint=False))-1
    
    ego_spikes = np.zeros(hd_bins)
    ego_occ = np.zeros(hd_bins)
    for i in range(len(center_x)):
        ego_spikes[ego_bins[i]] += ani_spikes[i]
        ego_occ[ego_bins[i]] += 1./framerate
    ego_curve = ego_spikes/ego_occ
    
    rayleigh, mean_angle = rayleigh_r(np.arange(6,366,360/hd_bins),ego_curve)
    
    ego_angles = np.linspace(0+np.deg2rad(6),2.*np.pi+np.deg2rad(6),hd_bins)
    mean_guess = np.mean(ego_curve)
    k_guess = np.max(ego_curve) - mean_guess
    phase_guess = -mean_angle

    obj = lambda x: x[0] + x[1] * np.cos(ego_angles + x[2]) - ego_curve
    mean, k, phase = leastsq(obj, [mean_guess,k_guess,phase_guess])[0]
    fit_curve = mean + k * np.cos(ego_angles + phase)
    r,p = pearsonr(fit_curve,ego_curve)
    
    cluster_data['center_ego_r'] = r
    cluster_data['center_ego_p'] = p
    cluster_data['center_ego_fit'] = fit_curve
    cluster_data['center_ego_curve'] = ego_curve
    cluster_data['center_ego_rayleigh'] = rayleigh
    cluster_data['center_ego_mean_angle'] = mean_angle
    cluster_data['ego_occ'] = ego_occ
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_center_ego'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    
    radial_bins = np.digitize(radial_dists,np.linspace(0,np.max(radial_dists),20,endpoint=False))-1
    
    radial_spikes = np.zeros(20)
    radial_occ = np.zeros(20)
    for i in range(len(center_x)):
        radial_spikes[radial_bins[i]] += ani_spikes[i]
        radial_occ[radial_bins[i]] += 1./framerate
    radial_curve = radial_spikes/radial_occ
    
    xvals = np.linspace(np.min(radial_dists),np.max(radial_dists),20)
    
    slope, intercept, r_value,p_value,std_err = linregress(xvals,radial_curve)
    fit_y = []
    for i in range(len(xvals)):
        fit_y.append(slope*xvals[i] + intercept)
    
    cluster_data['center_dist_curve'] = radial_curve
    cluster_data['center_dist_xvals'] = xvals
    cluster_data['center_dist_fit'] = fit_y
    cluster_data['center_dist_r'] = r_value
    cluster_data['center_dist_p'] = p_value
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_center_dist'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    cluster_data['center_ego_angles'] = center_ego_angles
    cluster_data['center_ego_dists'] = radial_dists
    center_ego_spike_angles = []
    center_ego_spike_dists = []
    for i in range(len(center_x)):
        for j in range(int(ani_spikes[i])):
            center_ego_spike_angles.append(center_ego_angles[i])
            center_ego_spike_dists.append(radial_dists[i])
            
    spike_data['center_ego_angles'] = np.array(center_ego_spike_angles)
    spike_data['center_ego_dists'] = np.array(center_ego_spike_dists)
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_center_ego_hd_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_center_ego_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
#    plot_boundary_field(ops,adv,trial_data,cluster_data,spike_data,self)
    plot_mh_ego(ops,adv,trial_data,cluster_data,spike_data,self)
    
    slopes = center_y/center_x
    wy1 = slopes * np.min(center_x)
    wx1 = np.min(center_x)
    w1_dists = np.sqrt((wy1 - center_y)**2 + (wx1 - center_x)**2)
    
    wy2 = slopes * np.max(center_x)
    wx2 = np.max(center_x)
    w2_dists = np.sqrt((wy2 - center_y)**2 + (wx2 - center_x)**2)
    
    wx3 = np.min(center_y) / slopes
    wy3 = np.min(center_y)
    w3_dists = np.sqrt((wy3 - center_y)**2 + (wx3 - center_x)**2)
    
    wx4 = np.max(center_y) / slopes
    wy4 = np.max(center_y)
    w4_dists = np.sqrt((wy4 - center_y)**2 + (wx4 - center_x)**2)
    
    all_dists = np.stack([w1_dists,w2_dists,w3_dists,w4_dists])
    wall_dists = np.min(all_dists,axis=0)
    
    dist_bins = np.digitize(wall_dists,np.linspace(0,np.max(wall_dists),20,endpoint=False))-1
    
    spikes = np.zeros(20)
    occ = np.zeros(20)
    for i in range(len(center_x)):
        spikes[dist_bins[i]] += ani_spikes[i]
        occ[dist_bins[i]] += 1./framerate
    curve = spikes/occ
    
    xvals = np.linspace(np.min(wall_dists),np.max(wall_dists),20)
    
    slope, intercept, r_value,p_value,std_err = linregress(xvals,curve)
    fit_y = []
    for i in range(len(xvals)):
        fit_y.append(slope*xvals[i] + intercept)
    
    egopath = cluster_data['new_folder']+'/egocentric'
    import os
    import matplotlib.pyplot as plt
    if not os.path.isdir(egopath):
        os.makedirs(egopath)
    fig=plt.figure()
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(curve))+10
    xmax = np.nanmax(xvals)
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.plot(xvals,curve,'k-')
    ax.plot(xvals,fit_y,'b-')
    ax.text(.1,.9,'fit r^2 = %s' % r_value**2,transform=ax.transAxes)
    ax.text(.1,.8,'fit p = %s' % p_value,transform=ax.transAxes)
    fig.savefig(egopath + '/polar_wall_dist_plot.png')
    
    
    return cluster_data

def calc_wall_dists(center_x,center_y,angles,xres,yres):
    
    xcoords = np.linspace(np.min(center_x),np.max(center_x),xres,endpoint=False)
    ycoords = np.linspace(np.min(center_y),np.max(center_y),yres,endpoint=False)
    w1 = np.stack((xcoords,np.repeat(np.max(ycoords),xres)))
    w3 = np.stack((xcoords,np.repeat(np.min(ycoords),xres)))
    w2 = np.stack((np.repeat(np.max(xcoords),yres),ycoords))
    w4 = np.stack((np.repeat(np.min(xcoords),yres),ycoords))
    
    all_walls = np.concatenate((w1,w2,w3,w4),axis=1)
    
    wall_x = all_walls[0]
    wall_y = all_walls[1]
    
    wall_dists = np.zeros((len(wall_x),len(center_x)))
    wall_angles = np.zeros((len(wall_x),len(center_x)))

    
    for i in range(len(center_x)):
        wall_dists[:,i] = np.sqrt((wall_x-center_x[i])**2 + (wall_y-center_y[i])**2)
        wall_angles[:,i] = (np.rad2deg(np.arctan2(wall_y-center_y[i],wall_x-center_x[i]))%360 - angles[i])%360

        
    return wall_x, wall_y, wall_dists, wall_angles


def plot_mh_ego(ops,adv,trial_data,cluster_data,spike_data,self):

    
    for direction_variable in ['hd','md']:
        #assign grid_res and hd_bins to shorter var for easier reading
        gr = adv['grid_res']
        framerate = adv['framerate']
    
        #grab appropriate data
        center_x = np.asarray(trial_data['center_x'])
        center_y = np.asarray(trial_data['center_y'])
        if direction_variable == 'hd':
            angles = np.asarray(trial_data['angles'])
        else:
            angles = np.array(trial_data['movement_directions'])
    #    mds = np.asarray(trial_data['movement_directions'])
        spike_train = np.asarray(spike_data['ani_spikes'])
    #    from astropy.convolution.kernels import Gaussian1DKernel
    #    angles = convolve(angles,Gaussian1DKernel(stddev=2))
        
        speeds = np.asarray(trial_data['speeds'])
        center_x = center_x[speeds>10]
        center_y = center_y[speeds>10]
        angles = angles[speeds>10]
    #    mds = mds[speeds>10]
        spike_train = spike_train[speeds>10]
    
        center_x -= np.min(center_x)
        center_y -= np.min(center_y)
        center_x -= (np.max(center_x) - np.min(center_x))/2
        center_y -= (np.max(center_y) - np.min(center_y))/2
        
        xcoords = np.linspace(np.min(center_x),np.max(center_x),gr,endpoint=False)
        ycoords = np.linspace(np.min(center_y),np.max(center_y),gr,endpoint=False)
        w1 = np.stack((xcoords,np.repeat(np.max(ycoords),gr)))
        w3 = np.stack((xcoords,np.repeat(np.min(ycoords),gr)))
        w2 = np.stack((np.repeat(np.max(xcoords),gr),ycoords))
        w4 = np.stack((np.repeat(np.min(xcoords),gr),ycoords))
        
        all_walls = np.concatenate((w1,w2,w3,w4),axis=1)
        
        wall_x = np.zeros((len(all_walls[0]),len(center_x)))
        wall_y = np.zeros((len(all_walls[1]),len(center_y)))
        
        for i in range(len(center_x)):
            wall_x[:,i] = all_walls[0] - center_x[i]
            wall_y[:,i] = all_walls[1] - center_y[i]
            
        wall_ego_angles = (np.rad2deg(np.arctan2(wall_y,wall_x))%360 - angles)%360
        wall_ego_dists = np.sqrt(wall_x**2 + wall_y**2)
        
        ref_angles = np.linspace(0,360,60,endpoint=False)
        radii = np.linspace(0,np.min((np.max(center_x)-np.min(center_x),np.max(center_y)-np.min(center_y)))/2.,20)
        dist_bins = np.digitize(wall_ego_dists,radii) - 1
        
        map_vals = np.zeros((60,len(center_x)))
        occ = np.zeros((60,20))
        spikes = np.zeros((60,20))
        cutoff = np.min((np.max(center_x)-np.min(center_x),np.max(center_y)-np.min(center_y)))/2.
        
        for i in range(len(center_x)):
            for a in range(len(ref_angles)):
                diffs = np.abs(wall_ego_angles[:,i] - ref_angles[a])
                closest_pt = np.where(diffs==np.min(diffs))[0][0]
                if wall_ego_dists[closest_pt,i] < cutoff:
                    map_vals[a,i] = dist_bins[closest_pt,i]
                    occ[a,dist_bins[closest_pt,i]] += 1./framerate
                    spikes[a,dist_bins[closest_pt,i]] += spike_train[i]
#                else:
#                    map_vals[a,i] = 1000
    
        heatmap = spikes/occ
        
        ref_angles = np.deg2rad(ref_angles)
    
        h_interp = heatmap.copy()
        v_interp = heatmap.copy()
        
        for i in range(len(heatmap)):
            nans,x = np.isnan(h_interp[i]), lambda z: z.nonzero()[0]
            try:
                h_interp[i][nans]= np.interp(x(nans), x(~nans), h_interp[i][~nans])
            except:
                pass
            
        for i in range(len(heatmap[0])):
            nans,x = np.isnan(v_interp[:,i].flatten()), lambda z: z.nonzero()[0]
            try:
                v_interp[:,i][nans]= np.interp(ref_angles[x(nans)], ref_angles[x(~nans)], v_interp[:,i][~nans],period=2*np.pi)
            except:
                pass
            
        histm = (h_interp + v_interp)/2.
        
        hist3 = np.concatenate((histm,histm,histm),axis=0)
        hist3 = convolve(hist3,Gaussian2DKernel(stddev=2))
        new_hist = hist3[len(histm):len(histm)*2]
    
        xvals,yvals = np.meshgrid(ref_angles,radii)
            
        mr = np.nansum(new_hist.T*np.exp(1j*xvals))/(np.sum(new_hist))
        mrl = np.abs(mr)
        mra = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))
    
    
        egopath = cluster_data['new_folder']+'/egocentric'
        import os
        import matplotlib.pyplot as plt
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
        fig=plt.figure()
        ax = fig.add_subplot(111,projection='polar')
        ax.set_theta_zero_location("N")
        ax.text(.1,.9,'MRL = %s' % mrl,transform=ax.transAxes)
        ax.text(.1,.8,'MRA = %s' % mra,transform=ax.transAxes)
    
        ax.pcolormesh(ref_angles,radii,new_hist.T,vmin=0)
    
        if direction_variable == 'hd':
            fig.savefig('%s/Full Wall Map.png' % egopath,dpi=adv['pic_resolution'])
        else:
            fig.savefig('%s/Full Wall Map_md.png' % egopath,dpi=adv['pic_resolution'])

        
        top_angle,top_dist = np.where(new_hist==np.max(new_hist))
        
        if direction_variable == 'hd':
        
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(new_hist[top_angle].flatten())
            ax.set_ylim(0,1.2*np.max(new_hist[top_angle])+10)
            fig.savefig('%s/top wall dists.png' % egopath,dpi=adv['pic_resolution'])
            
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(new_hist[:,top_dist].flatten())
            ax.set_ylim(0,1.2*np.max(new_hist[:,top_dist])+10)
            fig.savefig('%s/top wall angles.png' % egopath,dpi=adv['pic_resolution'])
            
        elif direction_variable == 'md':
        
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(new_hist[top_angle].flatten())
            ax.set_ylim(0,1.2*np.max(new_hist[top_angle])+10)
            fig.savefig('%s/top wall dists_md.png' % egopath,dpi=adv['pic_resolution'])
            
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(new_hist[:,top_dist].flatten())
            ax.set_ylim(0,1.2*np.max(new_hist[:,top_dist])+10)
            fig.savefig('%s/top wall angles_md.png' % egopath,dpi=adv['pic_resolution'])
            
            
        wall_x, wall_y, wall_dists, wall_angles = calc_wall_dists(center_x,center_y,angles,30,30)
        
        top_dists = top_dist - wall_dists
        top_angles = (top_angle - wall_angles)%360
        
        top_dists[top_dists>50] = 70
        
#        np.sqrt(wall_dists**2 + top_dist**2 - 2. * wall_dists * top_dist * np.cos(np.deg2rad(top_angle) - np.deg2rad(wall_angles)))
        
            
        wd_bins = np.digitize(top_dists,np.linspace(np.min(top_dists),np.max(top_dists),40,endpoint=False))-1
        wd_bins[top_dists==70] = -1
        wa_bins = np.digitize(top_angles,np.linspace(0,360,60,endpoint=False))-1
        
        spikes = np.zeros((60,40))
        occ = np.zeros((60,40))
        for i in range(len(spike_train)):
            spikes[wa_bins[:,i][wd_bins[:,i]>=0],wd_bins[:,i][wd_bins[:,i]>=0]] += spike_train[i]
            occ[wa_bins[:,i][wd_bins[:,i]>=0],wd_bins[:,i][wd_bins[:,i]>=0]] += 1./30.
            
        hmap = spikes/occ
        
        plt.imshow(hmap)




def plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    #assign grid_res and hd_bins to shorter var for easier reading
    gr = adv['grid_res']
    hd_bins = adv['hd_bins']
    framerate = adv['framerate']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    w1 = center_x - np.min(center_x)
    w2 = -(center_x - np.max(center_x))
    w3 = center_y - np.min(center_y)
    w4 = -(center_y - np.max(center_y))
    
    all_dists = np.stack([w1,w2,w3,w4])
    wall_dists = np.min(all_dists,axis=0)
    wall_ids = np.argmin(all_dists,axis=0)
    wall_angles = np.array([180.,0.,270.,90.])
    wall_ego_angles = wall_angles[wall_ids]
    wall_ego_angles = (wall_ego_angles - angles)%360
    
    dist_bins = np.digitize(wall_dists,np.linspace(0,np.max(wall_dists),20,endpoint=False))-1
    wall_ego_bins = np.digitize(wall_ego_angles,np.linspace(0,360,hd_bins,endpoint=False))-1
    
    wall_dist_spikes = np.zeros(20)
    wall_dist_occ = np.zeros(20)
    wall_ego_spikes = np.zeros(hd_bins)
    wall_ego_occ = np.zeros(hd_bins)
    
    for i in range(len(center_x)):
        wall_dist_spikes[dist_bins[i]] += ani_spikes[i]
        wall_dist_occ[dist_bins[i]] += 1./framerate
        wall_ego_spikes[wall_ego_bins[i]] += ani_spikes[i]
        wall_ego_occ[wall_ego_bins[i]] += 1./framerate
        
    wall_ego_curve = wall_ego_spikes/wall_ego_occ
    dist_curve = wall_dist_spikes/wall_dist_occ
    
    rayleigh, mean_angle = rayleigh_r(np.arange(6,366,360/hd_bins),wall_ego_curve)
    
    ego_angles = np.linspace(0+np.deg2rad(6),2.*np.pi+np.deg2rad(6),hd_bins)
    mean_guess = np.mean(wall_ego_curve)
    k_guess = np.max(wall_ego_curve) - mean_guess
    phase_guess = -mean_angle

    obj = lambda x: x[0] + x[1] * np.cos(ego_angles + x[2]) - wall_ego_curve
    mean, k, phase = leastsq(obj, [mean_guess,k_guess,phase_guess])[0]
    fit_curve = mean + k * np.cos(ego_angles + phase)
    r,p = pearsonr(fit_curve,wall_ego_curve)
    
    cluster_data['wall_ego_curve'] = wall_ego_curve
    cluster_data['wall_ego_rayleigh'] = rayleigh
    cluster_data['wall_ego_mean_angle'] = mean_angle
    cluster_data['wall_ego_fit'] = fit_curve
    cluster_data['wall_ego_r'] = r
    cluster_data['wall_ego_p'] = p
    cluster_data['wall_ids'] = wall_ids
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_wall_ego'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    
    xvals = np.linspace(np.min(wall_dists),np.max(wall_dists),20)
    
    slope, intercept, r_value,p_value,std_err = linregress(xvals,dist_curve)
    fit_y = []
    for i in range(len(xvals)):
        fit_y.append(slope*xvals[i] + intercept)

    cluster_data['wall_dist_curve'] = dist_curve
    cluster_data['wall_dist_xvals'] = xvals
    cluster_data['wall_dist_fit'] = fit_y
    cluster_data['wall_dist_r'] = r_value
    cluster_data['wall_dist_p'] = p_value
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_wall_dist'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    cluster_data['wall_ego_angles'] = wall_ego_angles
    cluster_data['wall_ego_dists'] = wall_dists
    wall_ego_spike_angles = []
    wall_ego_spike_dists = []
    for i in range(len(center_x)):
        for j in range(int(ani_spikes[i])):
            wall_ego_spike_angles.append(wall_ego_angles[i])
            wall_ego_spike_dists.append(wall_dists[i])
            
    spike_data['wall_ego_angles'] = np.array(wall_ego_spike_angles)
    spike_data['wall_ego_dists'] = np.array(wall_ego_spike_dists)
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_wall_ego_hd_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #note that autocorr data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_wall_ego_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    
    return cluster_data


def plot_boundary_field(ops,adv,trial_data,cluster_data,spike_data,self):
    
    #assign grid_res and hd_bins to shorter var for easier reading
    gr = adv['grid_res']
    hd_bins = adv['hd_bins']
    framerate = adv['framerate']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    x_pos = center_x > 0
    y_pos = center_y > 0
    
    x_walls = np.zeros(len(center_x))
    x_walls[x_pos] = np.max(center_x)+1
    x_walls[~x_pos] = np.min(center_x)-1
    
    y_walls = np.zeros(len(center_y))
    y_walls[y_pos] = np.max(center_y)+1
    y_walls[~y_pos] = np.min(center_y)-1
    
    x_dists = np.abs(x_walls) - np.abs(center_x)
    y_dists = np.abs(y_walls) - np.abs(center_y)
    
    y_points = np.zeros(len(center_x))
    x_points = np.zeros(len(center_y))
    
    y_log_weights = 1. - np.log(y_dists)/(np.log(y_dists) + np.log(x_dists))
    x_log_weights = 1. - np.log(x_dists)/(np.log(y_dists) + np.log(x_dists))
    
    y_weights = 1. - y_dists/(y_dists + x_dists)
    x_weights = 1. - x_dists/(y_dists + x_dists)
    
#    y_points = y_walls*y_weights + center_y*x_weights
#    x_points = x_walls*x_weights + center_x*y_weights
    
    y_points = y_walls*y_log_weights + center_y*x_log_weights
    x_points = x_walls*x_log_weights + center_x*y_log_weights
    
    dists = y_dists * y_log_weights + x_dists * x_log_weights
    
#    dists = np.sqrt((center_x-x_points)**2 + (center_y-y_points)**2)
    wall_angles = (np.rad2deg(np.arctan2(y_points-center_y,x_points-center_x)))%360
    wall_angles = (wall_angles-angles)%360
#    center_ego_mds = (center_ego_angles - np.asarray(trial_data['movement_directions']))
    
    dist_bins = np.digitize(dists,np.linspace(0,np.max(dists),20,endpoint=False))-1
    wall_ego_bins = np.digitize(wall_angles,np.linspace(0,360,hd_bins,endpoint=False))-1
    
    wall_dist_spikes = np.zeros(20)
    wall_dist_occ = np.zeros(20)
    wall_ego_spikes = np.zeros(hd_bins)
    wall_ego_occ = np.zeros(hd_bins)
    
    for i in range(len(center_x)):
        wall_dist_spikes[dist_bins[i]] += ani_spikes[i]
        wall_dist_occ[dist_bins[i]] += 1./framerate
        wall_ego_spikes[wall_ego_bins[i]] += ani_spikes[i]
        wall_ego_occ[wall_ego_bins[i]] += 1./framerate
        
    ego_curve = wall_ego_spikes/wall_ego_occ
    dist_curve = wall_dist_spikes/wall_dist_occ
    
    
    trial_data['boundary_field_angles'] = wall_angles
    
    rayleigh, mean_angle = rayleigh_r(np.arange(6,366,360/hd_bins),ego_curve)
    
    cluster_data['boundary_field_curve'] = ego_curve
    cluster_data['boundary_field_rayleigh'] = rayleigh
    cluster_data['boundary_field_mean_angle'] = mean_angle
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.93)
    #set y limit according to highest firing rate
    ymax = int(1.5*max(ego_curve))+10
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    ax.set_xlabel('boundary field bearing (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,45))
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % rayleigh,transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s' % mean_angle,transform=ax.transAxes)
    ax.plot(np.linspace(0,360,hd_bins),ego_curve,'k-') #,cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],max(cluster_data['hd_rates']),'r*',cluster_data['hd_angles'],cluster_data['gauss_rates'],'r--')

    egopath = cluster_data['new_folder']+'/egocentric'

    fig.savefig('%s/boundary field bearing plot.png' % egopath,dpi=adv['pic_resolution'])
    
    
    xvals = np.linspace(np.min(dists),np.max(dists),20)
    
    slope, intercept, r_value,p_value,std_err = linregress(xvals,dist_curve)
    fit_y = []
    for i in range(len(xvals)):
        fit_y.append(slope*xvals[i] + intercept)

    cluster_data['boundary_field_dist_curve'] = dist_curve
    cluster_data['boundary_field_dist_xvals'] = xvals
    cluster_data['boundary_field_dist_fit'] = fit_y
    cluster_data['boundary_field_dist_r'] = r_value
    cluster_data['boundary_field_dist_p'] = p_value
    
    fig.clf()
    ax = fig.add_subplot(111)  
    fig.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(dist_curve))+10
    xmax = np.nanmax(xvals)
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.set_xlabel('distance (pixels)')
    ax.set_ylabel('firing rate (hz)')

    ax.plot(xvals,dist_curve,'k-',xvals,fit_y,'b-')
    ax.text(.1,.9,'fit r^2 = %s' % r_value**2,transform=ax.transAxes)
    ax.text(.1,.8,'fit p = %s' % p_value,transform=ax.transAxes)
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    fig.savefig('%s/boundary field distance.png' % egopath,dpi=adv['pic_resolution'])
    
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import colors as mplcolors
    
    fig.clf()
    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.

    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    fig=plt.figure()
    #make the figure
    ax = fig.add_subplot(111)
#    fig.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x])
    
    spike_angles = []
    for i in range(len(wall_angles)):
        for j in range(ani_spikes[i]):
            spike_angles.append(wall_angles[i])

    #make a scatter plot of spike locations colored by head direction
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_angles,cmap=colormap,norm=norm)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
    
#    plt.axis('equal')
    
    egopath = cluster_data['new_folder']+'/egocentric'
        
    fig.savefig('%s/boundary field hd map.png' % egopath,dpi=adv['pic_resolution'])

    

def plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,ego=False,weighted=False,view=False):

    #assign grid_res and hd_bins to shorter var for easier reading
    gr = adv['grid_res']
    hd_bins = adv['hd_bins']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
#    from scipy.ndimage.filters import gaussian_filter1d
#    a=np.array(np.roll(center_y,1)-center_y,dtype=np.float)
#    a[0] = 0
#    a=gaussian_filter1d(a,sigma=5)
#    
#    b=np.array(np.roll(center_x,1)-center_x,dtype=np.float)
#    b[0] = 0
#    b=gaussian_filter1d(b,sigma=5)
#    for i in range(len(a)):
#        if a[i]==0 and b[i]==0:
#            a[i]=np.nan
#            b[i]=np.nan
#    
#    angles = np.rad2deg(np.arctan2(a,b))%360
    
#    yvals = np.linspace(np.min(center_x),np.max(center_x),gr)
#    xvals = np.linspace(np.min(center_y),np.max(center_y),gr)
#    xcoords,ycoords = np.meshgrid(xvals,yvals)
#    print ycoords
#    print xcoords
#    rat_y = np.swapaxes(np.tile(center_y,(gr,gr,1)),0,2)
#    rat_x = np.swapaxes(np.tile(center_x,(gr,gr,1)),0,2)
#    
#    new_angles = np.swapaxes((np.rad2deg(np.arctan2(ycoords-rat_y,xcoords-rat_x)))%360,0,2)
#    ego_angles = (new_angles-angles)%360
#    
#    ego_bins = np.swapaxes(np.digitize(ego_angles,np.linspace(0,360,hd_bins,endpoint=False))-1,0,2)
#    
#    del rat_y
#    del rat_x
#    del new_angles
#    del ego_angles
#    
#    @nb.jit(nogil=True)
#    def next_step(gr,hd_bins,ego_bins,ani_spikes,framerate):
#    
#        ego_spikes = np.zeros((gr,gr,hd_bins))
#        ego_occ = np.zeros((gr,gr,hd_bins))
#        for i in range(gr):
#            for j in range(gr):
#                for k in range(len(center_x)):
#                    if k%100 == 0:
#                        print k
#                    ego_spikes[i,j,ego_bins[i,j,k]] += ani_spikes[k]
#                    ego_occ[i,j,ego_bins[i,j,k]] += 1./framerate
#        tuning_curves = ego_spikes/ego_occ
#        
#        return tuning_curves
#    
#    tuning_curves = next_step(gr,hd_bins,ego_bins,ani_spikes,adv['framerate'])
    
#    rayleighs = np.zeros((gr,gr))
##    rxs = np.zeros((gr,gr))
##    rys = np.zeros((gr,gr))
#    mean_angles = np.zeros((gr,gr))   
#    for i in range(gr):
#        for j in range(gr):
#            r, mean_angle = rayleigh_r(np.linspace(0,360,hd_bins,endpoint=False),tuning_curves[i][j])
    
#    hist_angles = np.reshape(ego_angles,(gr**2,len(center_y)))
#    occ_hist,occ_bins = np.histogramdd(hist_angles,bins=hd_bins)
#    occ_hist = occ_hist.reshape((gr,gr,len(center_y)))
#    spike_hist_angles = 
    
    @nb.njit(nogil=True)
    def ego_loop(center_y,center_x,angles,framerate):
        """ transform allocentric to egocentric angles and assign dwell times """

        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(gr)
        ycoords = np.zeros(gr)
        
        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(gr):
            xcoords[x] = (np.float(x)/np.float(gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x)) + np.float(np.max(center_x)-np.min(center_x))/np.float(gr)
            ycoords[x] = np.float(np.max(center_y)) - (np.float(x)/np.float(gr))*np.float((np.max(center_y)-np.min(center_y))) - np.float(np.max(center_y)-np.min(center_y))/np.float(gr)

        if ego:
            
            #make arrays to hold egocentric angles and combos (allo,ego,spatial) for each video frame,
            #along with dwell times and spike counts for each bin
#            ego_bins = np.zeros((gr,gr,len(center_x)),dtype=nb.types.int64)
            ego_dwells = np.zeros((gr,gr,hd_bins))
            ego_spikes = np.zeros((gr,gr,hd_bins))
            
            tuning_curves = np.zeros((gr,gr,hd_bins))
            rayleighs = np.zeros((gr,gr))
            rxs = np.zeros((gr,gr))
            rys = np.zeros((gr,gr))
            mean_angles = np.zeros((gr,gr))
            
            if weighted:
                
                thresh_dist = 150
                
        if view:
            view_spikes = np.zeros((gr,gr))
            view_dwells = np.zeros((gr,gr))

        #for each y position...
        for i in range(gr):
            print(i)

            cue_y = ycoords[i]
    
            #for each x position...
            for j in range(gr):
                                        
                cue_x = xcoords[j]
                  
                #calc array of egocentric angles of this bin from pos x axis centered 
                #on animal using arctan
                new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
                #calculate ego angles by subtracting allocentric
                #angles from egocentric angles
                ego_angles = (new_angles-angles)%360

                
                if view:
                    view_bins = ego_angles.copy()
                    view_bins[np.where(np.logical_and(view_bins>15, view_bins<345))] = np.nan
                    view_bins /= (360/hd_bins)
                
                if ego:
                    #assign to bin
                    ego_bins = ego_angles/(360/hd_bins)
                    
                    if weighted:
                        dists = np.sqrt((cue_y - center_y)**2 + (cue_x - center_x)**2)
                        ego_bins[np.where(dists > thresh_dist)] = np.nan
                        ego_bins[np.where(dists < 10)] = np.nan
                        dists[np.where(dists > thresh_dist)] = np.nan
                        dists[np.where(dists < 10)] = np.nan
                                            
                        weights=dists
                        weights = np.sin(np.deg2rad(90+((weights - (np.nanmin(weights)))/(np.nanmax(weights)-np.nanmin(weights)))*90))
    
                    else:
                        weights = np.ones(len(center_x))
                        
#                    ego_bins[i][j] = epd_bins
                        
                for k in range(len(center_x)):
                                      
                    if ego and not np.isnan(ego_bins[k]) and not np.isnan(weights[k]):
                        #add one framelength to the dwell time for that ego hd bin
                        ego_dwells[i][j][np.int(ego_bins[k])] += (1./framerate) * weights[k]
                        #add the number of spikes for that frame to the appropriate ego bin
                        ego_spikes[i][j][np.int(ego_bins[k])] += ani_spikes[k] * weights[k]
                    
                    if view and not np.isnan(view_bins[k]):
                        view_dwells[i][j] += 1./framerate
                        view_spikes[i][j] += ani_spikes[k]
                        
                if ego:
                        

    #                #take away bins where the rat never sampled
    #                ego_dwells[i][j][ego_dwells[i][j]==0] = np.nan
    #                ego_spikes[i][j][np.isnan(ego_dwells[i][j])] = np.nan
    #                
    #                #calculate a matrix of firing rates
    #                fr_mat = ego_spikes[i][j] / ego_dwells[i][j]
    #                #calculate the mean firing rate
    #                mean_fr = np.nansum(ego_spikes[i][j]) / np.nansum(ego_dwells[i][j])
    #                
    #                #total occupancy time
    #                tot_occ = np.nansum(ego_dwells[i][j])
    #                #calc a matrix of occupancy probabilities
    #                prob_mat = ego_dwells[i][j] / tot_occ
    #            
    #                #calculate the information content
    #                info = np.nansum( prob_mat * (fr_mat / mean_fr) * np.log2(fr_mat / mean_fr) )
    #                
    #                rayleighs[i][j] = info
    #                
    #                
    #                ego_dwells[i][j][ego_dwells[i][j] == 0] = np.nan
                    rates = ego_spikes[i][j]/ego_dwells[i][j]
    
                    tuning_curves[i][j] = rates
                    hd_angles = np.arange(6,366,6)
                
                    #start vars for x and y rayleigh components
                    rx = 0
                    ry = 0
                    
                    #convert spike angles into x and y coordinates, sum up the results -- 
                    #if firing rates are provided along with HD plot edges instead of spike angles,
                    #do the same thing but with those
                    for m in range(len(hd_angles)):
                        rx += np.cos(np.deg2rad(hd_angles[m]))*rates[m]
                        ry += np.sin(np.deg2rad(hd_angles[m]))*rates[m]
                
                    if np.nansum(rates) > 0:
                        #calculate average x and y values for vector coordinates
                        rx = rx/np.nansum(rates)
                        ry = ry/np.nansum(rates)
                    else:
                        rx = ry = 0
                
                    #calculate vector length
                    r = np.sqrt(rx**2 + ry**2)
                    
                    #calculate the angle the vector points (rayleigh pfd)
                    #piecewise because of trig limitations
                    if rx == 0:
                        mean_angle = 0
                    if rx > 0:
                        mean_angle = np.rad2deg(np.arctan(ry/rx))
                    if rx < 0:
                        mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
                        
                    if mean_angle < 0:
                        mean_angle = mean_angle + 360
                        
                    rayleighs[i][j] = r
                    rxs[i][j] = rx
                    rys[i][j] = ry
                    mean_angles[i][j] = mean_angle

        if view:
            view_map = view_spikes/view_dwells
            
        if ego and not view:
            view_map = np.zeros((gr,gr))
        if view and not ego:
            tuning_curves = np.zeros((gr,gr,hd_bins))
            rayleighs = np.zeros((gr,gr))
            rxs = np.zeros((gr,gr))
            rys = np.zeros((gr,gr))
            mean_angles = np.zeros((gr,gr))

        return rayleighs,rxs,rys,mean_angles,tuning_curves,view_map
            

    rayleighs,rxs,rys,mean_angles,tuning_curves,view_map = ego_loop(center_y,center_x,angles,adv['framerate'])

    if ego:
        cluster_data['ego_curves'] = tuning_curves
#        cluster_data['rayleigh_rxs'] = rxs
#        cluster_data['rayleigh_rys'] = rys
        cluster_data['ego_rayleighs'] = rayleighs
        cluster_data['ego_mean_angles'] = mean_angles
    if view:
        cluster_data['view_map'] = view_map
    
#    cluster_data = allocentrize_ego(ops,adv,trial_data,cluster_data,spike_data,self)
           
    if ego:
        #note that ego rayleigh r plotting data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_ego'] = True
        #send updated data dicts to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
        #note that ego mean angle plotting data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_ego_angle'] = True
        #send updated data dicts to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    if view:
        #note that ego score plotting data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['plot_view'] = True
        #send updated data dicts to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    #return the data!
    return cluster_data

def allocentrize_ego(ops,adv,trial_data,cluster_data,spike_data,self,weighted=False):
    
    thresh_dist = 150
    
    tuning_curves = np.asarray(cluster_data['ego_curves'])
    gr = adv['grid_res']
    hd_bins = adv['hd_bins']
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    
    rayleighs = np.zeros((gr,gr))
    mean_angles = np.zeros((gr,gr))
    rxs = np.zeros((gr,gr))
    rys = np.zeros((gr,gr))

    #create arrays for x and y coords of spatial bins
    xcoords = np.zeros(gr)
    ycoords = np.zeros(gr)
    #assign coordinates for x and y axes for each bin
    for x in range(gr):
        xcoords[x] = (float(x)/float(gr))*(max(center_x)-min(center_x)) + min(center_x)
        ycoords[x] = max(center_y) - (float(x)/float(gr))*(max(center_y)-min(center_y))

#    allocentrized_curves = np.zeros((gr,gr,hd_bins))
    
    for i in range(gr):
        current_y = ycoords[i]
        for j in range(gr):
            current_x = xcoords[j]
            
            num_responsible = np.zeros(hd_bins)
            rates = np.zeros(hd_bins)
            
            if weighted:
                xgrid,ygrid = np.meshgrid(xcoords,ycoords)
                
                dists = np.sqrt((xgrid - current_x)**2 + (ygrid - current_y)**2)
                dists[dists > thresh_dist] = np.nan
                weights = np.sin(np.deg2rad(90+((dists - (np.nanmin(dists)))/(np.nanmax(dists)-np.nanmin(dists)))*90))
                weights[np.isnan(weights)] = 0
            
            else:
                weights = np.ones((gr,gr))
            
            for l in range(gr):
                other_y = ycoords[l]
                for m in range(gr):
  
                    other_x = xcoords[m] 
                    new_angle = np.rad2deg(np.arctan2((other_y-current_y),(other_x-current_x)))

                    for k in range(hd_bins):
                        hd_angle = k*360/float(hd_bins)
                        ego_bin = np.int(((new_angle - hd_angle)%360)/(360/hd_bins))
                        rates[k] += tuning_curves[l][m][ego_bin] * weights[l][m]
                        num_responsible[k] += weights[l][m]
                        
            rates /= num_responsible

            hd_angles = np.arange(6,366,6)
        
            #start vars for x and y rayleigh components
            rx = 0
            ry = 0
            
            #convert spike angles into x and y coordinates, sum up the results -- 
            #if firing rates are provided along with HD plot edges instead of spike angles,
            #do the same thing but with those
            for m in range(len(hd_angles)):
                rx += np.cos(np.deg2rad(hd_angles[m]))*rates[m]
                ry += np.sin(np.deg2rad(hd_angles[m]))*rates[m]
        
            if np.nansum(rates) > 0:
                #calculate average x and y values for vector coordinates
                rx = rx/np.nansum(rates)
                ry = ry/np.nansum(rates)
            else:
                rx = ry = 0
        
            #calculate vector length
            r = np.sqrt(rx**2 + ry**2)
            
            #calculate the angle the vector points (rayleigh pfd)
            #piecewise because of trig limitations
            if rx == 0:
                mean_angle = 0
            if rx > 0:
                mean_angle = np.rad2deg(np.arctan(ry/rx))
            if rx < 0:
                mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
                
            if mean_angle < 0:
                mean_angle = mean_angle + 360
                
            rayleighs[i][j] = r
            rxs[i][j] = rx
            rys[i][j] = ry
            mean_angles[i][j] = mean_angle


    cluster_data['allocentrized_rayleighs'] = rayleighs
    cluster_data['allocentrized_rxs'] = rxs
    cluster_data['allocentrized_rys'] = rys
    cluster_data['allocentrized_mean_angles'] = mean_angles
    cluster_data['ego_xcoords'] = xcoords
    cluster_data['ego_ycoords'] = ycoords
    return cluster_data

def subplot_designer(ops,adv,trial_data,cluster_data,spike_data,self):
    ''' call the subplot making program '''
    
    #note that subplot data is ready (doesn't totally make sense)
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']][trial_data['current_cam_ind']]['dataready']['subplot_designer'] = True
    #send updated data dicts to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    



    