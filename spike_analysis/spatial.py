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
from scipy.optimize import curve_fit
from scipy.fftpack import fft
from skimage.transform import rotate
from astropy.convolution.kernels import Gaussian2DKernel, Gaussian1DKernel
from astropy.convolution import convolve
import copy
import os
import warnings
import numba as nb

#####################

def interp_points(raw_vdata,trial_data,adv):
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
                        
    if adv['hd_calc'] == 'Neuralynx':
        #interpolate nondetects in head angle data
        if interpd_angles[i] == 450:
            x = 0
            count = 1
            while x == 0:
                if i+count < len(interpd_pos):
                    if interpd_angles[i+count] == 450:
                        count +=1
                    else:
                        true_diff = (interpd_angles[i+count] - interpd_angles[i-1] + 180.)%360. - 180.
                        interpd_angles[i] = (interpd_angles[i-1] + true_diff/count)%360.
                        x=1
                else:
                    interpd_angles[i] = interpd_angles[i-1]
                    x=1

        
    elif adv['hd_calc'] == 'LED positions':
        interpd_angles = []
        
        for row in interpd_pos:
            angle = (360 - np.rad2deg(np.arctan2(row[2]-row[4],row[1]-row[3])))%360
            
            interpd_angles.append(angle)
        
    trial_data['angles'] = np.array(interpd_angles)
        

            
    #return interpolated LED location and head angle data
    trial_data['positions'] = interpd_pos
#    trial_data['angles'] = interpd_angles
    
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
        
        # if ops['acq'] == 'neuralynx' or ops['acq']=='taube':
        center_x.append(x)
        center_y.append(y)
        # elif ops['acq'] == 'openephys':
        #     center_y.append(y)
        #     center_x.append(x)
        
    center_x = np.array(center_x)
    center_y = np.array(center_y)
    
    if ops['acq'] == 'neuralynx' or ops['acq'] == 'openephys':
        #now we have to flip neuralynx's inverted y-axis
        center_y = -center_y + np.max(center_y)
            
    # elif ops['acq'] == 'openephys':
    #     #now we have to flip the x axis (oops)
    #     center_x = -center_x + np.max(center_x)

    #interpolates new timestamps between existing timestamps for spike time     
    #analyses to reach temporal precision given by bin_size variable 
    spike_timestamps = np.arange(timestamps[0],timestamps[len(timestamps)-1],adv['bin_size']*1000.)

    #return coordinates of head center and timestamps for spatial and spike
    #analysis
    
    trial_data['center_x'] = center_x
    trial_data['center_y'] = center_y
    trial_data['timestamps'] = np.array(timestamps)
    trial_data['spike_timestamps'] = spike_timestamps
    
    return trial_data

def scale_tracking_data(adv,trial_data,trial):
    
    adv['dist_measurement'] = 'pixels'
    
    if adv['arena_x'] is not None and adv['arena_y'] is not None:
        adv['dist_measurement'] = 'cm'

    if '1.2m' in os.path.basename(trial):
        adv['arena_x'] = 120.
        adv['arena_y'] = 120.
        adv['dist_measurement'] = 'cm'
    elif '1m' in trial and 'rot' not in os.path.basename(trial):
        adv['arena_x'] = 100.
        adv['arena_y'] = 100.
        adv['dist_measurement'] = 'cm'
    elif '1m' in trial and 'rot' in os.path.basename(trial):
        adv['arena_x'] = 100.*np.sqrt(2.)
        adv['arena_y'] = 100.*np.sqrt(2.)
        adv['dist_measurement'] = 'cm'
    elif '.6m' in os.path.basename(trial):
        adv['arena_x'] = 60.
        adv['arena_y'] = 60.
        adv['dist_measurement'] = 'cm'
    elif 'rect' in os.path.basename(trial):
        adv['arena_x'] = 120.
        adv['arena_y'] = 60.
        adv['dist_measurement'] = 'cm'

    if adv['dist_measurement'] == 'cm':
        center_x = trial_data['center_x']
        center_y = trial_data['center_y']
        
        center_x -= np.min(center_x)
        center_x = center_x * adv['arena_x'] / np.max(center_x)
        
        center_y -= np.min(center_y)
        center_y = center_y * adv['arena_y'] / np.max(center_y)

        trial_data['center_x'] = center_x
        trial_data['center_y'] = center_y
        
    if trial_data['dlc_file'] is not None:
        
        led_dist = 3. #in cm
        side = -1 #1 for left, -1 for right
        angles = np.deg2rad(trial_data['angles'])

        center_x = center_x + side * led_dist * np.sin(angles)
        center_y = center_y - side * led_dist * np.cos(angles)
        
        center_x -= np.min(center_x)
        center_x = center_x * adv['arena_x'] / np.max(center_x)
        
        center_y -= np.min(center_y)
        center_y = center_y * adv['arena_y'] / np.max(center_y)
        
        trial_data['center_x'] = center_x
        trial_data['center_y'] = center_y

    return adv, trial_data
    
def create_spike_lists(ops,adv,trial_data,cluster_data):
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
        
            if spike_ind < len(spike_train):
                #add 1 to spike train at appropriate spot
                spike_train[spike_ind] = 1
            
    if adv['speed_cutoff'] > 0:
        ani_spikes = ani_spikes[trial_data['og_speeds'] > adv['speed_cutoff']]
    
    for ind in range(len(ani_spikes)):
        for i in range(ani_spikes[ind]):
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
    
    return spike_data, cluster_data
  
def plot_path(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots rat's running path and spike locations"""
    
    #note that path plot data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_path'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

def plot_heat(ops,adv,trial_data,cluster_data,spike_data,self,test=False):
    """plots raw, interpolated, and smoothed spatial heatmaps"""
    
    #grab appropriate tracking data
    if not test:
        center_x = trial_data['center_x']
        center_y = trial_data['center_y']
        spike_x = spike_data['spike_x']
        spike_y = spike_data['spike_y']

    elif test:
        center_x = trial_data['center_x']
        center_y = trial_data['center_y']
        spike_x = spike_data['spike_x']
        spike_y = spike_data['spike_y']
    
    #calculate 2D histogram of spatial occupancies for the rat's path, break 
    #arena into bins assigned by bin size parameter
    h,xedges,yedges = np.histogram2d(center_x,center_y,bins=[np.arange(np.min(center_x),np.max(center_x),adv['spatial_bin_size']),np.arange(np.min(center_y),np.max(center_y),adv['spatial_bin_size'])],range=[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    #calculate 2D histogram of spikes per bin
    spikeh,spikexedges,spikeyedges = np.histogram2d(spike_x,spike_y,[np.arange(np.min(center_x),np.max(center_x),adv['spatial_bin_size']),np.arange(np.min(center_y),np.max(center_y),adv['spatial_bin_size'])],range=[[min(center_x),max(center_x)],[min(center_y),max(center_y)]]) 
    #only include bins with sampling greater than set by sample_cutoff parameter
    
    for i in range(len(h)):
        for j in range(len(h[i])):
            if float(h[i][j])/adv['framerate'] < adv['sample_cutoff']:
                h[i][j] = 0
                spikeh[i][j] = 0
                
#    info,sparsity = spatial_info(adv,center_x,center_y,spike_x,spike_y)
#    print('spatial info and sparsity:')
#    print(info)
#    print(sparsity)

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
#    interpd_heatmap = interp_raw_heatmap(raw_heatmap)
        
    stddev = 4. / adv['spatial_bin_size']
    #create a smoothed heatmap using convolution with a Gaussian kernel
    smoothed_heatmap = convolve(raw_heatmap, Gaussian2DKernel(x_stddev=stddev,y_stddev=stddev))

    if test:
        return smoothed_heatmap

    #add heatmaps to cluster data
    cluster_data['heat_xedges'] = xedges
    cluster_data['heat_yedges'] = yedges
    cluster_data['raw_heatmap'] = raw_heatmap
#    cluster_data['interpd_heatmap'] = interpd_heatmap
    cluster_data['smoothed_heatmap'] = smoothed_heatmap
    
    #note that raw_heatmap data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_raw_heat'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
#    #note that interpd_heatmap data is ready
#    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_interpd_heat'] = True
#    #send updated data to the gui
#    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
    #note that smoothed_heatmap data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_smoothed_heat'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
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


def plot_hd(ops,adv,trial_data,cluster_data,spike_data,self,test=False):
    """makes a cartesian head direction plot"""
    
    #grab appropriate tracking and spike data
    angles = np.asarray(trial_data['angles'])
    spike_angles = np.asarray(spike_data['spike_angles'])

    #create histogram of spikes per direction bin specified by hd_bins parameter
    spike_hd_hist = np.histogram(spike_angles,bins=adv['hd_bins'],range=(0,360))
    #create histogram of frames spent in each bin
    hd_hist = np.histogram(angles,bins=adv['hd_bins'],range=(0,360))
    
#    info,sparsity = hd_info(adv,angles,spike_angles)
#    print('hd info and sparsity:')
#    print(info)
#    print(sparsity)
    
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
    hd_angles = np.array(hd_hist[1].tolist())
    
    #fit a curve to the plot, collect fit pfd and firing rates
#    pfd, gauss_rates = hd_fit_curve(hd_angles,rates)

    #calculate rayleigh r and rayleigh angle
    real_hd_angles = hd_angles + (hd_angles[1] - hd_angles[0])/2.
    r, mean_angle = rayleigh_r(real_hd_angles,rates)
    
    if test:
        return rates

    #assign new data to cluster/spike data dicts
    cluster_data['hd_angles'] = hd_angles
    cluster_data['hd_rates'] = np.array(rates)
#    cluster_data['gauss_rates'] = gauss_rates
#    cluster_data['pfd'] = pfd
    cluster_data['angles'] = angles
    spike_data['spike_angles'] = spike_angles
    cluster_data['rayleigh'] = r
    cluster_data['rayleigh_angle'] = mean_angle
    
    #note that HD data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_hd'] = True
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
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_half_hds'] = True
    #send updated data to gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #return data
    return cluster_data


def plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots each spike colored by head direction, along with HD vectors over space"""
        
    #note that HD map data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_hd_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data
    
    
def plot_hd_vectors(ops,adv,trial_data,cluster_data,spike_data,self):
    
    hd_bins = 12
    framerate = adv['framerate']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
    y_gr = int(np.ceil((np.max(center_y) - np.min(center_y)) / 15.))
    x_gr = int(np.ceil((np.max(center_x) - np.min(center_x)) / 15.))

    x_edges = np.linspace(np.min(center_x),np.max(center_x),x_gr+1,endpoint=True)
    y_edges = np.linspace(np.min(center_y),np.max(center_y),y_gr+1,endpoint=True)
    
    curves = np.zeros((x_gr,y_gr,hd_bins))
    
    
    for i in range(x_gr):
        for j in range(y_gr):
            
            xmin = x_edges[i]
            xmax = x_edges[i+1]
            ymin = y_edges[j]
            ymax = y_edges[j+1]

            done = False
            while not done:
                
                inds = [(center_x >= xmin) & (center_x <= xmax) & (center_y >= ymin) & (center_y <= ymax)]
                
                bin_angles = angles[tuple(inds)]
                bin_spikes = ani_spikes[tuple(inds)]
                
                angle_bins = np.digitize(bin_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
                
                spikes = np.zeros(hd_bins)
                occ = np.zeros(hd_bins)
                
                for k in range(len(bin_angles)):
                    occ[angle_bins[k]] += 1./framerate
                    spikes[angle_bins[k]] += bin_spikes[k]
                    
                if np.any(occ < 0.2):
                    xmin -= 0.5
                    xmax += 0.5
                    ymin -= 0.5
                    ymax += 0.5
                    
                elif np.all(occ >= 0.2) or (xmax - xmin >= 30.) or (ymax - ymin >= 30.):
                    
                    curve = spikes/occ
                    curves[i][j] = curve
                    
                    done = True
                    
                    
    rs = np.zeros((x_gr,y_gr))
    rxs = np.zeros_like(rs)
    rys = np.zeros_like(rs)
    mean_angles = np.zeros_like(rs)
    
    angle_edges = np.linspace(15,375,hd_bins,endpoint=False)
    
    for i in range(x_gr):
        for j in range(y_gr):
            rs[i][j], rxs[i][j], rys[i][j], mean_angles[i][j] = rayleigh_r(angle_edges,curves[i][j],ego=True)
            
            
    cluster_data['hd_rs'] = rs
    cluster_data['hd_rxs'] = rxs
    cluster_data['hd_rys'] = rys
    cluster_data['hd_vector_x_gr'] = x_gr
    cluster_data['hd_vector_y_gr'] = y_gr
    cluster_data['hd_vector_curves'] = curves
    cluster_data['hd_vector_mean_angles'] = mean_angles
    
    #note that hd vector data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_hd_vectors'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #return data
    return cluster_data
    

def spatial_autocorrelation(ops,adv,trial_data,cluster_data,spike_data,self):
    """makes a spatial autocorrelation plot for grid cells"""
    
    #grab the smoothed heatmap
    smoothed_heatmap = cluster_data['smoothed_heatmap']

    x_gr = len(smoothed_heatmap)
    y_gr = len(smoothed_heatmap[0])

    #make a matrix of zeros 2x the length and width of the smoothed heatmap (in bins)
    corr_matrix = np.zeros((2*x_gr,2*y_gr))
    
    #for every possible overlap between the smoothed heatmap and its copy, 
    #correlate those overlapping bins and assign them to the corresponding index
    #in the corr_matrix
    for i in range(-len(smoothed_heatmap),len(smoothed_heatmap)):
        for j in range(-len(smoothed_heatmap[0]),len(smoothed_heatmap[0])):
            if i < 0:
                if j < 0:
                    array1 = smoothed_heatmap[(-i):(x_gr),(-j):(y_gr)]
                    array2 = smoothed_heatmap[0:(x_gr+i),0:(y_gr+j)]
                elif j >= 0:
                    array1 = smoothed_heatmap[(-i):(x_gr),0:(y_gr-j)]
                    array2 = smoothed_heatmap[0:(x_gr+i),(j):y_gr]
            elif i >= 0:
                if j < 0:
                    array1 = smoothed_heatmap[0:(x_gr-i),(-j):(y_gr)]
                    array2 = smoothed_heatmap[(i):x_gr,0:(y_gr+j)]
                elif j >= 0:
                    array1 = smoothed_heatmap[0:(x_gr-i),0:(y_gr-j)]
                    array2 = smoothed_heatmap[(i):x_gr,(j):y_gr]
            
            #this will give us annoying warnings for issues that don't matter --
            #we'll just ignore them
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
            #get the pearson r for the overlapping arrays
            try:
                corr,p = pearsonr(np.ndarray.flatten(array1),np.ndarray.flatten(array2))
            except:
                corr = np.nan
            #assign the value to the appropriate spot in the autocorr matrix
            corr_matrix[x_gr+i][y_gr+j] = corr

    #return the spatial autocorrelation matrix for later use
    cluster_data['spatial_autocorr'] = corr_matrix
    #note that spatial autocorrelation data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_spatial_autocorr'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    return cluster_data
    
def grid_score(ops,adv,trial_data,cluster_data,spike_data,self):
    """computes gridness score for the cluster"""
    
    #grab the autocorrelation matrix and grid resolution
    corr_matrix = cluster_data['spatial_autocorr']
        
    #assign grid resolution to shorter variable for easier readability
    x_gr = int(len(corr_matrix)/2)
    y_gr = int(len(corr_matrix[0])/2)
    
    #make x and y grids size of corr_matrix for defining rings
    x,y = np.meshgrid(range(-y_gr,y_gr),range(-x_gr,x_gr))
    
    min_gr = np.min([x_gr,y_gr])
    #set default indices for donut rings in case NOT a grid cell
    outer_radius = int(min_gr*.9)
    inner_radius = int(min_gr*.3)

    #start counters for keeping track of negative and positive average correlation values
    neg_count = 0
    pos_count = 0
    #for just about every possible radius from the center of the corr_matrix
    for i in range(1,min_gr-2):
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
        outer_radius = min_gr - 1
            
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
    cluster_data['rot_angles'] = rot_angles
    cluster_data['rot_values'] = rot_values
    cluster_data['gridness'] = gridness
    #note that gridness data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_grid_score'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    return cluster_data


def calc_movement_direction(adv,trial_data):
    
    #grab appropriate tracking data
    center_x=trial_data['center_x']
    center_y=trial_data['center_y']

    smooth_x = convolve(center_x,kernel=Gaussian1DKernel(stddev=5))
    smooth_y = convolve(center_y,kernel=Gaussian1DKernel(stddev=5))
    
    dx = smooth_x - np.roll(smooth_x,1)
    dy = smooth_y - np.roll(smooth_y,1)
    
    mds = np.rad2deg(np.arctan2(dy,dx))%360
    
    trial_data['movement_directions'] = mds
    
    return trial_data

def calc_speed(adv,trial_data):
    """calculates 'instantaneous' linear speeds for each video frame"""
    
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    
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
    
def plot_speed(ops,adv,trial_data,cluster_data,spike_data,self):
    """plots firing rate vs speed"""
    
    #grab our calculated speed data
    speeds = trial_data['speeds']
    spike_train = spike_data['ani_spikes']
    bin_size = adv['speed_bin_size']
    
    max_bin = 100
    num_bins = len(np.arange(0,max_bin,bin_size))
    
    speed_bins = np.digitize(speeds,np.arange(0,max_bin,bin_size)) - 1
    
    spikes = np.zeros(num_bins)
    occ = np.zeros(num_bins)
    for i in range(len(speed_bins)):
        spikes[speed_bins[i]] += spike_train[i]
        occ[speed_bins[i]] += 1./adv['framerate']
        
    with warnings.catch_warnings():
        #nans are fine here stop bothering me!
        warnings.simplefilter("ignore")
        
        speed_rates = spikes/occ
        
    speed_rates[len(speed_rates)-1] = np.nan
        
    #throw out bins sampled less than 1 sec
    speed_rates[occ<1] = np.nan
    
    #calculate speed x values
    speed_edges = np.arange(0,max_bin,bin_size) + bin_size/2.
    #plot a linear regression and grab the y values for plotting, along with r and p values
    slope, intercept, r_value,p_value,std_err = linregress(speed_edges[~np.isnan(speed_rates)],speed_rates[~np.isnan(speed_rates)])
    fit_y = []
    for i in range(len(speed_edges[~np.isnan(speed_rates)])):
        fit_y.append(slope*speed_edges[~np.isnan(speed_rates)][i] + intercept)
    

    #return data for plotting later
    cluster_data['speed_edges'] = speed_edges[~np.isnan(speed_rates)]
    cluster_data['speed_rates'] = speed_rates[~np.isnan(speed_rates)]
    cluster_data['max_speed'] = max_bin
    cluster_data['speed_fit_y'] = np.array(fit_y)
    cluster_data['speed_r'] = r_value
    cluster_data['speed_p'] = p_value
    
    #note that speed data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_speed'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data


def calc_ahv(adv,trial_data):
    """calculates 'instantaneous' angular head velocity for each video frame"""
    
    #grab appropriate tracking data
    angles=np.array(trial_data['angles'])
    #TODO: smooth angles?
    #make an array of zeros to assign ahvs to
    ahvs = np.zeros(len(angles),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(ahvs)-2):
        #start a list of head directions with the direction 2 frames ago
        angle_list = [angles[i-2]]
        #for each frame until angle_list is 5 angles long...
        for j in range(-1,3):
            diff = angles[i+j] - angle_list[j+1]
            true_diff = (diff+180.)%360. - 180.
            angle_list.append(angle_list[j+1]+true_diff)

        #find a line of best fit for the 5 directions in angle_list (slope is AHV)
        fitline = np.polyfit(np.arange(0,5),angle_list,1)
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
    spike_train = spike_data['ani_spikes']
    bin_size = adv['ahv_bin_size']
    
    #we'll limit ourselves to maximum 402 deg/sec with bin size 6 deg/sec
    min_bin = -402
    max_bin = 402
    num_bins = int((max_bin-min_bin)/bin_size)
    
    ahvs[ahvs<-402] = -401
    ahv_bins = np.digitize(ahvs,np.linspace(-402,402,num_bins,endpoint=False)) - 1
    
    spikes = np.zeros(num_bins)
    occ = np.zeros(num_bins)
    for i in range(len(ahv_bins)):
        spikes[ahv_bins[i]] += spike_train[i]
        occ[ahv_bins[i]] += 1./adv['framerate']
        
    ahv_rates = spikes/occ
    #throw out bins sampled less than 1 sec
    ahv_rates[occ<1] = np.nan
    
    ahv_angles = np.linspace(-402,402,num_bins,endpoint=False)
    
    #return data for plotting later
    cluster_data['ahv_angles'] = ahv_angles
    cluster_data['ahv_rates'] = ahv_rates
    cluster_data['max_ahv'] = max_bin
    
    #note that AHV data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_ahv'] = True
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
    
    isi_width = .5
    
    nbins = int(isi_width * 1000. /adv['bin_size'])
    
    #make a histogram of the isi's
    isi_hist = np.histogram(isi_list,bins=nbins,range=[0,isi_width])
    
    isi_xvals = np.linspace(0,isi_width,nbins,endpoint=False)
        
    #add the ISI hist to cluster data dict
    cluster_data['isi_hist'] = isi_hist
    cluster_data['isi_xvals'] = isi_xvals
    
    #note that ISI data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_isi'] = True
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
    
    
    ac_copy = copy.deepcopy(ac_matrix)
    ac_copy[bins_per_window] = np.max(ac_copy)

    Fs = 200.

    ac = np.zeros(int(Fs))
    for i in range(len(ac)):
        ac[i] = np.sum(ac_copy[5*i:(5*i+5)])
                
    # Fs = 1000./adv['bin_size']
    
    output = np.abs(fft(ac,n=2**16))
    increment = Fs/(2**16)
    theta_range = output[int(5./increment):int(11./increment)]
    top_theta = np.where(theta_range==np.max(theta_range))[0][0]
    theta_power = np.mean(output[int(5./increment+top_theta-1./increment):int(5./increment+top_theta+1./increment)])
    baseline_power = np.mean(output[int(1./increment):int(125./increment)])
    
    theta_index = theta_power/baseline_power
                
    #choose our x-ticks
    x_vals=np.arange(-adv['autocorr_width'],adv['autocorr_width']+float(adv['bin_size'])/1000.,float(adv['bin_size'])/1000.)

    #return data for plotting
    cluster_data['ac_xvals'] = copy.deepcopy(x_vals)
    cluster_data['ac_vals'] = copy.deepcopy(ac_matrix)
    cluster_data['theta_index'] = copy.deepcopy(theta_index)
    
    ''' bonus! '''
    
    if ops['save_all']:
        
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
        
    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_spike_autocorr'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    return cluster_data


#def spatial_info(adv,center_x,center_y,spike_x,spike_y):
#    
#    gr = adv['grid_res']
#    
#    #calculate 2D histogram of spatial occupancies for the rat's path, break 
#    #arena into bins assigned by grid_res parameter
#    occ_hist,xedges,yedges = np.histogram2d(center_x,center_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
#    #calculate 2D histogram of spikes per bin
#    spike_hist,spikexedges,spikeyedges = np.histogram2d(spike_x,spike_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]]) 
#
#    #make sure we're using floats bc integer counts won't work
#    occ_hist = np.asarray(occ_hist,dtype=np.float)
#    spike_hist = np.asarray(spike_hist,dtype=np.float)
#
#    #remove bins where animal never sampled
#    occ_hist[occ_hist==0] = np.nan
#    spike_hist[np.isnan(occ_hist)] = np.nan
#    
#    #calculate a matrix of firing rates
#    fr_mat = spike_hist / occ_hist
#    #calculate the mean firing rate
#    mean_fr = np.nansum(spike_hist) / np.nansum(occ_hist)
#    
#    #total occupancy time
#    tot_occ = np.nansum(occ_hist)
#    #calc a matrix of occupancy probabilities
#    prob_mat = occ_hist / tot_occ
#
#    with warnings.catch_warnings():
#        #log errors are annoying and don't matter since we use nansum
#        warnings.simplefilter("ignore")
#        
#        #calculate the information content
#        info = np.nansum( prob_mat * (fr_mat / mean_fr) * np.log2(fr_mat / mean_fr))
#        
#        #calculate sparsity of the signal
#        sparsity = ( np.nansum(prob_mat * fr_mat)**2 ) / np.nansum(prob_mat * fr_mat**2)
#
#    return info, sparsity
#
#
#def hd_info(adv,angles,spike_angles):
#    
#    hd_bins = adv['hd_bins']
#    
#    angles = np.asarray(angles,dtype=np.float)
#    spike_angles = np.asarray(spike_angles,dtype=np.float)
#    
#    #create histogram of spikes per direction bin specified by hd_bins parameter
#    spike_hist, _ = np.histogram(spike_angles,bins=hd_bins,range=(0,360))
#    #create histogram of frames spent in each bin
#    occ_hist, _ = np.histogram(angles,bins=hd_bins,range=(0,360))
#        
#    #make sure we're using floats because integer counts won't work
#    occ_hist = np.asarray(occ_hist,dtype=np.float)
#    spike_hist = np.asarray(spike_hist,dtype=np.float)
#
#    #take away bins where the rat never sampled
#    occ_hist[occ_hist==0] = np.nan
#    spike_hist[np.isnan(occ_hist)] = np.nan
#    
#    #calculate a matrix of firing rates
#    fr_mat = spike_hist / occ_hist
#    #calculate the mean firing rate
#    mean_fr = np.nansum(spike_hist) / np.nansum(occ_hist)
#    
#    #total occupancy time
#    tot_occ = np.nansum(occ_hist)
#    #calc a matrix of occupancy probabilities
#    prob_mat = occ_hist / tot_occ
#
#    with warnings.catch_warnings():
#        #log errors are annoying and don't matter since we use nansum
#        warnings.simplefilter("ignore")
#        
#        #calculate the information content
#        info = np.nansum( prob_mat * (fr_mat / mean_fr) * np.log2(fr_mat / mean_fr))
#       
#        #calculate sparsity of the signal
#        sparsity = ( np.nansum(prob_mat * fr_mat)**2 ) / np.nansum(prob_mat * fr_mat**2)
#
#    return info, sparsity

def calc_center_ego(adv,trial_data):
        
    #grab appropriate data
    center_x = copy.deepcopy(trial_data['center_x'])
    center_y = copy.deepcopy(trial_data['center_y'])
    angles = copy.deepcopy(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2.
    center_y -= (np.max(center_y) - np.min(center_y))/2.

    center_dists = np.sqrt(center_x**2 + center_y**2)
    center_bearings = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_bearings = (center_bearings-angles)%360
    
    trial_data['center_bearings'] = center_bearings
    trial_data['center_dists'] = center_dists
    
    return trial_data

def plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    #assign =hd_bins to shorter var for easier reading
    hd_bins = adv['hd_bins']
    dist_bin_size = adv['ego_dist_bin_size']
    framerate = adv['framerate']
    
    center_bearings = trial_data['center_bearings']
    center_dists = trial_data['center_dists']
    ani_spikes = spike_data['ani_spikes']

    bearing_bins = np.digitize(center_bearings,np.linspace(0,360,hd_bins,endpoint=False))-1
    
    spikes = np.zeros(hd_bins)
    occ = np.zeros(hd_bins)
    for i in range(len(bearing_bins)):
        spikes[bearing_bins[i]] += ani_spikes[i]
        occ[bearing_bins[i]] += 1./framerate
    bearing_curve = spikes/occ
    
    ref_angles = np.linspace(0,360,hd_bins,endpoint=False)
    real_ref_angles = ref_angles + (ref_angles[1] - ref_angles[0])/2.
    rayleigh, mean_angle = rayleigh_r(real_ref_angles,bearing_curve)
    
    cluster_data['center_bearing_curve'] = bearing_curve
    cluster_data['center_bearing_rayleigh'] = rayleigh
    cluster_data['center_bearing_mean_angle'] = mean_angle
    cluster_data['center_bearing_occ'] = occ
    
    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_center_bearing'] = True
    #send updated data to the guself.canvas_label.setText(self.current_cluster)i
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    xvals = np.arange(0,np.max(center_dists),dist_bin_size)    
    dist_bins = np.digitize(center_dists,xvals)-1
    
    num_bins = len(xvals)
    
    spikes = np.zeros(num_bins)
    occ = np.zeros(num_bins)
    for i in range(len(dist_bins)):
        spikes[dist_bins[i]] += ani_spikes[i]
        occ[dist_bins[i]] += 1./framerate
    dist_curve = spikes/occ
    
    true_xvals = xvals + dist_bin_size/2.
    
#    slope, intercept, r_value,p_value,std_err = linregress(xvals,radial_curve)
#    fit_y = []
#    for i in range(len(xvals)):
#        fit_y.append(slope*xvals[i] + intercept)
    
    cluster_data['center_dist_curve'] = dist_curve
    cluster_data['center_dist_xvals'] = true_xvals
#    cluster_data['center_dist_fit'] = fit_y
#    cluster_data['center_dist_r'] = r_value
#    cluster_data['center_dist_p'] = p_value
    
    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_center_dist'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
    center_ego_spike_angles = []
    center_ego_spike_dists = []
    for i in range(len(center_bearings)):
        for j in range(int(ani_spikes[i])):
            center_ego_spike_angles.append(center_bearings[i])
            center_ego_spike_dists.append(center_dists[i])
            
    spike_data['center_bearings'] = np.array(center_ego_spike_angles)
    spike_data['center_dists'] = np.array(center_ego_spike_dists)
    

    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_center_ego_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
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


def plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self,direction_variable='hd'):

    framerate = adv['framerate']
    ebc_dist_bin_size = adv['ebc_dist_bin_size']
    ebc_bearing_bin_size = adv['ebc_bearing_bin_size']
    
    x_gr = int((np.max(trial_data['center_x']) - np.min(trial_data['center_x'])) / adv['spatial_bin_size'])
    y_gr = int((np.max(trial_data['center_y']) - np.min(trial_data['center_y'])) / adv['spatial_bin_size'])
    
    #grab appropriate data
    center_x = copy.deepcopy(trial_data['center_x'])
    center_y = copy.deepcopy(trial_data['center_y'])
    if direction_variable == 'hd':
        angles = np.asarray(trial_data['angles'])
    else:
        angles = np.array(trial_data['movement_directions'])
    spike_train = np.asarray(spike_data['ani_spikes'])

    if direction_variable == 'md':
        speeds = np.asarray(trial_data['speeds'])
        center_x = center_x[speeds>5]
        center_y = center_y[speeds>5]
        angles = angles[speeds>5]
        spike_train = spike_train[speeds>5]

    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    xcoords = np.linspace(np.min(center_x)-1.,np.max(center_x)+1.,x_gr+1,endpoint=True)
    ycoords = np.linspace(np.min(center_y)-1.,np.max(center_y)+1.,y_gr+1,endpoint=True)
    w1 = np.stack((xcoords,np.repeat(np.max(ycoords)+1.,x_gr+1)))
    w3 = np.stack((xcoords,np.repeat(np.min(ycoords)-1.,x_gr+1)))
    w2 = np.stack((np.repeat(np.max(xcoords)+1.,y_gr+1),ycoords))
    w4 = np.stack((np.repeat(np.min(xcoords)-1.,y_gr+1),ycoords))
    
    all_walls = np.concatenate((w1,w2,w3,w4),axis=1)
    
    wall_x = np.zeros((len(all_walls[0]),len(center_x)))
    wall_y = np.zeros((len(all_walls[1]),len(center_y)))
    
    for i in range(len(center_x)):
        wall_x[:,i] = all_walls[0] - center_x[i]
        wall_y[:,i] = all_walls[1] - center_y[i]
        
    wall_ego_angles = (np.rad2deg(np.arctan2(wall_y,wall_x))%360 - angles)%360
    wall_ego_dists = np.sqrt(wall_x**2 + wall_y**2)
    
    cutoff = np.max((np.max(center_x)-np.min(center_x),np.max(center_y)-np.min(center_y)))/2.

    num_bearing_bins = int(360. / ebc_bearing_bin_size)
    
    ref_angles = np.linspace(0,360,num_bearing_bins,endpoint=False)
    radii = np.arange(0,cutoff,ebc_dist_bin_size)
    dist_bins = np.digitize(wall_ego_dists,radii) - 1
    
    num_dist_bins = len(radii)
    
    occ = np.zeros((num_bearing_bins,num_dist_bins))
    spikes = np.zeros((num_bearing_bins,num_dist_bins))
    
    wall_ego_dists[wall_ego_dists>cutoff] = np.nan
    
    @nb.njit(nogil=True)
    def loop(wall_ego_angles,ref_angles,wall_ego_dists,dist_bins,spikes,occ,spike_train):
        for i in range(len(wall_ego_angles[0])):
            for a in range(len(ref_angles)):
                diffs = np.abs(wall_ego_angles[:,i] - ref_angles[a])
                closest_pt = np.where(diffs==np.min(diffs))[0][0]
                if not np.isnan(wall_ego_dists[closest_pt,i]):# < cutoff:
                    occ[a,dist_bins[closest_pt,i]] += 1./framerate
                    spikes[a,dist_bins[closest_pt,i]] += spike_train[i]
                    
        return spikes, occ
                
    spikes, occ = loop(wall_ego_angles,ref_angles,wall_ego_dists,dist_bins,spikes,occ,spike_train)

    heatmap = spikes/occ
    
    ref_angles = np.deg2rad(ref_angles)
   
    hist3 = np.concatenate((heatmap,heatmap,heatmap),axis=0)
    hist3 = convolve(hist3,Gaussian2DKernel(x_stddev=2,y_stddev=2))
    new_hist = hist3[len(heatmap):len(heatmap)*2]

    real_ref_angles = ref_angles + (ref_angles[1] - ref_angles[0])/2.
    real_radii = radii + (radii[1] - radii[0])/2.
    xvals,yvals = np.meshgrid(real_ref_angles,real_radii)
        
    mr = np.nansum(new_hist.T*np.exp(1j*xvals))/(np.sum(new_hist))
    mrl = np.abs(mr)
    mra = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))

    ref_angles = np.concatenate((ref_angles, 2.*np.pi + ref_angles[0,np.newaxis]))

    angle0 = new_hist[0]
    ebc_hist = np.concatenate((new_hist, angle0[np.newaxis,:]))
    
    cluster_data['ebc_ref_angles'] = ref_angles
    cluster_data['ebc_radii'] = radii
    
    if direction_variable == 'hd':
        cluster_data['ebc_mr'] = mr
        cluster_data['ebc_mrl'] = mrl
        cluster_data['ebc_mra'] = mra
        cluster_data['ebc_hist'] = ebc_hist
        
        #note that data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_ebc'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
        
    elif direction_variable == 'md':
        cluster_data['ebc_md_mr'] = mr
        cluster_data['ebc_md_mrl'] = mrl
        cluster_data['ebc_md_mra'] = mra
        cluster_data['ebc_md_hist'] = ebc_hist
        
        #note that data is ready
        self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_md_ebc'] = True
        #send updated data to the gui
        self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    return cluster_data

def calc_wall_ego(adv,trial_data):
    
    #grab appropriate data
    center_x = copy.deepcopy(trial_data['center_x'])
    center_y = copy.deepcopy(trial_data['center_y'])
    angles = copy.deepcopy(trial_data['angles'])
    
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
    wall_bearings = wall_angles[wall_ids]
    wall_bearings = (wall_bearings - angles)%360
    
    trial_data['wall_bearings'] = wall_bearings
    trial_data['wall_dists'] = wall_dists
    
    return trial_data

def plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    #assign hd_bins to shorter var for easier reading
    hd_bins = adv['hd_bins']
    framerate = adv['framerate']
    dist_bin_size = adv['ego_dist_bin_size']
    
    wall_dists = trial_data['wall_dists']
    wall_bearings = trial_data['wall_bearings']
    ani_spikes = spike_data['ani_spikes']
    
    num_bins = len(np.arange(0,np.max(wall_dists),dist_bin_size))
    
    dist_bins = np.digitize(wall_dists,np.arange(0,np.max(wall_dists),dist_bin_size))-1
    bearing_bins = np.digitize(wall_bearings,np.linspace(0,360,hd_bins,endpoint=False))-1
    
    wall_dist_spikes = np.zeros(num_bins)
    wall_dist_occ = np.zeros(num_bins)
    wall_bearing_spikes = np.zeros(hd_bins)
    wall_bearing_occ = np.zeros(hd_bins)
    
    for i in range(len(wall_dists)):
        wall_dist_spikes[dist_bins[i]] += ani_spikes[i]
        wall_dist_occ[dist_bins[i]] += 1./framerate
        wall_bearing_spikes[bearing_bins[i]] += ani_spikes[i]
        wall_bearing_occ[bearing_bins[i]] += 1./framerate
        
    bearing_curve = wall_bearing_spikes/wall_bearing_occ
    dist_curve = wall_dist_spikes/wall_dist_occ
    
    ref_angles = np.linspace(0,360,hd_bins,endpoint=False)
    real_ref_angles = ref_angles + (ref_angles[1] - ref_angles[0])/2.
    rayleigh, mean_angle = rayleigh_r(real_ref_angles,bearing_curve)
    
    cluster_data['wall_bearing_curve'] = bearing_curve
    cluster_data['wall_bearing_rayleigh'] = rayleigh
    cluster_data['wall_bearing_mean_angle'] = mean_angle

    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_wall_bearing'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    xvals = np.arange(0,np.max(wall_dists),dist_bin_size) + dist_bin_size/2.

#    
#    slope, intercept, r_value,p_value,std_err = linregress(xvals,dist_curve)
#    fit_y = []
#    for i in range(len(xvals)):
#        fit_y.append(slope*xvals[i] + intercept)

    cluster_data['wall_dist_curve'] = dist_curve
    cluster_data['wall_dist_xvals'] = xvals
#    cluster_data['wall_dist_fit'] = fit_y
#    cluster_data['wall_dist_r'] = r_value
#    cluster_data['wall_dist_p'] = p_value
    
    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_wall_dist'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    wall_ego_spike_angles = []
    wall_ego_spike_dists = []
    for i in range(len(wall_bearings)):
        for j in range(int(ani_spikes[i])):
            wall_ego_spike_angles.append(wall_bearings[i])
            wall_ego_spike_dists.append(wall_dists[i])
            
    spike_data['wall_bearings'] = np.array(wall_ego_spike_angles)
    spike_data['wall_dists'] = np.array(wall_ego_spike_dists)
    
    #note that data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_wall_ego_map'] = True
    #send updated data to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    
    return cluster_data


def plot_ego(ops,adv,trial_data,cluster_data,spike_data,self):

    ref_bin_size = adv['ego_ref_bin_size'] 
    
    hd_bins = adv['hd_bins']
    
    #grab appropriate data
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    ani_spikes = np.asarray(spike_data['ani_spikes'])
    
    y_gr = int((np.max(center_y) - np.min(center_y)) / ref_bin_size)
    x_gr = int((np.max(center_x) - np.min(center_x)) / ref_bin_size)

    @nb.njit(nogil=True)
    def ego_loop(center_y,center_x,angles,framerate):
        """ transform allocentric to egocentric angles and assign dwell times """

        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(x_gr)
        ycoords = np.zeros(y_gr)
        
        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(x_gr):
            xcoords[x] = (np.float(x)/np.float(x_gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x)) + np.float(np.max(center_x)-np.min(center_x))/np.float(x_gr)
        for y in range(y_gr):
            ycoords[y] = np.float(np.max(center_y)) - (np.float(y)/np.float(y_gr))*np.float((np.max(center_y)-np.min(center_y))) - np.float(np.max(center_y)-np.min(center_y))/np.float(y_gr)

        #make arrays to hold egocentric angles and combos (allo,ego,spatial) for each video frame,
        #along with dwell times and spike counts for each bin
        ego_dwells = np.zeros((y_gr,x_gr,hd_bins))
        ego_spikes = np.zeros((y_gr,x_gr,hd_bins))
        
        tuning_curves = np.zeros((y_gr,x_gr,hd_bins))
        rayleighs = np.zeros((y_gr,x_gr))
        rxs = np.zeros((y_gr,x_gr))
        rys = np.zeros((y_gr,x_gr))
        mean_angles = np.zeros((y_gr,x_gr))

        #for each y position...
        for i in range(y_gr):

            cue_y = ycoords[i]
    
            #for each x position...
            for j in range(x_gr):
                                        
                cue_x = xcoords[j]
                  
                #calc array of egocentric angles of this bin from pos x axis centered 
                #on animal using arctan
                new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
                #calculate ego angles by subtracting allocentric
                #angles from egocentric angles
                ego_angles = (new_angles-angles)%360

                #assign to bin
                ego_bins = ego_angles/(360/hd_bins)

                weights = np.ones(len(center_x))
                                                
                for k in range(len(center_x)):
                                      
                    if not np.isnan(ego_bins[k]) and not np.isnan(weights[k]):
                        #add one framelength to the dwell time for that ego hd bin
                        ego_dwells[i][j][np.int(ego_bins[k])] += (1./framerate) * weights[k]
                        #add the number of spikes for that frame to the appropriate ego bin
                        ego_spikes[i][j][np.int(ego_bins[k])] += ani_spikes[k] * weights[k]

                rates = ego_spikes[i][j]/ego_dwells[i][j]
    
                tuning_curves[i][j] = rates
                hd_angles = np.linspace(0,360,hd_bins)
                hd_angles = hd_angles + (hd_angles[1] - hd_angles[0])/2.
            
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

        return rayleighs,rxs,rys,mean_angles,tuning_curves
            
    rayleighs,rxs,rys,mean_angles,tuning_curves = ego_loop(center_y,center_x,angles,adv['framerate'])

    cluster_data['ego_curves'] = tuning_curves
    cluster_data['ego_rayleighs'] = rayleighs
    cluster_data['ego_mean_angles'] = mean_angles

    #note that ego rayleigh r plotting data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_ego'] = True
    #send updated data dicts to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)
    
    #note that ego mean angle plotting data is ready
    self.metadata[trial_data['current_trial']][trial_data['current_cluster']]['dataready']['plot_ego_angle'] = True
    #send updated data dicts to the gui
    self.worker.plotting_data.emit(ops,adv,trial_data,cluster_data,spike_data)

    #return the data!
    return cluster_data
