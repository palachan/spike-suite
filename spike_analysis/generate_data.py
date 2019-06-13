# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:10:13 2017

makes random cells by randomly sampling tracking frames from real data

@author: Patrick
"""

import main
import spatial
import tkFileDialog
import bisect
import numpy.random as rand
import numpy as np
import os
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
from skimage.transform import rotate
import copy

hd_bins = 60
random = 0
head_direction = 3
grid = 3
theta = 0
conjunctive = 0
conjunctive_properties = ['head_direction','grid','theta']


ops={}
adv={}
ops['labview'] = False
ops['load_data'] = False
ops['single_cluster'] = False
adv['framerate'] = 30.
adv['bin_size'] = 1
adv['grid_res'] = 64

def init_data():
    ''' get things started '''
    
    #make a list of the cells we need to make
    cell_list = []
    cell_list += ['random']*random
    cell_list += ['grid']*grid
    cell_list += ['head_direction']*head_direction
    cell_list += ['theta']*theta
    cell_list += ['conjunctive']*conjunctive
    
    #return it
    return cell_list

def make_random_cells(trial_data):
    ''' create random timestamps '''
    
    #grab the first and last tracking timestamps
    first = trial_data['timestamps'][0]
    last = trial_data['timestamps'][len(trial_data['timestamps'])-1]
    
    #total number of spikes is random choice (default between 800 and 5000)
    spike_number = rand.choice(range(2000,8000))

    #choose random numbers between the first and last timestamps
    random_timestamps = rand.randint(int(first),int(last),size=spike_number)
    #sort the timestamps in ascending order
    random_timestamps = np.sort(random_timestamps)
        
    #return the timestamps
    return random_timestamps

def partition_speeds(trial_data):
    
    speeds = trial_data['speeds']
    
    #choose how many bins to divide speed into for modifying theta power
    #(default = 10)
    power_bins = 10
    #find the bin edges
    speed_bins = np.percentile(speeds, np.arange(0, 100, power_bins))
    #make an array to match speeds with theta powers
    speed_powers = np.zeros(len(speeds),dtype=np.int)
    #assign a power bin to each speed in the session
    for i in range(len(speeds)):
        speed_powers[i] = bisect.bisect_right(speed_bins,speeds[i])-1
        
    trial_data['speed_powers'] = speed_powers
    
    return trial_data

    
def make_theta_cells(adv,trial_data,prob_dict,speed_coupled=True):
    ''' create randomized timestamps modulated by theta oscillations '''
    
    #create a dict to hold probability waveforms for each theta frequency
    waveforms = {}

    #randomly choose a theta frequency to work with (default 6-8 hz)
    freq = rand.choice([6,7,8])
    #dist between peaks in frames is ~equal to how many times frequency fits
    #into framerate
    wavelength = adv['framerate']/float(freq)
    
    #start a list for probabilities
    waveform = []
    
    #find point for the peak of sine wave (half of wavelength variable)
    peak = float(wavelength)/2.
    #for each frame in wavelength...
    for i in range(int(wavelength)):
        #if on the upslope, add sine of current frame over peak point location
        if i < peak:
            waveform.append(np.sin((np.pi/2.)*(float(i)/peak)))
        #if on downslope, add 1 - sine of (current frame minus peak location over peak location)
        else:
            waveform.append(np.sin((np.pi/2.)*(1.-(float(i)-peak)/peak)))
    
    #if we're correlating theta power with speed
    if speed_coupled:
        #grab our speed data and partitioned speeds 
        speeds = trial_data['speeds']
        speed_powers = trial_data['speed_powers']
        #for every theta power partition...
        for j in range(max(speed_powers)+1):
            #make a copy of the base waveform
            power_wave = copy.deepcopy(waveform)
            #for each point in the waveform...
            for i in range(int(wavelength)):
                #scale the waveform by the theta power (this brings all values closer
                #to zero -- not exactly what we want but a start)
                power_wave[i] = waveform[i]*float(j)/float(max(speed_powers))
            #take the average of the waveform
            avg = np.mean(power_wave)
            #for each point in the now reduced waveform...
            for i in range(int(wavelength)):
                #add a constant that brings the average value up to 0.5 (because
                #we want to reduce the range of probabilities in the wave without 
                #reducing the average probability, which should be around 0.5)
                power_wave[i] += (0.5-avg)
                #we can't have negative probabilities, so set negs to 0
                if power_wave[i] < 0:
                    power_wave[i] = 0
            #assign the wave to the waveforms dictionary named by the speed
            #partition (theta power bin) it belongs to
            waveforms[str(int(j))] = power_wave
            
        #start a list to hold our waves for the whole session
        wave_list = []
        #until we run out of speeds to base our theta power on...
        while len(wave_list) < len(speeds):
            #append waveforms to the list based on the current speed bin
            wave_list += waveforms[str(speed_powers[len(wave_list)])]

    else:
        #otherwise, just repeat the same waveform for the whole session
        wave_list = []
        while len(wave_list) < len(trial_data['timestamps']):
            wave_list += waveform

    #make an array for our official probabilities
    theta_probs = np.zeros(len(speeds))
    #for each frame, add the corresponding value from wave_list
    for i in range(len(theta_probs)):
        theta_probs[i] = wave_list[i]
        
    #divide probs by sum to make add to 1
    theta_probs = theta_probs/np.full(len(theta_probs),np.sum(theta_probs))
        
    #assign to prob_dict
    prob_dict['theta'] = theta_probs
    
    #return it!
    return prob_dict
            
def make_hd_cells(adv,trial_data,prob_dict): 
    ''' create randomized timestamps modulated (or tuned) by head direction '''

    #find bin edges for head direction tuning curve
    head_directions = np.arange(0,361,360/hd_bins)
    
    #choose a stdev (width) for the tuning curve from the normal distribution
    #(default centered on 0.05 with stdev 0.2)
    stdev =  rand.normal(0.05,0.2)
    #if the stdev ends up negative, make it positive
    if stdev < 0: stdev = -stdev
    #randomly choose values from the normal distribution centered on 0 with 
    #stdev calculated above and size spike_number
    norm_vals = rand.normal(0,stdev,size=1000)
    
    #width_modifier determines how far out from the  max and min of norm_vals we'll
    #look when we create a histogram -- otherwise the tuning curve will just be 
    #a bell curve that spans the entire head direction spectrum
    width_modifier = rand.normal(0.5,0.25)
    #make it positive if negative
    if width_modifier < 0: width_modifier = -width_modifier
    
    #create a histogram of the norm_vals binned into hd_bins -- this is the basis
    #for the tuning curve we want
    norm_hist = np.histogram(norm_vals,hd_bins,range=[min(norm_vals)-width_modifier,max(norm_vals)+width_modifier])
    
    #choose an amount to shift the curve by (so the pfd isn't always in the center)
    curve_shifter = rand.choice(range(360))

    #grab our angles from the tracking data
    angles = trial_data['angles']
    #for each angle in the session...
    for i in range(len(angles)):
        #shift it by the amount we just decided
        angles[i] += curve_shifter
        #keep it between 0 and 360
        if angles[i] >= 360:
            angles[i] -= 360
            
    #make an array for probabilities
    hd_probs = np.zeros(len(angles))
   
    #for every angle in the session...
    for i in range(len(angles)):
        #find the HD bin the angle belongs to
        hd_bin = bisect.bisect_left(head_directions, angles[i])-1
        #assign the corresponding norm_hist value to the probability array
        hd_probs[i] = norm_hist[0][hd_bin]
        
    #divide probability array by its sum to make the probabilities add to 1
    hd_probs = hd_probs/np.full(len(hd_probs),np.sum(hd_probs))
    
    prob_dict['head_direction'] = hd_probs
    
    return prob_dict
        
def make_grid_cells(adv,trial_data,prob_dict):
    ''' create randomized timestamps that create a grid pattern '''

    #grab data we need from data dicts
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    gr = adv['grid_res']
    
    #start an enormous array for making a huge grid pattern
    base_grid = np.zeros([5*gr,5*gr])
    #choose the grid spacing randomly between 4 choices (default 15,20,30,40)
    distance = rand.choice([15,20,30,40])

    #start a list for keeping track of node locations         
    nodes = []
    #set x and y equal to 0 to make the first node
    x=0
    y=0
    #helpful counting helpers
    x_start=0
    x_counter = 0
    y_counter = 1
    
    while True:
        #set x coordinate equal to x_counter * node spacing
        x = x_start + distance*x_counter
        #if we're beyond the x limit of the array...
        if x >= 5*gr:
            #bring x back to zero
            x_counter = 0
            #if we're on an odd sweep along the x axis...
            if y_counter % 2 != 0:
                #start the next sweep at an offset (half of node spacing)
                x_start = int(np.ceil(distance/2))
            else:
                #otherwise, start next sweep at zero
                x_start=0
            #trig to set y coordinate based on y_counter
            y = int(np.ceil(y_counter*np.sqrt(3)*distance/2))
            #increment the y_counter
            y_counter += 1
            #if we've exceeded the y limit of the array...
            if y >= 5*gr:
                #break the loop
                break
        else:
            #otherwise, add the node coordinates to the list
            nodes.append([x,y])
            #increment the x_counter to start again
            x_counter += 1
            
    #for each node...
    for node in nodes:
        #set the corresponding base_grid entry equal to 1
        base_grid[node[0],node[1]] = 1
        
    #choose a random angle to rotate the grid pattern
    rotation = rand.choice(range(360))

    #rotate the grid pattern (this will leave empty space at the corners)
    base_grid = rotate(base_grid,rotation,preserve_range=True)
    
    #choose a standard deviation for gaussian smoothing based on node spacing
    gauss_stddev = np.ceil(2.+float(distance)/30.)
    
    #smooth the array using a gaussian kernel -- this leaves us with perfect
    #circular grid nodes
    base_grid = convolve(base_grid, Gaussian2DKernel(stddev=gauss_stddev))
    
    #choose a random spot in the center of the enormous grid pattern to cut
    #out our regular sized arena (so we don't have rotation corner issues)
    start_x = rand.choice(range(2*gr,int(2.5*gr)))
    start_y = rand.choice(range(2*gr,int(2.5*gr)))
    #assign that section of the arena to a new array
    spike_coords = base_grid[start_x:(start_x+gr),start_y:(start_y+gr)]   

    #start an array for timestamp probabilities
    grid_probs = np.zeros(len(center_x))
    
    x_bins = np.digitize(center_x,np.linspace(np.min(center_x),np.max(center_x),gr,endpoint=False))-1
    y_bins = np.digitize(center_x,np.linspace(np.min(center_y),np.max(center_y),gr,endpoint=False))-1
    #for each frame...
    for i in range(len(center_x)):
#        #figure out which spatial bin the animal's position falls in
#        x_bin = bisect.bisect_left(xedges, center_x[i])-1
#        y_bin = gr-1-(bisect.bisect_left(yedges, center_y[i])-1)
        #assign the corresponding value from spike_coords array to
        #grid_probs array
        grid_probs[i] = spike_coords[y_bins[i],x_bins[i]]
                
    #divide probability array by its sum so probabilities add up to 1
    grid_probs = grid_probs/np.full(len(grid_probs),np.sum(grid_probs))
    
    #assign to probability dict
    prob_dict['grid'] = grid_probs
    
    #return it!
    return prob_dict

def join_conjunctive(trial_data,prob_dict):
    ''' combine probabiity arrays into conjunctive array '''
    
    #make a list of ones for multiplying against
    conjunctive_probs = np.ones(len(trial_data['timestamps']))
    #for each property specified...
    for prop in conjunctive_properties:
        #multiply the new prob array by prob array for each property
        conjunctive_probs = conjunctive_probs * prob_dict[prop]
    
    #divide array by its sum so it adds up to 1
    conjunctive_probs = conjunctive_probs/np.full(len(conjunctive_probs),np.sum(conjunctive_probs))
      
    #assign to prob_dict
    prob_dict['conjunctive'] = conjunctive_probs
    
    #return the probabilities
    return prob_dict

def choose_timestamps(trial_data,prob_dict,celltype):

    #choose a 'random' number of cells based on cell type
    if celltype == 'grid':
        spike_number = rand.choice(range(800,5000))
    elif celltype == 'head_direction':
        spike_number = 200
        #spike_number = rand.choice(range(2000,10000))
    elif celltype == 'theta':
        spike_number = rand.choice(range(5000,20000))
    elif celltype == 'conjunctive':
        spike_number = rand.choice(range(2000,10000))
    
    #'randomly' choose timestamps according to probability array (and spike_number)
    spike_timestamps = rand.choice(trial_data['timestamps'],size=spike_number,replace=True,p=prob_dict[celltype])
   
    #jitter the timestamps a little
    for i in range(len(spike_timestamps)):
        spike_timestamps[i] = int(rand.normal(spike_timestamps[i],10000))
        
    #make it into a list so we can append noise timestamps
    spike_timestamps = spike_timestamps.tolist()
    
    #return the list!
    return spike_timestamps

def add_noise(spike_timestamps,celltype):
    
    if celltype == 'grid':
        avg_noise_level = 0.2
    elif celltype == 'head_direction':
        avg_noise_level = 10
        #avg_noise_level = 10
    elif celltype == 'theta':
        avg_noise_level = 0
    elif celltype == 'conjunctive':
        avg_noise_level = 0.2
    
    #figure out how many true tuned spikes there are
    spike_number = len(spike_timestamps)
    #choose how much noise we want based on avg_noise_level
    noise_modifier = avg_noise_level*(1+rand.random_sample()*2)
    #figure out how many noise timestamps we want
    noise_level = int(rand.random_sample()*spike_number*noise_modifier)
    
    #randomly choose times between the first and last timestamps
    noise_timestamps = rand.randint(trial_data['timestamps'][0],trial_data['timestamps'][len(trial_data['timestamps'])-1],size=noise_level)
    #this is necessary for some reason
    noise_timestamps = np.sort(noise_timestamps).tolist()
    
    #add noise timestamps to spike timestamps
    spike_timestamps += noise_timestamps
    
    #return the data!
    return spike_timestamps
    
def save_spikefile(spike_timestamps,fdir,num,cell_type):
    ''' save the timestamps as a .txt file '''
    
    #make it an array
    spike_timestamps = np.asarray(spike_timestamps)
    #sort the timestamps
    spike_timestamps = np.sort(spike_timestamps)
    #back to list for isi stuff
    spike_timestamps = spike_timestamps.tolist()
    
    #throw out impossible ISIs
    false_stamps = np.zeros(len(spike_timestamps))
    for i in range(1,len(spike_timestamps)):
        #calculate the isi between each spike
        isi = spike_timestamps[i] - spike_timestamps[i-1]
        #if the isi is less than 1.5 ms...
        if isi < 1500:
            #set the second value equal to zero
            false_stamps[i] = 1
            
    for i in range(len(spike_timestamps)):
        if false_stamps[i] == 1:
            spike_timestamps[i] = 0
            
    spike_timestamps = [time for time in spike_timestamps if time != 0]
            
    #looks cooler in filenames
    if cell_type == 'head_direction':
        cell_type = 'HD'
    
    #create a file named after cell type and iteration number
    ts_file = open(fdir+'/TT'+str(cell_type)+str(num)+'.txt', 'w')
    #write the file
    for ts in spike_timestamps:
        ts_file.write("%s\n" % ts)
        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
if __name__ == '__main__':
    
    #ask for a video file
    vidfile = tkFileDialog.askopenfilename()
    #grab the directory of the video file
    fdir = os.path.split(vidfile)[0]
    
    #figure out which cells we need to make
    cell_list = init_data()
    
#    #collect relevant tracking data
#    trial_data = main.tracking_stuff(ops,adv,fdir,fdir)
    
    ops['acq'] = 'openephys'
    #read relevant files to extract raw video data/multi-cam info
    trial_data,raw_vdata,ops = main.read_files(ops,fdir,fdir)

    print('processing tracking data for this session...')
    #interpolate nondetects using extracted angles and coordinates
    trial_data = spatial.interp_points(raw_vdata,trial_data)
    #calculate centerpoints and timestamp vectors
    trial_data = spatial.center_points(ops,adv,trial_data)
    
    #if we're making theta cells, collect speeds and separate them into bins
    if 'theta' in cell_list or ('conjunctive' in cell_list and 'theta' in conjunctive_properties):
        print('processing speed data for theta')
        trial_data = spatial.calc_speed(adv,trial_data)
        trial_data = partition_speeds(trial_data)
    
    #start a counter for naming successive files of each celltype
    celltype_tracker = 0
    
    #for each cell...
    for i in range(len(cell_list)):
        
        #make a dictionary for holding probabilities
        prob_dict = {}
        
        #if this is the same celltype we made last time, tick the celltype_tracker
        if i > 0 and cell_list[i] == cell_list[i - 1]:
            celltype_tracker += 1
        #otherwise, reset it to zero
        elif i > 0 and cell_list[i] != cell_list[i - 1]:
            celltype_tracker = 0
            
        #easier way to keep track of the current celltype
        celltype = cell_list[i]
            
        #print our progress
        print('making %s cell %d' % (celltype,celltype_tracker))

        #if we're not making random cells right now...
        if celltype != 'random':
            #if grid cell, calc grid cell probabilities
            if celltype == 'grid':
                prob_dict = make_grid_cells(adv,trial_data,prob_dict)
            #if head direction, calc head direction probabilities
            elif celltype == 'head_direction':
                prob_dict = make_hd_cells(adv,trial_data,prob_dict)
            #if theta, calc theta cell probabilities
            elif celltype == 'theta':
                prob_dict = make_theta_cells(adv,trial_data,prob_dict)
            #if conjunctive, figure out probabilities for each property we want to represent
            elif celltype == 'conjunctive':
                if 'grid' in conjunctive_properties:
                    prob_dict = make_grid_cells(adv,trial_data,prob_dict)
                if 'head_direction' in conjunctive_properties:
                    prob_dict = make_hd_cells(adv,trial_data,prob_dict)
                if 'theta' in conjunctive_properties:
                    prob_dict = make_theta_cells(adv,trial_data,prob_dict)
                #join the probabilities into one (conjunctive) probability array
                prob_dict = join_conjunctive(trial_data,prob_dict)
            
            #select our timestamps for the spike file
            spike_timestamps = choose_timestamps(trial_data,prob_dict,celltype)
            
            #add some noise
            spike_timestamps = add_noise(spike_timestamps,celltype)
        
        else:
            #otherwise, make random cells
            spike_timestamps = make_random_cells(trial_data)
            
        #save the file!
        save_spikefile(spike_timestamps,fdir,celltype_tracker,celltype)