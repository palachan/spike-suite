# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:45:04 2017

@author: Patrick
"""
import numpy as np
import matplotlib as mpl
# mpl.rcParams['backend.qt']='PySide6'
#mpl.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mplcolors
import copy

from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve

def animated_heatmap(ops,adv,trial_data,cluster_data,spike_data):
    
    ani_speed = adv['ani_speed']
    
    center_x = copy.deepcopy(trial_data['center_x'])
    center_y = copy.deepcopy(trial_data['center_y'])
    
    n_frames = int(float(len(center_x)) / float(ani_speed))
    positions = np.swapaxes(np.asarray(copy.deepcopy(trial_data['positions'])),0,1)
    
    bins=[np.arange(np.min(center_x),np.max(center_x),adv['spatial_bin_size']),np.arange(np.min(center_y),np.max(center_y),adv['spatial_bin_size'])]
    
    xbins = np.digitize(center_x, bins[0]) - 1
    ybins = np.digitize(center_y, bins[1]) - 1
    
    x_gr = len(bins[0])
    y_gr = len(bins[1])
    
    stddev = 4. / adv['spatial_bin_size']
    
    h,xedges,yedges = np.histogram2d(center_x,center_y,bins=bins,range=[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    
    center_x = -center_x + np.max(center_x)
    center_y = -center_y + np.max(center_y)
        
    mid_x = (np.max(center_x) - np.min(center_x))/2.
    mid_y = (np.max(center_y) - np.min(center_y))/2.
    
    red_x = positions[1]
    red_y = positions[2]
    green_x = positions[3]
    green_y = positions[4]
  
            
    if adv['speed_cutoff'] > 0:
        red_x = red_x[trial_data['og_speeds'] > adv['speed_cutoff']]
        red_y = red_y[trial_data['og_speeds'] > adv['speed_cutoff']]
        green_x = green_x[trial_data['og_speeds'] > adv['speed_cutoff']]
        green_y = green_y[trial_data['og_speeds'] > adv['speed_cutoff']]
            
    unscaled_center_x = (red_x + green_x)/2.
    unscaled_center_y = (red_y + green_y)/2.
    
    xscale = (np.max(unscaled_center_x) - np.min(unscaled_center_x)) / (np.max(center_x) - np.min(center_x))
    yscale = (np.max(unscaled_center_y) - np.min(unscaled_center_y)) / (np.max(center_y) - np.min(center_y))
    
    red_x = (red_x - np.min(unscaled_center_x)) / xscale
    green_x = (green_x - np.min(unscaled_center_x)) / xscale

    red_y = (red_y - np.min(unscaled_center_y)) / yscale
    green_y = (green_y - np.min(unscaled_center_y)) / yscale
    
    #flip y on LED positions
    if ops['acq'] == 'neuralynx' or ops['acq'] == 'openephys':
        red_y = -red_y + 2. * mid_y
        green_y = -green_y + 2. * mid_y  
    
    fig,ax = plt.subplots()
    plt.axis('scaled')
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min(center_y),max(center_y)])
    ax.set_xlim([min(center_x),max(center_x)])
    ax.axis('off')

    spikes = np.zeros([x_gr,y_gr])
    occ = np.zeros([x_gr,y_gr])
    
    #assign plots for the heatmap and the red and green led positions
    line2 = plt.imshow(np.zeros((x_gr,y_gr)),origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],animated=True)
    liner, = plt.plot([],[],'r*',markersize=13,zorder=2,clip_on=False)
    lineg, = plt.plot([],[],'g*',markersize=13,zorder=2,clip_on=False)
    
    #initiation function for animation
    def animateinit():
        
        liner.set_data(red_x[0],red_y[0])
        lineg.set_data(green_x[0],green_y[0])
        
        return liner,lineg,line2,
    
    #main animation function
    def animate(n):
        #print what percent of the video is done
        perc = int(n_frames/10.)
        done = 10*int(n)/int(perc)
        if int(n)%perc==0:
            print('[animation %s percent done]' % done)
        #update spikes and occ
        for k in range(ani_speed * n, ani_speed * (n+1)):
            occ[xbins[k],ybins[k]] += 1./adv['framerate']
            spikes[xbins[k],ybins[k]] += spike_data['ani_spikes'][k]
        #creates a heatmap by dividing spikes by occupancy time per bin
        heatmap = spikes/occ
        #smooth the heatmap
        smoothed_heatmap = convolve(heatmap, Gaussian2DKernel(x_stddev=stddev,y_stddev=stddev),boundary='extend')
        #set data for the LEDs and the updated heatmap
        liner.set_data(red_x[ani_speed * n],red_y[ani_speed * n])
        lineg.set_data(green_x[ani_speed * n],green_y[ani_speed * n])
        line2.set_array(smoothed_heatmap.T)
        line2.set_clim(vmax=np.nanmax(smoothed_heatmap))
        #return plots to the animation function
        return liner,lineg,line2,
    
    print('Starting heatmap animation')
    #call FuncAnimation to start the process
    ani = FuncAnimation(fig, animate, init_func = animateinit, frames=n_frames, interval=0, blit=True)
    #save the video as an mp4 at 2x original speed
    ani.save('%s/Animated heatmap.mp4' % cluster_data['new_folder'],fps=int(adv['framerate']),dpi=150)
    
    print('Done making animation')
    
    
def animated_path_spike(ops,adv,trial_data,cluster_data,spike_data,spike_hd=False):
    
    ani_speed = adv['ani_speed']
    
    center_x = copy.deepcopy(trial_data['center_x'])
    center_y = copy.deepcopy(trial_data['center_y'])
    angles = copy.deepcopy(trial_data['angles'])
    
    n_frames = int(float(len(center_x)) / float(ani_speed))
    positions = np.swapaxes(np.asarray(copy.deepcopy(trial_data['positions'])),0,1)
    
    mid_x = (np.max(center_x) - np.min(center_x))/2.
    mid_y = (np.max(center_y) - np.min(center_y))/2.
    
    red_x = positions[1]
    red_y = positions[2]
    green_x = positions[3]
    green_y = positions[4]    
            
    if adv['speed_cutoff'] > 0:
        red_x = red_x[trial_data['og_speeds'] > adv['speed_cutoff']]
        red_y = red_y[trial_data['og_speeds'] > adv['speed_cutoff']]
        green_x = green_x[trial_data['og_speeds'] > adv['speed_cutoff']]
        green_y = green_y[trial_data['og_speeds'] > adv['speed_cutoff']]
            
    unscaled_center_x = (red_x + green_x)/2.
    unscaled_center_y = (red_y + green_y)/2.
    
    xscale = (np.max(unscaled_center_x) - np.min(unscaled_center_x)) / (np.max(center_x) - np.min(center_x))
    yscale = (np.max(unscaled_center_y) - np.min(unscaled_center_y)) / (np.max(center_y) - np.min(center_y))
    
    red_x = (red_x - np.min(unscaled_center_x)) / xscale
    green_x = (green_x - np.min(unscaled_center_x)) / xscale

    red_y = (red_y - np.min(unscaled_center_y)) / yscale
    green_y = (green_y - np.min(unscaled_center_y)) / yscale
    
    #flip y on LED positions
    if ops['acq'] == 'neuralynx' or ops['acq'] == 'openephys':
        red_y = -red_y + 2. * mid_y
        green_y = -green_y + 2. * mid_y  

    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    #assign a color to each spike in spike_angles
    if not spike_hd:
        angles = np.zeros_like(angles)
        
    #set up the plots and make sure they're scaled correctly
    fig = plt.figure()
    #this list will contain spike positions
    spike_x = []
    spike_y = []
    spike_colors = []
    path_array = []
    
    axes = plt.gca()
    plt.axis('scaled')
    plt.yticks([])
    plt.xticks([])
    axes.set_ylim([min(center_y),max(center_y)])
    axes.set_xlim([min(center_x),max(center_x)])
    axes.axis('off')
    
    line1, = plt.plot([],[],color='.7',zorder=0,clip_on=False)
    line2 = plt.scatter(spike_x,spike_y,c=spike_colors,cmap=colormap,norm=norm,zorder=1,clip_on=False)
    liner, = plt.plot([],[],'r*',markersize=13,zorder=2,clip_on=False)
    lineg, = plt.plot([],[],'g*',markersize=13,zorder=2,clip_on=False)
    
    #initiation function for animation
    def animateinit():

        line1.set_data(path_array)
        line2.set_data(spike_x,spike_y)
        liner.set_data(red_x[0],red_y[0])
        lineg.set_data(green_x[0],green_y[0])

        #assign plots for spike positions, red led, and green led
        return liner,lineg,line2,line1,

    #main animation function
    def animate(n):

        #print what percent of the video is done
        perc = n_frames/10.
        done = int(10*n/perc)
        if int(n)%int(perc)==0:
            print('[animation %s percent done]' % done)

        #set data for red and green led locations for this frame
        liner.set_data(red_x[ani_speed * n],red_y[ani_speed * n])
        lineg.set_data(green_x[ani_speed * n],green_y[ani_speed * n])
        line1.set_data(center_x[:ani_speed * n],center_y[:ani_speed * n])
        #if there is at least one spike during this frame...            
        for k in range(ani_speed * n, ani_speed * (n+1)):
            #add the position to the position list
            for i in range(spike_data['ani_spikes'][k]):
                spike_x.append(center_x[k])
                spike_y.append(center_y[k])
                spike_colors.append(angles[k])
            #update the offsets (data) for the scatter plot and return it
            line2.set_offsets(np.c_[spike_x,spike_y])
            line2.set_array(np.array(spike_colors))

        #return the led positions
        return liner,lineg,line2,line1,
        
    if spike_hd:
        print('Starting path & spike x hd animation')
    else:
        print('Starting path & spike animation')
    
    #call FuncAnimation to start the process
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=0, blit=True)
    if spike_hd:
        ani.save('%s/Animated spike location x HD.mp4' % cluster_data['new_folder'],fps=int(adv['framerate']),dpi=150)
    else:
        ani.save('%s/Animated path & spike.mp4' % cluster_data['new_folder'],fps=int(adv['framerate']),dpi=150)
        
    print('Done making animation')

    
    #save the video as an mp4 at Xx original speed
