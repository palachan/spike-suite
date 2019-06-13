# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:45:04 2017

@author: Patrick
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend.qt4']='PySide'
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as mplcolors

from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
import numba as nb
import pickle
#import tkFileDialog

import main

def animated_heatmap(ops,adv,trial_data,cluster_data,spike_data,gui):
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    h,xedges,yedges = np.histogram2d(center_x,center_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    
    mid_y = (max(center_y)+min(center_y))/2
    #now flip all the y-coordinates!
    for i in range(len(center_y)):
        center_y[i] = 2 * mid_y - center_y[i]
    
    
    fig,ax = plt.subplots()
    plt.axis('scaled')
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min(center_y),max(center_y)])
    ax.set_xlim([min(center_x),max(center_x)])
    #these lists will contain the sampled positions and spike locations
    sampled_x = []
    sampled_y = []
    sx = []
    sy = []
    
    #assign plots for the heatmap and the red and green led positions
    line2 = plt.imshow(np.zeros((adv['grid_res'],adv['grid_res'])),extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],animated=True)
    liner, = ax.plot([],[],'r.')
    lineg, = ax.plot([],[],'g.')
    
    #initiation function for animation
    def animateinit():
        return liner,lineg,line2,
    
    #main animation function
    def animate(n):
        #print what percent of the video is done
        perc = int(len(center_x)/100)
        done = int(n)/int(perc)
        if int(n)%perc==0:
            print('[video %s percent done]' % done)
        #add current position to list of sampled positions
        sampled_x.append(center_x[n])
        sampled_y.append(center_y[n])
        #if spikes during this frame, add their locations to lists
        if spike_data['ani_spikes'][n] > 0:
            for i in range(spike_data['ani_spikes'][n]):
                sx.append(center_x[n])
                sy.append(center_y[n]) 
        #calculates 2d histogram of occupancies for each bin
        h,xedges,yedges = np.histogram2d(sampled_x,sampled_y,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
        #calculates 2D histogram of spikes per bin
        spikeh,spikexedges,spikeyedges = np.histogram2d(sx,sy,adv['grid_res'],[[min(center_x),max(center_x)],[min(center_y),max(center_y)]]) 
        #creates a heatmap by dividing spikes by occupancy time per bin
        heatmap = spikeh/h
        #corrects axes of heatmap (backwards from histogram function)
        heatmap=np.swapaxes(heatmap,0,1).tolist()[::-1]
        heatmap = np.asarray(heatmap)
        #smooth the heatmap
        smoothed_heatmap = convolve(heatmap, Gaussian2DKernel(stddev=2),boundary='extend')
        #set data for the LEDs and the updated heatmap
        liner.set_data(trial_data['positions'][n][1],trial_data['positions'][n][2])
        lineg.set_data(trial_data['positions'][n][3],trial_data['positions'][n][4])
        line2.set_array(smoothed_heatmap)
        #return plots to the animation function
        return liner,lineg,line2,
    #call FuncAnimation to start the process
    ani = FuncAnimation(fig, animate, init_func = animateinit, frames=len(center_x), interval=0, blit=False)
    #save the video as an mp4 at 2x original speed
    ani.save('%s/heatmap_animation.mp4' % cluster_data['new_folder'],fps=60,dpi=adv['pic_resolution'])
    
def animated_hd_map(ops,adv,trial_data,cluster_data,spike_data,gui):
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    print(np.shape(np.asarray(trial_data['positions'])))
    positions = np.swapaxes(np.asarray(trial_data['positions']),0,1)
    if ops['acq'] == 'neuralynx':
        print(np.shape(positions))
        red_x = positions[1]
        red_y = positions[2]
        mid_red_y = (max(red_y)+min(red_y))/2
        green_x = positions[3]
        green_y = positions[4]
        mid_green_y = (max(green_y)+min(green_y))/2
        #now flip the y coordinates
        for i in range(len(red_y)):
            red_y[i] = 2 * mid_red_y - red_y[i]
            green_y[i] = 2 * mid_green_y - green_y[i]
        
    elif ops['acq'] == 'openephys':
        print(np.shape(positions))
        red_x = positions[2]
        red_y = positions[1]
        mid_red_x = (max(red_x)+min(red_x))/2
        green_x = positions[4]
        green_y = positions[3]
        mid_green_x = (max(green_x)+min(green_x))/2
        for i in range(len(red_y)):
            red_x[i] = 2 * mid_red_x - red_x[i]
            green_x[i] = 2 * mid_green_x - green_x[i]
    
#    if ops['acq'] == 'neuralynx':

    
#    norm = mpl.colors.Normalize(vmin=0, vmax=359)
#    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    #assign a color to each spike in spike_angles
    colors = np.array(spike_data['spike_angles'])
    
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
    
    line1, = plt.plot([],[],color='.7',zorder=0)
#    line2, = plt.plot([],[],'r.')
    line2 = plt.scatter(spike_x,spike_y,c=spike_colors,cmap=colormap,norm=norm)
    liner, = plt.plot([],[],'r*')
    lineg, = plt.plot([],[],'g*')
    
    #initiation function for animation
    def animateinit():

        line1.set_data(path_array)
        line2.set_data(spike_x,spike_y)
        liner.set_data([],[])
        lineg.set_data([],[])

        #assign plots for spike positions, red led, and green led
        return liner,lineg,line2,line1,

    #main animation function
    def animate(n):

        #print what percent of the video is done
        perc = int(len(center_x)/100)
        done = int(n)/int(perc)
        if int(n)%perc==0:
            print(n)
            print('[video %s percent done]' % done)

        #set data for red and green led locations for this frame
        liner.set_data(red_x[n],red_y[n])
        lineg.set_data(green_x[n],green_y[n])
        line1.set_data(center_x[:n],center_y[:n])
        #if there is at least one spike during this frame...
        if spike_data['ani_spikes'][n] > 0:
            #add the position to the position list
            for i in range(spike_data['ani_spikes'][n]):
                spike_x.append(center_x[n])
                spike_y.append(center_y[n])
                spike_colors.append(colors[len(spike_x)])
            #update the offsets (data) for the scatter plot and return it
            line2.set_offsets(np.c_[spike_x,spike_y])
            line2.set_array(np.array(spike_colors))

        #return the led positions
        return liner,lineg,line2,line1,
        
    #call FuncAnimation to start the process
    ani = FuncAnimation(fig, animate, frames=len(center_x),interval=0, blit=False)
    print('%s/hd_map_animation.mp4' % cluster_data['new_folder'])
    ani.save('%s/hd_mapstar_animation.mp4' % cluster_data['new_folder'],fps=90,dpi=300)
    
    #save the video as an mp4 at 2x original speed
    #ani.save('%s/hd_map_animation.mp4' % cluster_data['new_folder'],fps=2*adv['framerate'],dpi=adv['pic_resolution'])
    

def animated_egomap(ops,adv,trial_data,cluster_data,spike_data):
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])

    angles = np.asarray( trial_data['angles'])
    ani_spikes =np.asarray( spike_data['ani_spikes'])
    
    gr = 20
    hd_bins = 30
    framerate = adv['framerate']
    
    
    positions = np.swapaxes(np.asarray(trial_data['positions']),0,1)
    red_x = positions[1]
    red_y = positions[2]
    mid_red_y = (max(red_y)+min(red_y))/2
    green_x = positions[3]
    green_y = positions[4]
    mid_green_y = (max(green_y)+min(green_y))/2
    

    #now flip the y coordinates
    for i in range(len(red_y)):
        red_y[i] = 2 * mid_red_y - red_y[i]
        green_y[i] = 2 * mid_green_y - green_y[i]
    
    
#    
#     
#    
#    mid_y = (max(center_y)+min(center_y))/2
#    #now flip all the y-coordinates!
#    for i in range(len(center_y)):
#        center_y[i] = 2 * mid_y - center_y[i]
        
        
    h,xedges,yedges = np.histogram2d(center_x,center_y,gr,[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])   
        
        
    @nb.autojit(nopython=True)
    def ego_loop(center_y,center_x,angles,ani_spikes,framerate):
        """ transform allocentric to egocentric angles and assign dwell times """

        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(gr)
        ycoords = np.zeros(gr)
        
        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(gr):
            xcoords[x] = (np.float(x)/np.float(gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
            ycoords[x] = np.float(np.max(center_y)) - (np.float(x)/np.float(gr))*np.float((np.max(center_y)-np.min(center_y)))
                        
        #make arrays to hold egocentric angles and combos (allo,ego,spatial) for each video frame,
        #along with dwell times and spike counts for each bin
        ego_bins = np.zeros((gr,gr,len(center_x)),dtype=nb.types.int64)
        ego_dwells = np.zeros((len(center_x),gr,gr,hd_bins))
        ego_spikes = np.zeros((len(center_x),gr,gr,hd_bins))
        
        print('len center x')
        print(len(center_x))
        
        #for each y position...
        for i in range(gr):
    
            #fill an array with the current y coord
            cue_y = np.zeros(len(center_y))
            for l in range(len(cue_y)):
                cue_y[l] = ycoords[i]
    
            #for each x position...
            for j in range(gr):
                                        
                #fill an array with the current x coord
                cue_x = np.zeros(len(center_x))
                for l in range(len(cue_x)):
                    cue_x[l] = xcoords[j]
                  
                #calc array of egocentric angles of this bin from pos x axis centered 
                #on animal using arctan
                new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
                #calculate ecd angles by subtracting allocentric
                #angles from egocentric angles
                ecd_angles = (new_angles-angles)%360
                #assign to bin
                ecd_bins = ecd_angles/(360/hd_bins)
                #add to ego_bins array
                ego_bins[i][j] = ecd_bins
            
        #for every video frame...
        for k in range(len(center_x)):
            #for all egocentric (spatial) bins...
            for i in range(gr):
                for j in range(gr):
                    ego_dwells[k][i][j][np.int(ego_bins[i][j][k])] = 1.
                    ego_spikes[k][i][j][np.int(ego_bins[i][j][k])] += ani_spikes[k]
                    
                    
        return ego_dwells,ego_spikes
    
    @nb.autojit(nopython=True)
    def rayleigh_rs(ego_spikes,ego_dwells):
        
        ego_spikes = np.sum(ego_spikes,axis=0)
        ego_dwells = np.sum(ego_dwells,axis=0)
                        
        rayleighs = np.zeros((gr,gr))
        mean_angles = np.zeros((gr,gr))
        
        for i in range(gr):
            for j in range(gr):
#                ego_dwells[ego_dwells==0] = np.nan
                rates = ego_spikes[i][j]/ego_dwells[i][j]
                
                hd_angles = np.arange(0,360,12)
            
                #start vars for x and y rayleigh components
                rx = 0
                ry = 0
                
                #convert spike angles into x and y coordinates, sum up the results -- 
                #if firing rates are provided along with HD plot edges instead of spike angles,
                #do the same thing but with those
                for m in range(len(hd_angles)):
                    rx += np.cos(np.deg2rad(hd_angles[m]))*rates[m]
                    ry += np.sin(np.deg2rad(hd_angles[m]))*rates[m]
            
                #calculate average x and y values for vector coordinates
                if np.sum(rates) == 0 or np.isnan(np.sum(rates)):
                    rx = ry = 0
                else:
                    rx = rx/np.sum(rates)
                    ry = ry/np.sum(rates)
            
                #calculate vector length
                r = np.sqrt(rx**2 + ry**2)
                
                #calculate the angle the vector points (rayleigh pfd)
                #piecewise because of trig limitations
#                if rx == 0:
#                    mean_angle = 0
#                elif rx > 0:
#                    mean_angle = np.rad2deg(np.arctan(ry/rx))
#                elif rx < 0:
#                    mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
#                    
#                if mean_angle < 0:
#                    mean_angle = mean_angle + 360
                    
                rayleighs[i][j] = r
#                mean_angles[i][j] = mean_angle
        
        #return appropriate arrays
        return rayleighs,mean_angles
    
    
    
    plt.rcParams['image.cmap'] = 'jet'
    fig,ax = plt.subplots()
    plt.axis('scaled')
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min(center_y),max(center_y)])
    ax.set_xlim([min(center_x),max(center_x)])
    #these lists will contain the sampled positions and spike locations
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    
    #assign plots for the heatmap and the red and green led positions
    line2 = ax.imshow(np.zeros((gr,gr)),extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],animated=True)
    liner, = ax.plot([],[],'r.')
    lineg, = ax.plot([],[],'g.')
    
    fig.colorbar(line2, cax=cax, orientation='vertical')
    
    ego_dwells,ego_spikes = ego_loop(center_y, center_x, angles, ani_spikes, framerate)
    
    #initiation function for animation
    def animateinit():
        return liner,lineg,line2,
    
    #main animation function
    def animate(n):
        #print what percent of the video is done
#        perc = int(len(center_x)/100)
#        done = int(n)/int(perc)
#        if int(n)%perc==0:
#            print('[video %s percent done]' % done)

                
#        if n < 120*framerate:
        rayleighs,mean_angles = rayleigh_rs(ego_spikes[:n+1],ego_dwells[:n+1])
            
#        else:
#            rayleighs,mean_angles = rayleigh_rs(ego_spikes[np.int(n-120*framerate):n],ego_dwells[np.int(n-120*framerate):n])
            
        if n%100 == 0:
            print(n)
            print(rayleighs)
            

        #set data for the LEDs and the updated heatmap
        
        liner.set_data(red_x[n],red_y[n])
        lineg.set_data(green_x[n],green_y[n])
        
#        liner.set_data(trial_data['positions'][n][1],trial_data['positions'][n][2])
#        lineg.set_data(trial_data['positions'][n][3],trial_data['positions'][n][4])
        line2.set_clim(vmin=np.min(rayleighs),vmax=np.max(rayleighs))
        line2.set_array(rayleighs)
#        line2.set_data(rayleighs)
        #return plots to the animation function
        return liner,lineg,line2,
    
    #call FuncAnimation to start the process
    ani = FuncAnimation(fig, animate, init_func = animateinit, frames=len(center_x), interval=0, blit=True)
    #save the video as an mp4 at 2x original speed
    ani.save('%s/ego_animation_full.mp4' % cluster_data['new_folder'],fps=60,dpi=adv['pic_resolution'])
    
    
def load_data(fname):
    ''' load pickled numpy arrays '''

    try:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    except:
        print('couldn\'t open data file! try again!!')
    
    return data

def animate_model_ego():
    
    max_n = data['Xe'].shape[0]
    
    def animate(n):
        
        if n%100 == 0:
            print(n)
        
        c=np.zeros((8,8))
        a= np.swapaxes(data['Xe'][n].todense(),0,1).flatten()[0].tolist()[0]
        b=np.reshape(a,(8,8,30))
        for i in range(8):
            
            for j in range(8):
                if np.sum(b[i][j]) > 0:
                    c[i][j] = np.arange(0,360,12)[np.where(b[i][j] > 0)[0]][0]
                else:
                    c[i][j] = np.nan
                    
        im.set_array(c)
        
        return im
    from matplotlib import colors as mplcolors
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
                    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.zeros((8,8)),cmap=colormap,norm=norm,extent=[0,8,0,8])

    ani = FuncAnimation(fig, animate, frames=max_n, interval=0, blit=False)
    ani.save('C:/Users/Jeffrey_Taube/Desktop/ego_animation.mp4',fps=60)
    
    
    
    max_n = len(a)
    
    def animate(n):
        
        if n%100 == 0:
            print(n)
            
        im.set_array(a[n])
        
        return im
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(a[0])
    
    ani = FuncAnimation(fig, animate, frames=max_n, interval=0, blit=False)
    ani.save('C:/Users/Jeffrey_Taube/Desktop/pos_animation.mp4',fps=60)
    
    
    
def animate_decoded(binned_x,binned_y,decoded_x,decoded_y):
    
    
    fig = plt.figure()
    plt.axis('square')
    ax = fig.add_subplot(111)
    
    liner, = ax.plot([],[],'r.',markersize=15)
    lineg, = ax.plot([],[],'g*',markersize=15)
    ax.set_xlim([0,120])
    ax.set_xticks([])
    ax.set_ylim([0,120])
    ax.set_yticks([])

#    ax.axis('equal')
    
    
    def animate(n):

        if n%100 == 0:
            print(n)            

        lineg.set_data(binned_x[n], binned_y[n])
        liner.set_data(decoded_x[n],decoded_y[n])
        
        return liner,lineg,
    
    
    ani = FuncAnimation(fig, animate, frames=len(binned_x), interval=0, blit=False)
    ani.save('C:/Users/Jeffrey_Taube/Desktop/decoded_animation.mp4',fps=20)
    
 
if __name__ == '__main__':

    #ask for the directory we'll be operating in
#    fdir = tkFileDialog.askdirectory(initialdir='G:\mm44\PoS')
    fdir = 'C:/Users/Jeffrey_Taube/Desktop/view cell'

    #figure out what sessions (trials) are in the directory
    ops,trials = main.find_trials({'multi_session':False},fdir)
    #workaround
    ops['labview'] = False
    ops['single_cluster'] = False
    ops['acq'] = 'openephys'
            
    #for every session...
    for trial in trials:
                
        #grab the name of the data file we need
        fname = trial+'/all_trial_data.pickle'
        #load the data file
        all_trial_data = load_data(fname)
        #grab the advanced options and collect the framerate
        adv = all_trial_data['adv']
        framerate=adv['framerate']

        #collect names of the clusters we'll be analyzing
        trial_data = main.read_files(ops,fdir,trial,metadata=True)
        cluster_names = trial_data['filenames']

        #for each cluster...
        for name in cluster_names:

            #report which cluster we're working on                      
            print(name)
            
            #grab appropriate data from the data file
            trial_data = all_trial_data[name][0]['trial_data']
            cluster_data = all_trial_data[name][0]['cluster_data']
            spike_data = all_trial_data[name][0]['spike_data']

            animated_egomap(ops, adv, trial_data, cluster_data, spike_data)