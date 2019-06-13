# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:12:57 2018

@author: Jeffrey_Taube
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve

import collect_data

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

def run_throughs():
    
    base = 'H:/Patrick/PL16/V2L'
    trial = '2017-03-15_16-15-36'
    cells = ['TT4_SS_01']
    
    tracking_fdir = base + '/' + trial
    
    trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)
    
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    
    minx = np.min(center_x) + .3*(np.max(center_x)-np.min(center_x))
    maxx = np.max(center_x) - .3*(np.max(center_x)-np.min(center_x))
    miny = np.min(center_y) + .3*(np.max(center_y)-np.min(center_y))
    maxy = np.max(center_y) - .3*(np.max(center_y)-np.min(center_y))
    
    plt.figure()
    plt.plot(center_x,center_y)
    plt.vlines(minx,np.min(center_y),np.max(center_y),'r')
    plt.vlines(maxx,np.min(center_y),np.max(center_y),'r')
    plt.hlines(miny,np.min(center_x),np.max(center_x),'r')
    plt.hlines(maxy,np.min(center_x),np.max(center_x),'r')
    plt.show()

def radial_modulation():
    
    trial_dict = {}
    heatmaps = []
    
    base = 'H:/Patrick/PL16/V2L'
    
    trial = '2017-03-17_14-55-12'
    cells = ['TT4_SS_01']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-16_16-48-44 s1 box'
    cells = ['TT2_SS_02']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-16_14-26-25'
    cells = ['TT2_SS_01','TT2_SS_02']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-15_16-15-36'
    cells = ['TT2_SS_05','TT4_SS_03']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-13_16-35-19'
    cells = ['TT4_SS_01']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-20_11-31-56'
    cells = ['TT3_SS_02']
    
    trial_dict[trial] = cells
    
    trial = '2017-06-26_16-07-58'
    cells = ['TT2_SS_01','TT2_SS_02']
    
    for trial in trial_dict:
        
        tracking_fdir = base + '/' + trial
        
        #collect names of the clusters we'll be analyzing
        trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)
        
        for cell in trial_dict[trial]:
            fname = tracking_fdir+ '/' + cell + '.txt'
            cluster_data = {}
            spike_list = collect_data.ts_file_reader(fname)
            spike_train = collect_data.create_spike_train(trial_data,spike_list)
    
            center_y = np.array(trial_data['center_y'])
            
            center_x = np.array(trial_data['center_x'])
            
            xbins = np.digitize(center_x,np.arange(min(center_x),max(center_x),(max(center_x)-min(center_x))/64.)) - 1
            ybins = np.digitize(center_y,np.arange(min(center_y),max(center_y),(max(center_y)-min(center_y))/64.)) - 1
                        
            occ = np.zeros((64,64))
            spikes = np.zeros((64,64))

            
            for i in range(len(spike_train)):
                occ[xbins[i]][ybins[i]] += 1.
                spikes[xbins[i]][ybins[i]] += spike_train[i]
                
            raw_heatmap = spikes/occ
            
            interpd_heatmap = interp_raw_heatmap(raw_heatmap)
            #create a smoothed heatmap using convolution with a Gaussian kernel
            smoothed_heatmap = convolve(interpd_heatmap, Gaussian2DKernel(stddev=2))
        
            heatmaps.append(smoothed_heatmap)
            
    avg_heatmap = np.zeros((64,64))
    for hmap in heatmaps:
        avg_heatmap += hmap
        
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
        
    x_vals = np.arange(64) - 32
    y_vals = np.arange(64) - 32
    
    x,y = np.meshgrid(x_vals,y_vals)
    
    ax.plot_trisurf(x.flatten(),y.flatten(),avg_heatmap.flatten(),cmap=plt.cm.jet)
    plt.show()
    
    
    plt.figure()
    plt.imshow(avg_heatmap)
    plt.set_cmap('viridis')
    plt.show()
    

    
    rho_vals = np.zeros((64,64))
    phi_vals = np.zeros((64,64))
    curve = np.zeros(30)
    curve_occ = np.zeros(30)
    
    for i in range(64):
        for j in range(64):
            rho_vals[i][j] = np.sqrt(x_vals[i]**2 + y_vals[j]**2)
            phi_vals[i][j] = np.rad2deg(np.arctan2(y_vals[j],x_vals[i]))%360
            
    phi_bins = np.digitize(phi_vals,np.arange(np.min(phi_vals),np.max(phi_vals),(np.max(phi_vals)-np.min(phi_vals))/30.)) - 1
    rho_bins = np.digitize(rho_vals,np.arange(np.min(rho_vals),np.max(rho_vals),(np.max(rho_vals)-np.min(rho_vals))/30.)) - 1





    directions = np.arange(0,360,12)
    for d in directions:
        inds = np.where((phi_vals>d) & (phi_vals<(d+6)))
        for i in range(len(inds[0])):
            curve[int(rho_bins[inds[0][i]][inds[1][i]])] += avg_heatmap[inds[0][i]][inds[1][i]]
            curve_occ[int(rho_bins[inds[0][i]][inds[1][i]])] += 1.
            
            
if __name__ == '__main__':      
    trial_dict = {}
    heatmaps = []
    
    base = 'H:/Patrick/PL16/V2L'
    
    trial = '2017-03-16_16-48-44 s1 box'
    cells = ['TT2_SS_02','TT2_SS_04']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-16_14-26-25'
    cells = ['TT2_SS_02']
    
    trial_dict[trial] = cells
    
    trial = '2017-03-17_14-55-12'
    cells = ['TT4_SS_01']
    
    trial_dict[trial] = cells
    
    for trial in trial_dict:
        
        tracking_fdir = base + '/' + trial
        
        #collect names of the clusters we'll be analyzing
        trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)
        
        for cell in trial_dict[trial]:
            fname = tracking_fdir+ '/' + cell + '.txt'
            cluster_data = {}
            spike_list = collect_data.ts_file_reader(fname)
            spike_train = collect_data.create_spike_train(trial_data,spike_list)
    
            center_y = np.array(trial_data['center_y'])
            
            center_x = np.array(trial_data['center_x'])
            
            xbins = np.digitize(center_x,np.arange(min(center_x),max(center_x),(max(center_x)-min(center_x))/64.)) - 1
            ybins = np.digitize(center_y,np.arange(min(center_y),max(center_y),(max(center_y)-min(center_y))/64.)) - 1
                        
            occ = np.zeros((64,64))
            spikes = np.zeros((64,64))
    
            
            for i in range(len(spike_train)):
                occ[xbins[i]][ybins[i]] += 1.
                spikes[xbins[i]][ybins[i]] += spike_train[i]
                
            raw_heatmap = spikes/occ
            
            interpd_heatmap = interp_raw_heatmap(raw_heatmap)
            #create a smoothed heatmap using convolution with a Gaussian kernel
            smoothed_heatmap = convolve(interpd_heatmap, Gaussian2DKernel(stddev=2))
        
            heatmaps.append(smoothed_heatmap)
            
    avg_heatmap = np.ones((64,64))
    avg_heatmap += 2*heatmaps[0]
    avg_heatmap += 2*heatmaps[1]
    avg_heatmap -= 2*heatmaps[2]
    avg_heatmap += heatmaps[3]
    
    plt.imshow(avg_heatmap)
    
