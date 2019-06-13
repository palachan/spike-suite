# -*- coding: utf-8 -*-
"""
Created on Sun Oct 07 13:43:55 2018

egocentric functions for spike_analysis

@author: Patrick
"""

import os
import warnings
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as mplcolors
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
from spatial import interp_raw_heatmap

def plot_conjunctive(ops,adv,trial_data,cluster_data,spike_data):
    
    angles = np.array(trial_data['angles'])
    ego_angles = np.array(trial_data['center_ego_angles'])
    spike_train = np.array(spike_data['ani_spikes'])
    
    bins = 30
    
    angle_bins = np.digitize(angles,np.linspace(0,360,bins,endpoint=False)) - 1
    ego_bins = np.digitize(ego_angles,np.linspace(0,360,bins,endpoint=False)) - 1
    
    spikes = np.zeros((bins,bins))
    occ = np.zeros((bins,bins))
    
    for i in range(len(spike_train)):
        occ[angle_bins[i],ego_bins[i]] += 1./adv['framerate']
        spikes[angle_bins[i],ego_bins[i]] += spike_train[i]
        
    occ[occ<5./adv['framerate']] = 0
    occ[occ==0] = np.nan
    heatmap = spikes/occ
    
    heatmap = interp_raw_heatmap(heatmap)
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(heatmap) 
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    fig.savefig('%s/conjunctive angles.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
    
    plt.close()


def plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.93)
    #set y limit according to highest firing rate
    ymax = int(1.5*np.nanmax(cluster_data['center_ego_curve']))+10
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    ax.set_xlabel('egocentric bearing (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,45))
    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % cluster_data['center_ego_rayleigh'],transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s' % cluster_data['center_ego_mean_angle'],transform=ax.transAxes)
    ax.text(.1,.7,'fit r^2 = %s' % cluster_data['center_ego_r']**2,transform=ax.transAxes)
    ax.text(.1,.6,'fit p = %s' % cluster_data['center_ego_p'],transform=ax.transAxes)


    egopath = cluster_data['new_folder']+'/egocentric'
    if not os.path.isdir(egopath):
        os.makedirs(egopath)

    ax.plot(np.arange(0,360,360/adv['hd_bins']),cluster_data['center_ego_curve'],'k-',np.arange(0,360,360/adv['hd_bins']),cluster_data['center_ego_fit'],'r--') #,cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],max(cluster_data['hd_rates']),'r*',cluster_data['hd_angles'],cluster_data['gauss_rates'],'r--')
    if ops['save_all'] and not cluster_data['saved']['plot_center_ego']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Center Ego Plot trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Center Ego Plot.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_ego'] = True
        
    plt.close()
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    angles = np.linspace(0,2*np.pi,adv['hd_bins'],endpoint=False)
    angles = np.append(angles,2*np.pi)
    curve = np.append(cluster_data['center_ego_curve'],cluster_data['center_ego_curve'][0])
    ax.text(.8,.9,'%s Hz' % str(round(np.nanmax(curve),1)),transform=ax.transAxes,fontsize=14)
    ax.yaxis.grid(False)
    ax.xaxis.grid(linewidth=2,color='k')
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.))
    ax.set_xticklabels([0,90,180,270],fontsize=12)
    ax.set_yticklabels([])
    ax.set_theta_offset(np.pi/2.)
    ax.plot(angles,curve,'k-',linewidth=3)
    
    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/Center Ego Polar Plot trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            fig.savefig('%s/Center Ego Polar Plot.png' % egopath,dpi=adv['pic_resolution'])
     
    plt.close()
        
    occ_hist = cluster_data['ego_occ']
    occ_hist = np.append(occ_hist,occ_hist[0])
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    angles = np.linspace(0,2*np.pi,adv['hd_bins'],endpoint=False)
    angles = np.append(angles,2*np.pi)
    ax.yaxis.grid(False)
    ax.xaxis.grid(linewidth=2,color='k')
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.))
    ax.set_xticklabels([0,90,180,270],fontsize=12)
    ax.set_yticklabels([])
    ax.plot(angles,occ_hist,'k-',linewidth=3)
    
    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/Center Ego Occupancy Polar Plot trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            fig.savefig('%s/Center Ego Occupancy Polar Plot.png' % egopath,dpi=adv['pic_resolution'])
            
    ego_direction_dependent(ops,adv,trial_data,cluster_data,spike_data)
    plot_conjunctive(ops,adv,trial_data,cluster_data,spike_data)
    
    plt.close()
            
def ego_direction_dependent(ops,adv,trial_data,cluster_data,spike_data):
    
    gr=30
    sd = 1.5
    
    hmaps = np.zeros((9,gr,gr))
    
#    angles = np.array(trial_data['center_ego_angles'])
#    spike_angles = []
#    for i in range(len(angles)):
#        for j in range(int(spike_data['ani_spikes'][i])):
#            spike_angles.append(angles[i])
#    spike_angles = np.array(spike_angles)
     
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    spike_x = np.array(spike_data['spike_x'])
    spike_y = np.array(spike_data['spike_y'])
    spike_angles = np.array(spike_data['spike_angles'])
        
    fig = plt.figure()
    ax9 = fig.add_subplot(339)
    ax9.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax9.plot(spike_x[(spike_angles>292.5)&(spike_angles<337.5)],spike_y[(spike_angles>292.5)&(spike_angles<337.5)],'r.')
        
    fig2 = plt.figure()
    ax92 = fig2.add_subplot(339)
    
    h,xedges,yedges = np.histogram2d(center_x,center_y,bins=gr)
    
    hist,_,_ = np.histogram2d(center_x[(angles>292.5)&(angles<337.5)],center_y[(angles>292.5)&(angles<337.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>292.5)&(spike_angles<337.5)],spike_y[(spike_angles>292.5)&(spike_angles<337.5)],bins=[xedges,yedges])
    
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[8]=smoothed_heatmap
    
    ax6 = fig.add_subplot(336)
    ax6.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax6.plot(spike_x[(spike_angles>337.5)|(spike_angles<22.5)],spike_y[(spike_angles>337.5)|(spike_angles<22.5)],'r.')
        
    
    ax62 = fig2.add_subplot(336)
    hist,_,_ = np.histogram2d(center_x[(angles>337.5)|(angles<22.5)],center_y[(angles>337.5)|(angles<22.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>337.5)|(spike_angles<22.5)],spike_y[(spike_angles>337.5)|(spike_angles<22.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[5] = smoothed_heatmap    
    
    ax3 = fig.add_subplot(333)
    ax3.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax3.plot(spike_x[(spike_angles>22.5)&(spike_angles<67.5)],spike_y[(spike_angles>22.5)&(spike_angles<67.5)],'r.')
      
    ax32 = fig2.add_subplot(333)
    hist,_,_ = np.histogram2d(center_x[(angles>22.5)&(angles<67.5)],center_y[(angles>22.5)&(angles<67.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>22.5)&(spike_angles<67.5)],spike_y[(spike_angles>22.5)&(spike_angles<67.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[2] = smoothed_heatmap    
    
    ax8 = fig.add_subplot(338)
    ax8.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax8.plot(spike_x[(spike_angles>247.5)&(spike_angles<292.5)],spike_y[(spike_angles>247.5)&(spike_angles<292.5)],'r.')
      
    ax82 = fig2.add_subplot(338)
    hist,_,_ = np.histogram2d(center_x[(angles>247.5)&(angles<292.5)],center_y[(angles>247.5)&(angles<292.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>247.5)&(spike_angles<292.5)],spike_y[(spike_angles>247.5)&(spike_angles<292.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[7] = smoothed_heatmap    
    
    ax5 = fig.add_subplot(335)
    ax5.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax5.plot(spike_x,spike_y,'r.')
    
    ax52 = fig2.add_subplot(335)
    hist,_,_ = np.histogram2d(center_x,center_y,bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x,spike_y,bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[4] = smoothed_heatmap    
    
    ax2 = fig.add_subplot(332)
    ax2.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax2.plot(spike_x[(spike_angles>67.5)&(spike_angles<112.5)],spike_y[(spike_angles>67.5)&(spike_angles<112.5)],'r.')

    ax22 = fig2.add_subplot(332)
    hist,_,_ = np.histogram2d(center_x[(angles>67.5)&(angles<112.5)],center_y[(angles>67.5)&(angles<112.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>67.5)&(spike_angles<112.5)],spike_y[(spike_angles>67.5)&(spike_angles<112.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[1] = smoothed_heatmap

    ax7 = fig.add_subplot(337)
    ax7.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax7.plot(spike_x[(spike_angles>202.5)&(spike_angles<247.5)],spike_y[(spike_angles>202.5)&(spike_angles<247.5)],'r.')

    ax72 = fig2.add_subplot(337)
    hist,_,_ = np.histogram2d(center_x[(angles>202.5)&(angles<247.5)],center_y[(angles>202.5)&(angles<247.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>202.5)&(spike_angles<247.5)],spike_y[(spike_angles>202.5)&(spike_angles<247.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[6] = smoothed_heatmap

    ax4 = fig.add_subplot(334)
    ax4.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax4.plot(spike_x[(spike_angles>157.5)&(spike_angles<202.5)],spike_y[(spike_angles>157.5)&(spike_angles<202.5)],'r.')

    ax42 = fig2.add_subplot(334)
    hist,_,_ = np.histogram2d(center_x[(angles>157.5)&(angles<202.5)],center_y[(angles>157.5)&(angles<202.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>157.5)&(spike_angles<202.5)],spike_y[(spike_angles>157.5)&(spike_angles<202.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[3] = smoothed_heatmap

    ax1 = fig.add_subplot(331)
    ax1.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax1.plot(spike_x[(spike_angles>112.5)&(spike_angles<157.5)],spike_y[(spike_angles>112.5)&(spike_angles<157.5)],'r.')

    ax12 = fig2.add_subplot(331)
    hist,_,_ = np.histogram2d(center_x[(angles>112.5)&(angles<157.5)],center_y[(angles>112.5)&(angles<157.5)],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>112.5)&(spike_angles<157.5)],spike_y[(spike_angles>112.5)&(spike_angles<157.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(stddev=sd))
    hmaps[0] = smoothed_heatmap

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax9.set_xticks([])
    ax9.set_yticks([])
    
    vmin = 0
    vmax = np.nanmax(hmaps)
    
    ax12.imshow(hmaps[0],vmin=vmin,vmax=vmax)
    ax22.imshow(hmaps[1],vmin=vmin,vmax=vmax)
    ax32.imshow(hmaps[2],vmin=vmin,vmax=vmax)
    ax42.imshow(hmaps[3],vmin=vmin,vmax=vmax)
    ax52.imshow(hmaps[4],vmin=vmin,vmax=vmax)
    ax62.imshow(hmaps[5],vmin=vmin,vmax=vmax)
    ax72.imshow(hmaps[6],vmin=vmin,vmax=vmax)
    ax82.imshow(hmaps[7],vmin=vmin,vmax=vmax)
    ax92.imshow(hmaps[8],vmin=vmin,vmax=vmax)
    
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax22.set_xticks([])
    ax22.set_yticks([])
    ax32.set_xticks([])
    ax32.set_yticks([])
    ax42.set_xticks([])
    ax42.set_yticks([])
    ax52.set_xticks([])
    ax52.set_yticks([])
    ax62.set_xticks([])
    ax62.set_yticks([])
    ax72.set_xticks([])
    ax72.set_yticks([])
    ax82.set_xticks([])
    ax82.set_yticks([])
    ax92.set_xticks([])
    ax92.set_yticks([])
    
    egopath = cluster_data['new_folder']+'/egocentric'

    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/place by direction trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
            fig2.savefig('%s/place by direction hmap trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])

        else:
            fig.savefig('%s/place by direction.png' % egopath,dpi=adv['pic_resolution'])
            fig2.savefig('%s/place by direction hmap.png' % egopath,dpi=adv['pic_resolution'])

    plt.close()
    
    dd_autocorr(hmaps,cluster_data,adv,ops)
    

def dd_autocorr(hmaps,cluster_data,adv,ops):
    
    gr = 30
    
    #make a matrix of zeros 2x the length and width of the smoothed heatmap (in bins)
    corr_matrix = np.zeros((len(hmaps),2*gr,2*gr))
    
    for k in range(len(hmaps)):
        smoothed_heatmap = hmaps[k]
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
                    corr_matrix[k][gr+i][gr+j] = corr
         
    fig = plt.figure()
    ax9 = fig.add_subplot(339)
    
    ax6 = fig.add_subplot(336)
    
    ax3 = fig.add_subplot(333)
    
    ax8 = fig.add_subplot(338)

    ax5 = fig.add_subplot(335)

    ax2 = fig.add_subplot(332)

    ax7 = fig.add_subplot(337)
    
    ax4 = fig.add_subplot(334)

    ax1 = fig.add_subplot(331)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax9.set_xticks([])
    ax9.set_yticks([])

    
    ax1.imshow(corr_matrix[0])
    ax2.imshow(corr_matrix[1])
    ax3.imshow(corr_matrix[2])
    ax4.imshow(corr_matrix[3])
    ax5.imshow(corr_matrix[4])
    ax6.imshow(corr_matrix[5])
    ax7.imshow(corr_matrix[6])
    ax8.imshow(corr_matrix[7])
    ax9.imshow(corr_matrix[8])
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/place by direction autocorrs trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])

        else:
            fig.savefig('%s/place by direction autocorrs.png' % egopath,dpi=adv['pic_resolution'])
            
            
    plt.close()

    
def plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(cluster_data['center_dist_curve']))+10
    xmax = np.nanmax(cluster_data['center_dist_xvals'])
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.set_xlabel('distance (pixels)')
    ax.set_ylabel('firing rate (hz)')

    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    ax.plot(cluster_data['center_dist_xvals'],cluster_data['center_dist_curve'],'k-',cluster_data['center_dist_xvals'],cluster_data['center_dist_fit'],'b-')
    ax.text(.1,.9,'fit r^2 = %s' % cluster_data['center_dist_r']**2,transform=ax.transAxes)
    ax.text(.1,.8,'fit p = %s' % cluster_data['center_dist_p'],transform=ax.transAxes)
    
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    if ops['save_all'] and not cluster_data['saved']['plot_center_dist']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Center Distance trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Center Distance.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_dist'] = True
        
        
def plot_center_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    #make the figure
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min(trial_data['center_y']),max(trial_data['center_y'])])
    ax.set_xlim([min(trial_data['center_x']),max(trial_data['center_x'])]) 

    #make a scatter plot of spike locations colored by head direction
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['center_ego_angles'],cmap=colormap,norm=norm)
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_center_ego_hd_map']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spatial x Center Ego %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spatial x Center Ego.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_ego_hd_map'] = True
    #show it!
    
def plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    import spatial
    spatial.plot_center_ego_ahv(ops,adv,trial_data,cluster_data,spike_data,self)
    
    center_angles = cluster_data['center_ego_angles']
    center_dists = cluster_data['center_ego_dists']
    center_spike_angles = spike_data['center_ego_angles']
    center_spike_dists = spike_data['center_ego_dists']
    
    occ_hist,xedges,yedges = np.histogram2d(center_angles,center_dists,bins=[60,60])
    occ_hist[occ_hist<3] = 0
    spike_hist,xedges,yedges = np.histogram2d(center_spike_angles,center_spike_dists,bins=[xedges,yedges])
    hist = spike_hist/occ_hist
    hist[np.isinf(hist)] = np.nan
    
    h_interp = hist.copy()
    v_interp = hist.copy()
    
    for i in range(len(hist)):
        nans,x = np.isnan(h_interp[i]), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            h_interp[i][nans]= np.interp(x(nans), x(~nans), h_interp[i][~nans])
        
    for i in range(len(hist[0])):
        nans,x = np.isnan(v_interp[:,i].flatten()), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            v_interp[:,i][nans]= np.interp(np.deg2rad(xedges[x(nans)]), np.deg2rad(xedges[x(~nans)]), v_interp[:,i][~nans],period=2*np.pi)
        
    histm = (h_interp + v_interp)/2.
    
    hist3 = np.concatenate((histm,histm,histm),axis=0)
    hist3 = convolve(hist3,Gaussian2DKernel(stddev=2))
    new_hist = hist3[len(histm):len(histm)*2]
        
    ax = self.figure.add_subplot(111,projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_yticks([])
    self.figure.tight_layout(pad=2.5)

    print new_hist
    ax.pcolormesh(np.deg2rad(xedges),yedges,new_hist.T) 
        
    egopath = cluster_data['new_folder']+'/egocentric'
    
    if ops['save_all'] and not cluster_data['saved']['plot_center_ego_map']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Center Ego Map %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Center Ego Map.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_ego_map'] = True
    
        
def plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.93)
    #set y limit according to highest firing rate
    ymax = int(1.5*np.nanmax(cluster_data['wall_ego_curve']))+10
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    ax.set_xlabel('egocentric bearing (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,45))
    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % cluster_data['wall_ego_rayleigh'],transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s' % cluster_data['wall_ego_mean_angle'],transform=ax.transAxes)
    ax.text(.1,.7,'fit r^2 = %s' % cluster_data['wall_ego_r']**2,transform=ax.transAxes)
    ax.text(.1,.6,'fit p = %s' % cluster_data['wall_ego_p'],transform=ax.transAxes)

    
    egopath = cluster_data['new_folder']+'/egocentric'
    if not os.path.isdir(egopath):
        os.makedirs(egopath)

    ax.plot(np.arange(0,360,360/adv['hd_bins']),cluster_data['wall_ego_curve'],'k-',np.arange(0,360,360/adv['hd_bins']),cluster_data['wall_ego_fit'],'r--') #,cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],max(cluster_data['hd_rates']),'r*',cluster_data['hd_angles'],cluster_data['gauss_rates'],'r--')
    if ops['save_all'] and not cluster_data['saved']['plot_wall_ego']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Wall Ego Plot trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Wall Ego Plot.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_ego'] = True
        
def plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(cluster_data['wall_dist_curve']))+10
    xmax = np.nanmax(cluster_data['wall_dist_xvals'])
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.set_xlabel('distance (pixels)')
    ax.set_ylabel('firing rate (hz)')

    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    ax.plot(cluster_data['wall_dist_xvals'],cluster_data['wall_dist_curve'],'k-',cluster_data['wall_dist_xvals'],cluster_data['wall_dist_fit'],'b-')
    ax.text(.1,.9,'fit r^2 = %s' % cluster_data['wall_dist_r']**2,transform=ax.transAxes)
    ax.text(.1,.8,'fit p = %s' % cluster_data['wall_dist_p'],transform=ax.transAxes)
    
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
#    plot_wall_voronoi(ops,adv,trial_data,cluster_data,spike_data)
    
    if ops['save_all'] and not cluster_data['saved']['plot_wall_dist']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Wall Distance trial %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Wall Distance.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_dist'] = True
        
def plot_wall_voronoi(ops,adv,trial_data,cluster_data,spike_data):
    
    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=4)
    
    fig = plt.figure()
    
    #make the figure
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=2.5)
    
#    divider = make_axes_locatable(ax)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x]) 
    
#    center_mass_x = sum(spike_data['spike_x'])/len(spike_data['spike_x'])
#    center_mass_y = sum(spike_data['spike_y'])/len(spike_data['spike_y'])
    ax.scatter(trial_data['center_x'],trial_data['center_y'],c=cluster_data['wall_ids'],cmap=colormap,norm=norm)
    
    plt.show()

    #make a scatter plot of spike locations colored by head direction
#    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['spike_angles'],cmap=colormap,norm=norm,zorder=1)
#    ax.plot(center_mass_x,center_mass_y,'kx',markersize=15)

        
def plot_wall_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    #make the figure
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min(trial_data['center_y']),max(trial_data['center_y'])])
    ax.set_xlim([min(trial_data['center_x']),max(trial_data['center_x'])]) 

    #make a scatter plot of spike locations colored by head direction
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['wall_ego_angles'],cmap=colormap,norm=norm)
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_wall_ego_hd_map']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spatial x Wall Ego %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spatial x Wall Ego.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_ego_hd_map'] = True
        
def plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    wall_angles = cluster_data['wall_ego_angles']
    wall_dists = cluster_data['wall_ego_dists']
    wall_spike_angles = spike_data['wall_ego_angles']
    wall_spike_dists = spike_data['wall_ego_dists']
    
    occ_hist,xedges,yedges = np.histogram2d(wall_angles,wall_dists,bins=[np.linspace(0,360,60),np.linspace(0,np.nanmax(wall_dists),60)])
    occ_hist[occ_hist<3] = 0
    spike_hist,xedges,yedges = np.histogram2d(wall_spike_angles,wall_spike_dists,bins=[xedges,yedges])
    hist = spike_hist/occ_hist
    hist[np.isinf(hist)] = np.nan
    
    h_interp = hist.copy()
    v_interp = hist.copy()
    
    for i in range(len(hist)):
        nans,x = np.isnan(h_interp[i]), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            h_interp[i][nans]= np.interp(x(nans), x(~nans), h_interp[i][~nans])
        
    for i in range(len(hist[0])):
        nans,x = np.isnan(v_interp[:,i].flatten()), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            v_interp[:,i][nans]= np.interp(np.deg2rad(xedges[x(nans)]), np.deg2rad(xedges[x(~nans)]), v_interp[:,i][~nans],period=2*np.pi)
        
    histm = (h_interp + v_interp)/2.
    
    hist3 = np.concatenate((histm,histm,histm),axis=0)
    hist3 = convolve(hist3,Gaussian2DKernel(stddev=2))
    new_hist = hist3[len(histm):len(histm)*2]
        
    ax = self.figure.add_subplot(111,projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_yticks([])
    self.figure.tight_layout(pad=2.5)

    print xedges
    print yedges
    ax.pcolormesh(np.deg2rad(xedges),yedges,new_hist.T)

    egopath = cluster_data['new_folder']+'/egocentric'
    
    if ops['save_all'] and not cluster_data['saved']['plot_wall_ego_map']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Wall Ego Map %s.png' % (egopath,str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Wall Ego Map.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_ego_map'] = True
        
        
        
        
        
def plot_grating():
    
    #given allocentric bearings (theta) and distances (rho) from the center of environment...
    
    b = 3.
    
    theta = np.linspace(0,2*np.pi,360)
    rho = np.linspace(0,10,100)
    
    xvals,yvals = np.meshgrid(theta,rho)
    vals1 = yvals*np.cos(xvals) #+ yvals*np.sin(xvals)
    vals1 = np.cos(b*vals1)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='polar')
    ax.pcolormesh(xvals,yvals,vals1)


    vals2 = yvals*np.cos(xvals-np.pi/3.) #+ yvals*np.sin(xvals-np.pi/3.)
    vals2 = np.cos(b*vals2)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='polar')
    ax.pcolormesh(xvals,yvals,vals2)


    vals3 = yvals*np.cos(xvals-2*np.pi/3.) #+ yvals*np.sin(xvals-2*np.pi/3.)
    vals3 = np.cos(b*vals3)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='polar')
    ax.pcolormesh(xvals,yvals,vals3)
    
    
    grid = vals1+vals2+vals3
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='polar')
    ax.pcolormesh(xvals,yvals,grid)
    
    