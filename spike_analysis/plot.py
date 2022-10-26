# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:44:25 2017

script containing plotting functions

@author: Patrick
"""
import matplotlib as mpl
# mpl.rcParams['backend.qt']='PySide6'
mpl.use('QtAgg')
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['image.cmap'] = 'jet'
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import numpy as np
import copy

from spike_analysis import ego

def plot_hd(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
    #set y limit according to highest firing rate
    ymax = int(1.2*np.nanmax(cluster_data['hd_rates']))+5
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    ax.set_xlabel('head direction (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,90))
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.5,.9,'rayleigh r = %s' % round(cluster_data['rayleigh'],5),transform=ax.transAxes)
    ax.text(.5,.8,'rayleigh angle = %s$^\circ$' % round(cluster_data['rayleigh_angle'],5),transform=ax.transAxes)
    ax.plot(cluster_data['hd_angles'],cluster_data['hd_rates'],'k-')
    
    if ops['save_all'] and not cluster_data['saved']['plot_hd']:
        self.figure.savefig('%s/Head direction.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_hd'] = True
        
    ax.set_title('Head direction')

        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    ax.yaxis.grid(False)
    ax.xaxis.grid(linewidth=2,color='k')
    ax.text(.8,.9,'%s Hz' % str(round(np.nanmax(cluster_data['hd_rates']),1)),transform=ax.transAxes,fontsize=14)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.))
    ax.set_xticklabels([0,90,180,270],fontsize=12)
    ax.set_yticklabels([])
    ax.plot(np.deg2rad(cluster_data['hd_angles']),cluster_data['hd_rates'],'k-',linewidth=3)
    plt.tight_layout()
    
    if ops['save_all']:
        fig.savefig('%s/Head direction polar.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
             
    plt.close()
        
    spikes = np.zeros(adv['hd_bins'])
    occ = np.zeros(adv['hd_bins'])
    
    speeds = np.array(trial_data['speeds'])
    
    mds = trial_data['movement_directions'][speeds>5]
    md_spikes = np.array(spike_data['ani_spikes'])[speeds>5]
    
    md_bins = np.digitize(mds,np.linspace(0,360,adv['hd_bins'],endpoint=False)) - 1
    
    for i in range(len(md_bins)):
        spikes[md_bins[i]] += md_spikes[i]
        occ[md_bins[i]] += 1./adv['framerate']
        
    curve = spikes/occ

    mr = np.nansum(curve*np.exp(1j*np.deg2rad(np.linspace(0,360,adv['hd_bins'],endpoint=False))))/(np.nansum(curve))
    mrl = np.abs(mr)
    mra = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))
    
    curve = np.concatenate((curve,curve[0,np.newaxis]))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ymax = int(1.2*np.max(curve[np.isfinite(curve)]))+5
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    angles = np.linspace(0,361,adv['hd_bins']+1,endpoint=True)
    ax.set_xticks(np.arange(0,361,90))
    ax.text(.1,.9,'rayleigh r = %s' % np.round(mrl,4),transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s$^\circ$' % np.round(mra,2),transform=ax.transAxes)
    ax.plot(angles,curve,'k-')
    plt.tight_layout()
    
    if ops['save_all']:
        fig.savefig('%s/Movement direction.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        
    plt.close()


def plot_half_hds(ops,adv,trial_data,cluster_data,spike_data,self):

    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
#    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*max(cluster_data['half_hd_rates'][0]+cluster_data['half_hd_rates'][1]))+5
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,360])
    ax.set_xticks(range(0,361,90))
    ax.set_xlabel('head direction (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.plot(cluster_data['half_hd_angles'][0],cluster_data['half_hd_rates'][0],'k-',label='1st half')
    ax.plot(cluster_data['half_hd_angles'][1],cluster_data['half_hd_rates'][1],'r--',label='2nd half')
    ax.legend(loc='best')
    
    if ops['save_all'] and not cluster_data['saved']['plot_half_hds']:
        self.figure.savefig('%s/HD session halves.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_half_hds'] = True
        
    ax.set_title('Head direction - session halves')

        
def plot_path(ops,adv,trial_data,cluster_data,spike_data,self):
    
    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x])
    #plot it! path is a black line, spikes are red dots
    ax.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax.plot(spike_data['spike_x'],spike_data['spike_y'],'r.')
    ax.axis('off')
    
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_path']:
        self.figure.savefig('%s/Path & spike plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_path'] = True
    
def plot_raw_heat(ops,adv,trial_data,cluster_data,spike_data,self):
    
    xedges = cluster_data['heat_xedges']
    yedges = cluster_data['heat_yedges']

    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    colormap = plt.get_cmap('jet')
    norm = mplcolors.Normalize(vmin=0, vmax=np.nanmax(cluster_data['raw_heatmap']))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(cluster_data['raw_heatmap'],cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
        
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    ax.axis('off')
    
    if ops['save_all'] and not cluster_data['saved']['plot_raw_heat']:
        self.figure.savefig('%s/Raw heatmap.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_raw_heat'] = True
    
    ax.axis('off')
    
    
def plot_interpd_heat(ops,adv,trial_data,cluster_data,spike_data,self):
    
    xedges = cluster_data['heat_xedges']
    yedges = cluster_data['heat_yedges']

    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    colormap = plt.get_cmap('jet')
    norm = mplcolors.Normalize(vmin=0, vmax=np.nanmax(cluster_data['interpd_heatmap']))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(cluster_data['interpd_heatmap'],cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
    
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    ax.axis('off')
    
    if ops['save_all'] and not cluster_data['saved']['plot_interpd_heat']:
        self.figure.savefig('%s/Interpd heatmap.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_interpd_heat'] = True
    

def plot_smoothed_heat(ops,adv,trial_data,cluster_data,spike_data,self):   
    
    xedges = cluster_data['heat_xedges']
    yedges = cluster_data['heat_yedges']
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    colormap = plt.get_cmap('jet')
    norm = mplcolors.Normalize(vmin=0, vmax=np.nanmax(cluster_data['smoothed_heatmap']))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(cluster_data['smoothed_heatmap'],cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
    
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    ax.axis('off')
    
    if ops['save_all'] and not cluster_data['saved']['plot_smoothed_heat']:
        self.figure.savefig('%s/Smoothed heatmap.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_smoothed_heat'] = True


def plot_spatial_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.imshow(cluster_data['spatial_autocorr'])
    
    ax.axis('off')
    
    if ops['save_all'] and not cluster_data['saved']['plot_spatial_autocorr']:
        self.figure.savefig('%s/Spatial autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spatial_autocorr'] = True
    
    
def plot_grid_score(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.set_xlabel('Rotation (degrees)')
    ax.set_ylabel('Correlation')
    ax.set_xticks(range(0,181,30))
    ax.set_xlim((0,180))
    ax.text(.25,.9,'gridness score = %s' % np.round(cluster_data['gridness'],3),transform=ax.transAxes)  
    ax.plot(cluster_data['rot_angles'],cluster_data['rot_values'],'k-')
    
    if ops['save_all'] and not cluster_data['saved']['plot_grid_score']:
        self.figure.savefig('%s/Gridness.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_grid_score'] = True
        
    
def plot_ahv(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ymax = 1.2*np.max(cluster_data['ahv_rates'][np.isfinite(cluster_data['ahv_rates'])])
    ax.set_ylim([0,ymax])
    ax.set_xlim([-400,400])
    ax.set_xlabel('angular head velocity (degrees/sec)')
    ax.set_ylabel('firing rate (hz)')
    ax.plot(cluster_data['ahv_angles'],cluster_data['ahv_rates'],'ko',clip_on=False)
    
    if ops['save_all'] and not cluster_data['saved']['plot_ahv']:
        self.figure.savefig('%s/AHV.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ahv'] = True
        
    ax.set_title('Angular head velocity')

    
def plot_speed(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = 1.2*np.max(cluster_data['speed_rates'][np.isfinite(cluster_data['speed_rates'])])+5
    xmax = max(cluster_data['speed_edges']) + 5
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.set_xlabel('speed (%s/sec)' % adv['dist_measurement'])
    ax.set_ylabel('firing rate (hz)')
    ax.text(.1,.9,'r^2 = %f' % cluster_data['speed_r']**2,transform=ax.transAxes)
    ax.text(.1,.8,'p = %f' % cluster_data['speed_p'],transform=ax.transAxes)
    ax.plot(cluster_data['speed_edges'],cluster_data['speed_rates'],'k-')
#    ax.plot(cluster_data['speed_edges'],cluster_data['speed_fit_y'],color='gray',linestyle='--',alpha=0.6)
    
    if ops['save_all'] and not cluster_data['saved']['plot_speed']:
        self.figure.savefig('%s/Speed.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_speed'] = True
        
    ax.set_title('Linear speed')
    
    
def plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):

    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x])
    
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    
    ax.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)

    #make a scatter plot of spike locations colored by head direction
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['spike_angles'],cmap=colormap,norm=norm,zorder=1,clip_on=False)

    cbar = self.figure.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
    
    ax.axis('off')
    
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_hd_map']:
        self.figure.savefig('%s/Spike location x HD.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_hd_map'] = True
        
    ax.set_title('Spike location x HD')
    
    
def plot_hd_vectors(ops,adv,trial_data,cluster_data,spike_data,self):
    
    x_gr = cluster_data['hd_vector_x_gr']
    y_gr = cluster_data['hd_vector_y_gr']
    curves = cluster_data['hd_vector_curves']
    
    #make the figure
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    y,x = np.meshgrid(np.arange(y_gr),np.arange(x_gr))
    
    ax.quiver(x, y, cluster_data['hd_rxs'], cluster_data['hd_rys'], pivot='mid', clip_on=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    ax.axis('equal')
    ax.set_title('Max MVL: %f' % np.nanmax(cluster_data['hd_rs']))
        
    if ops['save_all'] and not cluster_data['saved']['plot_hd_vectors']:
        self.figure.savefig('%s/HD vectors.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_hd_vectors'] = True
        
    if ops['save_all']:
        
        max_y = 1.1*np.nanmax(curves)
        
        fig_x = (np.max(trial_data['center_x']) - np.min(trial_data['center_x'])) / 24.
        fig_y = (np.max(trial_data['center_y']) - np.min(trial_data['center_y'])) / 24.

        fig, axes = plt.subplots(y_gr,x_gr,figsize=(fig_x,fig_y))
        for i in range(x_gr):
            for j in range(y_gr):
                curve = list(curves[i][j])
                curve.append(curve[0])
                axes[y_gr-1-j,i].plot(curve,color='black')
                axes[y_gr-1-j,i].set_xticks([])
                axes[y_gr-1-j,i].set_yticks([])
                axes[y_gr-1-j,i].set_ylim([0,max_y])
                axes[y_gr-1-j,i].set_xlim([0,len(curve)-1])
    
    
        plt.tight_layout()
        fig.savefig('%s/HD vector curves.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        plt.close()
        
        plot_direction_dependent(ops,adv,trial_data,cluster_data,spike_data)
        
        
def plot_direction_dependent(ops,adv,trial_data,cluster_data,spike_data):

    sd = 6./adv['spatial_bin_size']

    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    spike_x = np.array(spike_data['spike_x'])
    spike_y = np.array(spike_data['spike_y'])
    spike_angles = np.array(spike_data['spike_angles'])
    
    h,xedges,yedges = np.histogram2d(center_x,center_y,bins=[np.arange(np.min(center_x),np.max(center_x),adv['spatial_bin_size']),np.arange(np.min(center_y),np.max(center_y),adv['spatial_bin_size'])],range=[[min(center_x),max(center_x)],[min(center_y),max(center_y)]])    
    x_gr = len(xedges)
    y_gr = len(yedges)
    
    hmaps = np.zeros((9,y_gr,x_gr))

    fig = plt.figure(figsize=(5,5))
    
    ''' southeast facing '''
    ax9 = fig.add_subplot(339)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>292.5)&(angles<337.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None

    ax9.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax9.plot(spike_x[(spike_angles>292.5)&(spike_angles<337.5)],spike_y[(spike_angles>292.5)&(spike_angles<337.5)],'r.')
        
    fig2 = plt.figure(figsize=(5,5))
    ax92 = fig2.add_subplot(339)
    
    h,xedges,yedges = np.histogram2d(center_x,center_y,bins=[x_gr,y_gr])
    
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>292.5)&(spike_angles<337.5)],spike_y[(spike_angles>292.5)&(spike_angles<337.5)],bins=[xedges,yedges])
    
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[8]=smoothed_heatmap.T
    
    
    ''' east facing '''
    ax6 = fig.add_subplot(336)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>337.5)|(angles<22.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax6.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax6.plot(spike_x[(spike_angles>337.5)|(spike_angles<22.5)],spike_y[(spike_angles>337.5)|(spike_angles<22.5)],'r.')
    
    ax62 = fig2.add_subplot(336)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>337.5)|(spike_angles<22.5)],spike_y[(spike_angles>337.5)|(spike_angles<22.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[5] = smoothed_heatmap.T
    
    
    ''' northeast facing '''
    ax3 = fig.add_subplot(333)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>22.5)&(angles<67.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None

    ax3.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax3.plot(spike_x[(spike_angles>22.5)&(spike_angles<67.5)],spike_y[(spike_angles>22.5)&(spike_angles<67.5)],'r.')
      
    ax32 = fig2.add_subplot(333)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>22.5)&(spike_angles<67.5)],spike_y[(spike_angles>22.5)&(spike_angles<67.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[2] = smoothed_heatmap.T
    
    ''' south facing '''
    ax8 = fig.add_subplot(338)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>247.5)&(angles<292.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax8.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax8.plot(spike_x[(spike_angles>247.5)&(spike_angles<292.5)],spike_y[(spike_angles>247.5)&(spike_angles<292.5)],'r.')
      
    ax82 = fig2.add_subplot(338)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>247.5)&(spike_angles<292.5)],spike_y[(spike_angles>247.5)&(spike_angles<292.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[7] = smoothed_heatmap.T
    
    
    ''' all HDs '''
    ax5 = fig.add_subplot(335)
    ax5.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    ax5.plot(spike_x,spike_y,'r.')
    
    ax52 = fig2.add_subplot(335)
    hist,_,_ = np.histogram2d(center_x,center_y,bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x,spike_y,bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[4] = smoothed_heatmap.T
    
    
    ''' north facing '''
    ax2 = fig.add_subplot(332)

    good_inds = np.zeros_like(angles)
    good_inds[(angles>67.5)&(angles<112.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax2.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax2.plot(spike_x[(spike_angles>67.5)&(spike_angles<112.5)],spike_y[(spike_angles>67.5)&(spike_angles<112.5)],'r.')

    ax22 = fig2.add_subplot(332)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>67.5)&(spike_angles<112.5)],spike_y[(spike_angles>67.5)&(spike_angles<112.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[1] = smoothed_heatmap.T


    ''' southwest facing '''
    ax7 = fig.add_subplot(337)

    good_inds = np.zeros_like(angles)
    good_inds[(angles>202.5)&(angles<247.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax7.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax7.plot(spike_x[(spike_angles>202.5)&(spike_angles<247.5)],spike_y[(spike_angles>202.5)&(spike_angles<247.5)],'r.')

    ax72 = fig2.add_subplot(337)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>202.5)&(spike_angles<247.5)],spike_y[(spike_angles>202.5)&(spike_angles<247.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[6] = smoothed_heatmap.T


    ''' west facing '''
    ax4 = fig.add_subplot(334)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>157.5)&(angles<202.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax4.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax4.plot(spike_x[(spike_angles>157.5)&(spike_angles<202.5)],spike_y[(spike_angles>157.5)&(spike_angles<202.5)],'r.')

    ax42 = fig2.add_subplot(334)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>157.5)&(spike_angles<202.5)],spike_y[(spike_angles>157.5)&(spike_angles<202.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[3] = smoothed_heatmap.T


    ''' northwest facing '''
    ax1 = fig.add_subplot(331)
    
    good_inds = np.zeros_like(angles)
    good_inds[(angles>112.5)&(angles<157.5)] = 1
    good_inds = good_inds.astype(bool)
    
    plot_x = copy.deepcopy(center_x)
    plot_x[~good_inds] = None
    plot_y = copy.deepcopy(center_y)
    plot_y[~good_inds] = None
    
    ax1.plot(plot_x,plot_y,color='gray',alpha=0.5,zorder=0)
    ax1.plot(spike_x[(spike_angles>112.5)&(spike_angles<157.5)],spike_y[(spike_angles>112.5)&(spike_angles<157.5)],'r.')

    ax12 = fig2.add_subplot(331)
    hist,_,_ = np.histogram2d(center_x[good_inds],center_y[good_inds],bins=[xedges,yedges])
    spike_hist,_,_ = np.histogram2d(spike_x[(spike_angles>112.5)&(spike_angles<157.5)],spike_y[(spike_angles>112.5)&(spike_angles<157.5)],bins=[xedges,yedges])
    heatmap = spike_hist/hist
    smoothed_heatmap = convolve(heatmap,Gaussian2DKernel(x_stddev=sd,y_stddev=sd))
    hmaps[0] = smoothed_heatmap.T

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('equal')
    ax1.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('equal')
    ax2.axis('off')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('equal')
    ax3.axis('off')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.axis('equal')
    ax4.axis('off')
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.axis('equal')
    ax5.axis('off')
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.axis('equal')
    ax6.axis('off')
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.axis('equal')
    ax7.axis('off')
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.axis('equal')
    ax8.axis('off')
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax9.axis('equal')
    ax9.axis('off')

    
    vmin = 0
    vmax = np.nanmax(hmaps)
    
    ax12.imshow(hmaps[0],vmin=vmin,vmax=vmax,origin='lower')
    ax22.imshow(hmaps[1],vmin=vmin,vmax=vmax,origin='lower')
    ax32.imshow(hmaps[2],vmin=vmin,vmax=vmax,origin='lower')
    ax42.imshow(hmaps[3],vmin=vmin,vmax=vmax,origin='lower')
    ax52.imshow(hmaps[4],vmin=vmin,vmax=vmax,origin='lower')
    ax62.imshow(hmaps[5],vmin=vmin,vmax=vmax,origin='lower')
    ax72.imshow(hmaps[6],vmin=vmin,vmax=vmax,origin='lower')
    ax82.imshow(hmaps[7],vmin=vmin,vmax=vmax,origin='lower')
    ax92.imshow(hmaps[8],vmin=vmin,vmax=vmax,origin='lower')
    
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
            
    fig.savefig('%s/Place x HD.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
    fig2.savefig('%s/Place x HD heatmap.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])

    plt.close('all')
    
    
def plot_isi(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
    ax.set_xlim([0,max(cluster_data['isi_hist'][1])])
    ax.set_ylim([0,1.2*max(cluster_data['isi_hist'][0])])
    ax.set_xlabel('ISI (s)')
    ax.set_ylabel('count')
    ax.vlines(cluster_data['isi_xvals'],0,cluster_data['isi_hist'][0],colors='black')
    if ops['save_all'] and not cluster_data['saved']['plot_isi']:
        self.figure.savefig('%s/ISI hist.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_isi'] = True
        
    ax.set_title('ISI histogram')

    
def plot_spike_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
    ax.set_xlim([min(cluster_data['ac_xvals']),max(cluster_data['ac_xvals'])])
    ax.set_ylim([0,1.2*max(cluster_data['ac_vals'])])
    ax.set_ylabel('count')
    ax.set_xlabel('seconds')
    ax.vlines(cluster_data['ac_xvals'],ymin=0,ymax=cluster_data['ac_vals'],colors='black')
    ax.set_title('Theta index: %f' % cluster_data['theta_index'])

    if ops['save_all'] and not cluster_data['saved']['plot_spike_autocorr']:
        self.figure.savefig('%s/Spike autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_autocorr'] = True
        
    if ops['save_all']:
        fig = plt.figure()  
        ax = fig.add_subplot(111)
        fig.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
        ax.set_xlim([min(cluster_data['small_ac_xvals']),max(cluster_data['small_ac_xvals'])])
        ax.set_ylim([0,1.2*max(cluster_data['small_ac_vals'])])
        ax.set_ylabel('count')
        ax.set_xlabel('ms')
        ax.set_xticks(np.linspace(-.1,.1,21,endpoint=True))
        ax.set_xticklabels(['-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10'])
        ax.vlines(cluster_data['small_ac_xvals'],ymin=0,ymax=cluster_data['small_ac_vals'],colors='black')
        fig.savefig('%s/Narrow spike autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
    
        plt.close()
    

def plot_view(ops,adv,trial_data,cluster_data,spike_data,self):

    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    view_map = cluster_data['view_map']

    im = ax.imshow(view_map,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    self.figure.colorbar(im, cax=cax, orientation='vertical')

    if ops['save_all'] and not cluster_data['saved']['plot_view']:
        self.figure.savefig('%s/view plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_view'] = True
        
    ax.set_title('\"View\" plot')

def plot_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_ego(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_ego_angle(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_ego_angle(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_center_bearing(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_bearing(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):

    ego.plot_center_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_bearing(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_bearing(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):

    ego.plot_wall_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_md_ebc(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_md_ebc(ops,adv,trial_data,cluster_data,spike_data,self)
    