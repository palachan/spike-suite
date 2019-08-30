# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:44:25 2017

script containing plotting functions

@author: Patrick
"""
import matplotlib as mpl
mpl.rcParams['backend.qt4']='PySide'
mpl.use('Qt4Agg')
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['image.cmap'] = 'jet'
import numpy as np
import heapq

import ego


def plot_hd(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.93)
    #set y limit according to highest firing rate
    ymax = int(1.5*max(cluster_data['hd_rates']))+10
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    ax.set_xlabel('head direction (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,45))
    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % cluster_data['rayleigh'],transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s' % cluster_data['rayleigh_angle'],transform=ax.transAxes)
    ax.text(.1,.7,'fit pfd = %s' % cluster_data['pfd'],transform=ax.transAxes)
    ax.text(.1,.75,'observed pfd = %s' % cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],transform=ax.transAxes)
    ax.plot(cluster_data['hd_angles'],cluster_data['hd_rates'],'k-') #,cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],max(cluster_data['hd_rates']),'r*',cluster_data['hd_angles'],cluster_data['gauss_rates'],'r--')
    
    
    if ops['save_all'] and not cluster_data['saved']['plot_hd']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Head Direction Plot trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Head Direction Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_hd'] = True
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    ax.yaxis.grid(False)
    ax.xaxis.grid(linewidth=2,color='k')
    ax.text(.8,.9,'%s Hz' % str(round(np.max(cluster_data['hd_rates']),1)),transform=ax.transAxes,fontsize=14)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.))
    ax.set_xticklabels([0,90,180,270],fontsize=12)
    ax.set_yticklabels([])
    ax.plot(np.deg2rad(cluster_data['hd_angles']),cluster_data['hd_rates'],'k-',linewidth=3)

    
    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/Head Direction Polar Plot trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            fig.savefig('%s/Head Direction PolarPlot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
     
        
    spikes = np.zeros(adv['hd_bins'])
    occ = np.zeros(adv['hd_bins'])
    
    speeds = np.array(trial_data['speeds'])
    
    mds = trial_data['movement_directions'][speeds>10]
    md_spikes = np.array(spike_data['ani_spikes'])[speeds>10]
    
    md_bins = np.digitize(mds,np.linspace(0,360,adv['hd_bins'],endpoint=False)) - 1
    
    for i in range(len(md_bins)):
        spikes[md_bins[i]] += md_spikes[i]
        occ[md_bins[i]] += 1./adv['framerate']
        
    curve = spikes/occ
    
    mr = np.nansum(curve*np.exp(1j*np.deg2rad(np.linspace(0,360,adv['hd_bins'],endpoint=False))))/(np.nansum(curve))
    mrl = np.abs(mr)
    mra = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ymax = int(1.5*np.nanmax(curve))+10
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    angles = np.linspace(0,360,adv['hd_bins'],endpoint=False)
    ax.set_xticks(np.arange(0,360,90))
    ax.text(.1,.9,'rayleigh r = %s' % mrl,transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s' % mra,transform=ax.transAxes)
    ax.plot(angles,curve,'k-',linewidth=3)
    
    if ops['save_all']:
        if ops['multi_cam']:
            fig.savefig('%s/Movement direction trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            fig.savefig('%s/Movement direction.png' % cluster_data['new_folder'])
            
        
        
#    inds = np.array([ind for ind,val in enumerate(spike_data['ani_spikes']) if val > 0],dtype=np.int)
#    print inds
    
#    center_x2 = np.roll(trial_data['center_x'],1)
#    center_y2 = np.roll(trial_data['center_y'],1)
#    
#    x_movement = np.abs(trial_data['center_x'] - center_x2)
#    y_movement = np.abs(trial_data['center_y'] - center_y2)
#    
#    movement = np.cumsum(np.sqrt(x_movement**2 + y_movement**2))
#    spike_movements = movement[inds]
#    movement2 = spike_movements - np.roll(spike_movements,-1)
        
#    self.plot_something((inds,np.array(trial_data['angles'])[inds]))
    

def plot_half_hds(ops,adv,trial_data,cluster_data,spike_data,self):

    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.98)
    ymax = int(1.5*max(cluster_data['half_hd_rates'][0]+cluster_data['half_hd_rates'][1]))+10
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,360])
    ax.set_xticks(range(0,361,45))
    ax.set_xlabel('head direction (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.plot(cluster_data['half_hd_angles'][0],cluster_data['half_hd_rates'][0],'k-',label='1st half')
    ax.plot(cluster_data['half_hd_angles'][1],cluster_data['half_hd_rates'][1],'r--',label='2nd half')
    ax.legend(loc='best')
    if ops['save_all'] and not cluster_data['saved']['plot_half_hds']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Half HD Plots trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Half HD Plots trial.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_half_hds'] = True
        

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
        
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_path']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Path plot %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Path plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_path'] = True
    
def plot_raw_heat(ops,adv,trial_data,cluster_data,spike_data,self):
    
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']

    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(cluster_data['raw_heatmap'],extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
        
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    
def plot_interpd_heat(ops,adv,trial_data,cluster_data,spike_data,self):
    
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']

    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(cluster_data['interpd_heatmap'],extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
    
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    if ops['save_all'] and not cluster_data['saved']['plot_interpd_heat']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Interpd Heat Map %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Interpd Heat Map.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_interpd_heat'] = True
    
    
    
def plot_smoothed_heat(ops,adv,trial_data,cluster_data,spike_data,self):   
    
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(cluster_data['smoothed_heatmap'],extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
    
    self.figure.colorbar(im, cax=cax, orientation='vertical')
    
    if ops['save_all'] and not cluster_data['saved']['plot_smoothed_heat']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Smoothed Heat Map %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Smoothed Heat Map.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_smoothed_heat'] = True


def plot_spatial_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.imshow(cluster_data['spatial_autocorr'])
    
    if ops['save_all'] and not cluster_data['saved']['plot_spatial_autocorr']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spatial Autocorrelation %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spatial Autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spatial_autocorr'] = True
    
    
    
    
def plot_grid_score(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.set_xlabel('Rotation (degrees)')
    ax.set_ylabel('Correlation')
    ax.set_xticks(range(0,181,30))
    ax.set_xlim((0,180))
    ax.text(.25,.9,'gridness score = %s' % cluster_data['gridness'],transform=ax.transAxes)  
    ax.plot(cluster_data['rot_angles'],cluster_data['rot_values'],'k-')
    
    if ops['save_all'] and not cluster_data['saved']['plot_grid_score']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Gridness Plot %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Gridness Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_grid_score'] = True
    
    
    
def plot_ahv(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.5*np.max(cluster_data['ahv_rates']))
    ax.set_ylim([0,ymax])
    ax.set_xlim([-400,400])
    ax.set_xlabel('angular head velocity (degrees/sec)')
    ax.set_ylabel('firing rate (hz)')
    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    ax.plot(cluster_data['ahv_angles'],cluster_data['ahv_rates'],'ko')
    
    if ops['save_all'] and not cluster_data['saved']['plot_ahv']:
        if ops['multi_cam']:
            self.figure.savefig('%s/AHV Plot trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/AHV Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ahv'] = True
    
    
    
    
def plot_speed(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*np.max(cluster_data['speed_rates']))+10
    xmax = max(cluster_data['speed_edges']) + 10
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xmax])
    ax.set_xlabel('speed (pixels/sec)')
    ax.set_ylabel('firing rate (hz)')
    ax.text(.1,.9,'r^2 = %f' % cluster_data['speed_r']**2,transform=ax.transAxes)
    ax.text(.1,.8,'p = %f' % cluster_data['speed_p'],transform=ax.transAxes)
    if ops['multi_cam']:
        ax.set_title('camera %s' % trial_data['cam_id'])
    ax.plot(cluster_data['speed_edges'],cluster_data['speed_rates'],'k-',cluster_data['speed_edges'],cluster_data['speed_fit_y'],'b-')
    
    if ops['save_all'] and not cluster_data['saved']['plot_speed']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Speed Plot trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Speed Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_speed'] = True
    
    
    
    
    
def plot_novelty(ops,adv,trial_data,cluster_data,spike_data,self):
    
    firing_rates = cluster_data['novelty_rates']
    vals = cluster_data['novelty_vals']
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    
    ax.plot(vals,firing_rates,'k-')
    ymax = int(1.2*np.max(firing_rates))+10
    ax.set_xlabel('bin occupancy (s)')
    ax.set_ylabel('firing rate (Hz)')
    ax.set_ylim([0,ymax])
    
    if ops['save_all'] and not cluster_data['saved']['plot_novelty']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Novelty Plot trial %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Novelty Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_novelty'] = True
    
    
    
    
    
def plot_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    #make the figure
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x]) 
    
#    center_mass_x = sum(spike_data['spike_x'])/len(spike_data['spike_x'])
#    center_mass_y = sum(spike_data['spike_y'])/len(spike_data['spike_y'])
    ax.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)

    #make a scatter plot of spike locations colored by head direction
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['spike_angles'],cmap=colormap,norm=norm,zorder=1)
#    ax.plot(center_mass_x,center_mass_y,'kx',markersize=15)

    cbar = self.figure.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
    
    #save it
    if ops['save_all'] and not cluster_data['saved']['plot_hd_map']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spatial x HD %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spatial x HD.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_hd_map'] = True
    #show it!
    
    
def plot_isi(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.98)
    ax.set_xlim([0,max(cluster_data['isi_hist'][1])])
    ax.set_ylim([0,1.2*max(cluster_data['isi_hist'][0])])
    ax.set_xlabel('ISI (s)')
    ax.set_ylabel('count')
    ax.vlines(cluster_data['isi_xvals'],0,cluster_data['isi_hist'][0])
    if ops['save_all'] and not cluster_data['saved']['plot_isi']:
        if ops['multi_cam']:
            self.figure.savefig('%s/ISI hist %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/ISI hist.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_isi'] = True
    
    
def plot_spike_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.98)
    ax.set_xlim([min(cluster_data['ac_xvals']),max(cluster_data['ac_xvals'])])
    ax.set_ylim([0,1.2*max(cluster_data['ac_vals'])])
    ax.set_ylabel('count')
    ax.set_xlabel('seconds')
    ax.vlines(cluster_data['ac_xvals'],ymin=0,ymax=cluster_data['ac_vals'])
    if ops['save_all'] and not cluster_data['saved']['plot_spike_autocorr']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spike Autocorrelation %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spike Autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_autocorr'] = True
        
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    fig.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.11, top=.80, right=0.98)
    ax.set_xlim([min(cluster_data['small_ac_xvals']),max(cluster_data['small_ac_xvals'])])
    ax.set_ylim([0,1.2*max(cluster_data['small_ac_vals'])])
    ax.set_ylabel('count')
    ax.set_xlabel('ms')
    ax.set_xticks(np.linspace(-.1,.1,21,endpoint=True))
    ax.set_xticklabels(['-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10'])
    ax.vlines(cluster_data['small_ac_xvals'],ymin=0,ymax=cluster_data['small_ac_vals'])
    fig.savefig('%s/small Spike Autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])

    
def plot_spike_timed(ops,adv,trial_data,cluster_data,spike_data,self):
    
    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
#    
#    ax = self.figure.add_subplot(111)
#    self.figure.tight_layout(pad=2.5)
#    ax.set_ylim([min(center_y)-max(center_y),max(center_y)-min(center_y)])
#    ax.set_xlim([min(center_x)-max(center_x),max(center_x)-min(center_x)]) 
#    
#    ax.plot(cluster_data['st_long_xs'][::10],cluster_data['st_long_ys'][::10],'k-',cluster_data['st_long_sxs'],cluster_data['st_long_sys'],'r.')
#    
#    #save it!
    if ops['save_all'] and not cluster_data['saved']['plot_spike_timed']:
#        if ops['multi_cam']:
#            self.figure.savefig('%s/Spike-timed Path %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
#        else:
#            self.figure.savefig('%s/Spike-timed Path.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_timed'] = True
    
    
def plot_spike_timed_heat(ops,adv,trial_data,cluster_data,spike_data,self):   
    
    xedges = cluster_data['st_xedges']
    yedges = cluster_data['st_yedges']
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.imshow(cluster_data['st_smoothed_heatmap'],extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) 
    
    if ops['save_all'] and not cluster_data['saved']['plot_spike_timed_heat']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spike-timed Heat Map %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spike-timed Heat Map.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_timed_heat'] = True
    
    
    
def plot_spike_timed_autocorr(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.imshow(cluster_data['st_spatial_autocorr'])
    
    if ops['save_all'] and not cluster_data['saved']['plot_spike_timed_autocorr']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spike-timed Autocorrelation %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spike-timed Autocorrelation.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_timed_autocorr'] = True
    
def plot_spike_timed_gridness(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    ax.set_xlabel('Rotation (degrees)')
    ax.set_ylabel('Correlation')
    ax.set_xticks(range(0,181,30))
    ax.set_xlim((0,180))
    ax.text(.25,.9,'gridness score = %s' % cluster_data['st_gridness'],transform=ax.transAxes)  
    ax.plot(cluster_data['st_rot_angles'],cluster_data['st_rot_values'],'k-')
    if ops['save_all'] and not cluster_data['saved']['plot_spike_timed_gridness']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Spike-timed Gridness Plot %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Spike-timed Gridness Plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_spike_timed_gridness'] = True
        
def plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,raw=False,center_mass=False,quiver=False):
    
    center_mass_x = sum(spike_data['spike_x'])/len(spike_data['spike_x'])
    center_mass_y = sum(spike_data['spike_y'])/len(spike_data['spike_y'])
    
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    if raw:
        ego_rayleighs = cluster_data['ego_raw_rayleighs']
    else:
        ego_rayleighs = cluster_data['ego_rayleighs']
            
    if not quiver:
        im = ax.imshow(ego_rayleighs,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        if center_mass:
            ax.plot(center_mass_x,center_mass_y,'kx')
        #
        self.figure.colorbar(im, cax=cax, orientation='vertical')
        #ax.plot(spike_data['spike_x'],spike_data['spike_y'],'r.')
    
    else:
        xcoords = cluster_data['ego_xcoords']
        ycoords = cluster_data['ego_ycoords']
        
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        self.figure.tight_layout(pad=2.5)
        
        x,y=np.meshgrid(xcoords,ycoords)
        
        ax.quiver(x, y, cluster_data['allocentrized_rxs'], cluster_data['allocentrized_rys'])
        ax.set_yticks([])
        ax.set_xticks([])
        #ax.text(.1,.9,'max rayleigh = %s' % np.max(cluster_data['allocentrized_rayleighs']),transform=ax.transAxes) 
    
    if ops['save_all'] and not cluster_data['saved']['plot_ego']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Ego Rayleighs %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Ego Rayleighs.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ego'] = True
    
def plot_ego_angle(ops,adv,trial_data,cluster_data,spike_data,self,raw=False,center_mass=False,pinwheel_center=False):
    
#    center_points = cluster_data['ego_center_points']
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    
    center_mass_x = sum(spike_data['spike_x'])/len(spike_data['spike_x'])
    center_mass_y = sum(spike_data['spike_y'])/len(spike_data['spike_y'])
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    #ax.plot(spike_data['spike_x'],spike_data['spike_y'],'r.')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    if raw:
        ego_mean_angles = cluster_data['ego_raw_mean_angles']
    else:
        ego_mean_angles = cluster_data['ego_mean_angles']

    im = ax.imshow(ego_mean_angles,cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    if center_mass:
        ax.plot(center_mass_x,center_mass_y,'kx')
        
    if pinwheel_center:
        center_xcoord = xedges[np.where(np.flipud(center_points) == 1)[1]+1]
        center_ycoord = yedges[np.where(np.flipud(center_points) == 1)[0]+1]
        ax.plot(center_xcoord,center_ycoord,'k+')
        if np.sum(center_points) == 1 and len(cluster_data['oct_points']) > 0:
            oct_points = cluster_data['oct_points']
            ax.plot(oct_points[1],oct_points[0],'k-')

    self.figure.colorbar(im, cax=cax, orientation='vertical')
        
    if ops['save_all'] and not cluster_data['saved']['plot_ego_angle']:
        if ops['multi_cam']:
            self.figure.savefig('%s/Ego Mean Angles %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/Ego Mean Angles.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ego_angle'] = True
        
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
        if ops['multi_cam']:
            self.figure.savefig('%s/view plot %s.png' % (cluster_data['new_folder'],str(cluster_data['cam']+1)),dpi=adv['pic_resolution'])
        else:
            self.figure.savefig('%s/view plot.png' % cluster_data['new_folder'],dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_view'] = True
        
def plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_ego(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):

    ego.plot_center_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self)
    
def plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_ego(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self):

    ego.plot_wall_ego_hd_map(ops,adv,trial_data,cluster_data,spike_data,self)
        
def plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ego.plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self)
    
  
def subplot_designer(ops,adv,trial_data,cluster_data,spike_data,self):
    
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize':'small'})
    plt.rcParams.update({'ytick.labelsize':'small'})
    
    rows = 0
    spaces = 0
    if ops['run_hd']:
        if spaces%16 == 0:
            rows += 2
        spaces += 16
    if ops['run_spatial']:
        if spaces%16 == 0:
            rows += 2
        spaces += 16
    if ops['run_grid']:
        if spaces%16 == 0:
            rows += 2
        spaces += 4
        if spaces%16 == 0:
            rows += 2
        spaces += 4
    if ops['run_autocorr']:
        if spaces%16 == 0:
            rows += 2
        spaces += 4
        if spaces%16 == 0:
            rows += 2
        spaces += 4
    if ops['run_speed']:
        if spaces%16 == 0:
            rows += 2
        spaces += 4
    if ops['run_ahv']:
        if spaces%16 == 0:
            rows += 2
        spaces += 4
    if ops['hd_map']:
        if spaces%16 == 0:
            rows += 2
        spaces += 4
        
    if spaces >= 16:
        cols = 8
    else:
        cols = spaces/2
        
    spaces_taken = 0

    if ops['run_hd']:
        
        ax1 = plt.subplot2grid((rows, cols), (0, 1), rowspan=2, colspan=3)
        ax1.set_ylim([0,max(cluster_data['hd_rates'])+10])
        ax1.set_xlim([0,360])
        #ax1.set_xlabel('head direction (degrees)')
        ax1.set_ylabel('firing rate (hz)')
        ax1.set_xticks([])
#        if ops['multi_cam']:
#            plt.title('camera %s' % trial_data['cam_id'])
        #print rayleigh r, rayleigh pfd, fit pfd values on graph
        ax1.text(.1,.9,'rayleigh r = %s' % round(cluster_data['rayleigh'],4),transform=ax1.transAxes)
        ax1.text(.1,.8,'fit pfd = %s' % cluster_data['pfd'],transform=ax1.transAxes)
        #ax1.text(.1,.75,'observed pfd = %s' % cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],transform=ax1.transAxes)
        ax1.plot(cluster_data['hd_angles'],cluster_data['hd_rates'],'k-',cluster_data['hd_angles'][cluster_data['hd_rates'].index(max(cluster_data['hd_rates']))],max(cluster_data['hd_rates']),'r*',cluster_data['hd_angles'],cluster_data['gauss_rates'],'r--')
        
        spaces_taken+=4
        
        ax2 = plt.subplot2grid((rows, cols), (0, 4), rowspan=2, colspan=3)
        #max y set to peak rate + 10 over both halves, max x to 360
        ymax = max(cluster_data['half_hd_rates'][0]+cluster_data['half_hd_rates'][1]) + 10
        ax2.set_ylim([0,ymax])
        ax2.set_xlim([0,360])
        #plot and save!
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.plot(cluster_data['half_hd_angles'][0],cluster_data['half_hd_rates'][0],'k-',label='1st half')
        ax2.plot(cluster_data['half_hd_angles'][1],cluster_data['half_hd_rates'][1],'r--',label='2nd half')
        
        spaces_taken+=4
    if ops['run_spatial']:

        ax3 = plt.subplot2grid((rows, cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))),rowspan=2, colspan=2)
        ax3.axis('scaled')
        #axes set to min and max x and y values in dataset
        ax3.set_ylim([min(trial_data['center_y']),max(trial_data['center_y'])])
        ax3.set_xlim([min(trial_data['center_x']),max(trial_data['center_x'])])
        #plot it! path is a black line, spikes are red dots
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.plot(trial_data['center_x'],trial_data['center_y'],'k-',spike_data['spike_x'],spike_data['spike_y'],'r.')
        spaces_taken += 2
        
        
        ax4 = plt.subplot2grid((rows, cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)  
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.imshow(cluster_data['raw_heatmap']) 
        spaces_taken += 2
        ax5 = plt.subplot2grid((rows, cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.imshow(cluster_data['interpd_heatmap']) 
        spaces_taken += 2
        ax6 = plt.subplot2grid((rows, cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)  
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.imshow(cluster_data['smoothed_heatmap']) 
        spaces_taken += 2

    if ops['run_grid']:

        ax7 = plt.subplot2grid((rows, cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax7.set_xticks([])
        ax7.set_yticks([])
        ax7.imshow(cluster_data['spatial_autocorr'])
        spaces_taken += 2
        
        ax8 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax8.text(.15,.9,'g_score = %s' % round(cluster_data['gridness'],4),transform=ax8.transAxes)    
        ax8.set_xlabel('Rotation (degrees)')
        ax8.set_ylabel('Correlation')
        ax8.set_xticks(range(0,181,30))
        ax8.set_xlim((0,180))
        ax8.plot(cluster_data['rot_angles'],cluster_data['rot_values'],'k-')
        spaces_taken += 2
        
    if ops['run_autocorr']:
        ax9 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax9.set_xlim([min(cluster_data['ac_xvals']),max(cluster_data['ac_xvals'])])
        ax9.set_ylim([0,heapq.nlargest(3,cluster_data['ac_vals'])[2]])
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax9.bar(cluster_data['ac_xvals'],cluster_data['ac_vals'],width=cluster_data['ac_bar_width'],align='center')
        spaces_taken += 2

        ax10 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax10.set_xlim([0,max(cluster_data['isi_list'])])
        ax10.set_xticks([])
        ax10.set_yticks([])
        ax10.hist(cluster_data['isi_list'],bins=1000,range=[0,max(cluster_data['isi_list'])])
        spaces_taken += 2
        
    if ops['run_speed']:
        ax11 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ymax = int(1.5*np.max(cluster_data['speed_rates']))
        xmax = max(cluster_data['speed_edges']) + 10
        ax11.set_ylim([0,ymax])
        ax11.set_xlim([0,xmax])
        ax11.text(.1,.9,'r^2 = %f' % cluster_data['speed_r']**2,transform=ax11.transAxes)
        ax11.text(.1,.8,'p = %f' % cluster_data['speed_p'],transform=ax11.transAxes)
        ax11.set_xticks([0,xmax/2,xmax])
        ax11.set_yticks([0,ymax/2,ymax])   
        ax11.plot(cluster_data['speed_edges'],cluster_data['speed_rates'],'k-',cluster_data['speed_edges'],cluster_data['speed_fit_y'],'b-')
        spaces_taken += 2
        
    if ops['run_ahv']:
        ax12 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ymax = int(1.5*np.max(cluster_data['ahv_rates']))
        ax12.set_ylim([0,ymax])
        ax12.set_xlim([-400,400])
        ax12.set_xticks([-400,-200,0,200,400])
        ax12.set_yticks([0,ymax/2,ymax])  
        ax12.plot(cluster_data['ahv_angles'],cluster_data['ahv_rates'],'ko')
        spaces_taken += 2
        
    if ops['hd_map']:
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        
        ax13 = plt.subplot2grid((rows,cols), ((int(2*np.floor(spaces_taken/cols)), int(spaces_taken%cols))), rowspan=2, colspan=2)
        ax13.axis('scaled')
        ax13.set_ylim([min(trial_data['center_y']),max(trial_data['center_y'])])
        ax13.set_xlim([min(trial_data['center_x']),max(trial_data['center_x'])])  
        ax13.set_xticks([])
        ax13.set_yticks([])           
        ax13.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['spike_angles'],cmap=colormap,norm=norm)
        spaces_taken += 2

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    self.figure.clear()

    plt.show()

