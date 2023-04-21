# -*- coding: utf-8 -*-
"""
Created on Sun Oct 07 13:43:55 2018

egocentric functions for spike_analysis

@author: Patrick
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as mplcolors
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
from spike_analysis import spatial

def plot_ego(ops,adv,trial_data,cluster_data,spike_data,self,raw=False,center_mass=False,quiver=False):
    
    if ops['save_all']:
        egopath = cluster_data['new_folder'] + '/egocentric'
    
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
    
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
        
    colormap = plt.get_cmap('jet')
    norm = mplcolors.Normalize(vmin=0, vmax=np.nanmax(ego_rayleighs))

    im = ax.imshow(ego_rayleighs,cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    if center_mass:
        ax.plot(center_mass_x,center_mass_y,'kx')

    self.figure.colorbar(im, cax=cax, orientation='vertical')

    if ops['save_all'] and not cluster_data['saved']['plot_ego']:
        self.figure.savefig('%s/Ego Rayleighs.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ego'] = True
        
    ax.set_title('Egocentric bearing MVLs')
        
    
def plot_ego_angle(ops,adv,trial_data,cluster_data,spike_data,self,raw=False,center_mass=False,pinwheel_center=False):
    
    if ops['save_all']:
        egopath = cluster_data['new_folder'] + '/egocentric'
    
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
    
    xedges = trial_data['heat_xedges']
    yedges = trial_data['heat_yedges']
    
    center_mass_x = sum(spike_data['spike_x'])/len(spike_data['spike_x'])
    center_mass_y = sum(spike_data['spike_y'])/len(spike_data['spike_y'])
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    ax = self.figure.add_subplot(111)
    self.figure.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    if raw:
        ego_mean_angles = cluster_data['ego_raw_mean_angles']
    else:
        ego_mean_angles = cluster_data['ego_mean_angles']

    im = ax.imshow(ego_mean_angles,cmap=colormap,norm=norm,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    if center_mass:
        ax.plot(center_mass_x,center_mass_y,'kx')

    cbar = self.figure.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
        
    if ops['save_all'] and not cluster_data['saved']['plot_ego_angle']:
        self.figure.savefig('%s/Ego Mean Angles.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ego_angle'] = True
        
    ax.set_title('Egocentric bearing mean angles')

def plot_ebc(ops,adv,trial_data,cluster_data,spike_data,self):
    
    if ops['save_all']:
        egopath = cluster_data['new_folder'] + '/egocentric'
    
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
        
    ebc_hist = cluster_data['ebc_hist']
    top_angle,top_dist = np.where(ebc_hist==np.max(ebc_hist))
    
    real_xvals = cluster_data['ebc_radii'] + (cluster_data['ebc_radii'][1] - cluster_data['ebc_radii'][0])/2.

    ax = self.figure.add_subplot(111,projection='polar')  
    self.figure.tight_layout(pad=2.5)
    ax.set_theta_zero_location("N")
    
    ax.text(-.2,1.1,'Pref dist = %scm' % np.round(float(real_xvals[top_dist]),2),transform=ax.transAxes)
    ax.text(-.2,1.05,'MRL = %s' % np.round(cluster_data['ebc_mrl'],4),transform=ax.transAxes)
    ax.text(-.2,1,'MRA = %s$^\circ$' % np.round(cluster_data['ebc_mra'],2),transform=ax.transAxes)

    ax.pcolormesh(cluster_data['ebc_ref_angles'],cluster_data['ebc_radii'],ebc_hist.T,vmin=0)


    if ops['save_all'] and not cluster_data['saved']['plot_ebc']:

        self.figure.savefig('%s/EBC.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_ebc'] = True
        
    ax.text(.33,1.1,'EBC ratemap',transform=ax.transAxes)

    if ops['save_all']:
        
        fig=plt.figure()
        ax = fig.add_subplot(111)
        top_angle_rates = ebc_hist[top_angle].flatten()
        ax.plot(cluster_data['ebc_radii'], top_angle_rates, 'k-')
        ax.set_ylim([0, 1.2 * np.max(top_angle_rates) + 5])
        ax.set_xlim([0,np.max(cluster_data['ebc_radii'])])
        ax.set_xlabel('egocentric distance (cm)')
        ax.set_ylabel('firing rate (hz)')
        ax.text(.5,.9,'peak dist = %s' % round(float(real_xvals[top_dist]),2),transform=ax.transAxes)
        fig.savefig('%s/EBC dists along best angle.png' % egopath,dpi=adv['pic_resolution'])
        
        plt.close()
        
        top_dist_rates = ebc_hist[:,top_dist].flatten()
        
        real_angles = cluster_data['ebc_ref_angles'] + (cluster_data['ebc_ref_angles'][1] - cluster_data['ebc_ref_angles'][0])/2.
        rayleigh, mean_angle = spatial.rayleigh_r(np.rad2deg(real_angles),top_dist_rates)
        
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.rad2deg(cluster_data['ebc_ref_angles']), top_dist_rates, 'k-')
        ax.set_ylim([0,1.2*np.max(top_dist_rates)+5])
        ax.set_xlim([0,360])
        ax.set_xticks([0,90,180,270,360])
        ax.set_xlabel('egocentric bearing (degrees)')
        ax.set_ylabel('firing rate (hz)')
        ax.text(.5,.9,'rayleigh r = %s' % round(rayleigh,5),transform=ax.transAxes)
        ax.text(.5,.8,'rayleigh angle = %s$^\circ$' % round(mean_angle,5),transform=ax.transAxes)
        fig.savefig('%s/EBC angles along best dist.png' % egopath,dpi=adv['pic_resolution'])
        
        plt.close()
    
def plot_md_ebc(ops,adv,trial_data,cluster_data,spike_data,self):
    
    if ops['save_all']:
        egopath = cluster_data['new_folder'] + '/egocentric'
    
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
        
    ebc_hist = cluster_data['ebc_md_hist']
        
    ax = self.figure.add_subplot(111,projection='polar')  
    self.figure.tight_layout(pad=2.5)
    ax.set_theta_zero_location("N")
    ax.text(-.2,1.1,'MRL = %s' % np.round(cluster_data['ebc_md_mrl'],4),transform=ax.transAxes)
    ax.text(-.2,1,'MRA = %s$^\circ$' % np.round(cluster_data['ebc_md_mra'],2),transform=ax.transAxes)

    ax.pcolormesh(cluster_data['ebc_ref_angles'],cluster_data['ebc_radii'],ebc_hist.T,vmin=0)

    if ops['save_all'] and not cluster_data['saved']['plot_md_ebc']:

        self.figure.savefig('%s/EBC_md.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_md_ebc'] = True

    top_angle,top_dist = np.where(ebc_hist==np.max(ebc_hist))
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ebc_hist[top_angle].flatten())
    ax.set_ylim(0,1.2*np.max(ebc_hist[top_angle])+5)
    fig.savefig('%s/EBC top wall dists_md.png' % egopath,dpi=adv['pic_resolution'])
    
    plt.close()
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ebc_hist[:,top_dist].flatten())
    ax.set_ylim(0,1.2*np.max(ebc_hist[:,top_dist])+5)
    fig.savefig('%s/EBC top wall angles_md.png' % egopath,dpi=adv['pic_resolution'])

    plt.close()

def plot_conjunctive(ops,adv,trial_data,cluster_data,spike_data):
    
    egopath = cluster_data['new_folder'] + '/egocentric'

    if not os.path.isdir(egopath):
        os.makedirs(egopath)
    
    angles = np.array(trial_data['angles'])
    center_bearings = np.array(trial_data['center_bearings'])
    spike_train = np.array(spike_data['ani_spikes'])
    
    bins = 30
    
    angle_bins = np.digitize(angles,np.linspace(0,360,bins,endpoint=False)) - 1
    ego_bins = np.digitize(center_bearings,np.linspace(0,360,bins,endpoint=False)) - 1
    
    spikes = np.zeros((bins,bins))
    occ = np.zeros((bins,bins))
    
    for i in range(len(spike_train)):
        occ[angle_bins[i],ego_bins[i]] += 1./adv['framerate']
        spikes[angle_bins[i],ego_bins[i]] += spike_train[i]
        
    occ[occ<5./adv['framerate']] = 0
    occ[occ==0] = np.nan
    heatmap = spikes/occ
    
    angle0 = heatmap[0]
    heatmap = np.concatenate((heatmap, angle0[np.newaxis,:]))
    
    angle0 = heatmap[:,0]
    heatmap = np.concatenate((heatmap,angle0[:,np.newaxis]),axis=1)
        
    heatmap = spatial.interp_raw_heatmap(heatmap)

    fig=plt.figure()
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=2.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(heatmap, origin='lower') 
    
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('center bearing (deg)')
    
    ax.set_yticks([0,7.5,15,22.5,30])
    ax.set_yticklabels([0,90,180,270,360])
    ax.set_ylabel('head direction (deg)')
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    fig.savefig('%s/HD x center bearing.png' % egopath,dpi=adv['pic_resolution'])
    
    plt.close()


def plot_center_bearing(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
    #set y limit according to highest firing rate
    ymax = int(1.2*np.nanmax(cluster_data['center_bearing_curve']))+5
    
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
        
    ax.set_xlabel('center bearing (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,90))
    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % np.round(cluster_data['center_bearing_rayleigh'],4),transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s$^\circ$' % np.round(cluster_data['center_bearing_mean_angle'],2),transform=ax.transAxes)

    if ops['save_all']:
        egopath = cluster_data['new_folder']+'/egocentric'
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
        
    curve = list(cluster_data['center_bearing_curve'])
    curve.append(curve[0])

    ax.plot(np.linspace(0,360,adv['hd_bins']+1,endpoint=True),curve,'k-')
    
    if ops['save_all'] and not cluster_data['saved']['plot_center_bearing']:
        self.figure.savefig('%s/Center bearing.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_bearing'] = True
        
    ax.set_title('Center bearing')

        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    angles = np.linspace(0,2*np.pi,adv['hd_bins'],endpoint=False)
    angles = np.append(angles,2*np.pi)
    curve = np.append(cluster_data['center_bearing_curve'],cluster_data['center_bearing_curve'][0])
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
        fig.savefig('%s/Center bearing polar.png' % egopath,dpi=adv['pic_resolution'])
     
    plt.close()
        
    occ_hist = cluster_data['center_bearing_occ']
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
        fig.savefig('%s/Center bearing occupancy polar.png' % egopath,dpi=adv['pic_resolution'])
            
    plt.close()
    
    if ops['save_all']:
        plot_center_bearing_hd_map(ops,adv,trial_data,cluster_data,spike_data)
        plot_conjunctive(ops,adv,trial_data,cluster_data,spike_data)
        

def plot_center_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(cluster_data['center_dist_curve']))+5
    ax.set_ylim([0,ymax])
    ax.set_xlabel('center distance (%s)' % adv['dist_measurement'])
    ax.set_ylabel('firing rate (hz)')

    ax.plot(cluster_data['center_dist_xvals'],cluster_data['center_dist_curve'],'k-')
#    ax.plot(cluster_data['center_dist_xvals'],cluster_data['center_dist_fit'],color='gray',linestyle='--',alpha=0.6)
        
    if ops['save_all'] and not cluster_data['saved']['plot_center_dist']:
        
        egopath = cluster_data['new_folder']+'/egocentric'

        self.figure.savefig('%s/Center distance.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_dist'] = True
        
    ax.set_title('Center distance')

        
def plot_center_bearing_hd_map(ops,adv,trial_data,cluster_data,spike_data):

    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)

    #make a scatter plot of spike locations colored by head direction
    ax.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['center_bearings'],cmap=colormap,norm=norm,clip_on=False)
    
    ax.axis('equal')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x])
    
    ax.axis('off')
    
    egopath = cluster_data['new_folder']+'/egocentric'
    
    #save it
    fig.savefig('%s/Spike location x center bearing.png' % egopath,dpi=adv['pic_resolution'])
    
    plt.close()
    
    
def plot_center_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
        
    center_bearings = trial_data['center_bearings']
    center_dists = trial_data['center_dists']
    center_spike_bearings = spike_data['center_bearings']
    center_spike_dists = spike_data['center_dists']
    
    hd_bins = adv['hd_bins']
    dist_bin_size = adv['ego_dist_bin_size']
    
    occ_hist,xedges,yedges = np.histogram2d(center_bearings,center_dists,bins=[np.linspace(0,360,hd_bins),np.arange(0,np.nanmax(center_dists),dist_bin_size)])

    occ_hist[occ_hist<3] = 0
    spike_hist,xedges,yedges = np.histogram2d(center_spike_bearings,center_spike_dists,bins=[xedges,yedges])
    hist = spike_hist/occ_hist
    hist[np.isinf(hist)] = np.nan
    
    hist3 = np.concatenate((hist,hist,hist),axis=0)
    hist3 = convolve(hist3,Gaussian2DKernel(x_stddev=2,y_stddev=2))
    new_hist = hist3[len(hist):len(hist)*2]
        
    ax = self.figure.add_subplot(111,projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_yticks([])
    self.figure.tight_layout(pad=2.5)

    ax.pcolormesh(np.deg2rad(xedges),yedges,new_hist.T) 
            
    if ops['save_all'] and not cluster_data['saved']['plot_center_ego_map']:
        
        egopath = cluster_data['new_folder']+'/egocentric'

        self.figure.savefig('%s/Center ego ratemap.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_center_ego_map'] = True
        
    ax.text(.23,1.1,'Egocentric ratemap - center',transform=ax.transAxes)

    
        
def plot_wall_bearing(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)
    self.figure.subplots_adjust(hspace=.20, wspace=0.20, bottom=0.18, left=0.13, top=.80, right=0.95)
    #set y limit according to highest firing rate
    ymax = 1.2*np.nanmax(cluster_data['wall_bearing_curve'])+5.
    
    ax.set_xlim([0,360])
    ax.set_ylim([0,ymax])
    
    ax.set_xlabel('wall bearing (degrees)')
    ax.set_ylabel('firing rate (hz)')
    ax.set_xticks(range(0,361,90))


    #print rayleigh r, rayleigh pfd, fit pfd values on graph
    ax.text(.1,.9,'rayleigh r = %s' % np.round(cluster_data['wall_bearing_rayleigh'],4),transform=ax.transAxes)
    ax.text(.1,.8,'rayleigh angle = %s$^\circ$' % np.round(cluster_data['wall_bearing_mean_angle'],2),transform=ax.transAxes)

    curve = list(cluster_data['wall_bearing_curve'])
    curve.append(curve[0])

    ax.plot(np.linspace(0,360,adv['hd_bins']+1,endpoint=True),curve,'k-')
    
    if ops['save_all']:
        egopath = cluster_data['new_folder']+'/egocentric'
        if not os.path.isdir(egopath):
            os.makedirs(egopath)
    
    if ops['save_all'] and not cluster_data['saved']['plot_wall_bearing']:
        self.figure.savefig('%s/Wall bearing.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_bearing'] = True
        
    ax.set_title('Closest wall bearing')
    
    if ops['save_all']:

        plot_wall_bearing_hd_map(ops,adv,trial_data,cluster_data,spike_data)
        
        
def plot_wall_dist(ops,adv,trial_data,cluster_data,spike_data,self):
    
    ax = self.figure.add_subplot(111)  
    self.figure.tight_layout(pad=2.5)
    ymax = int(1.2*np.nanmax(cluster_data['wall_dist_curve']))+5
    ax.set_ylim([0,ymax])
    ax.set_xlabel('wall distance (%s)' % adv['dist_measurement'])
    ax.set_ylabel('firing rate (hz)')

    ax.plot(cluster_data['wall_dist_xvals'],cluster_data['wall_dist_curve'],'k-')
#    ax.plot(cluster_data['wall_dist_xvals'],cluster_data['wall_dist_fit'],color='gray',linestyle='--',alpha=0.6)
        
    if ops['save_all'] and not cluster_data['saved']['plot_wall_dist']:
        
        egopath = cluster_data['new_folder']+'/egocentric'

        self.figure.savefig('%s/Wall distance.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_dist'] = True
        
    ax.set_title('Closest wall distance')

        
        
def plot_wall_bearing_hd_map(ops,adv,trial_data,cluster_data,spike_data):

    axis_range = np.max([np.max(trial_data['center_x'])-np.min(trial_data['center_x']),np.max(trial_data['center_y'])-np.min(trial_data['center_y'])])
    min_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) - axis_range/2.
    max_y = np.mean([np.max(trial_data['center_y']),np.min(trial_data['center_y'])]) + axis_range/2.
    min_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) - axis_range/2.
    max_x = np.mean([np.max(trial_data['center_x']),np.min(trial_data['center_x'])]) + axis_range/2.
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #create the color map for the plot according to the range of possible angles
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)

    #make a scatter plot of spike locations colored by head direction
    ax.plot(trial_data['center_x'],trial_data['center_y'],color='gray',alpha=0.5,zorder=0)
    im=ax.scatter(spike_data['spike_x'],spike_data['spike_y'],c=spike_data['wall_bearings'],cmap=colormap,norm=norm,zorder=1,clip_on=False)
    
    ax.axis('equal')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0,90,180,270,360])
    
    #axes set to min and max x and y values in dataset
    ax.set_ylim([min_y,max_y])
    ax.set_xlim([min_x,max_x])

    ax.axis('off')
        
    egopath = cluster_data['new_folder']+'/egocentric'
    
    #save it
    fig.savefig('%s/Spike location x wall bearing.png' % egopath,dpi=adv['pic_resolution'])
    
    plt.close()
        
def plot_wall_ego_map(ops,adv,trial_data,cluster_data,spike_data,self):
    
    wall_bearings = trial_data['wall_bearings']
    wall_dists = trial_data['wall_dists']
    wall_spike_bearings = spike_data['wall_bearings']
    wall_spike_dists = spike_data['wall_dists']
    
    hd_bins = adv['hd_bins']
    dist_bin_size = adv['ego_dist_bin_size']
    
    occ_hist,xedges,yedges = np.histogram2d(wall_bearings,wall_dists,bins=[np.linspace(0,360,hd_bins),np.arange(0,np.nanmax(wall_dists),dist_bin_size)])
    occ_hist[occ_hist<3] = 0
    spike_hist,xedges,yedges = np.histogram2d(wall_spike_bearings,wall_spike_dists,bins=[xedges,yedges])
    hist = spike_hist/occ_hist
    hist[np.isinf(hist)] = np.nan
    
    hist3 = np.concatenate((hist,hist,hist),axis=0)
    hist3 = convolve(hist3,Gaussian2DKernel(x_stddev=2,y_stddev=2))
    new_hist = hist3[len(hist):len(hist)*2]
        
    ax = self.figure.add_subplot(111,projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_yticks([])
    self.figure.tight_layout(pad=2.5)

    ax.pcolormesh(np.deg2rad(xedges),yedges,new_hist.T)
    
    if ops['save_all'] and not cluster_data['saved']['plot_wall_ego_map']:
        
        egopath = cluster_data['new_folder']+'/egocentric'

        self.figure.savefig('%s/Wall Ego Map.png' % egopath,dpi=adv['pic_resolution'])
        cluster_data['saved']['plot_wall_ego_map'] = True
        
    ax.text(.18,1.1,'Egocentric ratemap - closest wall',transform=ax.transAxes)

        