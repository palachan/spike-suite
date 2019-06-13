# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 16:31:23 2018

rat visual field simulator

@author: Patrick
"""

import os
import numpy as np
import collect_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage.filters import gaussian_filter1d

resolution = 64

def make_arena():

    wall_coords = np.zeros([resolution*4-4,2])
    gray_vals = np.zeros(resolution*4-4)
    
    for i in range(resolution):
        wall_coords[i][0] = i
    for i in range(resolution-1,(resolution*2-1)):
        wall_coords[i][0] = resolution-1
        wall_coords[i][1] = i - (resolution-1)
    for i in range((resolution*2-2),(resolution*3-2)):
        wall_coords[i][0] = 3*resolution - i - 3
        wall_coords[i][1] = resolution - 1
    for i in range((resolution*3-3),len(wall_coords)):
        wall_coords[i][1] = 4*resolution - i - 4
        
        
    gray_vals[:] = 122
    corners = [0,resolution-1,2*resolution-2,3*resolution-3]
    
    for c in corners:
        gray_vals[c-2] = 100
        gray_vals[c-1] = 50
        gray_vals[c] = 0
        gray_vals[c+1] = 50
        gray_vals[c+2] = 100
        
    gray_vals[int(resolution*.1):int(resolution*.9)] = 255
    
    return gray_vals, wall_coords

def animate_vis(trial_data,gray_vals,wall_coords,fdir):

    center_x = trial_data['center_x']
    center_y = trial_data['center_y']
    angles = trial_data['angles']
    
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

    field_angles = np.arange(-108,112,3)%360
    field_vals = np.zeros((len(center_x),len(field_angles)))
    
    real_coords = np.zeros_like(wall_coords)
    real_coords[:,0] = (np.max(center_x) - np.min(center_x)) * wall_coords[:,0].astype(np.float)/np.float(resolution) + np.min(center_x)
    real_coords[:,1] = (np.max(center_y) - np.min(center_y)) * wall_coords[:,1].astype(np.float)/np.float(resolution) + np.min(center_y)
    
    for i in range(len(center_x)):
        new_angles = np.rad2deg(np.arctan2((real_coords[:,1]-center_y[i]),(real_coords[:,0]-center_x[i])))%360
        #calculate ego angles by subtracting allocentric
        #angles from egocentric angles
        ego_angles = (new_angles-angles[i])%360
        
        for j in range(len(field_angles)):
            errors = np.abs(ego_angles-field_angles[j])
            field_vals[i][j] = gray_vals[np.min(np.where(errors == np.min(errors))[0])]
        
#        valid_spots[i] = np.where(np.logical_or(ego_angles<112, ego_angles>248))[0]
#        diffs = np.abs(np.ediff1d(valid_spots[i]))
#        valid_spots[i] = np.roll(valid_spots[i],-1*(np.where(diffs>1)[0]+1))
        
    
#    fig = plt.figure()
#    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
#    ax2 = plt.subplot2grid((1, 3), (0, 2))
#
#    ax2.set_ylim([np.min(center_y),np.max(center_y)])
#    ax2.set_xlim([np.min(center_x),np.max(center_x)])
#    
#    line = ax.imshow(np.stack([field_vals[0],field_vals[0]]),vmin=0,vmax=255,cmap='gray',extent=[-112,112, 0, 100])
#    liner, = ax2.plot([],[],'r.')
#    lineg, = ax2.plot([],[],'g.')
#    
#
#    ax.set_yticks([])
#    ax.set_xticks([-112,0,112])
##    ax.set_xticklabels([-112,0,112])
#    
##    x0,x1 = ax.get_xlim()
##    y0,y1 = ax.get_ylim()
#    ax.set_aspect('equal')
#    ax2.set_aspect('equal')
#    ax2.set_yticks([])
#    ax2.set_xticks([])
#    plt.tight_layout()
    
#    plt.axis('square')
    
        
#    def animateinit():
#        line.set_data(np.stack([field_vals[0],field_vals[0]]))
#        return line,
    
    def animate(n):
        #print what percent of the video is done
        perc = int(len(center_x)/100)
        done = int(n)/int(perc)
        if int(n)%perc==0:
            print('[video %s percent done]' % done)
            
#        data = gray_vals[valid_spots[n]]
#        data = np.stack([data,data])
        line.set_array(np.stack([field_vals[n][::-1],field_vals[n][::-1]]))
        liner.set_data(red_x[n],red_y[n])
        lineg.set_data(green_x[n],green_y[n])
#        line = plt.imshow(np.stack([field_vals[n][::-1],field_vals[n][::-1]]),aspect='auto',vmin=0,vmax=255)
        return line,liner,lineg,
#        
    
#    ani = FuncAnimation(fig, animate, frames=len(center_x), interval=0, blit=False)
#    #save the video as an mp4 at 2x original speed
#    ani.save('%s/heatmap_animation.mp4' % fdir,fps=30)
    
    
    
    return field_vals

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

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
c=[]
#c.append('H:/Patrick/PL16/V2L/2017-03-16_14-26-25/TT4_SS_03.txt')
fname='H:/Patrick/PL16/PoR/2017-03-24_16-21-20/TT2_SS_02.txt'
#c.append('C:/Users/Jeffrey_Taube/desktop/adn_hdc/PL5/ADN/trial_1/ST4_SS_01.txt')

#trial = '2017-03-16_14-26-25'
tracking_fdir = 'H:/Patrick/PL16/PoR/2017-03-24_16-21-20'

#collect names of the clusters we'll be analyzing
trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)

cluster_data = {}
spike_list = collect_data.ts_file_reader(fname)
spike_train = collect_data.create_spike_train(trial_data,spike_list)


center_y = np.array(trial_data['center_y'])
center_x = np.array(trial_data['center_x'])
angles = np.array(trial_data['angles'])


for i in range(8):
    
    spike_x = []
    spike_y = []
    
    
    if i == 0:
        angle_min = 337.5
        angle_max = 22.5
        for i in range(len(spike_train)):
            if (angles[i]>337.5 or angles[i]<22.5) and spike_train[i]>=1:
                spike_x.append(center_x[i])
                spike_y.append(center_y[i])
                
    else:
    
        angle_min = 22.5 + 45*i - 45
        angle_max = angle_min + 45
        
    
        
        for i in range(len(spike_train)):
            if angles[i] >= angle_min and angles[i] < angle_max and spike_train[i]>=1:
                spike_x.append(center_x[i])
                spike_y.append(center_y[i])
                
    plt.figure()
    plt.plot(spike_x,spike_y,'r.')
    plt.title('%d to %d' % (angle_min,angle_max))
    plt.show()
    
    



curves = np.zeros((8,8,30))
spikes = np.zeros((8,8,30))
occs = np.zeros((8,8,30))

xbins = np.digitize(center_x,np.arange(min(center_x),max(center_x),(max(center_x)-min(center_x))/8.)) - 1
ybins = np.digitize(center_y,np.arange(min(center_y),max(center_y),(max(center_y)-min(center_y))/8.)) - 1
hdbins = np.digitize(angles,np.arange(0,360,12)) - 1


for i in range(len(center_y)):
    occs[xbins[i],ybins[i],hdbins[i]] += 1.
    spikes[xbins[i],ybins[i],hdbins[i]] += spike_train[i]
    
curves = spikes/occs


def nan_helper(y_vals):
    ''' returns where NaNs are for use by np.interp function '''
    return np.isnan(y_vals), lambda z: z.nonzero()[0]

for i in range(8):
    for j in range(8):
        nans, x = nan_helper(curves[i][j])
        curves[i][j][nans] = np.interp(x(nans), x(~nans), curves[i][j][~nans], period=30)



rayleighs = np.zeros((8,8))
mean_angles = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        rayleighs[i][j],mean_angles[i][j]=rayleigh_r(np.arange(0,360,12),curves[i][j])     

#fig = plt.figure()
        
peaks = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        peaks[i][j] = np.where(curves[i][j]==np.max(curves[i][j]))[0][0] * 12

fig2,axes = plt.subplots(8,8)

max_param = np.max(curves)
for i in range(8):
    for j in range(8):
        axes[i,j].plot(curves[i][j])
        axes[i,j].set_yticks([])
        axes[i,j].set_xticks([])
        axes[i,j].set_ylim([0,max_param])
#fig2.savefig('%s/ego_plots' % cluster_img_dir,dpi=900)
plt.show()

cluster_names = c
spike_train = {}
#for each cluster...
for name in cluster_names:

    ts_file = name
    
    cluster_data = {}
    spike_list = collect_data.ts_file_reader(ts_file)
    spike_train[name] = collect_data.create_spike_train(trial_data,spike_list)

#spike_train[cluster_names[1]] = np.clip(spike_train[cluster_names[0]] + spike_train[cluster_names[1]] - 2,0,8)
center_x = np.array(trial_data['center_x'])
center_y = np.array(trial_data['center_y'])
angles = np.array(trial_data['angles'])




a=np.array(np.roll(center_y,1)-center_y,dtype=np.float)
a[0] = 0
a=gaussian_filter1d(a,sigma=5)

b=np.array(np.roll(center_x,1)-center_x,dtype=np.float)
b[0] = 0
b=gaussian_filter1d(b,sigma=5)
for i in range(len(a)):
    if a[i]==0 and b[i]==0:
        a[i]=np.nan
        b[i]=np.nan

movement_dirs = np.rad2deg(np.arctan2(a,b))%360

hdbins = np.arange(0,360,6)
dirbins = np.digitize(movement_dirs,hdbins)-1
hdvals = np.zeros(60)
hdocc = np.zeros(60)

train = spike_train[cluster_names[0]]
for i in range(len(train)):
    hdvals[dirbins[i]] += np.float(train[i])
    hdocc[dirbins[i]] += 1.
    
hd_curve = hdvals*30./hdocc
plt.figure()
plt.ylim([0,15])
plt.plot(hd_curve)
plt.show()



spike_x = []
spike_y = []

for n in range(len(spike_train[c[0]])):
    if spike_train[c[0]][n] > 0:
        spike_x.append(center_x[n])
        spike_y.append(center_y[n])

plt.figure()
plt.plot(spike_x,spike_y,'r.')
plt.show()

angles = trial_data['angles']

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
    
    
fig = plt.figure()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax2.set_ylim([np.min(center_y),np.max(center_y)])
ax2.set_xlim([np.min(center_x),np.max(center_x)])

ax.set_ylim([np.min(center_y),np.max(center_y)])
ax.set_xlim([np.min(center_x),np.max(center_x)])

ax.set_aspect('equal')
ax2.set_aspect('equal')

liner1, = ax.plot([],[],'r*')
lineg1, = ax.plot([],[],'g*')
lines1, = ax.plot([],[],'r.')
liner2, = ax2.plot([],[],'r*')
lineg2, = ax2.plot([],[],'g*')
lines2, = ax2.plot([],[],'g.')

spikes1 = []
spikes2 = []

def animate(n):
    #print what percent of the video is done
    perc = int(len(center_x)/100)
    done = int(n)/int(perc)
    if int(n)%perc==0:
        print('[video %s percent done]' % done)
        
    if spike_train[c[0]][n] > 0:
        spikes1.append((center_x[n],center_y[n]))
    s1=np.array(spikes1)
    if spike_train[c[1]][n] > 0:
        spikes2.append((center_x[n],center_y[n]))
    s2=np.array(spikes2)

    liner1.set_data(red_x[n],red_y[n])
    liner2.set_data(red_x[n],red_y[n])
    lineg1.set_data(green_x[n],green_y[n])
    lineg2.set_data(green_x[n],green_y[n])
    
    if len(s1) > 0:
        lines1.set_data(s1[:,0],s1[:,1])
    if len(s2) > 0:
        lines2.set_data(s2[:,0],s2[:,1])
#        line = plt.imshow(np.stack([field_vals[n][::-1],field_vals[n][::-1]]),aspect='auto',vmin=0,vmax=255)
    return lines1,liner1,lineg1,lines2,liner2,lineg2,
#        

ani = FuncAnimation(fig, animate, frames=len(center_x), interval=0, blit=False)
#save the video as an mp4 at 2x original speed
ani.save('%s/heatmap_animation.mp4' % tracking_fdir,fps=60)
    


if __name__ == '__main__':
    
    gray_vals, wall_coords = make_arena()
    arena_vals = np.stack([gray_vals]*20)
    
    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC']

    fdir = 'H:/Patrick'
    
    fig = plt.figure()
    
    for animal in os.listdir(fdir):
        animaldir = fdir + '/' + animal
        print animaldir
        
        for area in os.listdir(animaldir):
            if area in areas:
                areadir = animaldir + '/' + area
                print areadir
                
                for trial in os.listdir(areadir):
    
    
#    fdir = 'C:/Users/Jeffrey_Taube/Desktop/mec ego'
#    trial = fdir

                    #collect names of the clusters we'll be analyzing
                    trial_data = collect_data.tracking_stuff(areadir,areadir + '/' + trial)
                    cluster_names = trial_data['filenames']
                    
                    vs = animate_vis(trial_data,gray_vals,wall_coords,areadir + '/' + trial)
                    
                    cluster_names = trial_data['filenames']
                
                    #for each cluster...
                    for name in cluster_names:
                
                        ts_file = areadir + '/' + trial + '/' + name + '.txt'
                        
                        cluster_data = {}
                        cluster_data['spike_list'] = collect_data.ts_file_reader(ts_file)
                        spike_data, cluster_data = collect_data.create_spike_lists(trial_data,cluster_data)
                    
                        spike_train = spike_data['ani_spikes']
                        
                        result = np.zeros_like(vs[0])
                        
                        for i in range(len(vs[0])):
                            result[i] = np.sum(vs[:,i] * spike_train) / np.sum(spike_train)
                            
                        
                        plt.clf()
                        plt.imshow(np.stack([result,result]),vmin=0,vmax=255,cmap='gray',extent=[-112,112, 0, 100])
                        
                        savedir = areadir + '/' + trial + '/' + name
                        if not os.path.isdir(savedir):
                            os.makedirs(savedir)
                        
                        fig.savefig('%s/VIS.png' % savedir)
                    
    
    