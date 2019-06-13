# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:11:21 2018

@author: Patrick
"""

import numpy as np

def read_taube(fname):
    
    a=np.fromfile(fname,dtype=np.uint8)
    a=a.astype(np.float)
    a[a==255] = 0
    a[a==0] = 0
    
    red_x = a[0::6]
    red_y = a[1::6]
    green_x = a[2::6]
    green_y = a[3::6]
    spike_1 = a[4::6]
#    spike_2 = a[5::6]


    rows = []
    for i in range(len(red_x)):
        rows.append([i,float(red_x[i]),float(red_y[i]),float(green_x[i]),float(green_y[i])])
    
    angles = compute_angles(rows)
    
    raw_vdata = {}
    raw_vdata['positions'] = rows
    raw_vdata['angles'] = angles
    
    spike_data = {}
    ani_spikes = np.array(spike_1,dtype=np.int)
    ani_spikes[ani_spikes>=16] = 0
    spike_data['ani_spikes'] = ani_spikes
    
    return raw_vdata,spike_data


def write_taube(trial_data,spike_train):
    
    positions = np.array(trial_data['positions'])
    
#    timestamps = positions[:,0]
    red_x = positions[:,1]
    red_y = positions[:,2]
    green_x = positions[:,3]
    green_y = positions[:,4]
    
    min_x = np.min((np.min(red_x),np.min(green_x)))
    min_y = np.min((np.min(red_y),np.min(green_y)))
    
    red_x = red_x - min_x
    red_y = red_y - min_y
    green_x = green_x - min_x
    green_y = green_y - min_y

    max_x = np.max((np.max(red_x),np.max(green_x)))
    max_y = np.max((np.max(red_y),np.max(green_y)))
    
    red_x = red_x * 255. / max_x
    green_x = green_x * 255. / max_x
    red_y = 255. - red_y * 255. / max_y
    green_y = 255. - green_y * 255. / max_y
    
    data = []
    for i in range(len(spike_train)):
        data.append(red_x[i])
        data.append(red_y[i])
        data.append(green_x[i])
        data.append(green_y[i])
        data.append(spike_train[i])
        data.append(0)
        
    fname = 'C:/Users/Jeffrey_Taube/Desktop/labview_file'
    data = np.array(data,dtype=np.uint8)
    data.tofile(fname)
    


def compute_angles(rows):
    
    angles = []
    
    for row in rows:
        angle = (360 - np.rad2deg(np.arctan2(row[2]-row[4],row[1]-row[3])))%360
        angles.append(angle)
        
    return angles