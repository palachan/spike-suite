# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:51:38 2017

read files generated for/by the openephys system

@author: Patrick
"""

import csv
import numpy as np

def read_video_file(video_file,filename,trial_data):
    
    raw_vdata = {}

    timestamp = []
    red_x = []
    red_y = []
    green_x = []
    green_y = []
    
    reader = csv.reader(open(video_file,'r'))
    for row in reader:
        try:
            timestamp.append(float(row[0]))
            red_x.append(float(row[1]))
            red_y.append(float(row[2]))
            green_x.append(float(row[3]))
            green_y.append(float(row[4]))
        except:
            pass
        
    timestamp = np.asarray(timestamp).astype(np.int)

    rows = []
    for i in range(len(timestamp)):
        rows.append([int(timestamp[i]),int(red_x[i]),int(red_y[i]),int(green_x[i]),int(green_y[i])])
    
    angles = compute_angles(rows)
    
    raw_vdata['positions'] = rows
    raw_vdata['angles'] = angles
    trial_data['video_txt_file'] = video_file
    
    return trial_data,raw_vdata

def compute_angles(rows):
    
    angles = []
    
    for row in rows:
        angle = (360 - np.rad2deg(np.arctan2(row[2]-row[4],row[1]-row[3])))%360
        angles.append(angle)
        
    return angles