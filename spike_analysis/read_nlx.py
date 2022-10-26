# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:49:39 2017

-read_nvt pulled from og script:
"Extracts LED data from .NVT video file

adapted from matlab code written by Susan Schwarz"

@author: Patrick
"""

import numpy as np
import csv

def read_nvt(adv,filename,trial_data):
    #TODO: rewrite this in simpler way
    """set stuff up"""
    
    
    fid = open(filename, 'r')
    header = {}
    header_string = fid.read(1024).splitlines()
    for pair in header_string:
        if str(pair).startswith('-SamplingFrequency '):
            header['fs'] = np.float(pair[19:])
        elif str(pair).startswith('-CheetahRev '):
            header['cheetah_ver'] = str(pair[12:])
        elif str(pair).startswith('-EnableFieldEstimation '):
            header['field_est'] = str(pair[23:])
        elif str(pair).startswith('-DirectionOffset '):
            header['direction_offset'] = str(pair[17:])
            
    if 'fs' in header.keys():
        if 'field_est' in header and header['field_est'] == 'True':
            adv['framerate'] = header['fs'] * 2.
        else:
            adv['framerate'] = header['fs']
            
    if 'direction_offset' in header.keys():
        adv['offset'] = header['direction_offset']
    
    
    #specify the nvt datatypes
    nvt_dtype = np.dtype([ 
        ('swstx'  , '<u2'), 
        ('swid'  , '<u2'), 
        ('swdata_size', '<u2'), 
        ('timestamp', '<u8'),
        ('dwPoints', '<u4', (400,)),
        ('sncrc', '<i2'),
        ('dnextracted_x', '<i4'),
        ('dnextracted_y', '<i4'),
        ('dnextracted_angle', '<i4'),
        ('dntargets', '<i4', (50,)),
    ]) 
    #memmap the file
    mmap = np.memmap(filename, dtype=nvt_dtype, mode='r+', 
       offset=(16 * 2**10))
    
    timestamp = np.array(mmap['timestamp']).astype(np.float)
    dntargets = np.array(mmap['dntargets'])
    
    red_x = np.zeros_like(timestamp)
    red_y = np.zeros_like(timestamp)
    green_x = np.zeros_like(timestamp)
    green_y = np.zeros_like(timestamp)
    
    for i in range(len(timestamp)):
        if (i+1)%5000 == 0:
            print('processing record # %s of %d' % ((i+1),len(timestamp)))
        for j in range(4):
            target = dntargets[i,j]
            bin_target=format(target,'032b')
            red = bin_target[1]
            green = bin_target[2]
            y = int(bin_target[4:16],2)
            x = int(bin_target[20:],2)
            if red == '1':
                red_x[i] = x
                red_y[i] = y
            elif green == '1':
                green_x[i] = x
                green_y[i] = y
                
    angles = np.array(mmap['dnextracted_angle'])
    
    """write data to file"""
    video_txt_file = filename[:len(filename)-4] + '.txt'
    rows = []
    with open(video_txt_file,'w',newline='') as f:
        for i in range(len(red_x)): 
            rows.append([int(timestamp[i]),int(red_x[i]),int(red_y[i]),int(green_x[i]),int(green_y[i])])
            f.write('%i\t%i\t%i\t%i\t%i\n' % (int(timestamp[i]),int(red_x[i]),int(red_y[i]),int(green_x[i]),int(green_y[i])))
            
    #return rows,dnextracted_angle.tolist()[0]
    raw_vdata = {}
    
    raw_vdata['positions'] = rows
    
    nlx_angles = np.array(angles,dtype=np.float).flatten()
    offset = np.float(adv['offset'])
    nondetects = np.where(nlx_angles==0)[0]
    
    #this flips the angles to increase counterclockwise starting at the positive x-axis,
    #taking into account the HD offset value set in cheetah
    angles = (360. - (nlx_angles - offset - 180.))%360
    angles[nondetects] = 450
    
    raw_vdata['angles'] = angles
    trial_data['video_txt_file'] = video_txt_file
    
    return trial_data,raw_vdata,adv
    
def grab_nev_data(filename): 
    ''' get timestamps,ttl ids, and ttl messages from nev file '''
    
    #read file
    f = open(filename, 'rb')
    #skip past header
    f.seek(2 ** 14)
    #specity data types
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('id', '<h'),
                   ('nttl', '<h'), ('filler2', '<h', 3), ('extra', '<i', 8),
                   ('estr', np.dtype('a128'))])
    #grab the data
    temp = np.fromfile(f, dt) 
    #return it
    return temp['time'], temp['nttl'], temp['estr']
    
def read_nev(event_file,trial_data):
    ''' process camera (event) information from nev file '''
    
    #get data from nev file
    ttl_ts, ttl_ids, ttl_msgs = grab_nev_data(event_file)
    
    rows = []
    for i in range(len(ttl_ts)):
        rows.append([int(ttl_ts[i]),int(ttl_ids[i]),str(ttl_msgs[i])])
    #name the event txt file
    event_txt_file = trial_data['trial']+'/events_text.txt'
    #write the event txt file
    writer = csv.writer(open(event_txt_file, 'w', newline=''))
    writer.writerows(rows)
    
    trial_data['event_txt_file'] = event_txt_file
    
    return trial_data