# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:49:39 2017

-read_nvt pulled from og script:
"Extracts LED data from .NVT video file

adapted from matlab code written by Susan Schwarz"

@author: Patrick
"""

import numpy as np
import os
import csv

def read_nvt(video_file,filename,trial_data):
    #TODO: rewrite this in simpler way
    """set stuff up"""
    
    trial = trial_data['trial']
    
    HEADER_SIZE = 16384;
    RECORD_SIZE = 1828;
    NLX_VTREC_NUM_POINTS = 400;
    NLX_VTREC_NUM_TARGETS = 50;
    
    
    filesize = os.path.getsize(video_file)
    nrecs = int(np.floor((filesize-HEADER_SIZE)/RECORD_SIZE))
    
    fid = open(video_file, 'rb')
    header = np.fromfile(fid, dtype='uint8',count=16384)
    swstx = np.zeros((1,nrecs),dtype='uint16')
    swid = np.zeros((1,nrecs),dtype='uint16')
    swdata_size = np.zeros((1,nrecs),dtype='uint16')
    qwTimeStamp = np.zeros((1,nrecs),dtype='uint64')
    dwPoints = np.zeros((400,),dtype='uint32')
    sncrc = np.zeros((1,nrecs),dtype='int16')
    dnextracted_x = np.zeros((1,nrecs),dtype='int32');
    dnextracted_y = np.zeros((1,nrecs),dtype='int32');
    dnextracted_angle = np.zeros((1,nrecs),dtype='int32');
    dntargets = np.zeros((50,nrecs),dtype='int32');
    targets = np.zeros((1,50));

    """read in all data"""
    
    for i in range(nrecs):
        if (i+1)%5000 == 0:
            print('reading record # %s of %d' % ((i+1),nrecs))
        swstx[0,i] = np.fromfile(fid,dtype='uint16',count=1)
        swid[0,i]=np.fromfile(fid,'uint16',count=1)
        swdata_size[0,i]=np.fromfile(fid,'uint16',count=1)
        qwTimeStamp[0,i]=np.fromfile(fid,'uint64',count=1)
        dwPoints[0:NLX_VTREC_NUM_POINTS]=np.fromfile(fid,'uint32',count=NLX_VTREC_NUM_POINTS)
        sncrc[0,i]=np.fromfile(fid,'int16',count=1)
        dnextracted_x[0,i]=np.fromfile(fid,'int32',count=1)
        dnextracted_y[0,i]=np.fromfile(fid,'int32',count=1)
        dnextracted_angle[0,i]=np.fromfile(fid,'int32',count=1)
        targets= np.fromfile(fid,'int32',count=NLX_VTREC_NUM_TARGETS)
        dntargets[:,i]=targets;
        
    print('finished reading video tracking file')
            
    """extracting x/y positions etc"""
    
    target_data_filename = filename[:3] + '.txt'
    print('starting to process video tracking data file.')
    
    timestamp = np.zeros((1,nrecs))
    red_x = np.zeros((1,nrecs))
    red_y = np.zeros((1,nrecs))
    green_x = np.zeros((1,nrecs))
    green_y = np.zeros((1,nrecs))
    
    
    for i in range(nrecs):
        if (i+1)%5000 == 0:
            print('processing record # %s of %d' % ((i+1),nrecs))
        for j in range(4):
            target = dntargets[j,i]
            bin_target=format(target,'032b')
            red = bin_target[1]
            green = bin_target[2]
            y = int(bin_target[4:16],2)
            x = int(bin_target[20:],2)
            if red == '1':
                red_x[0,i] = str(x)
                red_y[0,i] = y
            elif green == '1':
                green_x[0,i] = x
                green_y[0,i] = y
        timestamp[0,i] = qwTimeStamp[0,i]
    
    """write data to file"""
    
    rows = []
    for i in range(nrecs):
        rows.append([int(timestamp[0,i]),int(red_x[0,i]),int(red_y[0,i]),int(green_x[0,i]),int(green_y[0,i])])
    
    video_txt_file = trial+'/'+target_data_filename
    with open(video_txt_file,'w') as f:
        writer = csv.writer(f,dialect='excel-tab')
        writer.writerows(rows)
    
    #return rows,dnextracted_angle.tolist()[0]
    raw_vdata = {}
    
    raw_vdata['positions'] = rows
    raw_vdata['angles'] = dnextracted_angle.tolist()[0]
    trial_data['video_txt_file'] = video_txt_file
    
    return trial_data,raw_vdata
    
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
    cam_ts,ttl_ids,estr = grab_nev_data(event_file)
    #set starting cam_id == 3 (will be changed to 1 later)
    cam_id = [3]
    #make lists for camera timestamps and timestamps required by labview program
    cam_timestamp = []
    jeff_timestamp = []
    #for each message associated with a ttl pulse....
    for i in range(len(estr)):
        #if it signals start of the session, collect it for labview
        if estr[i].startswith('Starting Recording'):
            jeff_timestamp.append(cam_ts[i])
        #if it signals a camera switch...
        elif estr[i].startswith('TTL Input on AcqSystem1_0 board 0 port 0 value'):
            #collect timestamp and camera id
            cam_id.append(ttl_ids[i])
            cam_timestamp.append(cam_ts[i])
            jeff_timestamp.append(cam_ts[i])
    #add final timestamp to timestamp list
    cam_timestamp.append(cam_ts[len(cam_ts)-1])
    
    #switch camera numbers so they make sense (odd number order comes from neuralynx)
    for i in range(len(cam_id)):
        if cam_id[i] == 3:
            cam_id[i] = 1
        elif cam_id[i] == 1:
            cam_id[i] = 2
        elif cam_id[i] == 4:
            cam_id[i] = 3
        elif cam_id[i] == 0:
            cam_id[i] = 4
            
    #write timestamps and camera ids in format labview program can read
    cam_rows = []
    for i in range(len(cam_id)):
        cam_rows.append([int(jeff_timestamp[i]),int(cam_id[i])])
    #name the event txt file
    event_txt_file = trial_data['trial']+'/events_text.txt'
    #write the event txt file
    writer = csv.writer(open(event_txt_file, 'wb'),dialect='excel-tab')
    writer.writerows(cam_rows)
    
    #return relevant data
    trial_data['cam_id'] = cam_id
    trial_data['cam_timestamp'] = cam_timestamp
    trial_data['event_txt_file'] = event_txt_file
    
    return trial_data