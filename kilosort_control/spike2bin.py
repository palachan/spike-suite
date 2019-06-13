# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 13:44:38 2017

@author: Patrick
"""
import numpy as np
import copy

def stitch_waveforms(waveforms,timestamps,fs,trodenum):
    """ stitch the waveforms into one array """
    
    #swap some axes to make the data more intuitive
    waveforms = np.swapaxes(waveforms,0,2)
    waveforms = np.swapaxes(waveforms,1,2)
    
    #figure out the session length in samples
    sample_rate = fs
    session_length = np.int(max(timestamps) * sample_rate / 1000000.)
    session_length = np.int(session_length)
        
    #start an array of zeros, one dimension for channels and one for samples
    stitched = np.zeros((trodenum,int(session_length)))
    
    #for each channel...
    for i in range(len(waveforms)):
        #for each spike on this channel...
        for j in range(len(waveforms[0])):
            #grab our spike data
            data = waveforms[i][j]
            #figure out which sample number we need to insert at to 
            #preserve timestamp validity
            ts = timestamps[j]
            if np.int(ts*sample_rate/1000000.) + 24 < session_length:
                if ts*sample_rate/1000000. < 8:
                    start_samp = sample_rate*ts/1000000.
                else:
                    start_samp = sample_rate*ts/1000000. - 8
                #add the data
                stitched[i][np.int(start_samp):(np.int(start_samp)+32)] = data
                
    #return it  
    return stitched

def write_bin(stitched,filename,writemode):
    """ write long waveform array to file """
    #get a file ready
    with open(filename,writemode) as f:
        #write it!
        stitched = -stitched.astype(np.int16)
        stitched.flatten('F').tofile(f)
