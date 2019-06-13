# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:48:35 2018

load neuralynx .NTT file

@author: Patrick
"""

import numpy as np

def mmap_tt_spike_file(filename): 
    """ Memory map the Neuralynx .ntt file """ 
    #specify the NTT datatypes
    ntt_dtype = np.dtype([ 
        ('timestamp'  , '<u8'), 
        ('sc_number'  , '<u4'), 
        ('cell_number', '<u4'), 
        ('params'     , '<u4',   (8,)),
        ('waveforms'  , '<i2', (32,4)),
    ]) 
    #memmap the file
    mmap = np.memmap(filename, dtype=ntt_dtype, mode='r+', 
       offset=(16 * 2**10))
    
    waveforms = mmap['waveforms']
    timestamps = mmap['timestamp'].astype(np.float)
    fs,gain = load_tt_header(filename)
    
    #return the data
    return waveforms, timestamps, fs, gain

def load_tt_header(filename):
        
    counter = 0
    gain = '1000 1000 1000 1000'
    with open(filename,'rb') as f:
        
        for line in f:
            counter += 1
            if counter > 32:
                break
            line = line.decode("utf-8")
            if line.startswith("-SamplingFrequency"):
                fs = line[len("-SamplingFrequency"):]
                fs = int(fs.strip())
            elif line.startswith("-AmpGain"):
                gain = str(line[len("-AmpGain"):])
        f.close()
        
    return np.float(fs),gain

def mmap_st_spike_file(filename): 
    """ Memory map the Neuralynx .ntt file """ 
    #specify the NTT datatypes
    ntt_dtype = np.dtype([ 
        ('timestamp'  , '<u8'), 
        ('sc_number'  , '<u4'), 
        ('cell_number', '<u4'), 
        ('params'     , '<u4',   (8,)),
        ('waveforms'  , '<i2', (32,2)),
    ]) 
    #memmap the file
    mmap = np.memmap(filename, dtype=ntt_dtype, mode='r+', 
       offset=(16 * 2**10))
    
    waveforms = mmap['waveforms']
    timestamps = mmap['timestamp'].astype(np.float)
    fs,gain = load_st_header(filename)
    
    #return the data
    return waveforms, timestamps, fs, gain

def load_st_header(filename):
        
    counter = 0
    gain = '1000 1000'
    with open(filename,'rb') as f:
        
        for line in f:
            counter += 1
            if counter > 32:
                break
            line = line.decode("utf-8")
            if line.startswith("-SamplingFrequency"):
                fs = line[len("-SamplingFrequency"):]
                fs = int(fs.strip())
            elif line.startswith("-AmpGain"):
                gain = str(line[len("-AmpGain"):])
        f.close()
        
    return np.float(fs),gain