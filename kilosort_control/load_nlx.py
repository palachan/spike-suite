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
    fs,gain,inverted = load_tt_header(filename)
        
    #return the data
    return waveforms, timestamps, fs, gain, inverted

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
            elif line.startswith("-InputInverted"):
                inverted = str(line[len("-InputInverted"):])
                inverted = inverted.strip() == 'True'
        f.close()
        
    return np.float(fs),gain,inverted

def load_full_header(filename, nlx_headersize=16*2**10):
    
    counter = 0
    with open(filename,'rb') as f:
        
        for line in f:
            counter += 1
            if counter > 32:
                break
            line = line.decode("utf-8")
            if counter == 1:
                header = line
            else:
                header = header + line
            
    offset = int(nlx_headersize - len(header))
    header = header.ljust(offset, '\x00')

    return header.encode()

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
    fs,gain,inverted = load_st_header(filename)
    
    #return the data
    return waveforms, timestamps, fs, gain, inverted

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
            elif line.startswith("-InputInverted"):
                inverted = str(line[len("-InputInverted"):])
                inverted = inverted.strip() == 'True'
        f.close()
        
    return np.float(fs),gain,bool(inverted)

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
