# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:38:51 2018

load open-ephys .Spikes file,
return waveforms, timestamps, and sampling rate (fs)

@author: Patrick
"""

import numpy as np

def load_spikefile(filename):
    ''' loads an openephys .spikes file '''

    #read the header
#    fs,numChannels = load_header(filename)
    fs,numChannels = 30000.,4
    numSamples = 40 # **NOT CURRENTLY WRITTEN TO HEADER**
            
    #define the data types for reading the file
    spike_dtype = np.dtype([('eventType', np.dtype('<u1')), ('timestamps', np.dtype('<i8')), ('software_timestamp', np.dtype('<i8')),
           ('source', np.dtype('<u2')), ('numChannels', np.dtype('<u2')), ('numSamples', np.dtype('<u2')),
           ('sortedId', np.dtype('<u2')), ('electrodeId', np.dtype('<u2')), ('channel', np.dtype('<u2')),
           ('color', np.dtype('<u1'), 3), ('pcProj', np.float32, 2), ('sampleFreq', np.dtype('<u2')), ('waveforms', np.dtype('<u2'), numChannels*numSamples),
           ('gain', np.float32,numChannels), ('thresh', np.dtype('<u2'), numChannels), ('recNum', np.dtype('<u2'))])
    
    #grab the data
    data = np.memmap(filename, dtype=spike_dtype, mode='r+',
                         offset=(1024))
            
    #create an array for holding waveforms
    spikes = np.zeros((len(data), numSamples, numChannels))
    #for each spike in waveforms array...
    for i in range(len(data['waveforms'])):
        #reshape to be in shape (numChannels,numSamples)
        wv = np.reshape(data['waveforms'][i], (numChannels, numSamples))
        #for each channel...
        for ch in range(numChannels):
            #convert values to volts and assign to spikes dict
            spikes[i,:,ch] = (np.float64(wv[ch])-32768)
#        spikes[i][(spikes[i] < -800) | (spikes[i] > 800)] = 0
        spikes[i] = np.clip(spikes[i],a_min = -1000,a_max=1000)
    
    waveforms = spikes[:,:32,:]
    timestamps = data['timestamps'].astype(np.float) * 1000000. / np.float(fs)
    gain = str(data['gain'][0][0]) + ' ' + str(data['gain'][0][1]) + ' ' + str(data['gain'][0][2]) + ' ' + str(data['gain'][0][3])

    #return our data
    return -waveforms, timestamps, fs, gain

def load_header(filename):
        
    counter = 0
    with open(filename,'rb') as f:
        for line in f:
            counter += 1
            if counter > 10:
                break
            line = line.decode("utf-8")
            if line.startswith("header.sampleRate ="):
                fs = line[len("header.sampleRate ="):len(line)-2]
                fs = np.int(fs.strip())
            elif line.startswith("header.num_channels ="):
                numChannels = line[len("header.num_channels ="):len(line)-2]
                numChannels = np.int(numChannels.strip())
        f.close()
    
    return fs, numChannels
