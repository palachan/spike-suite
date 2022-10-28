# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:33:47 2018

@author: Patrick
"""

import numpy as np
from itertools import groupby
from kilosort_control.write_nlx_headers import write_ntt_header, write_nst_header
from kilosort_control import load_nlx

def write_ntt(stitched, spike_times, spike_clusters, dirname, trode, fs, ampgain, acq, trodename, og_filename, first_timestamp):    
    
    fname = trodename
 
    if acq == 'neuralynx':
        filename = dirname + '/' + trodename
        header = load_nlx.load_full_header(og_filename,nlx_headersize=16*2**10)

    elif acq == 'openephys':
        admax = '1000'
        filename = dirname + '/' + trodename[:(len(trodename)-len('.spikes'))] + '.NTT'

        header = write_ntt_header(nlx_headersize=16*2**10, name=filename, t_open=None, t_close=None,
                             filetype='Spike', fileversion='3.3.0', recordsize='304',
                             cheetahrev='5.6.3', hardwaresubname='AcqSystem1', hardwaresubtype='Cheetah64',
                             samplingfreq=str(int(fs)), admaxvalue=admax, adbitvolts='0.000000004577776380187970 0.000000004577776380187970 0.000000004577776380187970 0.000000004577776380187970',
                             acqentname='TT1', numadchannels='4', adchannel='0 1 2 3',
                             inputrange='300 300 300 300', inputinverted='True', amplowcut='600',
                             amphicut='6000',ampgain=ampgain.strip(),waveformlen='32', alignmentpt='8',
                             threshval='0 0 0 0', minretriggertime=None, spikeretriggertime=None,
                             dualthresh=None, featurepeak1=None, featurepeak2=None,
                             featurepeak3=None, featurepeak4=None, featurevalley1=None,
                             featurevalley2=None, featurevalley3=None, featurevalley4=None)
    

    #define out ncs datatypes
    ntt_dtype = np.dtype([ 
        ('timestamp'  , '<u8'), 
        ('sc_number'  , '<u4'), 
        ('cell_number', '<u4'), 
        ('params'     , '<u4',   (8,)),
        ('waveforms'  , '<i2', (32,4)),
    ]) 
    
    f = open(filename,"wb")
    f.write(header)
    f.flush()
    f.close()

    #make the new file
    extracted = np.memmap(filename, dtype=ntt_dtype, mode='readwrite', 
       offset=(16 * 2**10), shape = len(spike_times))
    
    spike_clusters = np.unique(spike_clusters, return_inverse=True)[1]
    
    #for each chunk of 512 samples...
    for i in range(len(spike_times)):
        #grab appropriate data
        timestamp = first_timestamp + int(spike_times[i]*1000000./fs)
        sc_number = 1
        
        params = np.asarray((1000,3456,5432,5577,6543,2345,3435,4567))
        waveforms = np.swapaxes(stitched[fname][:,int(spike_times[i]-8):int(spike_times[i]+24)],0,1) # * np.asarray([1.0,1.1,1.2,1.3])
        
        summed_waves = np.sum(waveforms,axis=1).tolist()
        edge = False
        for j in groupby(summed_waves):
            if len(list(j[1])) > 5 and j[0] == 0:
#                print summed_waves
                edge = True
        if edge:            
            cell_number = 0
        else:
            cell_number = spike_clusters[i]
        
        #write the data to file
        extracted[i] = (timestamp,sc_number,cell_number,params,waveforms)
        
    #flush the changes to disk
    extracted.flush()
        
    del extracted
    
def write_nst(stitched, spike_times, spike_clusters, dirname, trode, fs, ampgain, acq, trodename, og_filename, first_timestamp):    
    
    fname = trodename
 
    if acq == 'neuralynx':
        filename = dirname + '/' + trodename
        header = load_nlx.load_full_header(og_filename,nlx_headersize=16*2**10)

    elif acq == 'openephys':
        admax = '1000'
        filename = dirname + '/' + trodename[:(len(trodename)-len('.spikes'))] + '.NST'
        header = write_nst_header(nlx_headersize=16*2**10, name=filename, t_open=None, t_close=None,
                             filetype='Spike', fileversion='3.3.0', recordsize='176',
                             cheetahrev='5.6.3', hardwaresubname='AcqSystem1', hardwaresubtype='Cheetah64',
                             samplingfreq=str(int(fs)), admaxvalue=admax, adbitvolts='0.000000004577776380187970 0.000000004577776380187970',
                             acqentname='ST1', numadchannels='2', adchannel='0 1',
                             inputrange='300 300', inputinverted='True', amplowcut='600',
                             amphicut='6000',ampgain=ampgain.strip(),waveformlen='32', alignmentpt='8',
                             threshval='0 0', minretriggertime=None, spikeretriggertime=None,
                             dualthresh=None, featurepeak1=None, featurepeak2=None,
                             featurepeak3=None, featurepeak4=None, featurevalley1=None,
                             featurevalley2=None, featurevalley3=None, featurevalley4=None)
    

    #define out ncs datatypes
    ntt_dtype = np.dtype([ 
        ('timestamp'  , '<u8'), 
        ('sc_number'  , '<u4'), 
        ('cell_number', '<u4'), 
        ('params'     , '<u4',   (8,)),
        ('waveforms'  , '<i2', (32,2)),
    ]) 
    
    f = open(filename,"wb")
    f.write(header)
    f.flush()
    f.close()

    #make the new file
    extracted = np.memmap(filename, dtype=ntt_dtype, mode='readwrite', 
       offset=(16 * 2**10), shape = len(spike_times))
    
    spike_clusters = np.unique(spike_clusters, return_inverse=True)[1]
    
    #for each chunk of 512 samples...
    for i in range(len(spike_times)):
        #grab appropriate data
        timestamp = first_timestamp + int(spike_times[i]*1000000./fs)
        sc_number = 1
        
        params = np.asarray((1000,3456,5432,5577,6543,2345,3435,4567))
        waveforms = np.swapaxes(stitched[fname][:,int(spike_times[i]-8):int(spike_times[i]+24)],0,1) # * np.asarray([1.0,1.1,1.2,1.3])
        
        summed_waves = np.sum(waveforms,axis=1).tolist()
        edge = False
        for j in groupby(summed_waves):
            if len(list(j[1])) > 5 and j[0] == 0:
#                print summed_waves
                edge = True
        if edge:            
            cell_number = 0
        else:
            cell_number = spike_clusters[i]
        
        #write the data to file
        extracted[i] = (timestamp,sc_number,cell_number,params,waveforms)
        
    #flush the changes to disk
    extracted.flush()
        
    del extracted