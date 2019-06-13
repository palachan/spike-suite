# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:30:01 2018

phase locking

@author: Patrick
"""

from scipy.signal import butter, filtfilt
import bisect
import numpy as np
import numba as nb

def load_ncs(filename):
    
    #define ncs datatypes
    ncs_dtype = np.dtype([ 
        ('timestamp'  , '<u8'), 
        ('chan_number'  , '<u4'), 
        ('sample_freq', '<u4'), 
        ('num_valid'     , '<u4'), 
        ('data'  , '<i2', (512,)), 
    ])

    #memmap the file
    ncs_data = np.memmap(filename, dtype=ncs_dtype, mode='r+', 
       offset=(16 * 2**10))
    
    #sample rate
    fs = ncs_data['sample_freq'][0]
    #flatten signal into one vector
    signal = ncs_data['data'].flatten()
    #same for timestamps
    timestamps = ncs_data['timestamp'].flatten()
    
    @nb.njit()
    def get_timestamps(signal):
        #vector for interpolated timestamps
        real_timestamps = np.zeros(len(signal))
        #calc gap between existing timestamps (512 samples apart)
        time_gap = timestamps[1] - timestamps[0]
        #for each sample in the signal...
        for i in range(len(signal)):
            #interpolate a timestamp
            real_timestamps[i] = timestamps[np.int(np.floor(i/512))] + i%512 * np.float(time_gap)/512.
        #return them
        return real_timestamps
    
    #interpolate timestamps
    real_timestamps = get_timestamps(signal)

    #return everything
    return fs, signal, real_timestamps

def filter_signal(signal,fs):
    ''' modified from http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html '''
    
    #theta frequency band
    lowcut = 5.
    highcut = 10.
    #order of butterworth filter
    order = 2
    
    #nyquist frequency
    nyq = 0.5 * fs
    #low value
    low = lowcut / nyq
    #high value
    high = highcut / nyq

    #create the butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    #filter the signal
    filtered_signal = filtfilt(b, a, signal)
    
    #return the filtered signal
    return filtered_signal

def phase_detect(filtered_signal):
    ''' find peaks and valleys of filtered signal and interpolate
    phases in between '''
    
    #number of bins to use in tuning curve
    phase_bins = 30
    #vector for holding phases
    phases = np.zeros(np.shape(filtered_signal))
    
    @nb.njit()
    def peaks_troughs(filtered_signal,phases):
        ''' finds peaks and troughs in signal, assigns phase_bins/2 to troughs
        and phase_bins to peaks, but to preserve circular structure these numbers are
        constantly increasing (+phase_bins for each new waveform) '''
        #counter for waveforms
        waves=0
        #find peaks and troughs, assign appropriate phase numbers
        for v in range(len(filtered_signal)):
            if v > 0 and v < len(filtered_signal)-1:
                if filtered_signal[v-1] < filtered_signal[v] and filtered_signal[v+1] < filtered_signal[v]:
                    waves += 1
                    phases[v] = waves * phase_bins
                elif filtered_signal[v-1] > filtered_signal[v] and filtered_signal[v+1] > filtered_signal[v]:
                    phases[v] = waves * phase_bins + phase_bins/2
        #return partial phase vector
        return phases
    
    #find peaks and troughs in filtered signal
    phases = peaks_troughs(filtered_signal,phases)
    #set bins in between to NaN
    phases[phases==0] = np.nan

    #interpolate the NaNs
    def nan_helper(y_vals):
        ''' returns where NaNs are for use by np.interp function '''
        return np.isnan(y_vals), lambda z: z.nonzero()[0]
    #interpolate empty allocentric HD spots
    nans, x = nan_helper(phases)
    phases[nans] = np.interp(x(nans), x(~nans), phases[~nans])
            
    #modulo phase_bins to bring everything back into 0-phase_bins range
    phases = phases%phase_bins

    #return phases
    return phases

def match_time(phases,timestamps,trial_data):
    ''' match up the phases to actual video tracking timestamps '''
    
    #grab tracking timestamps
    tracking_ts = trial_data['timestamps']
    #new vector for phases (length of tracking timestamps vector)
    new_phases = np.zeros(np.shape(tracking_ts))
    
    #assign the closest phase to each tracking timestamp
    for i in range(len(tracking_ts)):
        ind = bisect.bisect_left(timestamps,tracking_ts[i])
        new_phases[i] = phases[ind]
        
    #return phases
    return new_phases
    
def run_phase_analysis(filename,trial_data):
    
    #load CSC file    
    fs, signal, timestamps = load_ncs(filename)
    #filter for theta
    filtered_signal = filter_signal(signal,fs)
    #detect theta phases
    phases = phase_detect(filtered_signal)
    #match to video tracking timestamps
    phases = match_time(phases,timestamps,trial_data)
    #return theta phases
    return phases

