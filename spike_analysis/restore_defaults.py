# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:26:57 2017

@author: Patrick
"""

import pickle

#fdir = 'E:/PL5/2017-02-28_15-11-44/'
#image dpi
pic_resolution = 300
#spatial heatmap bins
grid_res=64
#spike train bin size in ms
bin_size = 1
#spike autocorr width in s
autocorr_width = .5
#camera framerate (float)
framerate = 30.
#bins to split HD data into
hd_bins = 60
#cutoff for binned occupancy time in seconds
sample_cutoff = 0

default_advanced = [300,64,1,.5,30.,60,0]

with open('default_advanced.pickle','wb') as f:
    pickle.dump(default_advanced,f)