# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:26:57 2017

@author: Patrick
"""

import pickle

#calculate HD from... 
hd_calc = 'Neuralynx' #or 'LED positions'
#arena size along x axis
arena_x = 100.
#arena size along y axis
arena_y = 100.
#image dpi
pic_resolution = 300
#spatial bin size in cm
spatial_bin_size = 2.
#speed bin size in cm/s
speed_bin_size = 2.
#ahv bin size in deg/s
ahv_bin_size = 6.
#ego dist bin size in cm
ego_dist_bin_size = 4.
#ebc dist bin size
ebc_dist_bin_size = 2.5
#ebc bearing bin size
ebc_bearing_bin_size = 3.
#separation among ego reference points
ego_ref_bin_size = 2.
#spike train bin size in ms
bin_size = 1
#spike autocorr width in s
autocorr_width = .5
#camera framerate (float)
framerate = 30.
#bins to split HD data into
hd_bin_size = 6.
#cutoff for binned occupancy time in seconds
sample_cutoff = 0
#cutoff for linear speed
speed_cutoff = 5
#animation speed relative to real time
ani_speed = 10

default_advanced = [hd_calc, arena_x, arena_y, pic_resolution, spatial_bin_size, 
                    speed_bin_size, ahv_bin_size, ego_dist_bin_size, ebc_dist_bin_size, 
                    ebc_bearing_bin_size, ego_ref_bin_size, bin_size, autocorr_width,
                    framerate, hd_bin_size, sample_cutoff, speed_cutoff, ani_speed]
#default_advanced = [300,64,1,.5,30.,60,0]

with open('default_advanced.pickle','wb') as f:
    pickle.dump(default_advanced,f)
    