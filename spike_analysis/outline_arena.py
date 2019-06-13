# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:51:15 2019

@author: Patrick
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import csv
import tkFileDialog

import collect_data

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        
        print self.x1
        print self.x0
        print self.y1
        print self.y0
        
def find_trials(fdir):
    ''' search the chosen directory for trial folders '''
    #start with an empty list
    trials = []
    #for every file in the directory...
    for file in os.listdir(fdir):
        #if there are any folders, check if they have timestamp files in them
        if os.path.isdir(fdir + '/' + file):
            #start by assuming we have no 
            count = 0
            video_file = False
            #for every file in folder...
            for nfile in os.listdir(fdir + '/' + file):
                #if there's a tetrode timestamp file...
                if nfile.endswith(".txt") and nfile.startswith('TT'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilosorted_spikefiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                #if there's a stereotrode timestamp file...
                elif nfile.endswith(".txt") and nfile.startswith('ST'):
                    #if saved in the kilosort folder, move it up a level
                    if file.endswith('kilosorted_spikefiles'):
                        shutil.move(fdir + '/' + file + '/' + nfile, fdir + '/' + file[:len(file)-21] + '/' + nfile)
                    #otherwise, tick one more timestamp file
                    else:
                        count +=1
                #if there's a video file, make a note of it!
                elif nfile.endswith('.nvt'):
                    video_file = True
#                elif ops['acq']=='openephys' and nfile.startswith('vt1') and nfile.endswith('.txt'):
#                    video_file = True
            #if the folder contains timestamp files and a video file, add it to the "trials" list
            if count > 0 and video_file:
                trials.append(fdir + '/' + file)

    #return our options (with multi-session entry) and trials list
    return trials
#
#a = Annotate()
#plt.show()


fdir = 'H:/Patrick/egocentric/PL62/MEC'

trials = find_trials(fdir)
    
#for every session...
for trial in trials:
    
    print trial
    tracking_fdir = fdir + '/' + trial
    if not os.path.isdir(tracking_fdir):
        continue
        
    
    trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)
    
    plt.plot(trial_data['center_x'],trial_data['center_y'])
    plt.show()