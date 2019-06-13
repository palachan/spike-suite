# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:58:54 2017

@author: Patrick
"""

import cv2
import numpy as np
from scipy import ndimage
import qimage2ndarray
import time
from PySide.QtGui import QPixmap

vidbuffer_size = 100

def run_video_acq(self,sock):
    """ acquire video """
    
    #start capturing video
    vc = cv2.VideoCapture(0)
    #set resolution to 720p
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    #set focus to 0 so it doesn't autofocus
    vc.set(cv2.CAP_PROP_FOCUS, 0)
    #set the exposure low so we only pick up leds
    vc.set(cv2.CAP_PROP_EXPOSURE, -9)
    #set the framerate to 30
    vc.set(cv2.CAP_PROP_FPS, 30)
    
    #if everything is working, collect the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    
    #if we have our first frame...
    if rval:
        
        #start counting frames and time so we can see our avg fps
        framecount = 0
        time_a = 0
        elapsed_sum = 0
        
        #start arrays for keeping track of LED locations for threshold overlay
        self.green_frame = np.zeros_like(frame)
        self.red_frame = np.zeros_like(frame)
        
        self.clear_button.setEnabled(True)
        self.overlay_button.setEnabled(True)
        
        #start looping
        while True:
            
            if not self.acquiring:
                self.workerThread.quit
                break
            
            if self.vt_file is None and self.recording:
                while 1:
                    try:
                        self.sock.send('GetRecordingPath')
                        self.recdir = str(self.sock.recv())
                    except:
                        continue
                    break
                
                print(self.recdir)
                self.vt_file = self.recdir + '/vt1.txt'
                
                f = open(self.recdir + '/experiment_info.txt','wb')
                f.write(self.experiment_info)
                f.flush()
                f.close()
            
            #read the next frame
            rval, frame = vc.read()
            #wait a millisecond so we can see our frame
            cv2.waitKey(1)
            while 1:
                try:
                    
                    #request a timestamp from the openephys gui
                    sock.send('timestamp')
                    #grab the timestamp and make it an integer
                    timestamp=np.float(sock.recv())
                except:
                    continue
                break
            
            #blur the image - this takes a while and we have some extra time
            #because video acq is so fast
            procframe = cv2.flip(cv2.GaussianBlur(frame, (11, 11), 0), -1)
            #send it to the video processing worker
            self.videoproc_worker.procvideo.emit(procframe,timestamp)
            
            resized_frame = frame[:,380:]    
#            #resize the frame to a usable size
            resized_frame = cv2.resize(resized_frame, (540,360))
            #switch to rgb for showing in gui
            show_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            #change to a different format?
            
            image = qimage2ndarray.array2qimage(show_frame)
            #show the raw frame
            self.collect_frame.emit(image)
            
            
            ##############################################
            #this all just calculates framerate
            if framecount == 0:
                time_a = float(timestamp)
                
            else:
                time_b = float(timestamp)

            if framecount > 0:
                elapsed = float(time_b - time_a)/self.fs
                elapsed_sum += elapsed
                avg_time = elapsed_sum/framecount
                if avg_time > 0:
                    avg_fps = 1./avg_time
                    if framecount%30==0:
                        print('%f fps' % avg_fps)
            framecount += 1
            time_a = float(timestamp)
            #############################################


def process_video(self,thresh,timestamp,vidwriter=None):

    #split into BGR channels
    blue_img, green_img, red_img = cv2.split(thresh)
    #also make a grayscale image for detecting brightness
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #threshold for red and green colors
    rbin, thresh_r = cv2.threshold(red_img, 100, 255, cv2.THRESH_TOZERO )
    rbin, thresh_g = cv2.threshold(green_img, 100, 255, cv2.THRESH_TOZERO)
#   rbin, thresh_b = cv2.threshold(blue_img, 100, 255, cv2.THRESH_TOZERO)
    
    #threshold for brightness
    rbin, thresh_bright = cv2.threshold(gray, 50, 200, cv2.THRESH_TOZERO)

    #remove pixels below brightness threshold
    thresh_g[thresh_bright == 0] = 0
    thresh_r[thresh_bright == 0] = 0

    #threshold for blue color (probably unnecessary)
#                thresh_r[thresh_b > 0] = 0
#                thresh_g[thresh_b > 0] = 0
    
    #dilate the red and green dots so we can see them
    thresh_r = cv2.dilate(thresh_r, None, iterations=2)
    thresh_g = cv2.dilate(thresh_g, None, iterations=2)
    
    #make sure green doesn't show up as red and vice versa
    thresh_r[thresh_g>thresh_r] = 0
    thresh_g[thresh_r>thresh_g] = 0
    
    #find the center of mass coordinates for the red led
    rlocs = np.where(thresh_r > 0)
    try:
        rx = np.sum(rlocs[0])/len(rlocs[0])
        ry = np.sum(rlocs[1])/len(rlocs[1])
    except:
        rx,ry=[0,0]
    
    #find the center of mass coordinates for the green led
    glocs = np.where(thresh_g > 0)
    try:
        gx = np.sum(glocs[0])/len(glocs[0])
        gy = np.sum(glocs[1])/len(glocs[1])
    except:
        gx,gy=[0,0]
  
    #transform timestamp to microseconds by dividing by samplerate and multiplying by 1e6
    timestamp = np.int(timestamp * 1000000./self.fs)
        
    vidrow = [timestamp,rx,ry,gx,gy]
    
    self.vidbuffer.append(vidrow)
    
    if vidwriter is not None and len(self.vidbuffer)==vidbuffer_size:
        #write the led coords and associated timestamp to a csv file
        for row in self.vidbuffer:
            vidwriter.writerow(row)
        self.vidbuffer = []

    if not self.overlay:
        self.green_frame = np.zeros_like(self.green_frame)
        self.red_frame = np.zeros_like(self.red_frame)
    
    #add the appropriate spots to the green overlay array
    self.green_frame[thresh_g > 0] = [0,255,0]
    
    #same with red
    self.red_frame[thresh_r > 0] = [0,0,255]
    
    #add the green and red frames together for displaying
    thresh =  self.green_frame + self.red_frame
    
    resized_thresh = thresh[:,:900]
    #resize frame to usable size
    resized_thresh = cv2.resize(resized_thresh,(540,360))
    
    show_thresh = cv2.cvtColor(resized_thresh, cv2.COLOR_BGR2RGB)
    #flip it so cue card is south
    show_thresh = cv2.flip(show_thresh, -1)
    #change to a different format?
    thresh_image = qimage2ndarray.array2qimage(show_thresh)
    

    self.collect_threshed.emit(thresh_image)
