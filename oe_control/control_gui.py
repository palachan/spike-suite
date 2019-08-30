# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 11:46:57 2017

@author: Patrick
"""
import os
os.environ['QT_API'] = 'pyside'

import sys
import zmq
import csv
import qimage2ndarray
import time
import numpy as np
import pickle
import subprocess
import shutil
import xml.etree.ElementTree as et

import video

#sys.path.append('C:/Users/Jeffrey_Taube/Desktop/Analysis/spike-suite/')
#import kilosort_control.sort_gui

from PySide.QtCore import (QProcess,QRect,Qt,QObject,Signal,Slot,QThread,QEventLoop,QTimer)
from PySide.QtGui import (QApplication, QMainWindow, QFrame, QLabel, QCheckBox, QLineEdit, QImage,
                          QAction, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,QTextCursor, QComboBox,
                          QTextEdit, QMenuBar, QMenu, QStatusBar, QStyle, QPushButton, QFileDialog, QDesktopWidget,QPixmap)


""""""""""""""""""
open_ephys_path = 'C:/Users/Jeffrey_Taube/Desktop/Analysis/plugin-GUI/Builds/VisualStudio2013/x64/Release64/bin/open-ephys.exe'
config_path =  'C:/Users/Jeffrey_Taube/Desktop/Analysis/plugin-GUI/Builds/VisualStudio2013/x64/Release64/bin'
default_config = config_path + '/lastConfig.xml'
data_save_dir = 'C:/Users/Jeffrey_Taube/Desktop/'

""""""""""""""""""

#class for catching standard output and standard error
class outputStream(QObject):
    #set up a signal 'message' for transmitting stdout/stderr
    message = Signal(str)
    
    def __init__(self, parent=None):
        #standard __init__
        super(outputStream, self).__init__(parent)
    
    def write(self, message):
        #send message to be printed in terminal window
        self.message.emit(str(message))
        
#class for doing calculations in parallel thread
class VideoWorker(QObject):
    procvideo = Signal(object,int)
    
    def __init__(self,gui,parent=None):
        #standard __init__
        super(VideoWorker, self).__init__(parent)
        #grab fname from Run function and MainWindow instance, assign to class
        self.gui = gui
        self.procvideo.connect(self.process_video)
        
    def process_video(self,frame,timestamp):
                
        #process the video frame
        if self.gui.recording and self.gui.vt_file is not None:
                                        
            #open a csv file for saving tracking data
            with open(self.gui.vt_file, 'a') as csvfile:
                #create a writer
                vidwriter = csv.writer(csvfile,dialect='excel-tab')
                #check if it's an empty file
                if os.path.getsize(self.gui.vt_file) == 0:
                    #write headers
                    vidwriter.writerow(['timestamp','red_x','red_y','green_x','green_y'])
                    
                try:
                    video.process_video(self.gui,frame,timestamp,vidwriter=vidwriter)
                except:
                    video.process_video(self.gui,frame,timestamp)
        else:
            video.process_video(self.gui,frame,timestamp)

#class for doing calculations in parallel thread
class Worker(QObject):
    
    def __init__(self,gui,parent=None):
        #standard __init__
        super(Worker, self).__init__(parent)
        #grab fname from Run function and MainWindow instance, assign to class
        self.gui = gui
        

    def acquire(self):

        self.gui.sock.send('GetSampleRate')
        self.gui.fs = np.float(self.gui.sock.recv())

        #use process to start Run function in main script
        self.gui.acquiring = True
        video.run_video_acq(self.gui,self.gui.sock)
            

#class for MainWindow instance
class MainWindow(QMainWindow):
    start_acq = Signal(str)
    start_rec = Signal(str)
    collect_frame = Signal(object)
    collect_threshed = Signal(object)
    
    def __init__(self, parent=None):
        ''' sets up the whole main window '''

        #standard init
        super(MainWindow, self).__init__(parent)

        #set the window title
        self.setWindowTitle('Open Ephys Control GUI')
        
        self.window_height = 700
        self.window_width = 1100
        
        self.screen2 = QDesktopWidget().screenGeometry(0)
        self.move(self.screen2.left()+(self.screen2.width()-self.window_width)/2.,self.screen2.top()+(self.screen2.height()-self.window_height)/2.)
        
        self.get_info()
        self.noinfo = True
        
        while self.noinfo:
            loop = QEventLoop()
            QTimer.singleShot(500.,loop.quit)
            loop.exec_()
        
        subprocess.Popen('start %s' % open_ephys_path,
                               shell=True)
        
        self.collect_frame.connect(self.update_frame)
        self.collect_threshed.connect(self.update_threshed)
        self.acquiring = False
        self.recording = False
        
        self.video_height = self.window_height*.52
        self.video_width = self.window_width*.48
        
        self.resize(self.window_width, self.window_height)
        
        #create QTextEdit window 'terminal' for receiving stdout and stderr
        self.terminal = QTextEdit(self)
        #set the geometry
        self.terminal.setGeometry(QRect(self.window_width*.02,self.window_height*.15+self.video_height,self.video_width*.96,150))
        
        #make widgets
        self.setup_video_frames()
        self.setup_thresh_buttons()
        
        self.overlay = True

        #create thread and worker for video processing
        self.videoThread = QThread(self)
        self.videoThread.start()
        self.videoproc_worker = VideoWorker(self)
        self.videoproc_worker.moveToThread(self.videoThread)
        
        self.vt_file = None        

        """""""""""""""""""""""""""""""""""""""""""""
        set up menus
        """""""""""""""""""""""""""""""""""""""""""""
        
        #create a QMenuBar and set geometry
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, self.window_width*.5, self.window_height*.03))
        #set the QMenuBar as menu bar for main window
        self.setMenuBar(self.menubar)


        #create a QStatusBar
        statusbar = QStatusBar(self)
        #set it as status bar for main window
        self.setStatusBar(statusbar)
        
        #create icon toolbar with default image
        iconToolBar = self.addToolBar("iconBar.png")
        
        #create a QAction for the acquire button
        self.action_Acq = QAction(self)
        #make it checkable
        self.action_Acq.setCheckable(True)
        #grab an icon for the button
        acq_icon=self.style().standardIcon(QStyle.SP_MediaPlay)
        #set the icon for the action
        self.action_Acq.setIcon(acq_icon)
        #when the button is pressed, call the Acquire function
        self.action_Acq.triggered.connect(self.Acquire)
        
        #create a QAction for the record button
        self.action_Record = QAction(self)
        #make it checkable
        self.action_Record.setCheckable(True)
        #grab an icon for the button
        record_icon=self.style().standardIcon(QStyle.SP_DialogYesButton)
        #set the icon for the action
        self.action_Record.setIcon(record_icon)
        #when the button is pressed, call advanced_settings function
        self.action_Record.triggered.connect(self.Record)
        
        #create QAction for stop button
        action_Stop = QAction(self)   
        #grab close icon
        stop_icon=self.style().standardIcon(QStyle.SP_MediaStop)
        #set icon for action
        action_Stop.setIcon(stop_icon)
        #when button pressed, close window
        action_Stop.triggered.connect(self.Stop)

        #show tips for each action in the status bar
        self.action_Acq.setStatusTip("Start acquiring")
        self.action_Record.setStatusTip("Start recording")
        action_Stop.setStatusTip("Stop acquiring/recording")
     
        #add actions to icon toolbar
        iconToolBar.addAction(self.action_Acq)
        iconToolBar.addAction(self.action_Record)
        iconToolBar.addAction(action_Stop)
        
#        self.sort_button = QPushButton('Sort Now',self)
#        self.sort_button.setGeometry(QRect(self.window_width*.85,0,self.window_width*.15,self.window_height*.05))
#        self.sort_button.clicked.connect(self.sort_now)
                
        #show the window if minimized by windows
        self.showMinimized()
        self.showNormal()
                
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        
    #this function acts as a slot to accept 'message' signal
    @Slot(str)
    def print_message(self, message):
        ''' print stdout and stderr to terminal window '''
        
        #move terminal cursor to end
        self.terminal.moveCursor(QTextCursor.End)
        #write message to terminal
        self.terminal.insertPlainText(message)
        
    def setup_thresh_buttons(self):
        ''' set up buttons for overlay/clearing thresh view '''
        
        self.button_frame = QFrame(self)
        self.button_frame.setGeometry(QRect(self.window_width*.52,self.window_height*.13+self.video_height,self.video_width*.98,50))
        button_layout = QHBoxLayout()
        self.button_frame.setLayout(button_layout)
        self.clear_button = QPushButton('Clear')
        self.overlay_button = QPushButton('Overlay')
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.overlay_button)
        
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_threshed)
        self.overlay_button.setEnabled(False)
        self.overlay_button.clicked.connect(self.overlay_threshed)
        
    def setup_video_frames(self):
        ''' set up spots for playing video frames '''

        filler_frame = np.zeros((360,540,3))
        filler_frame = qimage2ndarray.array2qimage(filler_frame)

        self.raw_frame = QFrame(self)
        self.raw_label = QLabel()
        self.raw_label.setText('raw')
        self.raw_frame.setGeometry(QRect(self.window_width*.01,self.window_height*.15,self.video_width,self.video_height))
        self.raw_frame
        raw_layout = QVBoxLayout()
        self.raw_frame.setLayout(raw_layout)
        raw_layout.addWidget(self.raw_label)
        
        self.threshed_frame = QFrame(self)
        self.threshed_label = QLabel()
        self.threshed_label.setText('Threshed')
        self.threshed_frame.setGeometry(QRect(self.window_width*.51,self.window_height*.15,self.video_width,self.video_height))
        threshed_layout = QVBoxLayout()
        self.threshed_frame.setLayout(threshed_layout)
        threshed_layout.addWidget(self.threshed_label)
        
        self.label_frame = QFrame(self)
        self.label_frame.setGeometry(QRect(self.window_width*.01,self.window_height*.11,self.video_width*2,50))
        self.label_rawlabel = QLabel()
        self.label_rawlabel.setText('Raw Video')
        self.label_threshedlabel = QLabel()
        self.label_threshedlabel.setText('Threshold View')
        label_layout = QHBoxLayout()
        self.label_frame.setLayout(label_layout)
        label_layout.addWidget(self.label_rawlabel)
        label_layout.addWidget(self.label_threshedlabel)

        self.raw_label.setPixmap(QPixmap.fromImage(filler_frame))
        self.threshed_label.setPixmap(QPixmap.fromImage(filler_frame))
        
        
    def Acquire(self):

        if self.action_Acq.isChecked():
            
            self.vidbuffer = []
            
            if self.recording:
                
                while 1:
                    try:
                        self.sock.send('StopRecord')
                        self.sock.recv()
                    except:
                        continue
                    break
                
                self.recording = False

            else:
                #create and start a thread to transport a worker to later
                self.workerThread = QThread(self)
                self.workerThread.start()
                #create a worker object based on Worker class and move it to our 
                #worker thread
                self.worker = Worker(self)
                self.worker.moveToThread(self.workerThread)
    
                try:
                    self.start_acq.disconnect()
                except:
                    pass
                                
                while 1:
                    try:
                        self.sock.send('StartAcquisition')
                        self.sock.recv()
                    except:
                        continue
                    break
                
                self.acquiring = True
                self.start_acq.connect(self.worker.acquire)
                self.start_acq.emit('start!')
            
            self.action_Acq.setEnabled(False)
            self.action_Record.setChecked(False)
            self.action_Record.setEnabled(True)
            
            record_icon=self.style().standardIcon(QStyle.SP_DialogYesButton)
            #set the icon for the action
            self.action_Record.setIcon(record_icon)

        
    def Record(self):
        
        if self.action_Record.isChecked():

            if not self.acquiring:
                self.workerThread = QThread(self)
                self.workerThread.start()
            
                self.worker = Worker(self)
                self.worker.moveToThread(self.workerThread)
                
                try:
                    self.start_rec.disconnect()
                except:
                    pass
                
                while 1:
                    try:
                        self.sock.send('StartAcquisition')
                        self.sock.recv()
                    except:
                        continue
                    break
                
                while 1:
                    try:
                        self.sock.send('StartRecord')
                        self.sock.recv()
                    except:
                        continue
                    break
            
                self.vidbuffer = []
                self.start_rec.connect(self.worker.acquire)
                self.recording = True
                self.start_rec.emit('start!')
                
            else:
                
                while 1:
                    try:
                        self.sock.send('StartRecord')
                        self.sock.recv()
                    except:
                        continue
                    break

                self.vidbuffer = []
                self.recording = True
            
            record_icon=self.style().standardIcon(QStyle.SP_DialogNoButton)
            #set the icon for the action
            self.action_Record.setIcon(record_icon)
            self.action_Record.setEnabled(False)
            
            self.action_Acq.setChecked(False)
            self.action_Acq.setEnabled(True)
            

    def Stop(self):
        
        self.acquiring = False
        self.recording = False
                        
        while 1:
            try:
                self.sock.send('isRecording')
                rec = self.sock.recv()  
            except:
                continue
            break
        
        if rec == '1':
            while 1:
                try:
                    self.sock.send('StopRecord')
                    self.sock.recv()
                except:
                    continue
                break

            self.action_Record.setEnabled(True)
            self.action_Record.setChecked(False)
    
        while 1:
            try:
                self.sock.send('isAcquiring')
                acq = self.sock.recv_string()
            except:
                continue
            break
        
        if acq == '1':
            while 1:
                try:
                    self.sock.send('StopAcquisition')
                    self.sock.recv()
                except:
                    continue
                break
            
        self.action_Acq.setEnabled(True)
        self.action_Acq.setChecked(False)
            
        try:
            #open a csv file for saving tracking data
            with open(self.vt_file, 'a') as csvfile:
                #create a writer
                vidwriter = csv.writer(csvfile,dialect='excel-tab')
                #check if it's an empty file
                for row in self.vidbuffer:
                    vidwriter.writerow(row)
                    
        except:
            pass

        record_icon=self.style().standardIcon(QStyle.SP_DialogYesButton)
        #set the icon for the action
        self.action_Record.setIcon(record_icon)
        
    def update_frame(self,image):
        self.raw_label.setPixmap(QPixmap.fromImage(image))
        
    def update_threshed(self,threshed_image):
        self.threshed_label.setPixmap(QPixmap.fromImage(threshed_image))
        
    def clear_threshed(self):
        self.green_frame = np.zeros_like(self.green_frame)
        self.red_frame = np.zeros_like(self.red_frame)
        
    def overlay_threshed(self):
        if self.overlay:
            self.overlay = False
        elif not self.overlay:
            self.overlay = True
            
#    def sort_now(self):
#        
#        if self.recdir is not None:
#            os.chdir('./kilosort_control')
#            #create and show the main window
#            self.sort_frame = kilosort_control.sort_gui.MainWindow()
#            self.sort_frame.show()
#        
#            #set up stream for stdout and stderr based on outputStream class
#            self.outputStream = kilosort_control.sort_gui.outputStream()
#            #when outputStream sends messages, connect to appropriate function for
#            #writing to terminal window
#            self.outputStream.message.connect(self.sort_frame.print_message)
#    
#            #connect stdout and stderr to outputStream
#            sys.stdout = self.outputStream
#            sys.stderr = self.outputStream
#            
#            self.sort_frame.run_now(self.recdir)
#            
#            self.close()
            
    def get_info(self):
        
        self.info_window = QWidget()
        self.info_window.resize(400, 350)
        #set title
        self.info_window.setWindowTitle('Session Info')
        #give layout
        info_layout = QVBoxLayout(self.info_window)
        
        with open('info_fields.pickle','rb') as f:
            default_fields = pickle.load(f)
            f.close()
        
        #set label for pic_resolution setting
        experimenter_label = QLabel('Experimenter:')
        #make a QLineEdit box for displaying/editing settings
        experimenter = QComboBox(self.info_window)
        experimenter.setEditable(True)
        experimenter.addItems(default_fields['experimenter'])
        #add label and box to current window
        info_layout.addWidget(experimenter_label)
        info_layout.addWidget(experimenter)
        
        #set label for pic_resolution setting
        whose_animal_label = QLabel('Whose animal?')
        #make a QLineEdit box for displaying/editing settings
        whose_animal = QComboBox(self.info_window)
        whose_animal.setEditable(True)
        whose_animal.addItems(default_fields['whose_animal'])
        #add label and box to current window
        info_layout.addWidget(whose_animal_label)
        info_layout.addWidget(whose_animal)
        
        animal_number_label = QLabel('Animal number:')
        animal_number = QComboBox(self.info_window)
        animal_number.setEditable(True)
        animal_number.addItems(default_fields['animal_number'])
        info_layout.addWidget(animal_number_label)
        info_layout.addWidget(animal_number)
        
        session_number_label = QLabel('Session number:')
        session_number = QTextEdit(self.info_window)
        session_number.setText('1')
        info_layout.addWidget(session_number_label)
        info_layout.addWidget(session_number)
        
        session_type_label = QLabel('Session type:')
        session_type = QComboBox(self.info_window)
        session_type.setEditable(True)
        session_type.addItems(default_fields['session_type'])
        info_layout.addWidget(session_type_label)
        info_layout.addWidget(session_type)
        
        def save_info(self):
            
            info_fields = {}
            info_fields['experimenter'] = [experimenter.itemText(i) for i in range(experimenter.count())]
            info_fields['whose_animal'] = [whose_animal.itemText(i) for i in range(whose_animal.count())]
            info_fields['animal_number'] = [animal_number.itemText(i) for i in range(animal_number.count())]
            info_fields['session_type'] = [session_type.itemText(i) for i in range(session_type.count())]
            
            with open('info_fields.pickle','wb') as f:
                pickle.dump(info_fields,f,protocol=2)
                f.close()
            
            current_experimenter = str(experimenter.currentText())
            current_whose_animal = str(whose_animal.currentText())
            current_animal_number = str(animal_number.currentText())
            current_session_number = str(session_number.toPlainText())
            current_session_type = str(session_type.currentText())
            
            recdir = data_save_dir + current_whose_animal + '/' + current_animal_number
            
            if not os.path.exists(recdir):
                os.makedirs(recdir)
                
            self.experiment_info = '###### Experiment Info ######\r\n'
            self.experiment_info += 'Experimenter: %s\r\n' % current_experimenter
            self.experiment_info += 'Whose animal? %s\r\n' % current_whose_animal
            self.experiment_info += 'Animal number: %s\r\n' % current_animal_number
            self.experiment_info += 'Session number: %s\r\n' % current_session_number
            self.experiment_info += 'Session type: %s\r\n' % current_session_type
            
            self.experiment_info = self.experiment_info.encode()
            
            config_file = config_path + '/' + current_animal_number + '.xml'
                
            if not os.path.exists(config_file):
                shutil.copy(default_config,config_file)
            
            tree = et.parse(config_file)
            root = tree.getroot()
            for child in root:
                if child.tag == 'CONTROLPANEL':
                    child.attrib['recordPath'] = recdir.replace('/', '\\')
            tree.write(config_file)
            tree.write(default_config)
            
            self.info_window.close()
            self.noinfo = False
                    
        ready_button = QPushButton('Ready!')
        ready_button.clicked.connect(lambda: save_info(self))
        info_layout.addWidget(ready_button)
        
        self.info_window.show()

            
if __name__ == '__main__':
    
    #get our context for communicating through ZeroMQ
    with zmq.Context() as ctx:
        #set up a socket to send and receive messages
        with ctx.socket(zmq.REQ) as sock:
            #connect to the socket (TCP, port 5556 default)
            port = 5556
            sock.connect('tcp://%s:%d' % ('localhost', port))
            time.sleep(1)
            
            #create a QApplication if one doesn't already exist
            app = QApplication.instance()
            if app == None:
                app = QApplication(['/Users/Patrick/anaconda/lib/python2.7/site-packages/spyderlib/widgets/externalshell/start_ipython_kernel.py'])
            
            #create and show the main window
            mwind = MainWindow()
            mwind.show()
            
            mwind.sock = sock
        
            #set up stream for stdout and stderr based on outputStream class
            outputStream = outputStream()
            #when outputStream sends messages, connect to appropriate function for
            #writing to terminal window
            outputStream.message.connect(mwind.print_message)
            
            #connect stdout and stderr to outputStream
            sys.stdout = outputStream  
            sys.stderr = outputStream
            
            #exit the app when we're all done
            sys.exit(app.exec_())
        
