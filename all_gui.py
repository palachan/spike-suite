# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:43:57 2018

gui to run everything

@author: Patrick
"""

#import necessary modules
import os
import sys
import subprocess
import spike_analysis.gui
import kilosort_control.sort_gui

#import GUI objects/Widgets
from PySide.QtCore import QRect,Qt
from PySide.QtGui import (QApplication, QMainWindow, QFrame, QLabel, QKeySequence, QSizePolicy,
                          QVBoxLayout, QHBoxLayout, QGridLayout, QShortcut, QWidget, QLineEdit,
                          QMenuBar, QMenu, QPushButton, QFileDialog, QDesktopWidget, QComboBox)

#make sure we're using the right qt API
os.environ['QT_API'] = 'pyside'


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Let's begin!
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#class for MainWindow instance
class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        ''' sets up the whole main window '''
        
        #standard init
        super(MainWindow, self).__init__(parent)

        #set the window title
        self.setWindowTitle('SpikeSuite')
        
        #get screen dimensions
        self.screen_height = QDesktopWidget().screenGeometry().height()
        self.screen_width = QDesktopWidget().screenGeometry().width()

        self.make_buttons()

        #create a QMenuBar and set geometry
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, self.screen_width*.5, self.screen_height*.03))
        #set the QMenuBar as menu bar for main window
        self.setMenuBar(self.menubar)
        
        #show the window fullscreen
        self.resize(600, 400)
        
    def make_buttons(self):
        
        buttons = QFrame(self)
        buttons_layout = QVBoxLayout()
        
        self.acquire_button = QPushButton('Acquire',buttons)
        self.sort_button = QPushButton('Sort',buttons)
        self.analyze_button = QPushButton('Analyze',buttons)
#        self.classify_button = QPushButton('Classify')
        
        self.acquire_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.sort_button.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Expanding)
        self.analyze_button.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Expanding)
#        self.classify_button.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Expanding)

        self.acquire_button.setStyleSheet('background-color: darkRed')
        self.sort_button.setStyleSheet('background-color: darkBlue')
        self.analyze_button.setStyleSheet('background-color: darkGreen')
#        self.classify_button.setStyleSheet('background-color: orange')
        
        self.analyze_button.clicked.connect(self.run_analysis)
        self.acquire_button.clicked.connect(self.run_acq)
        self.sort_button.clicked.connect(self.run_sort)
            
        buttonstyle = 'QPushButton{border-style: solid;border-width: 2px;border-radius: 40px;font: bold 30px; color: white}'
        buttonstyle += 'QPushButton:hover{border-width: 4px; border-color:black}'
        buttons.setStyleSheet(buttonstyle)

        buttons_layout.addWidget(self.acquire_button)
        buttons_layout.addWidget(self.sort_button)
        buttons_layout.addWidget(self.analyze_button)
#        buttons_layout.addWidget(self.classify_button)
        
        buttons.setLayout(buttons_layout)
        self.setCentralWidget(buttons)
#        
        
    def run_analysis(self):        
        os.chdir('./spike_analysis')
        #create and show the main window
        self.analysis_frame = spike_analysis.gui.MainWindow(launcher=self)
        self.analysis_frame.show()
    
        #set up stream for stdout and stderr based on outputStream class
        self.outputStream = spike_analysis.gui.outputStream()
        #when outputStream sends messages, connect to appropriate function for
        #writing to terminal window
        self.outputStream.message.connect(self.analysis_frame.print_message)

        #connect stdout and stderr to outputStream
        sys.stdout = self.outputStream
        sys.stderr = self.outputStream
        
        self.close()
        
    def run_sort(self):
        os.chdir('./kilosort_control')
        #create and show the main window
        self.sort_frame = kilosort_control.sort_gui.MainWindow(launcher=self)
        self.sort_frame.show()
    
        #set up stream for stdout and stderr based on outputStream class
        self.outputStream = kilosort_control.sort_gui.outputStream()
        #when outputStream sends messages, connect to appropriate function for
        #writing to terminal window
        self.outputStream.message.connect(self.sort_frame.print_message)

        #connect stdout and stderr to outputStream
        sys.stdout = self.outputStream
        sys.stderr = self.outputStream
        
        self.close()
        
    def run_acq(self):
        
        subprocess.Popen('python ./oe_control/control_gui.py',
                               shell=True) 
        
        self.close()

    def show_launcher(self):
        #create and show the main window
        os.chdir('../')
        self.show()
        
        try:
            self.analysis_frame.close()
        except:
            pass      
        
        try:
            self.sort_frame.close()
        except:
            pass     
        
if __name__ == '__main__':
    #create a QApplication if one doesn't already exist
    app = QApplication.instance()
    if app == None:
        app = QApplication(['/Users/Patrick/anaconda/lib/python2.7/site-packages/spyderlib/widgets/externalshell/start_ipython_kernel.py'])
    
    #create and show the main window
    frame = MainWindow()
    frame.show()
    
    #exit the app when we're all done
    sys.exit(app.exec_())