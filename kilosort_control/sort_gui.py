# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 11:46:57 2017

@author: Patrick
"""
import os
os.environ['QT_API'] = 'pyside'

import sys
import pickle

from kilosort_control import run_kilo

from PySide2.QtCore import (QRect,Qt,QObject,Signal,Slot,QThread)
from PySide2.QtWidgets import (QApplication, QButtonGroup,QMainWindow, QFrame, QLabel, QCheckBox, QLineEdit, QScrollArea,
                          QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QComboBox,
                          QTextEdit, QMenuBar, QStatusBar, QPushButton, QFileDialog)

from PySide2.QtGui import QTextCursor


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
class Worker(QObject):
    
    def __init__(self,gui,parent=None):
        #standard __init__
        super(Worker, self).__init__(parent)
        #grab fname from Run function and MainWindow instance, assign to class
        self.gui = gui

    def run_kilo(self,fname,config_ops,acq):
        #start a process
        self.isrunning = True
        run_kilo.run(self.gui,fname,config_ops,acq)
        
    def run_kilo_batch(self,fname,config_ops,acq):
        #start a process
        self.isrunning = True
        run_kilo.batch_run(self.gui,fname,config_ops,acq)

#class for MainWindow instance
class MainWindow(QMainWindow):
    start = Signal(str,dict,str)

    def __init__(self, parent=None, launcher=None):
        ''' sets up the whole main window '''
        
        self.launcher = launcher

        #standard init
        super(MainWindow, self).__init__(parent)

        #set the window title
        self.setWindowTitle('Automatic Sorting with KiloSort')
        
        self.window_height = 400
        self.window_width = 600
        
        self.resize(self.window_width, self.window_height)
        
        self.mainlayout = QGridLayout()
                
        #create QTextEdit window 'terminal' for receiving stdout and stderr
        self.terminal = QTextEdit(self)
        self.terminal.setGeometry(QRect(self.window_width*.1,self.window_height*.15,self.window_width*.8,self.window_height*.65))        
        
        self.all_button = QPushButton(self)
        self.all_button.setText('All Trodes')
        self.all_button.setGeometry(QRect(self.window_width*.075,self.window_height*.82,self.window_width*.25,self.window_height*.1))
        self.all_button.clicked.connect(lambda: self.run('all'))
        
        self.single_button = QPushButton(self)
        self.single_button.setText('Single Trode')
        self.single_button.setGeometry(QRect(self.window_width*.375,self.window_height*.82,self.window_width*.25,self.window_height*.1))
        self.single_button.clicked.connect(lambda: self.run('single'))
        
        self.batch_button = QPushButton(self)
        self.batch_button.setText('Batch')
        self.batch_button.setGeometry(QRect(self.window_width*.675,self.window_height*.82,self.window_width*.25,self.window_height*.1))
        self.batch_button.clicked.connect(lambda: self.run('batch'))
                
        
        """""""""""""""""""""""""""""""""""""""""""""
        set up menus
        """""""""""""""""""""""""""""""""""""""""""""
        
        #create a QMenuBar and set geometry
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, self.window_width*.5, self.window_height*.03))
        #set the QMenuBar as menu bar for main window
        self.setMenuBar(self.menubar)
        
        self.config_button = QPushButton('Config Options',self)
        self.config_button.clicked.connect(self.config_options)
        self.config_button.setGeometry(QRect(self.window_width*.65,self.window_height*.03,self.window_width*.25,self.window_height*.1))
        
        self.setup_acq_frame()
        
        #create a QStatusBar
        statusbar = QStatusBar(self)
        #set it as status bar for main window
        self.setStatusBar(statusbar)
        
        self.set_defaults()
        
        if self.launcher is not None:
            if isinstance(self.launcher.recdir,str):
                self.run_now(self.launcher.recdir)
                

                
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    def setup_acq_frame(self):
        
        self.acq_frame = QFrame(self)
        self.acq_frame.setGeometry(QRect(self.window_width*.1,self.window_height*.03,self.window_width*.5,self.window_height*.1))
        self.acq_frame.setStyleSheet('background-color:magenta')
        
        acq_label = QLabel('Acq System')
        
        self.nlx_box = QCheckBox('Neuralynx')
        self.openephys_box = QCheckBox('Open-Ephys')
        # self.old_box = QCheckBox('Taube')
        
        acq_buttons = QButtonGroup(self)
        acq_buttons.addButton(self.nlx_box)
        acq_buttons.addButton(self.openephys_box)
        # acq_buttons.addButton(self.old_box)
        acq_buttons.setExclusive(True)
        
        acq_layout = QHBoxLayout()
        acq_layout.addWidget(acq_label)
        acq_layout.addWidget(self.nlx_box)
        acq_layout.addWidget(self.openephys_box)
        # acq_layout.addWidget(self.old_box)
        
        self.acq_frame.setLayout(acq_layout)
        
    def run(self,button):
        
        if self.nlx_box.isChecked():
            acq = 'neuralynx'
        elif self.openephys_box.isChecked():
            acq = 'openephys'
        # elif self.old_box.isChecked():
        #     acq = 'taube'
        
        #if we're doing single cluster mode, ask for a text file
        if button == 'single':
            if acq == 'openephys':
                fname = QFileDialog.getOpenFileName(self, 'Open File', '','OpenEphys spike files (*.spikes)')[0]
            elif acq == 'neuralynx':
                fname = QFileDialog.getOpenFileName(self, 'Open File', '','Neuralynx spike files (*.NTT *.NST)')[0]
            else:
                fname = QFileDialog.getOpenFileName(self, 'Open File', '','')[0]

        #otherwiese, ask for a directory
        else:
            fname = QFileDialog.getExistingDirectory()
          
        with open('config_ops.pickle','rb') as f:
            config_ops = pickle.load(f)
            
        #create a worker object based on Worker class and move it to our 
        #worker thread
        self.workerThread = QThread(self)
        self.workerThread.start()

        self.worker = Worker(self)
        self.worker.moveToThread(self.workerThread)
        
        try:
            self.start.disconnect()
        except:
            pass
        
        if button != 'batch':
            self.start.connect(self.worker.run_kilo)
        else:
            self.start.connect(self.worker.run_kilo_batch)
        self.start.emit(fname,config_ops,acq)

    def run_now(self,fname):

        acq = 'openephys'
                        
        with open('config_ops.pickle','rb') as f:
            config_ops = pickle.load(f)
            
        #create a worker object based on Worker class and move it to our 
        #worker thread
        self.workerThread = QThread(self)
        self.workerThread.start()

        self.worker = Worker(self)
        self.worker.moveToThread(self.workerThread)
        
        try:
            self.start.disconnect()
        except:
            pass
        
        self.start.connect(self.worker.run_kilo)
        self.start.emit(fname,config_ops,acq)

        
    def set_defaults(self):
        
        #grab default advanced options
        with open('default_config_ops.pickle','rb') as f:
            default = pickle.load(f)
            
        #set default advanced options as current advanced options
        with open('config_ops.pickle','wb') as f:
            pickle.dump(default,f,protocol=2)
        
    def config_options(self):
        
        self.config_window = QWidget()
        self.config_window.resize(400, 350)
        #set title
        self.config_window.setWindowTitle('Config Options')
        #give layout
        config_layout = QFormLayout()
        
        scroll = QScrollArea(self.config_window)
        subwidget = QWidget(scroll)
        subwidget.setLayout(config_layout)
        
        with open('config_ops.pickle','rb') as f:
            default = pickle.load(f)
            f.close()
        
        gpu = QLineEdit(subwidget)
        gpu.setText(default['GPU'])
        config_layout.addRow('GPU:',gpu)
        
        parfor = QLineEdit(subwidget)
        parfor.setText(default['parfor'])
        config_layout.addRow('Parfor:',parfor)
        
        verbose = QLineEdit(subwidget)
        verbose.setText(default['verbose'])
        config_layout.addRow('Verbose:',verbose)
        
        showfigures = QLineEdit(subwidget)
        showfigures.setText(default['showfigures'])
        config_layout.addRow('Show Figures:',showfigures)
        
        datatype = QComboBox(subwidget)
        datatype.addItems(['bin','dat','OpenEphys'])
        datatype.setCurrentIndex(datatype.findText(default['datatype']))
        config_layout.addRow('Data Type:',datatype)
        
        Nfilt = QLineEdit(subwidget)
        Nfilt.setText(default['Nfilt'])
        config_layout.addRow('Nfilt:',Nfilt)
        
        whitening = QComboBox(subwidget)
        whitening.addItems(['full'])
        whitening.setCurrentIndex(whitening.findText(default['whitening']))
        config_layout.addRow('Whitening:',whitening)

        nSkipCov = QLineEdit(subwidget)
        nSkipCov.setText(default['nSkipCov'])
        config_layout.addRow('nSkipCov:',nSkipCov)
        
        whiteningRange = QLineEdit(subwidget)
        whiteningRange.setText(default['whiteningRange'])
        config_layout.addRow('Whitening range:',whiteningRange)
        
        criterionNoiseChannels = QLineEdit(subwidget)
        criterionNoiseChannels.setText(default['criterionNoiseChannels'])
        config_layout.addRow('Criterion noise channels:',criterionNoiseChannels)
        
        Nrank = QLineEdit(subwidget)
        Nrank.setText(default['Nrank'])
        config_layout.addRow('Nrank:',Nrank)
        
        nfullpasses = QLineEdit(subwidget)
        nfullpasses.setText(default['nfullpasses'])
        config_layout.addRow('N full passes:',nfullpasses)
        
        maxFR = QLineEdit(subwidget)
        maxFR.setText(default['maxFR'])
        config_layout.addRow('Max FR:',maxFR)
        
        fshigh = QLineEdit(subwidget)
        fshigh.setText(default['fshigh'])
        config_layout.addRow('High pass freq:',fshigh)
        
        ntbuff = QLineEdit(subwidget)
        ntbuff.setText(default['ntbuff'])
        config_layout.addRow('ntbuff:',ntbuff)
        
        scaleproc = QLineEdit(subwidget)
        scaleproc.setText(default['scaleproc'])
        config_layout.addRow('scaleproc:',scaleproc)
        
        Th1 = QLineEdit(subwidget)
        Th1.setText(default['Th'][0])
        Th2 = QLineEdit(subwidget)
        Th2.setText(default['Th'][1])
        Th3 = QLineEdit(subwidget)
        Th3.setText(default['Th'][2])
        config_layout.addRow('Th:',Th1)
        config_layout.addRow('',Th2)
        config_layout.addRow('',Th3)
        
        lam1 = QLineEdit(subwidget)
        lam1.setText(default['lam'][0])
        lam2 = QLineEdit(subwidget)
        lam2.setText(default['lam'][1])
        lam3 = QLineEdit(subwidget)
        lam3.setText(default['lam'][2])
        config_layout.addRow('lam:',lam1)
        config_layout.addRow('',lam2)
        config_layout.addRow('',lam3)
        
        nannealpasses = QLineEdit(subwidget)
        nannealpasses.setText(default['nannealpasses'])
        config_layout.addRow('N anneal passes:',nannealpasses)
        
        shuffle_clusters = QLineEdit(subwidget)
        shuffle_clusters.setText(default['shuffle_clusters'])
        config_layout.addRow('Shuffle clusters:',shuffle_clusters)
        
        mergeT = QLineEdit(subwidget)
        mergeT.setText(default['mergeT'])
        config_layout.addRow('Merge thresh:',mergeT)
        
        splitT = QLineEdit(subwidget)
        splitT.setText(default['splitT'])
        config_layout.addRow('Split thresh:',splitT)
        
        initialize = QComboBox(subwidget)
        initialize.addItems(['fromData','no'])
        initialize.setCurrentIndex(initialize.findText(default['initialize']))
        config_layout.addRow('Initialize:',initialize)
        
        spkTh = QLineEdit(subwidget)
        spkTh.setText(default['spkTh'])
        config_layout.addRow('Spike thresh:',spkTh)
        
        loc_range1 = QLineEdit(subwidget)
        loc_range1.setText(default['loc_range'][0])
        loc_range2 = QLineEdit(subwidget)
        loc_range2.setText(default['loc_range'][1])
        config_layout.addRow('Local range:',loc_range1)
        config_layout.addRow('',loc_range2)
        
        long_range1 = QLineEdit(subwidget)
        long_range1.setText(default['long_range'][0])
        long_range2 = QLineEdit(subwidget)
        long_range2.setText(default['long_range'][1])
        config_layout.addRow('Long range:',long_range1)
        config_layout.addRow('',long_range2)
        
        maskMaxChannels = QLineEdit(subwidget)
        maskMaxChannels.setText(default['maskMaxChannels'])
        config_layout.addRow('Mask max channels:',maskMaxChannels)
        
        crit = QLineEdit(subwidget)
        crit.setText(default['crit'])
        config_layout.addRow('Crit:',crit)
        
        nFiltMax = QLineEdit(subwidget)
        nFiltMax.setText(default['nFiltMax'])
        config_layout.addRow('N filt max:',nFiltMax)
        
        
        def save(savetype):
            ''' save adv settings for the current run cycle '''
            
            ops = {}
            ops['GPU'] = str(gpu.text())
            ops['parfor'] = str(parfor.text())
            ops['verbose'] = str(verbose.text())
            ops['showfigures'] = str(showfigures.text())
            ops['datatype'] = str(datatype.currentText())
            ops['Nfilt'] = str(Nfilt.text())
            ops['whitening'] = str(whitening.currentText())
            ops['nSkipCov'] = str(nSkipCov.text())
            ops['whiteningRange'] = str(whiteningRange.text())
            ops['criterionNoiseChannels'] = str(criterionNoiseChannels.text())
            ops['Nrank'] = str(Nrank.text())
            ops['nfullpasses'] = str(nfullpasses.text())
            ops['maxFR'] = str(maxFR.text())
            ops['fshigh'] = str(fshigh.text())
            ops['ntbuff'] = str(ntbuff.text())
            ops['scaleproc'] = str(scaleproc.text())
            ops['Th'] = (str(Th1.text()),str(Th2.text()),str(Th3.text()))
            ops['lam'] = (str(lam1.text()),str(lam2.text()),str(lam3.text()))
            ops['nannealpasses'] = str(nannealpasses.text())
            ops['shuffle_clusters'] = str(shuffle_clusters.text())
            ops['mergeT'] = str(mergeT.text())
            ops['splitT'] = str(splitT.text())
            ops['initialize'] = str(initialize.currentText())
            ops['spkTh'] = str(spkTh.text())
            ops['loc_range'] = (str(loc_range1.text()),str(loc_range2.text()))
            ops['long_range'] = (str(long_range1.text()),str(long_range2.text()))
            ops['maskMaxChannels'] = str(maskMaxChannels.text())
            ops['crit'] = str(crit.text())
            ops['nFiltMax'] = str(nFiltMax.text())
                    
            if savetype == 'default':
                
                #save settings as current settings
                with open('default_config_ops.pickle','wb') as f:
                    pickle.dump(ops,f,protocol=2)
                    f.close()
                    
            #save settings as current settings
            with open('config_ops.pickle','wb') as f:
                pickle.dump(ops,f,protocol=2)
                f.close()
                
            #close the window
            self.config_window.close()

        #make QPushButtons for "save" and "save as defaults" options
        usenow_button = QPushButton('Use Now')
        saveasdefault_button = QPushButton('Save as Defaults')
        usenow_button.clicked.connect(lambda: save('now'))
        saveasdefault_button.clicked.connect(lambda: save('default'))
        
        scroll.setWidget(subwidget)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        
        parent_layout = QVBoxLayout()
        self.config_window.setLayout(parent_layout)
        parent_layout.addWidget(scroll)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(usenow_button)
        button_layout.addWidget(saveasdefault_button)
        parent_layout.addLayout(button_layout)
        
        self.config_window.show()
        
    def closeEvent(self,event):
        
        if self.launcher is not None:
            self.launcher.show_launcher()  
            
        
                        
    #this function acts as a slot to accept 'message' signal
    @Slot(str)
    def print_message(self, message):
        ''' print stdout and stderr to terminal window '''
        
        #move terminal cursor to end
        self.terminal.moveCursor(QTextCursor.End)
        #write message to terminal
        self.terminal.insertPlainText(message)
     
if __name__ == '__main__':
    
    #create a QApplication if one doesn't already exist
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
    
    #create and show the main window
    frame = MainWindow()
    frame.show()
    
    #set up stream for stdout and stderr based on outputStream class
    outputStream = outputStream()
    #when outputStream sends messages, connect to appropriate function for
    #writing to terminal window
    outputStream.message.connect(frame.print_message)
    
    #connect stdout and stderr to outputStream
    sys.stdout = outputStream  
    sys.stderr = outputStream
    
    #exit the app when we're all done
    sys.exit(app.exec_())
        
