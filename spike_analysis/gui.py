# -*- coding: utf-8 -*-
"""
Created on Tues Aug 1 11:44:25 2017

script for running the main gui

@author: Patrick
"""

import os
#os.chdir('./spike_analysis')
os.environ['QT_API'] = 'pyside'
import sys
import pickle
import time

         
from PySide2.QtCore import (QRect,Qt,QObject,Signal,Slot,QThread,QEventLoop,QTimer)
from PySide2.QtWidgets import (QAction, QApplication, QMainWindow, QFrame, QLabel, QCheckBox, QLineEdit, QButtonGroup,
                          QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox,
                          QTextEdit, QMenuBar, QMenu, QStatusBar, QStyle, QPushButton, QFileDialog)

from PySide2.QtGui import QTextCursor

from spike_analysis import main, plot

import matplotlib.pyplot as plt



from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import warnings
warnings.filterwarnings('ignore')

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
    #signal for initializing metadata
    init_data = Signal(dict)
    #signal for updating plotting data
    plotting_data = Signal(dict,dict,dict,dict,dict)
    #signal for updating all_data dictionary
    data_dict = Signal(dict)
    #signal for stopping processes (#TODO: not currently fully functional)
    stop = Signal(str)
    
    def __init__(self,fname,gui,parent=None):
        #standard __init__
        super(Worker, self).__init__(parent)
        #grab fname from Run function and MainWindow instance, assign to class
        self.fname = fname
        self.gui = gui

    def run_full(self):
        #start a process
        self.isrunning = True
        main.Run(self.fname,self.gui)

#class for MainWindow instance
class MainWindow(QMainWindow):
    #signals for starting main script and initializing data
    start = Signal(str)
    initialized = Signal(str)
    plot_signal = Signal(object)
    
    def __init__(self, parent=None, launcher=None):
        ''' sets up the whole main window '''
        
        self.launcher = launcher
        
        #standard init
        super(MainWindow, self).__init__(parent)
        #set the window size in pixels
        self.resize(1100, 700)
        #set the window title
        self.setWindowTitle('')

        #set up buttons etc.
        self.set_presets()
        self.set_options()
        self.set_animations()
        self.setup_mpl_canvas()
        self.set_bonus_options()
        self.set_save_options()
        self.set_acq_options()
        self.set_classify_options()
        
        #create QTextEdit window 'terminal' for receiving stdout and stderr
        self.terminal = QTextEdit(self)
        #set the geometry
        self.terminal.setGeometry(QRect(30, 400, 530, 200))
        
        #create a stopbutton and set geometry
        self.stopbutton = QPushButton('Stop!',self)
        self.stopbutton.setGeometry(QRect(230,610,100,30))

        #call function to set default options
        self.set_defaults()

        #add the matplotlib NavigationToolbar above plotting canvas
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setGeometry(QRect(580, 37, 500, 60))

        #create a QMenuBar and set geometry
        menubar = QMenuBar(self)
        menubar.setGeometry(QRect(0, 0, 731, 29))
        #set the QMenuBar as menu bar for main window
        self.setMenuBar(menubar)
        #create a QMenu
        menu = QMenu(menubar)

        #create a QStatusBar
        statusbar = QStatusBar(self)
        #set it as status bar for main window
        self.setStatusBar(statusbar)
        
        #create icon toolbar with default image
        iconToolBar = self.addToolBar("iconBar.png")
        
        #create a QAction for the run button
        action_Run = QAction(self)  
        #grab an icon for the button
        run_icon=self.style().standardIcon(QStyle.SP_ArrowRight)
        #set the icon for the action
        action_Run.setIcon(run_icon)
        #when the button is pressed, call the Run function
        action_Run.triggered.connect(self.Run)
        
        #create a QAction for the advanced settings button
        actionAdvanced = QAction(self)
        #grab an icon for the button
        advanced_icon=self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        #set the icon for the action
        actionAdvanced.setIcon(advanced_icon)
        #when the button is pressed, call advanced_settings function
        actionAdvanced.triggered.connect(self.advanced_settings)
        
        #create QAction for close button
        action_Close = QAction(self)   
        #grab close icon
        close_icon=self.style().standardIcon(QStyle.SP_DialogCancelButton)
        #set icon for action
        action_Close.setIcon(close_icon)
        action_Close.triggered.connect(self.close) 

        #show tips for each action in the status bar
        actionAdvanced.setStatusTip("Edit advanced settings.")
        action_Run.setStatusTip("Run the program.")
        action_Close.setStatusTip("Close the program.")
#------------------------------------------------------
        
        #add actions to the menu
        menu.addAction(action_Run)
        menu.addAction(action_Close)
        menu.addAction(actionAdvanced)
        #add menu to menubar as its own action (what?)
        menubar.addAction(menu.menuAction())
     
        #add actions to icon toolbar
        iconToolBar.addAction(action_Run)
        iconToolBar.addAction(action_Close)
        iconToolBar.addAction(actionAdvanced)
        
        self.plot_signal.connect(self.plot_something)
            
        
    #this function acts as a slot to accept 'message' signal
    @Slot(str)
    def print_message(self, message):
        ''' print stdout and stderr to terminal window '''
        
        #move terminal cursor to end
        self.terminal.moveCursor(QTextCursor.End)
        #write message to terminal
        self.terminal.insertPlainText(message)
        
    def plot_something(self,something):
        
        plt.figure()
        plt.plot(something[0],something[1],'k.')
        plt.show()
        
    def setup_mpl_canvas(self):
        ''' set up the matplotlib plotting canvas '''
        
        #create QFrame with parent self
        plot_frame = QFrame(self)
        #make a label for the canvas
        self.canvas_label = QLabel()
        #make label empty for now
        self.canvas_label.setText('')
        #set geometry
        plot_frame.setGeometry(QRect(580, 100, 500, 500))
        #name the QFrame so we can give it a cool color
        plot_frame.setObjectName("canvas_widget")
        #give it a cool color
        plot_frame.setStyleSheet("#canvas_widget {background-color:white;}") 
        #give the QFrame a layout
        plot_layout = QVBoxLayout()
        plot_frame.setLayout(plot_layout)
        #create a matplotlib figure for plotting data to
        self.figure = Figure()
        #create a canvas to show the figure on
        self.canvas = FigureCanvas(self.figure)
        #add canvas and label to layout
        plot_layout.addWidget(self.canvas_label)
        plot_layout.addWidget(self.canvas)
        
        #make a QFrame for Next and Back buttons
        man_frame = QFrame(self)
        man_frame.setGeometry(QRect(580,605,500,50))
        man_frame.setObjectName("manual_widget")
        #make it white
        man_frame.setStyleSheet("#manual_widget {background-color:white;}")
        #layout
        man_layout = QHBoxLayout()
        man_frame.setLayout(man_layout)
        #next and back buttons
        self.backbutton = QPushButton('Back')
        self.nextbutton = QPushButton('Next')
        self.pausebutton = QPushButton('Pause')
        
        #disable next and back buttons while no plotting data is available
        self.nextbutton.setEnabled(False)
        self.backbutton.setEnabled(False)
        self.pausebutton.setEnabled(False)
        
        #if next button is ever pressed, call plot_following func with 'next' argument
        self.nextbutton.clicked.connect(lambda: self.plot_following('next'))
        #if back button, call with 'back' argument
        self.backbutton.clicked.connect(lambda: self.plot_following('back'))        
        
        #add widgets
        man_layout.addWidget(self.backbutton)
        man_layout.addWidget(self.pausebutton)
        man_layout.addWidget(self.nextbutton)
        
    def set_presets(self):
        ''' create presets frame '''
        
        #make QFrame
        presets = QFrame(self)
        #make label
        self.preset_label = QLabel(self)
        self.preset_label.setText('Presets')
        #set geometry and color
        presets.setGeometry(QRect(210, 300, 170, 90))
        presets.setObjectName("myWidget")
        presets.setStyleSheet("#myWidget {background-color:cyan;}") 
        #create and set layout
        preset_layout = QGridLayout()
        presets.setLayout(preset_layout)
        #make buttons for presets (currently Grid and Greatest Hits)
        self.gridbutton = QPushButton('Grid')
        self.greatestbutton = QPushButton('Greatest Hits')
        self.egobutton = QPushButton('Ego')
        self.allbutton = QPushButton('ALL')
        
        self.gridbutton.clicked.connect(self.grid_options)
        self.greatestbutton.clicked.connect(self.greatest_hits)
        self.egobutton.clicked.connect(self.ego_options)
        self.allbutton.clicked.connect(self.all_options)
        
        #add widgets to layout
        preset_layout.addWidget(self.preset_label,0,0)
        preset_layout.addWidget(self.gridbutton,1,0)
        preset_layout.addWidget(self.greatestbutton,2,0)
        preset_layout.addWidget(self.egobutton,1,1)
        preset_layout.addWidget(self.allbutton,2,1)

    
    def set_options(self):
        ''' create options frame '''
        
        #make QFrame
        options = QFrame(self)
        #make label
        options_label = QLabel()
        options_label.setText('Analyses')
        #set geometry and color
        options.setGeometry(QRect(30, 100, 170, 290))
        options.setObjectName("mysadfWidget")
        options.setStyleSheet("#mysadfWidget {background-color:gray;}") 
        #make and set layout
        options_layout = QVBoxLayout()
        options.setLayout(options_layout)
        
        #create check boxes for each plotting option
        self.hdbox = QCheckBox('Head Direction') 
        self.spatialbox = QCheckBox('Spatial')
        self.hdspatialbox = QCheckBox('HD x Spatial')
        self.gridbox = QCheckBox('Grid')
        self.speedbox = QCheckBox('Speed')
        self.ahvbox = QCheckBox('AHV')
        self.timebox = QCheckBox('Temporal')
        self.egobox = QCheckBox('Egocentric Bearings')
        self.ebcbox = QCheckBox('EBC')
        self.centeregobox = QCheckBox('Center Ego')
        self.wallegobox = QCheckBox('Closest Wall Ego')
        self.viewbox = QCheckBox('View')
        #create clear button
        self.clearbutton = QPushButton('Clear')
        #point other buttons to appropriate functions
        self.clearbutton.clicked.connect(self.clear_all)

        #add buttons to layout
        options_layout.addWidget(options_label)
        options_layout.addWidget(self.hdbox)
        options_layout.addWidget(self.spatialbox)
        options_layout.addWidget(self.hdspatialbox)
        options_layout.addWidget(self.gridbox)
        options_layout.addWidget(self.speedbox)
        options_layout.addWidget(self.ahvbox)
        options_layout.addWidget(self.timebox)
        options_layout.addWidget(self.egobox)
        options_layout.addWidget(self.ebcbox)
        options_layout.addWidget(self.centeregobox)
        options_layout.addWidget(self.wallegobox)
#        options_layout.addWidget(self.viewbox)
        options_layout.addWidget(self.clearbutton)


    def set_animations(self):
        ''' create animations frame '''
        
        #make QFrame
        animations = QFrame(self)
        #make label
        animations_label = QLabel()
        animations_label.setText('Animations')
        #set geometry and color
        animations.setGeometry(QRect(390, 100, 170, 95))
        animations.setObjectName("animationWidget")
        animations.setStyleSheet("#animationWidget {background-color:pink;}") 
        #create and set layout
        animations_layout = QVBoxLayout()
        animations.setLayout(animations_layout)
        #make checkboxes for animations
        self.ani_spikebox = QCheckBox('Path && Spike') 
        self.ani_hdbox = QCheckBox('Path && Spike x HD') 
        self.ani_heatbox = QCheckBox('Heat Map')
        #add label and boxes to layout
        animations_layout.addWidget(animations_label)     
        animations_layout.addWidget(self.ani_spikebox)
        animations_layout.addWidget(self.ani_hdbox)
        animations_layout.addWidget(self.ani_heatbox)
        
        
    def set_classify_options(self):
        ''' create classify frame '''
        
        #make QFrame
        classify_options = QFrame(self)
        #make label
        classify_label = QLabel()
        classify_label.setText('Classify')
        #set geometry and color
        classify_options.setGeometry(QRect(210, 100, 170, 190))
        classify_options.setObjectName("classifyWidget")
        classify_options.setStyleSheet("#classifyWidget {background-color:yellow;}") 
        #create and set layout
        classify_layout = QVBoxLayout()
        classify_options.setLayout(classify_layout)
        #make checkboxes for animations
        self.class_posbox = QCheckBox('2D Position') 
        self.class_hdbox = QCheckBox('HD')
        self.class_speedbox = QCheckBox('Speed')
        self.class_ahvbox = QCheckBox('AHV')
        self.class_bearingbox = QCheckBox('Bearing')
        self.class_distbox = QCheckBox('Dist')
        
        self.class_clearbutton = QPushButton('Clear')
        self.class_clearbutton.clicked.connect(self.clear_class_options)

        #add label and boxes to layout
        classify_layout.addWidget(classify_label)        
        classify_layout.addWidget(self.class_posbox)
        classify_layout.addWidget(self.class_hdbox)
        classify_layout.addWidget(self.class_speedbox)
        classify_layout.addWidget(self.class_ahvbox)
        classify_layout.addWidget(self.class_bearingbox)
        classify_layout.addWidget(self.class_distbox)
        classify_layout.addWidget(self.class_clearbutton)
        
        
    def set_save_options(self):
        ''' create save options frame '''
        
        #make QFrame
        save_options = QFrame(self)
        #label
        save_label = QLabel()
        save_label.setText('Save Options')
        #geometry and color
        save_options.setGeometry(QRect(390,205,170,95))
        save_options.setObjectName("saveWidget")
        save_options.setStyleSheet("#saveWidget {background-color:orange;}")
        #make and set layout
        save_layout = QVBoxLayout()
        save_options.setLayout(save_layout)
        #make check boxes for options
        self.savebox = QCheckBox('Save All Figures')
        self.saveglmbox = QCheckBox('Save GLM Profiles')
        self.savedatabox = QCheckBox('Save Data')

        #add to layout
        save_layout.addWidget(save_label)
        save_layout.addWidget(self.savebox)
        save_layout.addWidget(self.saveglmbox)
        save_layout.addWidget(self.savedatabox)

    def set_bonus_options(self):
        ''' create bonus options frame '''
        
        #make QFrame
        bonus_options = QFrame(self)
        #label
        bonus_label = QLabel()
        bonus_label.setText('Bonus Options')
        #geometry and color
        bonus_options.setGeometry(QRect(390,310,170,80))
        bonus_options.setObjectName("bonusWidget")
        bonus_options.setStyleSheet("#bonusWidget {background-color:yellow;}")
        #make and set layout
        bonus_layout = QVBoxLayout()
        bonus_options.setLayout(bonus_layout)
        #create check boxes for options
        self.singlebox = QCheckBox('Single Cluster')
        self.speedmodebox = QCheckBox('Speed Mode')
        #add to layout
        bonus_layout.addWidget(bonus_label)
        bonus_layout.addWidget(self.singlebox)
        bonus_layout.addWidget(self.speedmodebox)
        

    def set_acq_options(self):
        ''' create acquisition system frame '''
        
        #make QFrame
        acq_options = QFrame(self)
        #label
        acq_label = QLabel()
        acq_label.setText('Acq System')
        #geometry and color
        acq_options.setGeometry(QRect(100,50,400,30))
        acq_options.setObjectName("acqWidget")
        acq_options.setStyleSheet("#acqWidget {background-color:magenta;}")
        #make and set layout
        acq_layout = QHBoxLayout()
        acq_options.setLayout(acq_layout)
        #create check boxes for options
        self.nlx_box = QCheckBox('Neuralynx')
        self.openephys_box = QCheckBox('Open-Ephys')
        self.old_box = QCheckBox('Taube')
        
        acq_buttons = QButtonGroup(self)
        acq_buttons.addButton(self.nlx_box)
        acq_buttons.addButton(self.openephys_box)
        acq_buttons.addButton(self.old_box)
        acq_buttons.setExclusive(True)
        
        #add to layout
        acq_layout.addWidget(acq_label)
        acq_layout.addWidget(self.nlx_box)
        acq_layout.addWidget(self.openephys_box)
        acq_layout.addWidget(self.old_box)
    
    def set_defaults(self):
        ''' set default options '''
        
        #first load the options used last time, saved as a pickle file
        with open('options.pickle','rb') as f:
            [run_hd,run_spatial,hd_map,run_grid,run_speed,run_ahv,run_autocorr,
             run_ego,run_ebc,run_center_ego,run_wall_ego,run_view,animated_path_spike,animated_hd_map,heatmap_animation,save_all,
             singlecluster,speedmode,savedata,acq_system] = pickle.load(f)
            
        with open('classify_options.pickle','rb') as f:
            [classify_pos,classify_hd,classify_speed,classify_ahv,classify_bearing,classify_dist,save_profiles] = pickle.load(f)
            
        #grab default advanced options
        with open('default_advanced.pickle','rb') as f:
            default_advanced = pickle.load(f)
            
        #set default advanced options as current advanced options
        with open('advanced.pickle','wb') as f:
            pickle.dump(default_advanced,f,protocol=2)
            
        #set each checkbox according to loaded options
        self.hdbox.setChecked(run_hd)
        self.spatialbox.setChecked(run_spatial)
        self.hdspatialbox.setChecked(hd_map)
        self.gridbox.setChecked(run_grid)
        self.speedbox.setChecked(run_speed)
        self.ahvbox.setChecked(run_ahv)
        self.timebox.setChecked(run_autocorr)
        self.egobox.setChecked(run_ego)
        self.ebcbox.setChecked(run_ebc)
        self.centeregobox.setChecked(run_center_ego)
        self.wallegobox.setChecked(run_wall_ego)
        self.viewbox.setChecked(run_view)
        self.ani_spikebox.setChecked(animated_path_spike)
        self.ani_hdbox.setChecked(animated_hd_map)
        self.ani_heatbox.setChecked(heatmap_animation)
        self.savebox.setChecked(save_all)
        self.saveglmbox.setChecked(save_profiles)
        self.singlebox.setChecked(singlecluster)
        self.speedmodebox.setChecked(speedmode)
        self.savedatabox.setChecked(savedata)
        
        self.class_posbox.setChecked(classify_pos)
        self.class_hdbox.setChecked(classify_hd)
        self.class_speedbox.setChecked(classify_speed)
        self.class_ahvbox.setChecked(classify_ahv)
        self.class_bearingbox.setChecked(classify_bearing)
        self.class_distbox.setChecked(classify_dist)
        
        if acq_system == 'neuralynx':
            self.nlx_box.setChecked(True)
        elif acq_system == 'openephys':
            self.openephys_box.setChecked(True)
        elif acq_system == 'old':
            self.old_box.setChecked(True)
        
    def clear_all(self):
        ''' clear checkboxes in options frame '''
        
        self.hdbox.setCheckState(Qt.Unchecked)
        self.spatialbox.setCheckState(Qt.Unchecked)
        self.hdspatialbox.setCheckState(Qt.Unchecked)
        self.gridbox.setCheckState(Qt.Unchecked)
        self.speedbox.setCheckState(Qt.Unchecked)
        self.ahvbox.setCheckState(Qt.Unchecked)
        self.timebox.setCheckState(Qt.Unchecked)
        self.egobox.setCheckState(Qt.Unchecked)
        self.ebcbox.setCheckState(Qt.Unchecked)
        self.centeregobox.setCheckState(Qt.Unchecked)
        self.wallegobox.setCheckState(Qt.Unchecked)
        self.viewbox.setCheckState(Qt.Unchecked)
        
    def clear_class_options(self):
        ''' clear checkboxes in classify frame '''
        
        self.class_posbox.setCheckState(Qt.Unchecked)
        self.class_hdbox.setCheckState(Qt.Unchecked)
        self.class_speedbox.setCheckState(Qt.Unchecked)
        self.class_ahvbox.setCheckState(Qt.Unchecked)
        self.class_bearingbox.setCheckState(Qt.Unchecked)
        self.class_distbox.setCheckState(Qt.Unchecked)
        
    def grid_options(self):
        ''' set options checkboxes according to grid preset '''
        
        self.hdbox.setCheckState(Qt.Checked)
        self.spatialbox.setCheckState(Qt.Checked)
        self.hdspatialbox.setCheckState(Qt.Checked)
        self.gridbox.setCheckState(Qt.Checked)
        self.speedbox.setCheckState(Qt.Unchecked)
        self.ahvbox.setCheckState(Qt.Unchecked)
        self.timebox.setCheckState(Qt.Unchecked)
        self.egobox.setCheckState(Qt.Unchecked)
        self.ebcbox.setCheckState(Qt.Unchecked)
        self.centeregobox.setCheckState(Qt.Unchecked)
        self.wallegobox.setCheckState(Qt.Unchecked)
        
    def greatest_hits(self):
        ''' set options checkboxes according to greatest hits preset '''
        
        self.hdbox.setCheckState(Qt.Checked)
        self.spatialbox.setCheckState(Qt.Checked)
        self.hdspatialbox.setCheckState(Qt.Checked)
        self.gridbox.setCheckState(Qt.Checked)
        self.speedbox.setCheckState(Qt.Checked)
        self.ahvbox.setCheckState(Qt.Checked)
        self.timebox.setCheckState(Qt.Checked)
        self.egobox.setCheckState(Qt.Unchecked)
        self.ebcbox.setCheckState(Qt.Unchecked)
        self.centeregobox.setCheckState(Qt.Unchecked)
        self.wallegobox.setCheckState(Qt.Unchecked)
        
    def ego_options(self):
        ''' set options checkboxes according to ego preset '''
        
        self.hdbox.setCheckState(Qt.Unchecked)
        self.spatialbox.setCheckState(Qt.Unchecked)
        self.hdspatialbox.setCheckState(Qt.Unchecked)
        self.gridbox.setCheckState(Qt.Unchecked)
        self.speedbox.setCheckState(Qt.Unchecked)
        self.ahvbox.setCheckState(Qt.Unchecked)
        self.timebox.setCheckState(Qt.Unchecked)
        self.egobox.setCheckState(Qt.Checked)
        self.ebcbox.setCheckState(Qt.Checked)
        self.centeregobox.setCheckState(Qt.Checked)
        self.wallegobox.setCheckState(Qt.Checked)
        
    def all_options(self):
        ''' set options checkboxes according to ALL preset '''
        
        self.hdbox.setCheckState(Qt.Checked)
        self.spatialbox.setCheckState(Qt.Checked)
        self.hdspatialbox.setCheckState(Qt.Checked)
        self.gridbox.setCheckState(Qt.Checked)
        self.speedbox.setCheckState(Qt.Checked)
        self.ahvbox.setCheckState(Qt.Checked)
        self.timebox.setCheckState(Qt.Checked)
        self.egobox.setCheckState(Qt.Checked)
        self.ebcbox.setCheckState(Qt.Checked)
        self.centeregobox.setCheckState(Qt.Checked)
        self.wallegobox.setCheckState(Qt.Checked)
         
    def advanced_settings(self):
        ''' advanced settings window'''
        
        def save():
            ''' save adv settings for the current run cycle '''
            
            if str(self.arena_x.text()) == 'None' or self.arena_x.text() == None:
                arena_x = None
            else:
                arena_x = float(self.arena_x.text())
                
            if str(self.arena_y.text()) == 'None' or self.arena_y.text() == None:
                arena_y = None
            else:
                arena_y = float(self.arena_y.text())
            
            #grab settings with appropriate data types
            advanced=[str(self.hd_calc.currentText()),arena_x,arena_y,int(self.pic_resolution.text()),float(self.spatial_bin.text()),
                              float(self.speed_bin.text()),float(self.ahv_bin.text()),float(self.ego_dist_bin.text()),
                              float(self.ebc_dist.text()),float(self.ebc_bearing.text()),float(self.ego_ref.text()),int(self.bin_size.text()),float(self.autocorr_width.text()),
                              float(self.framerate.text()),float(self.hd_bin.text()),float(self.sample_cutoff.text()),float(self.speed_cutoff.text()),
                              int(self.ani_speed.text())] 
            #save settings as current settings
            with open('advanced.pickle','wb') as f:
                pickle.dump(advanced,f,protocol=2)
                f.close()
                
            #close the window
            self.wid.close()
                
        def save_as_defaults():
            ''' save adv settings for now and the future '''
            
            if str(self.arena_x.text()) == 'None' or self.arena_x.text() == None:
                arena_x = None
            else:
                arena_x = float(self.arena_x.text())
                
            if str(self.arena_y.text()) == 'None' or self.arena_y.text() == None:
                arena_y = None
            else:
                arena_y = float(self.arena_y.text())
            
            #grab settings with appropriate data types
            default_advanced=[str(self.hd_calc.currentText()),arena_x,arena_y,int(self.pic_resolution.text()),float(self.spatial_bin.text()),
                              float(self.speed_bin.text()),float(self.ahv_bin.text()),float(self.ego_dist_bin.text()),
                              float(self.ebc_dist.text()),float(self.ebc_bearing.text()),float(self.ego_ref.text()),int(self.bin_size.text()),float(self.autocorr_width.text()),
                              float(self.framerate.text()),float(self.hd_bin.text()),float(self.sample_cutoff.text()),float(self.speed_cutoff.text()),
                              int(self.ani_speed.text())]

            #save as default settings
            with open('default_advanced.pickle','wb') as f:
                pickle.dump(default_advanced,f,protocol=2)
            #save as current settings
            with open('advanced.pickle','wb') as f:
                pickle.dump(default_advanced,f,protocol=2)
            
            #close window
            self.wid.close()
        
        #load current settings from pickled file
        with open('advanced.pickle','rb') as f:
            [hdc,ax,ay,pr,spat_b,speed_b,ahv_b,dist_b,ebcd_b,ebcb_b,ego_ref_b,bs,ac,fr,hd_b,sac,spc,ans] = pickle.load(f)

        #create window as parentless QWidget, resize
        self.wid = QWidget()
        self.wid.resize(400, 350)
        #set title
        self.wid.setWindowTitle('Advanced Settings')
        #give layout
        self.widlayout = QGridLayout(self.wid)
        
        #HD calculation
        self.hd_calc_label = QLabel()
        self.hd_calc_label.setText('Head direction calculated from...')
        self.hd_calc = QComboBox()
        self.hd_calc.addItems(['Neuralynx','LED positions'])
        if hdc == 'Neuralynx':
            self.hd_calc.setCurrentIndex(self.hd_calc.findText('Neuralynx'))
        elif hdc == 'LED positions':
            self.hd_calc.setCurrentIndex(self.hd_calc.findText('LED positions'))
        self.widlayout.addWidget(self.hd_calc_label,0,0)
        self.widlayout.addWidget(self.hd_calc,1,0)
        
        #set label for arena_x setting
        self.arena_x_label = QLabel()
        self.arena_x_label.setText('Arena size along x axis (cm or None)')
        #make a QLineEdit box for displaying/editing settings
        self.arena_x = QLineEdit()
        #set text equal to loaded setting
        self.arena_x.setText(str(ax))
        #add label and box to current window
        self.widlayout.addWidget(self.arena_x_label,2,0)
        self.widlayout.addWidget(self.arena_x,3,0)
        
        #set label for arena_y setting
        self.arena_y_label = QLabel()
        self.arena_y_label.setText('Arena size along y ayis (cm or None)')
        #make a QLineEdit box for displaying/editing settings
        self.arena_y = QLineEdit()
        #set text equal to loaded setting
        self.arena_y.setText(str(ay))
        #add label and boy to current window
        self.widlayout.addWidget(self.arena_y_label,4,0)
        self.widlayout.addWidget(self.arena_y,5,0)
        
        #set label for ani_speed setting
        self.ani_speed_label = QLabel()
        self.ani_speed_label.setText('Animation speed multiplier')
        #make a QLineEdit box for displaying/editing settings
        self.ani_speed = QLineEdit()
        #set text equal to loaded setting
        self.ani_speed.setText(str(ans))
        #add label and box to current window
        self.widlayout.addWidget(self.ani_speed_label,8,2)
        self.widlayout.addWidget(self.ani_speed,9,2)
        
        #set label for pic_resolution setting
        self.pic_resolution_label = QLabel()
        self.pic_resolution_label.setText('Image Resolution (DPI)')
        #make a QLineEdit box for displaying/editing settings
        self.pic_resolution = QLineEdit()
        #set text equal to loaded setting
        self.pic_resolution.setText(str(pr))
        #add label and box to current window
        self.widlayout.addWidget(self.pic_resolution_label,10,2)
        self.widlayout.addWidget(self.pic_resolution,11,2)
        
        #spatial_bin
        self.spatial_bin_label = QLabel()
        self.spatial_bin_label.setText('2D position bin size (cm<sup>2</sup>)')
        self.spatial_bin = QLineEdit()
        self.spatial_bin.setText(str(spat_b))
        self.widlayout.addWidget(self.spatial_bin_label,0,1)
        self.widlayout.addWidget(self.spatial_bin,1,1)
        
        #speed_bin
        self.speed_bin_label = QLabel()
        self.speed_bin_label.setText('Speed bin size (cm/s)')
        self.speed_bin = QLineEdit()
        self.speed_bin.setText(str(speed_b))
        self.widlayout.addWidget(self.speed_bin_label,4,1)
        self.widlayout.addWidget(self.speed_bin,5,1)
        
        #ahv_bin
        self.ahv_bin_label = QLabel()
        self.ahv_bin_label.setText('AHV bin size (deg/s)')
        self.ahv_bin = QLineEdit()
        self.ahv_bin.setText(str(ahv_b))
        self.widlayout.addWidget(self.ahv_bin_label,6,1)
        self.widlayout.addWidget(self.ahv_bin,7,1)
        
        #ego_dist_bin
        self.ego_dist_bin_label = QLabel()
        self.ego_dist_bin_label.setText('Center/wall dist bin size (cm)')
        self.ego_dist_bin = QLineEdit()
        self.ego_dist_bin.setText(str(dist_b))
        self.widlayout.addWidget(self.ego_dist_bin_label,0,2)
        self.widlayout.addWidget(self.ego_dist_bin,1,2)
        
        #ebc_dist_bin
        self.ebc_dist_label = QLabel()
        self.ebc_dist_label.setText('EBC dist bin size (cm)')
        self.ebc_dist = QLineEdit()
        self.ebc_dist.setText(str(ebcd_b))
        self.widlayout.addWidget(self.ebc_dist_label,8,1)
        self.widlayout.addWidget(self.ebc_dist,9,1)
        
        #ebc_bearing_bin
        self.ebc_bearing_label = QLabel()
        self.ebc_bearing_label.setText('EBC angular bin size (deg)')
        self.ebc_bearing = QLineEdit()
        self.ebc_bearing.setText(str(ebcb_b))
        self.widlayout.addWidget(self.ebc_bearing_label,10,1)
        self.widlayout.addWidget(self.ebc_bearing,11,1)
        
        #ego_ref_bin
        self.ego_ref_label = QLabel()
        self.ego_ref_label.setText('Spacing of ego bearing reference points (cm)')
        self.ego_ref = QLineEdit()
        self.ego_ref.setText(str(ego_ref_b))
        self.widlayout.addWidget(self.ego_ref_label,2,2)
        self.widlayout.addWidget(self.ego_ref,3,2)
        
        #bin_size
        self.bin_size_label = QLabel()
        self.bin_size_label.setText('Bin size for temporal analyses (ms)')
        self.bin_size = QLineEdit()
        self.bin_size.setText(str(bs))
        self.widlayout.addWidget(self.bin_size_label,4,2)
        self.widlayout.addWidget(self.bin_size,5,2)
        
        #autocorr_width
        self.autocorr_width_label = QLabel()
        self.autocorr_width_label.setText('Spike autocorr width (seconds)')
        self.autocorr_width = QLineEdit()
        self.autocorr_width.setText(str(ac))
        self.widlayout.addWidget(self.autocorr_width_label,6,2)
        self.widlayout.addWidget(self.autocorr_width,7,2)
        
        #framerate
        self.framerate_label = QLabel()
        self.framerate_label.setText('Framerate')
        self.framerate = QLineEdit()
        self.framerate.setText(str(fr))
        self.widlayout.addWidget(self.framerate_label,6,0)
        self.widlayout.addWidget(self.framerate,7,0)
        
        #hd_bin
        self.hd_bin_label = QLabel()
        self.hd_bin_label.setText('Angular bin size (deg)')
        self.hd_bin = QLineEdit()
        self.hd_bin.setText(str(hd_b))
        self.widlayout.addWidget(self.hd_bin_label,2,1)
        self.widlayout.addWidget(self.hd_bin,3,1)
        
        #sample_cutoff
        self.sample_cutoff_label = QLabel()
        self.sample_cutoff_label.setText('Minimum occupancy for spatial heatmap (seconds)')
        self.sample_cutoff = QLineEdit()
        self.sample_cutoff.setText(str(sac))
        self.widlayout.addWidget(self.sample_cutoff_label,10,0)
        self.widlayout.addWidget(self.sample_cutoff,11,0)
        
        #speed_cutoff
        self.speed_cutoff_label = QLabel()
        self.speed_cutoff_label.setText('Minimum speed cutoff (cm/s)')
        self.speed_cutoff = QLineEdit()
        self.speed_cutoff.setText(str(spc))
        self.widlayout.addWidget(self.speed_cutoff_label,8,0)
        self.widlayout.addWidget(self.speed_cutoff,9,0)
        
        #make QPushButtons for "save" and "save as defaults" options
        self.save = QPushButton('Use Now')
        self.save_as_defaults = QPushButton('Save as Defaults')
        
        #add these buttons to window layout
        self.widlayout.addWidget(self.save,12,1)
        self.widlayout.addWidget(self.save_as_defaults,12,2)
        #call appropriate functions when buttons are pressed
        self.save.clicked.connect(save)
        self.save_as_defaults.clicked.connect(save_as_defaults)
        
        #show the window
        self.wid.show()
        
    def Run(self):
        '''Run the main analysis program!!'''
        
        #first check which acquisition system is being used
        if self.nlx_box.isChecked():
            acq_system = 'neuralynx'
        elif self.openephys_box.isChecked():
            acq_system = 'openephys'
        elif self.old_box.isChecked():
            acq_system = 'taube'
        
        #next we need to save (pickle) the options -- first collect whether boxes
        #are checked
        options = [self.hdbox.isChecked(),self.spatialbox.isChecked(),self.hdspatialbox.isChecked(),
                   self.gridbox.isChecked(),self.speedbox.isChecked(),self.ahvbox.isChecked(),
                   self.timebox.isChecked(),self.egobox.isChecked(),self.ebcbox.isChecked(),self.centeregobox.isChecked(),
                   self.wallegobox.isChecked(),self.viewbox.isChecked(),self.ani_spikebox.isChecked(),self.ani_hdbox.isChecked(),
                   self.ani_heatbox.isChecked(),self.savebox.isChecked(),
                   self.singlebox.isChecked(),self.speedmodebox.isChecked(),
                   self.savedatabox.isChecked(),acq_system]
      
        #then save the options
        with open('options.pickle', 'wb') as f:
            pickle.dump(options, f, protocol=2)
            
        classify_options = [self.class_posbox.isChecked(),self.class_hdbox.isChecked(),self.class_speedbox.isChecked(),
                   self.class_ahvbox.isChecked(),self.class_bearingbox.isChecked(),self.class_distbox.isChecked(),
                   self.saveglmbox.isChecked()]
            
        with open('classify_options.pickle', 'wb') as f:
            pickle.dump(classify_options, f, protocol=2)
        
        #if we're doing single cluster mode, ask for a text file
        if self.singlebox.isChecked():
            fname = QFileDialog.getOpenFileName(self, 'Open File', '','Timestamp txt files (*.txt)')
        #otherwiese, ask for a directory
        else:
            fname = QFileDialog.getExistingDirectory()
            
        #create a worker object based on Worker class and move it to our 
        #worker thread
        self.workerThread = QThread(self)
        self.workerThread.start()
 
        self.worker = Worker(fname,self)
        self.worker.moveToThread(self.workerThread)
        
        try:
            self.start.disconnect()
        except:
            pass

        #connect the start button to the worker's main task
        self.start.connect(self.worker.run_full)
            
        #take note that we haven't made a plot yet and we're not currently
        #making spike-triggered plots
        self.first_plot = False
        
        #start the worker!
        self.start.emit('start!')
            
        #connect incoming data dictionary signals to relevant functions
        self.worker.data_dict.connect(self.update_data_dict)
        self.worker.init_data.connect(self.init_plotting_data)
        self.worker.plotting_data.connect(self.update_plotting_data)
        
        #connect stop button to process-stoppoing function
        self.stopbutton.clicked.connect(self.stop_process)
        
        #set auto-advance parameter to true
        self.auto = True
        #when data is initialized, start plotting
        self.initialized.connect(self.auto_plot)
        #if next or back buttons are clicked, break the auto-plot cycle
#        self.backbutton.clicked.connect(self.break_auto)
#        self.nextbutton.clicked.connect(self.break_auto)
        
        self.pausebutton.clicked.connect(self.pause_resume)
            
        
    def plot_first(self):
        ''' draw the first plot '''
        
        #grab the function name (string) from the first element in metadata plot_list
        function = self.metadata['plot_list'][0][2]
        #change the string into a callable function from the 'plot' script
        plot_func = getattr(plot,function)
        
        #clear the plotting figure
        self.figure.clear()
        
        #call the plotting function with data dict arguments
        plot_func(self.ops,self.adv,self.trial_data,self.cluster_data,self.spike_data,self)
        
        #set canvas label to name of current cluster
        if self.cluster_data['best_model'] is not None:
            self.canvas_label.setText(self.trial_data['current_cluster'] + ' -- ' + ', '.join(list(self.cluster_data['best_model'])))
        else:
            self.canvas_label.setText(self.trial_data['current_cluster'])   
            
        #draw the figure
        self.canvas.draw()
        
        #start the timer for auto-advancing
        self.tstart = time.time()

        #enable the next and back buttons now that plotting has started
#        self.nextbutton.setEnabled(True)
#        self.backbutton.setEnabled(True)
        self.pausebutton.setEnabled(True)
            
        #note that the first plot has been made
        self.first_plot = True
        #signal to start auto-advancing the plots
        self.initialized.emit('initialized!')
   
    def plot_following(self,direction):
        ''' plot the next or last (or same) plot '''

        #if the back button was pushed and we're on our first plot,
        #tell the user we can't do anything
        if direction == 'back' and self.plot_counter == 0:
            print('this is the first plot, can\'t go back now!')
            
        #if the next button was pushed and we're on our last plot,
        #tell the user we can't do anything        
        elif direction == 'next' and self.plot_counter == self.tot_plots - 1:
            if not self.auto:
                print('no more plots to show!')
            
        #otherwise, depending on the direction requested, tick the plot counter
        #accordingly and move on with the plotting procedure
        else:
            if direction == 'next':
                self.plot_counter += 1
            elif direction == 'back':
                self.plot_counter -= 1
            elif direction == 'same':
                pass
                
            #grab the current trial and cluster from metadata plot_list
            #based on plot_counter value
            self.current_trial = self.metadata['plot_list'][self.plot_counter][0]
            self.current_cluster = self.metadata['plot_list'][self.plot_counter][1]
            
            #grab the plot name and make it into a useable function
            plot_to_make = self.metadata['plot_list'][self.plot_counter][2]
            plot_func = getattr(plot,plot_to_make)
            
            #if the data is ready for this plot (for this cluster for this trial)...
            if self.metadata[self.current_trial][self.current_cluster]['dataready'][plot_to_make]:
                #if we're currently calculating data for this cluster/trial in the worker thread...
                if self.current_trial == self.trial_data['current_trial'] and self.current_cluster == self.trial_data['current_cluster']:
                    #the data we need is in the data dicts sent from the worker thread
                    trial_data = self.trial_data
                    cluster_data = self.cluster_data
                    spike_data = self.spike_data
                else:
                    #otherwise grab the relevant data dicts from the all_data dict
                    trial_data = self.all_data[self.current_trial][self.current_cluster]['trial_data']
                    cluster_data = self.all_data[self.current_trial][self.current_cluster]['cluster_data']
                    spike_data = self.all_data[self.current_trial][self.current_cluster]['spike_data']
                
                #if we're auto-advancing and not in speed mode...
                if not self.ops['speedmode'] and self.auto:
                    #grab the current time
                    self.end = time.time()
                    #if time since last plot was shown is less than 1.5 sec...
                    if self.end - self.tstart < 1.:
                        #start an event loop with a QTimer that waits until the remaining
                        #part of 1.5 seconds has elapsed, then kill the event loop
                        loop = QEventLoop()
                        QTimer.singleShot(1000.*(1.-(self.end-self.tstart)),loop.quit)
                        loop.exec_()
                
                #clear the figure
                self.figure.clear()
                #call the plotting function
                plot_func(self.ops,self.adv,trial_data,cluster_data,spike_data,self)
               
                #set the canvas label to current cluster
                if cluster_data['best_model'] is not None:
                    self.canvas_label.setText(self.current_cluster + ' -- ' + ', '.join(list(cluster_data['best_model'])))
                else:
                    self.canvas_label.setText(self.current_cluster)

            
                #draw the figure
                self.canvas.draw()
                
                #start the timer again
                self.tstart = time.time()
                
            #if the data isn't ready for this plot...
            else:
                #if we're in auto-advance mode...
                if self.auto:
                    #start an event loop until we get new data from the worker,
                    #then kill the loop and run the plotting func again
                    loop = QEventLoop()
                    self.worker.plotting_data.connect(loop.quit)
                    loop.exec_()
                    self.plot_following('same')
                else:
                    #if this is a user trying to get the next plot, tell them
                    #the data just ain't ready
                    print('data not ready yet! try again soon')
                    self.plot_counter -= 1

                    
    def init_plotting_data(self,metadata):
        ''' grab the metadata dict from the worker thread '''
        
        #assign metadata to self
        self.metadata = metadata
        #note how many plots we need to make
        self.tot_plots = len(metadata['plot_list'])

        #start the plot counter
        self.plot_counter = 0
        
    def update_plotting_data(self,ops,adv,trial_data,cluster_data,spike_data):
        ''' grab other data dicts from worker thread when they are ready '''
        
        #assign dicts to self
        self.ops = ops
        self.adv = adv
        self.trial_data = trial_data
        self.cluster_data = cluster_data
        self.spike_data = spike_data
        
        #if we haven't made a plot yet, make the first one
        if not self.first_plot:
            self.plot_first()
            
    def update_data_dict(self,all_data):
        ''' grab all_data dict from worker thread '''
        
        self.all_data = all_data
            
    def auto_plot(self):
        ''' start the auto-plotting cycle '''
        
        for i in range(self.plot_counter,len(self.metadata['plot_list'])):
            if self.auto:
                self.plot_following('next')
                
    def pause_resume(self):
        ''' pause or resume auto-plotting '''
        
        if self.auto:
            self.auto = False
            self.pausebutton.setText('Resume')
            self.nextbutton.setEnabled(True)
            self.backbutton.setEnabled(True)
        else:
            self.auto = True
            self.pausebutton.setText('Pause')
            self.nextbutton.setEnabled(False)
            self.backbutton.setEnabled(False)
            self.auto_plot()

#    def break_auto(self):
#        ''' break the auto-plotting cycle '''
#        
#        self.auto = False
#        self.pausebutton.setText('Resume')
        
    def stop_process(self):
        ''' try to stop all processes '''
        
        print('stopping all processes!')
        self.worker.isrunning = False

    def closeEvent(self,event):
        
        if self.launcher is not None:
            self.launcher.show_launcher()        
        
            
''''''''''''''''''''''''''''''''''''''''''''''''''''''
######################################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''

if __name__ == '__main__':

    #create a QApplication if one doesn't already exist
    app = QApplication.instance()
    if app == None:
        app = QApplication()
    
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
