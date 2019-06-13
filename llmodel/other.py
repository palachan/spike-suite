# -*- coding: utf-8 -*-
"""
Created on Sat Jul 07 12:22:46 2018

model functions that might be useful

@author: Patrick
"""
import pickle
import numpy as np

from interpret import rayleigh_r

def load_data(fname):
    ''' load pickled numpy arrays '''

    try:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    except:
        print('couldn\'t open data file! try again!!')
    
    return data

def plot_contribs(overdir):
    betas = [1.5,20.,20.,150.,150.,150.]
    
    overdir = 'H:/Patrick'
    measure = 'corr_r'
    variables = [('allo',),('spatial',),('ahv',),('speed',),('theta',),('ego',)]
    
    allo = []
    ego = []
    colors = []
    
    same = 0.
    
    all_contribs = {}
    all_counts = {}
    total_count = 0.
    for var in variables:
        all_contribs[frozenset(var)] = []
        all_counts[frozenset(var)] = 0.
        
    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC']
    variables = [('allo',),('spatial',),('ahv',),('speed',),('theta',),('ego',)]
    full_model = frozenset(['allo','spatial','ahv','speed','theta','ego'])
    measures = ['corr_r','explained_var','llps']
    
    fig = plt.figure()
    
#    area_dict = {}
#    for a in areas:
#        area_dict[a] = {}
#        area_dict[a]['cell_count'] = 0.
#        for var in variables:
#            area_dict[a][var[0]] = {}
#            for measure in measures:
#                area_dict[a][var[0]][measure] = []
#            area_dict[a][var[0]]['cell_count'] = 0.

    for animal in os.listdir(overdir):
#        if animal != 'MM44' and animal != 'PL10':
        animaldir = overdir + '/' + animal
        print animaldir
        
        for area in os.listdir(animaldir):
            if area in areas:
                fdir = animaldir + '/' + area
                print fdir
    
                trials = collect_data.find_trials(fdir)
                
                if len(trials) == 0:
                    continue
#                        trials.append(fdir)
                        
                cell_labels = {}
                
                #for every session...
                for trial in trials:
                    
                    cell_labels[trial] = {}
            
            #        #collect names of the clusters we'll be analyzing
            #        trial_data = collect_data.read_files(fdir,trial,video=False)
            #        
            #        print trial_data.keys()
            #        
            #        cluster_names = trial_data['filenames']
                    
                    #collect names of the clusters we'll be analyzing
                    trial_data = collect_data.tracking_stuff(fdir,trial)
                    trial_data = collect_data.speed_stuff(trial_data)
                    trial_data = collect_data.calc_novelty(trial_data)
                    
                    cluster_names = trial_data['filenames']
            
                    #for each cluster...
                    for name in cluster_names:
                           
                        if name.startswith('TT'):
                            ttnum = int(name[2])
                            lfp_num = ttnum * 4
                            
                        elif name.startswith('ST'):
                            stnum = int(name[2])
                            lfp_num = stnum * 2
            #                
                        lfp_fname = trial + '/' + 'CSC%d.ncs' % lfp_num
            ##            lfp_fname = trial + '/' + '100_CH%d.continuous' % lfp_num
                        phases = phase_analysis.run_phase_analysis(lfp_fname,trial_data).astype(np.int)
            ##            phases = np.ones(len(allangles),dtype=np.int)
            #
                        cluster_data = {}
                        ts_file = trial + '/' + name + '.txt'
                        
                        cluster_data['spike_list'] = collect_data.ts_file_reader(ts_file)
                        spike_data, cluster_data = collect_data.create_spike_lists(trial_data,cluster_data)
                        
                        cdict = [[]]*10  
            
                        #report which cluster we're working on                      
                        print(name)
            #
                        #grab relevant tracking and spike data
                        allcenter_x = np.asarray(trial_data['center_x'])
                        allcenter_y = np.asarray(trial_data['center_y'])
                        allangles = np.asarray(trial_data['angles'])
                        all_speeds = np.asarray(trial_data['speeds'])
                        all_ahvs = np.asarray(trial_data['ahvs'])
                        all_novelties = np.asarray(trial_data['novelties'])
                        all_spikes = spike_data['ani_spikes']
            
                    #for each cluster...
            #        for name in cluster_names:
                        
                        fname = trial+'/best_model_%s.pickle' % name
                        print fname
                        model_dict = load_data(fname)
                        
                        cfname = trial+'/corrected_best_model_%s.pickle' % name
                        print cfname
                        corrected_model_dict = load_data(cfname)
                        
                        same += np.float(len(corrected_model_dict['best_model'] & model_dict['best_model'])) / np.float(len(corrected_model_dict['best_model'] | model_dict['best_model']))
                        print len(corrected_model_dict['best_model'] & model_dict['best_model'])
                                    
                        
                        data = llmodel.assign_data(allcenter_x, allcenter_y, allangles, all_speeds, all_ahvs, all_novelties, phases, all_spikes, [], [], cross_val=False)
                        
                        for i in ['opt_all','corrected']:
                            
                            if i == 'opt_all':
                                opt_all = True
                                best_model = corrected_model_dict['best_model']
                                
                            elif i == 'corrected':
                                opt_all = False
                                best_model = model_dict['best_model']
                            
                            print(tuple(best_model))
                            
                            cell_labels[trial][name] = tuple(best_model)
                            
                            #TODO:
#                            betas = [0,0,0,0,0,0]
                            
                            if opt_all:
                            
                                if 'uniform' not in best_model:
                                    best_params = optimize.newton_cg(data, best_model, betas)
                                else:
                                    best_params = {}
                                    
                                scale_factor = 1.
                                    
                            else:
                                
                                if 'uniform' not in best_model:
                                    best_params = optimize.newton_cg(data, full_model, betas)
                                    scale_factor = llmodel.calc_scale_factor(best_params,data,best_model)
                                else:
                                    best_params = {}
                                    scale_factor = 1.
                                
                                
                            
                            #run the model using our optimized parameters
                            model_dict = llmodel.run_model(scale_factor, data, data, best_params, best_model)
                            
                            contribs = {}
                            variables = [('allo',),('spatial',),('ahv',),('speed',),('theta',),('ego',)]
                            
                            if len(best_model) == 1 and 'uniform' not in best_model:
                                uniform_model_dict = llmodel.run_model(scale_factor, data, data, best_params, frozenset(('uniform',)))
                                contribs[best_model] = {}
                                contribs[best_model]['ll'] = model_dict['ll'] - uniform_model_dict['ll']
                                contribs[best_model]['llps'] = model_dict['llps'] - uniform_model_dict['llps']
                                contribs[best_model]['explained_var'] = model_dict['explained_var'] - uniform_model_dict['explained_var']
                                contribs[best_model]['corr_r'] = model_dict['corr_r'] - uniform_model_dict['corr_r']
                                contribs[best_model]['pseudo_r2'] = model_dict['pseudo_r2'] - uniform_model_dict['pseudo_r2']
                                
                                for var in variables:
                                    if frozenset(var) != best_model:
                                        new_model = frozenset(chain(list(best_model),list(var)))
                                        
                                        if opt_all:
                                            new_params = optimize.newton_cg(data, new_model, betas)
                                            scale_factor = 1.
                                        else:
                                            new_params = copy.deepcopy(best_params)
                                            for key in new_params:
                                                if key not in new_model:
                                                    new_params[key] = None
                                            scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                                            
                                        new_model_dict = llmodel.run_model(scale_factor, data, data, new_params, new_model)
                                        
                                        contribs[frozenset(var)] = {}
                                        contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                                        contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                                        contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                                        contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                                        contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
                                
                            elif len(best_model) > 1:
                                
                                for var in variables:
                                    
                                    if var[0] in best_model:
                                        
                                        new_model = []
                                        for i in best_model:
                                            if i != var[0]:
                                                new_model.append(i)
                                        new_model = frozenset(new_model)
                                        
                                        if opt_all:
                                            new_params = optimize.newton_cg(data, new_model, betas)
                                            scale_factor = 1.
                                            
                                        else:
                                            new_params = copy.deepcopy(best_params)
                                            for key in new_params:
                                                if key not in new_model:
                                                    new_params[key] = None
                                            scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                                            
                                        new_model_dict = llmodel.run_model(scale_factor, data, data, new_params, new_model)
                                        
                                        contribs[frozenset(var)] = {}
                                        contribs[frozenset(var)]['ll'] = model_dict['ll'] - new_model_dict['ll']
                                        contribs[frozenset(var)]['llps'] = model_dict['llps'] - new_model_dict['llps']
                                        contribs[frozenset(var)]['explained_var'] = model_dict['explained_var'] - new_model_dict['explained_var']
                                        contribs[frozenset(var)]['corr_r'] = model_dict['corr_r'] - new_model_dict['corr_r']
                                        contribs[frozenset(var)]['pseudo_r2'] = model_dict['pseudo_r2'] - new_model_dict['pseudo_r2']
                                        
                                    else:
                                        
                                        new_model = frozenset(chain(list(best_model),list(var)))
                                        
                                        if opt_all:
                                            new_params = optimize.newton_cg(data, new_model, betas)
                                            scale_factor = 1.
                                        else:
                                            new_params = copy.deepcopy(best_params)
                                            for key in new_params:
                                                if key not in new_model:
                                                    new_params[key] = None
                                            scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                                            
                                        new_model_dict = llmodel.run_model(scale_factor, data, data, new_params, new_model)
                                        
                                        contribs[frozenset(var)] = {}
                                        contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                                        contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                                        contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                                        contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                                        contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
                                    
                            model_dict['contribs'] = contribs
                            model_dict['best_model'] = best_model
                                      
                            if opt_all:
                                with open((trial+'/best_model_%s.pickle' % name),'wb') as f:
                                    pickle.dump(model_dict,f,protocol=2)
                            else:
                                with open((trial+'/corrected_best_model_%s.pickle' % name),'wb') as f:
                                    pickle.dump(model_dict,f,protocol=2)
             

                       
def run_thing():
    import collect_data
    allo_cell = 'C:/Users/Jeffrey_Taube/Desktop/adn hdc/PL5/ADN/trial_1/corrected_best_model_ST4_SS_01.pickle'
    ego_cell = 'H:/Patrick/PL16/V2L/2017-03-16_16-48-44 s1 box/corrected_best_model_TT2_SS_02.pickle'
    
    trial = '2017-03-16_16-48-44 s1 box'
    tracking_fdir = 'H:/Patrick/PL16/V2L/2017-03-16_16-48-44 s1 box'
    
    #calculate tracking data
    trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)
    
    allo_dict = load_data(allo_cell)
    ego_dict = load_data(ego_cell)
    
    allo_curve = allo_dict['allo_params']
    ego_curves = ego_dict['ego_params']
    spatial_curve = ego_dict['spatial_params']
    
    #rayleighs = np.zeros((8,8))
    #for i in range(8):
    #    for j in range(8):
    #        rayleighs[i][j],_ = rayleigh_r(np.arange(0,360,12),ego_curves[i][j])
    #        
    #top_spot = np.where(rayleighs == np.max(rayleighs))
    #ego_curve = ego_curves[top_spot].flatten()
    #
    #ego_x_bin = top_spot[0]
    #ego_y_bin = top_spot[1]
    
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    angles = np.asarray(trial_data['angles'])
    
    import numba as nb
    
    @nb.njit()
    def ego_loop(center_x,center_y,angles):
        
        ego_gr = 8
        hd_bins = 30
        ego_radius = 100
    
        #create arrays for x and y coords of spatial bins
        xcoords = np.zeros(ego_gr)
        ycoords = np.zeros(ego_gr)
        
        ego_bins = np.zeros((ego_gr,ego_gr,len(center_x)))
    
        #assign coordinates for x and y axes for each bin
        #(x counts up, y counts down)
        for x in range(ego_gr):
            xcoords[x] = (np.float(x+.5)/np.float(ego_gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
            ycoords[x] = np.float(np.max(center_y)) - (np.float(x+.5)/np.float(ego_gr))*np.float((np.max(center_y)-np.min(center_y)))
        
        #for each y position...
        for i in range(ego_gr):
    
            #fill an array with the current y coord
            cue_y = ycoords[i]
    
            #for each x position...
            for j in range(ego_gr):
                                                    
                #fill an array with the current x coord
                cue_x = xcoords[j]
    
                #calc array of egocentric angles of this bin from pos x axis centered 
                #on animal using arctan
                new_angles = np.rad2deg(np.arctan2((cue_y-center_y),(cue_x-center_x)))%360
                #calculate ecd angles by subtracting allocentric
                #angles from egocentric angles
                ecd_angles = (new_angles-angles)%360
                #assign to bin
                ecd_bins = ecd_angles/(360/hd_bins)
    
                ego_bins[i][j] = ecd_bins
                
        return ego_bins
    
    ego_bins = ego_loop(center_x,center_y,angles)
    
    #ego_x = np.min(center_x) + (np.max(center_x) - np.min(center_x)) * np.float(ego_x_bin + .5)/np.float(8)
    #ego_y = np.max(center_y) - (np.max(center_y) - np.min(center_y)) * np.float(ego_y_bin + .5)/np.float(8)
    #
    ##calc array of egocentric angles of this bin from pos x axis centered 
    ##on animal using arctan
    #new_angles = np.rad2deg(np.arctan2((ego_y-center_y),(ego_x-center_x)))%360
    ##calculate ecd angles by subtracting allocentric
    ##angles from egocentric angles
    #ecd_angles = (new_angles-angles)%360
    ##assign to bin
    #ecd_bins = ecd_angles/(360/30)
    
    bin_y = center_y -np.min(center_y)
    yconv = np.max(bin_y)/19.
    bin_y = bin_y.astype(np.float)/yconv
    bin_y = bin_y.astype(np.int)
    
    bin_x = center_x -np.min(center_x)
    xconv = np.max(bin_x)/19.
    bin_x = bin_x.astype(np.float)/xconv
    bin_x = bin_x.astype(np.int)
    
    allo_bins = np.zeros(len(center_x))
    spatial_bins = np.zeros([len(center_x),2])
    for i in range(len(center_x)):
        #calculate the head direction bin for this frame
        hd_bin = int(angles[i]/(360/30))
        spatial_bins[i,0] = bin_y[i]
        spatial_bins[i,1] = bin_x[i]
        #assign it the appropriate array
        allo_bins[i] = hd_bin
    
    ego_train = np.zeros(len(center_x))
    spatial_train = np.zeros(len(center_x))
    
    for i in range(len(center_x)):
    #    ego_train[i] = ego_curve[int(ecd_bins[i])]
        spatial_train[i] = spatial_curve[int(spatial_bins[i][0]),int(spatial_bins[i][1])]
        for j in range(8):
            for k in range(8):
                ego_train[i] += ego_curves[j][k][int(ego_bins[j][k][i])]
        
    egospatial_train = ego_train * spatial_train
    egospatial_train *= (5000./np.sum(egospatial_train))
    #ego_train *= (5000./np.sum(ego_train))
    #spatial_train *= (5000./np.sum(spatial_train))
    
    for i in range(20):
        allo_train = np.zeros(len(center_x))
        allo_curve = np.roll(allo_curve,1)
        for j in range(len(center_x)):
            allo_train[j] = allo_curve[int(allo_bins[j])]
        allo_train *= (5000./np.sum(allo_train))
    
        summed_train = allo_train + egospatial_train
            
        spikes = np.where(summed_train > .6)[0]
        
        spike_x = center_x[spikes]
        spike_y = center_y[spikes]
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(center_x,center_y,'k-',spike_x[::3],spike_y[::3],'ro')
        plt.show()
        

import collect_data

c=[]
c.append('H:/Patrick/PL16/V2L/2017-03-15_16-15-36/TT4_SS_01.txt')
c.append('H:/Patrick/PL16/V2L/2017-03-15_16-15-36/TT4_SS_02.txt')
c.append('H:/Patrick/PL16/V2L/2017-03-15_16-15-36/TT4_SS_03.txt')
c.append('H:/Patrick/PL16/V2L/2017-03-15_16-15-36/TT4_SS_04.txt')
#c.append()

fname='H:/Patrick/PL16/V2L/2017-03-16_16-48-44 s1 box/TT4_SS_01.txt'
tracking_fdir = 'H:/Patrick/PL16/V2L/2017-03-16_16-48-44 s1 box'

trial = '2017-03-15_16-15-36'
tracking_fdir = 'H:/Patrick/PL16/V2L/2017-03-15_16-15-36'

st_dict = {}

#calculate tracking data
trial_data = collect_data.tracking_stuff(tracking_fdir,tracking_fdir)

import csv
def ts_file_reader(ts_file):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'),dialect='excel-tab')

    for row in reader:
        spike_list.append(int(row[0]))
                
    #return it!
    return spike_list

import bisect
for fname in c:
    spike_list = ts_file_reader(fname)
    #creates array of zeros length of spike_timestamps to create spike train
    spike_train = np.zeros(len(trial_data['spike_timestamps']))
    
    #for each spike timestamp...
    for i in spike_list:
        #find closest video frame 
        ind = bisect.bisect_left(trial_data['timestamps'],i)
        #find closest entry in high precision 'spike timestamps' list
        spike_ind = bisect.bisect_left(trial_data['spike_timestamps'],i)
        
        if ind < len(trial_data['timestamps']):
            if spike_ind < len(spike_train):
                #add 1 to spike train at appropriate spot
                spike_train[spike_ind] = 1
                
    st_dict[fname] = spike_train

import matplotlib.pyplot as plt
fig_dict={}
for pair in ((c[0],c[1]),(c[0],c[2]),(c[1],c[2])):
    
    #figure out how many bins we want per window in our spike autocorrelation
    #default comes out to 500
    bins_per_window = 200
    
    #grab where the spikes are in our spike train
    spike_inds = np.where(st_dict[pair[0]] == 1)
    
    #start a vector for holding our autocorrelation values
    ac_matrix = np.zeros(2*bins_per_window + 1)
    
    #for each spike...
    for i in spike_inds[0]:
        #if we're at least one half window width from the beginning and end of the session...
        if (i-bins_per_window)>=0 and (i+bins_per_window)<len(spike_train):
            #for each bin in the window
            for j in range(-bins_per_window,bins_per_window+1):
                #add the corresponding value from the spike train
                ac_matrix[j+bins_per_window] += st_dict[pair[1]][i+j]
                
#    ac_matrix[bins_per_window] = 0
                
    #choose our x-ticks
    x_vals=np.arange(-.01,.01+float(.05)/1000.,float(.05)/1000.)
    
    fig_dict['fig'] = plt.figure()
    ax=fig_dict['fig'].add_subplot(111)
    ax.set_xlim([min(x_vals),max(x_vals)])
    ax.set_ylim([0,1.2*max(ac_matrix)])
    ax.set_ylabel('count')
    ax.set_xlabel('seconds')
    ax.set_title('%s by %s' % (pair[0][(len(pair[0])-8):],pair[1][(len(pair[0])-8):]))
    ax.vlines(x_vals,ymin=0,ymax=ac_matrix)
