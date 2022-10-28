# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:29:31 2018

@author: Patrick
"""
import sys
import os
import shutil
import numpy as np

import tkinter.filedialog as tkFileDialog

from kilosort_control import spike2bin, load_oe, load_nlx, kilo2ntt, write_config
import copy

#acq = 'openephys'
no_data = True
run_kilo = True
fs=30000.
zero_channels = [13,14,15]

def run_kilosort(fpath,bin_file,fs,n_trodes,trodetype,self):
        
    import subprocess
    p1 = subprocess.Popen('matlab -nodesktop -wait -r \"run_kilosort(\'%s\',\'%s\',%s,%s,\'%s\')\"' % (fpath,bin_file,fs,n_trodes,trodetype))
    p1.communicate()
    
    p1.wait()

def run(self,fpath,config_ops,acq):
    
#    acq = 'openephys'
    no_data = True
    run_kilo = True
    fs=30000.
    
    stitched = {}
    ttfiles = []
    session_length = []
    
    single_trode = False
    if os.path.isdir(fpath):
        dirname = fpath
    else:
        dirname = os.path.dirname(fpath)
        basename = os.path.basename(fpath)
        ttfiles = [basename]
        single_trode = True
        if basename.startswith('ST'):
            trodetype = 'stereotrode'
            trodenum = 2
        elif basename.startswith('TT'):
            trodetype = 'tetrode'
            trodenum = 4
    
    kilo_folder = dirname + '/kilofiles'
    if not os.path.exists(kilo_folder):
        os.makedirs(kilo_folder)
    elif run_kilo:
        shutil.rmtree(kilo_folder)
        os.makedirs(kilo_folder)
            
            
    if not single_trode:
        for fname in os.listdir(dirname):
            if acq == 'openephys' and fname.startswith('TT') and '-01.ntt' not in fname and fname.endswith('.spikes'):
                ttfiles.append(fname)
                trodetype='tetrode'
                trodenum = 4
            elif acq == 'neuralynx' and fname.startswith('TT') and '-01.ntt' not in fname and fname.endswith('.ntt'):
                ttfiles.append(fname)
                trodetype='tetrode'
                trodenum = 4
            elif acq == 'neuralynx' and fname.startswith('ST') and '-01.nst' not in fname and fname.endswith('.nst'):
                trodetype='stereotrode'
                trodenum = 2
                ttfiles.append(fname)
                
                
    if acq == 'neuralynx':
        event_file = dirname + '/events.nev'
        ttl_ts,ttl_ids,ttl_msgs = load_nlx.grab_nev_data(event_file)
        first_timestamp = ttl_ts[0]
        
    else:
        first_timestamp = 0
        
    ttfiles = sorted(ttfiles)
        
    if no_data:
        for fname in ttfiles:
            
            print('processing %s' % fname)
        
            filename = dirname + '/' + fname
            if acq == 'neuralynx':
                if trodetype == 'tetrode':
                    waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_tt_spike_file(filename)
                elif trodetype == 'stereotrode':
                    waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_st_spike_file(filename)
                    
                timestamps = timestamps - first_timestamp
                    
            elif acq == 'openephys':
                waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
                inverted = True
                
            if not inverted:
                waveforms = waveforms * -1.
                
            stitched[fname] = spike2bin.stitch_waveforms(waveforms,timestamps,fs,trodenum)
            
            session_length.append(len(stitched[fname][1]))
            
        max_session_length = max(session_length)
        
        ref_array = np.zeros((trodenum, max_session_length))
        
        for fname in ttfiles:
            if len(stitched[fname][1]) < max_session_length:
                new_array = copy.deepcopy(ref_array)
                new_array[:,:len(stitched[fname][1])] = stitched[fname]
                stitched[fname] = copy.deepcopy(new_array)

        counter = 0
        
        print('writing binary file')
        while counter < (max_session_length-10001):
                                
            all_stitched = stitched[ttfiles[0]][:,counter:(counter+10000)]
            if not single_trode:
                for fname in ttfiles[1:]:
                    all_stitched = np.concatenate((all_stitched,stitched[fname][:,counter:(counter+10000)]))
            if counter == 0:
                spike2bin.write_bin(all_stitched, dirname + '/data.bin','wb')
            else:
                spike2bin.write_bin(all_stitched, dirname + '/data.bin','ab')
            counter += 10000

            
        del stitched
        del all_stitched
        
    write_config.write_config(config_ops,kilo_folder + '/config.m')
    
    if run_kilo:
        print('starting kilosort')
#        self.kilo_done = False
        run_kilosort(dirname,dirname + '/data.bin',fs,len(ttfiles),trodetype,self)        
        print('done sorting')
        
    
    spike_times = np.load(kilo_folder+'/spike_times.npy', mmap_mode='r')
    spike_clusters = np.load(kilo_folder+'/spike_clusters.npy', mmap_mode='r')
    spike_templates = np.load(kilo_folder+'/spike_templates.npy', mmap_mode='r')
    templates = np.load(kilo_folder+'/templates.npy', mmap_mode='r')
    
    os.remove(dirname + '/data.bin')
    
    trode_clusters = {}
    for fname in ttfiles:
        trode_clusters[fname] = []
    
    channel_templates = np.max(templates,axis=1)
    for i in range(len(channel_templates)):
        try:
            top_channel = np.where(channel_templates[i] == np.max(channel_templates[i]))[0][0]
            trode_clusters[ttfiles[int(top_channel/trodenum)]].append(i)
        except:
            pass
        
    trode_data = {}
    for j in range(len(ttfiles)):
        inds = [i for i in range(len(spike_templates)) if spike_templates[i] in trode_clusters[ttfiles[j]]]
        trode_data[ttfiles[j]] = {}
        trode_data[ttfiles[j]]['spike_times'] = spike_times[inds]
        trode_data[ttfiles[j]]['spike_clusters'] = spike_clusters[inds]

    from scipy import stats
    

#    wave_dict = {}
    stitched = {}
    for fname in ttfiles:
        
        print('reloading %s' % fname)
    
        filename = dirname + '/' + fname
        if acq == 'neuralynx':
            if trodetype == 'tetrode':
                waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_tt_spike_file(filename)
            elif trodetype == 'stereotrode':
                waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_st_spike_file(filename)

        elif acq == 'openephys':
            waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
            
        stitched[fname] = spike2bin.stitch_waveforms(waveforms, timestamps - first_timestamp, fs, trodenum)
        
        if len(stitched[fname][1]) < max_session_length:
            new_array = copy.deepcopy(ref_array)
            new_array[:,:len(stitched[fname][1])] = stitched[fname]
            stitched[fname] = copy.deepcopy(new_array)
        
        wave_dict = {}
        ind_dict = {}
        for cluster in np.unique(trode_data[fname]['spike_clusters']):
            wave_dict[str(cluster)] = []
            ind_dict[str(cluster)] = []
        for i in range(len(trode_data[fname]['spike_times'])):
            wave_dict[str(trode_data[fname]['spike_clusters'][i][0])].append(np.swapaxes(stitched[fname][:,int(trode_data[fname]['spike_times'][i]-8):int(trode_data[fname]['spike_times'][i]+24)],0,1))
            ind_dict[str(trode_data[fname]['spike_clusters'][i][0])].append(i)
            
        for cluster in np.unique(trode_data[fname]['spike_clusters']):
            
            wave_dict[str(cluster)] = np.swapaxes(np.asarray(wave_dict[str(cluster)]),0,2).copy()
            ind_dict[str(cluster)] = np.asarray(ind_dict[str(cluster)])
            trode_data[fname]['spike_clusters'].setflags(write=1)
            
            for j in range(len(wave_dict[str(cluster)])):
#                print j
                for k in range(len(wave_dict[str(cluster)][j])):
                    zscores = np.abs(stats.zscore(wave_dict[str(cluster)][j][k]))
                    bad_inds = ind_dict[str(cluster)][np.where(zscores > 3)]
                    trode_data[fname]['spike_clusters'][bad_inds] = 0
                            
    
    for trode in range(len(ttfiles)):
        print('writing %d' % (trode+1))
        
        og_filename = dirname + '/' + ttfiles[trode]
        
        if trodetype == 'tetrode':
            kilo2ntt.write_ntt(stitched, trode_data[ttfiles[trode]]['spike_times'], trode_data[ttfiles[trode]]['spike_clusters'], kilo_folder, trode+1, fs, gain, acq, ttfiles[trode], og_filename, first_timestamp)
        elif trodetype == 'stereotrode':
            kilo2ntt.write_nst(stitched, trode_data[ttfiles[trode]]['spike_times'], trode_data[ttfiles[trode]]['spike_clusters'], kilo_folder, trode+1, fs, gain, acq, ttfiles[trode], og_filename, first_timestamp)
    
    print('Done!')
    
    
def batch_run(self,fpath,config_ops,acq):
    
    no_data = True
    run_kilo = True
    fs=30000.
    
    trials = os.listdir(fpath)
        
    session_boundaries=[0]
    
    for trial in trials:
        if not os.path.isdir(fpath+'/'+trial):
            continue
        if trial == 'kilofiles':
            continue
        
        stitched = {}
        
        dirname = fpath + '/' + trial

        ttfiles = []
        session_length = []
    
        kilo_folder = fpath + '/kilofiles'
        if not os.path.exists(kilo_folder):
            os.makedirs(kilo_folder)
        elif run_kilo:
            shutil.rmtree(kilo_folder)
            os.makedirs(kilo_folder)
#            
        for fname in os.listdir(dirname):
            if acq == 'openephys' and fname.startswith('TT') and fname.endswith('.spikes'):
                ttfiles.append(fname)
                trodetype='tetrode'
                trodenum = 4
            elif acq == 'neuralynx' and fname.startswith('TT') and fname.endswith('.ntt'):
                ttfiles.append(fname)
                trodetype='tetrode'
                trodenum = 4
            elif acq == 'neuralynx' and fname.startswith('ST') and fname.endswith('.nst'):
                trodetype='stereotrode'
                trodenum = 2
                ttfiles.append(fname)
                
        if acq == 'neuralynx':
            event_file = dirname + '/events.nev'
            ttl_ts,ttl_ids,ttl_msgs = load_nlx.grab_nev_data(event_file)
            first_timestamp = ttl_ts[0]
            
        else:
            first_timestamp == 0
                
        ttfiles = sorted(ttfiles)
            
        if no_data:
            for fname in ttfiles:
                
                print('processing %s' % fname)
            
                filename = dirname + '/' + fname
                if acq == 'neuralynx':
                    if trodetype == 'tetrode':
                        waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_tt_spike_file(filename)
                    elif trodetype == 'stereotrode':
                        waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_st_spike_file(filename)
                        
                    timestamps = timestamps - first_timestamp
                        
                elif acq == 'openephys':
                    waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
                    inverted = True
                    
                if not inverted:
                    waveforms = waveforms * -1.
                    
                stitched[fname] = spike2bin.stitch_waveforms(waveforms,timestamps,fs,trodenum)
                
                session_length.append(len(stitched[fname][1]))
                
        max_session_length = max(session_length)
        
        ref_array = np.zeros((trodenum, max_session_length))
        
        for fname in ttfiles:
            if len(stitched[fname][1]) < max_session_length:
                new_array = copy.deepcopy(ref_array)
                new_array[:,:len(stitched[fname][1])] = stitched[fname]
                stitched[fname] = copy.deepcopy(new_array)

        counter = 0
        
        print('writing binary file')
        while counter < (max_session_length-10001):
                                
            all_stitched = stitched[ttfiles[0]][:,counter:(counter+10000)]
            for fname in ttfiles[1:]:
                all_stitched = np.concatenate((all_stitched,stitched[fname][:,counter:(counter+10000)]))
            if len(session_boundaries)==1 and counter == 0:
                spike2bin.write_bin(all_stitched, fpath + '/data.bin','wb')
            else:
                spike2bin.write_bin(all_stitched, fpath + '/data.bin','ab')
            counter += 10000
    
        session_boundaries.append(session_boundaries[len(session_boundaries)-1]+counter)

        del stitched
        del all_stitched
        
        
    write_config.write_config(config_ops,kilo_folder + '/config.m')
    
    if run_kilo:
        print('starting kilosort')
#        self.kilo_done = False
        run_kilosort(fpath,fpath + '/data.bin',fs,len(ttfiles),trodetype,self)        
        print('done sorting')
        
    
    all_spike_times = np.load(kilo_folder+'/spike_times.npy', mmap_mode='r')
    all_spike_clusters = np.load(kilo_folder+'/spike_clusters.npy', mmap_mode='r')
    all_spike_templates = np.load(kilo_folder+'/spike_templates.npy', mmap_mode='r')
    templates = np.load(kilo_folder+'/templates.npy', mmap_mode='r')
    
    for bound in range(len(session_boundaries)-1):
        
        trial = trials[bound]
        dirname = fpath + '/' + trial
        
        ttfiles = []
        
        for fname in os.listdir(dirname):
            if acq == 'openephys' and fname.startswith('TT') and fname.endswith('.spikes'):
                ttfiles.append(fname)
            elif acq == 'neuralynx' and fname.startswith('TT') and fname.endswith('.ntt'):
                ttfiles.append(fname)
            elif acq == 'neuralynx' and fname.startswith('ST') and fname.endswith('.nst'):
                ttfiles.append(fname)
                
        if acq == 'neuralynx':
            event_file = dirname + '/events.nev'
            ttl_ts,ttl_ids,ttl_msgs = load_nlx.grab_nev_data(event_file)
            first_timestamp = ttl_ts[0]
            
        else:
            first_timestamp = 0
                
        ttfiles = sorted(ttfiles)
        
        good_inds = np.where((all_spike_times[:,0] >= session_boundaries[bound]) & (all_spike_times[:,0] < session_boundaries[bound+1]))[0]
        spike_times = all_spike_times[good_inds] - session_boundaries[bound]
        spike_clusters = all_spike_clusters[good_inds]
        spike_templates = all_spike_templates[good_inds]
        
        trode_clusters = {}
        for fname in ttfiles:
            trode_clusters[fname] = []
        
        channel_templates = np.max(templates,axis=1)
        for i in range(len(channel_templates)):
            try:
                top_channel = np.where(channel_templates[i] == np.max(channel_templates[i]))[0][0]
                trode_clusters[ttfiles[int(top_channel/trodenum)]].append(i)
            except:
                pass
            
        trode_data = {}
        for j in range(len(ttfiles)):
            inds = [i for i in range(len(spike_templates)) if spike_templates[i] in trode_clusters[ttfiles[j]]]
            trode_data[ttfiles[j]] = {}
            trode_data[ttfiles[j]]['spike_times'] = spike_times[inds]
            trode_data[ttfiles[j]]['spike_clusters'] = spike_clusters[inds]
    
        from scipy import stats
        
    
    #    wave_dict = {}
        stitched = {}
        for fname in ttfiles:
            
            print('reloading %s' % fname)
        
            filename = dirname + '/' + fname
            if acq == 'neuralynx':
                if trodetype == 'tetrode':
                    waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_tt_spike_file(filename)
                elif trodetype == 'stereotrode':
                    waveforms,timestamps,fs,gain,inverted = load_nlx.mmap_st_spike_file(filename)
            elif acq == 'openephys':
                waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
                
            stitched[fname] = spike2bin.stitch_waveforms(waveforms,timestamps - first_timestamp,fs,trodenum)
            wave_dict = {}
            ind_dict = {}
            for cluster in np.unique(trode_data[fname]['spike_clusters']):
                wave_dict[str(cluster)] = []
                ind_dict[str(cluster)] = []
            for i in range(len(trode_data[fname]['spike_times'])):
                wave_dict[str(trode_data[fname]['spike_clusters'][i][0])].append(np.swapaxes(stitched[fname][:,int(trode_data[fname]['spike_times'][i]-8):int(trode_data[fname]['spike_times'][i]+24)],0,1))
                ind_dict[str(trode_data[fname]['spike_clusters'][i][0])].append(i)
                
            for cluster in np.unique(trode_data[fname]['spike_clusters']):
                
                wave_dict[str(cluster)] = np.swapaxes(np.asarray(wave_dict[str(cluster)]),0,2).copy()
                ind_dict[str(cluster)] = np.asarray(ind_dict[str(cluster)])
                trode_data[fname]['spike_clusters'].setflags(write=1)
                
                for j in range(len(wave_dict[str(cluster)])):
    #                print j
                    for k in range(len(wave_dict[str(cluster)][j])):
                        zscores = np.abs(stats.zscore(wave_dict[str(cluster)][j][k]))
                        bad_inds = ind_dict[str(cluster)][np.where(zscores > 3)]
                        trode_data[fname]['spike_clusters'][bad_inds] = 0
                        
        kilo_folder = dirname + '/kilofiles'
        if not os.path.exists(kilo_folder):
            os.makedirs(kilo_folder)
        elif run_kilo:
            shutil.rmtree(kilo_folder)
            os.makedirs(kilo_folder)
                                
        for trode in range(len(ttfiles)):
            print('writing %d' % (trode+1))
            
            og_filename = dirname + '/' + ttfiles[trode]
            
            if trodetype == 'tetrode':
                kilo2ntt.write_ntt(stitched, trode_data[ttfiles[trode]]['spike_times'], trode_data[ttfiles[trode]]['spike_clusters'], kilo_folder, trode+1, fs, gain, acq, ttfiles[trode], og_filename, first_timestamp)
            elif trodetype == 'stereotrode':
                kilo2ntt.write_nst(stitched, trode_data[ttfiles[trode]]['spike_times'], trode_data[ttfiles[trode]]['spike_clusters'], kilo_folder, trode+1, fs, gain, acq, ttfiles[trode], og_filename, first_timestamp)
    
    print('Done!')
    
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
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
#if __name__ == '__main__':      
#    
#    
#    import pickle
#    with open('config_ops.pickle','rb') as f:
#        config_ops = pickle.load(f)
#    
##    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC','V1B','PoR','RSA']
###    areas = ['V2L']
##    
##    overdir = 'H:/Patrick'
###    overdir = 'C:/Users/Jeffrey_Taube/desktop/adn hdc'
##    
##    for animal in os.listdir(overdir):
##
##        animaldir = overdir + '/' + animal
##        print animaldir
##        
##        if animal not in ['mm42','mm43','mm45']:
##            continue
##
###        for area in os.listdir(animaldir):
###            if area in areas:
###                areadir = animaldir + '/' + area
###                print areadir
##                
##                #figure out what sessions (trials) are in the directory
#        
##    fdir = 'H:/Patrick/egocentric/PL62/MEC'
##    fdir = '//tsclient/C/Users/Jeffrey Taube/Desktop/Patrick/PL68/2019-4-22 try 2'
#
#    fdir = 'G:/Patrick cube/JLM17'
##    fdir = 'G:/Patrick cube/NB11 CHECK EXCLUDED/Ambiguous -- iso poor'
#    fdir = 'G:/Patrick cube/JRD54 CHECK EXCLUDED/Excluded Sessions 30hz'
##    fdir = 'D:/Patrick/PL73'
#    
##    for animal in os.listdir(fdir):
##        animaldir = fdir + '/' + animal
##        
##        if not os.path.isdir(animaldir):
##            continue
##        
##        for session in os.listdir(animaldir):
##            
##            sessiondir = animaldir + '/' + session
##            if not os.path.isdir(sessiondir):
##                continue
##            
###            ogdir = sessiondir + '/og'
#                        
#
#    
##    trials = find_trials(fdir)
#    
#    trials = []
#    for underdir in os.listdir(fdir):
#        trialdir = fdir + '/' + underdir
#        if not os.path.isdir(trialdir):
#            continue
##        print trialdir
#        count = 0
#        for fname in os.listdir(trialdir):
#            if fname.startswith('ST') & fname.endswith('.nst'): # & trialdir.endswith('no kilo'):# ~os.path.isdir(trialdir + '/kilofiles'):
#                count += 1
#                
#        if count == 8:
#            trials.append(trialdir)
#            
#            
#        count= 0
#        for fname in os.listdir(trialdir):
#            if fname.startswith('TT') & fname.endswith('.ntt'): # & trialdir.endswith('no kilo'):# ~os.path.isdir(trialdir + '/kilofiles'):
#                count += 1
#                
#        if count == 4:
#            trials.append(trialdir)
#        
#    #for every session...
#    for trial in trials:
#        
#        
#        
###                trialdir = sessiondir + '/' + trial
#        ogdir = trial + '/og'
#        
#        if not os.path.isdir(ogdir):
#            os.makedirs(ogdir)
#                
#        print trial
#                    
#        try:
#            run([],trial,config_ops,'neuralynx')
#        except:
#            continue
    

#    filename = tkFileDialog.askopenfilename()
#    filename = 'H:/PL37 grids/multi-level 10-13-17/2017-10-13_13-09-16 big box s1/TT1.NTT'
#    filename = 'D:/PL46/2018-04-05_13-42-59/TT8.spikes'
#    filename = 'C:/Users/Jeffrey_Taube/Desktop/Patrick/PL46/2018-04-19_11-56-39/TT8.spikes'
#    ttname = os.path.basename(filename)[:3]
#    dirname = os.path.dirname(filename)

#    dirname = 'C:/Users/Jeffrey_Taube/Desktop/Patrick/PL46/2018-04-25_17-22-54'
#
##    dirname = 'H:/PL37 grids/multi-level 10-13-17/2017-10-13_13-09-16 big box s1'
#    
#    stitched = {}
#    ttfiles = []
#    session_length = []
#    
#    if run_kilo:
#        if not os.path.exists(dirname + '/kilofiles'):
#            os.makedirs(dirname + '/kilofiles')
#        else:
#            shutil.rmtree(dirname + '/kilofiles')
#            os.makedirs(dirname + '/kilofiles')
#    for fname in os.listdir(dirname):
#        if acq == 'openephys' and fname.startswith('TT') and fname.endswith('.spikes'):
#            ttfiles.append(fname)
#        elif acq == 'neuralynx' and fname.startswith('TT') and fname.endswith('.ntt'):
#            ttfiles.append(fname)
#            
#    ttfiles = sorted(ttfiles)
#    
#    if no_data:
#        for fname in ttfiles:
#            
#            print('processing %s' % fname)
#        
#            filename = dirname + '/' + fname
#            if acq == 'neuralynx':
#                waveforms,timestamps,fs,gain = load_nlx.mmap_spike_file(filename)
#            elif acq == 'openephys':
#                waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
#                
#            stitched[fname] = spike2bin.stitch_waveforms(waveforms,timestamps,fs)
#            
#            session_length.append(len(stitched[fname][1]))
#            
#        min_session_length = min(session_length)
#        counter = 0
#        
#        print('writing binary file')
#        while counter < (min_session_length-10001):
#                                
#            all_stitched = stitched[ttfiles[0]][:,counter:(counter+10000)]
#            for fname in ttfiles[1:]:
#                all_stitched = np.concatenate((all_stitched,stitched[fname][:,counter:(counter+10000)]))
#            if counter == 0:
#                spike2bin.write_bin(all_stitched, dirname + '/data.bin','wb')
#            else:
#                spike2bin.write_bin(all_stitched, dirname + '/data.bin','ab')
#            counter += 10000
#            
#        del stitched
#        del all_stitched
#    
#    if run_kilo:
#        print('starting kilosort')
#        run_kilosort(dirname,dirname + '/data.bin',fs,len(ttfiles),'tetrode')
#        print('done sorting')
#    
#    kilo_folder = dirname + "/kilofiles"
#    spike_times = np.load(kilo_folder+'/spike_times.npy', mmap_mode='r')
#    spike_clusters = np.load(kilo_folder+'/spike_clusters.npy', mmap_mode='r')
#    spike_templates = np.load(kilo_folder+'/spike_templates.npy', mmap_mode='r')
#    templates = np.load(kilo_folder+'/templates.npy', mmap_mode='r')
#    
#    trode_clusters = {}
#    for fname in ttfiles:
#        trode_clusters[fname] = []
#    
#    channel_templates = np.max(templates,axis=1)
#    for i in range(len(channel_templates)):
#        try:
#            top_channel = np.where(channel_templates[i] == np.max(channel_templates[i]))[0][0]
#            trode_clusters[ttfiles[int(top_channel/4)]].append(i)
#        except:
#            pass
#        
#    trode_data = {}
#    for j in range(len(ttfiles)):
#        inds = [i for i in range(len(spike_templates)) if spike_templates[i] in trode_clusters[ttfiles[j]]]
#        trode_data[ttfiles[j]] = {}
#        trode_data[ttfiles[j]]['spike_times'] = spike_times[inds]
#        trode_data[ttfiles[j]]['spike_clusters'] = spike_clusters[inds]
#
#
#    stitched = {}
#    for fname in ttfiles:
#        
#        print('reloading %s' % fname)
#    
#        filename = dirname + '/' + fname
#        if acq == 'neuralynx':
#            waveforms,timestamps,fs,gain = load_nlx.mmap_spike_file(filename)
#        elif acq == 'openephys':
#            waveforms,timestamps,fs,gain = load_oe.load_spikefile(filename)
#            
#        stitched[fname] = spike2bin.stitch_waveforms(waveforms,timestamps,fs)
#    
#    for trode in range(len(ttfiles)):
#        print('writing %d' % (trode+1))
#        kilo2ntt.write_ntt(stitched, trode_data[ttfiles[trode]]['spike_times'], trode_data[ttfiles[trode]]['spike_clusters'], kilo_folder, trode+1, fs, gain, acq, ttfiles[trode])
#    