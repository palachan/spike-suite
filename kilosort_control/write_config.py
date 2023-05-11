# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:43:54 2018

write config file for kilosort

@author: Patrick
"""

import pickle

def write_config(config_ops,fpath):
    
    config = 'clear ops\r\n'
    config += 'ops.GPU                 = %s; %% whether to run this code on an Nvidia GPU (much faster, mexGPUall first)\r\n' % config_ops['GPU']
    config += 'ops.parfor              = %s; %% whether to use parfor to accelerate some parts of the algorithm\r\n' % config_ops['parfor']
    config += 'ops.verbose             = %s; %% whether to print command line progress\r\n' % config_ops['verbose']
    config += 'ops.showfigures         = %s; %% whether to plot figures during optimization\r\n' % config_ops['showfigures']
    
    config += '\r\n'
    
    config += 'ops.datatype            = \'%s\';  %% binary (\'dat\', \'bin\') or \'openEphys\'\r\n' % config_ops['datatype']
    if config_ops['datatype'] == 'dat':
        config += 'ops.fbinary             = \'%s\'; %% will be created for \'openEphys\'\r\n' % config_ops['bin_file']	
    else:
        config += 'ops.fbinary             = fullfile(fpath, \'data.bin\'); %% will be created for \'openEphys\'\r\n'	
    config += 'ops.fproc               = fullfile(savepath, \'temp_wh.dat\'); %% residual from RAM of preprocessed data\r\n'
    config += 'ops.root                = fpath; %% \'openEphys\' only: where raw files are\r\n'
    config += '%% define the channel map as a filename (string) or simply an array\r\n'
    config += 'ops.chanMap             = fullfile(savepath, \'chanMap.mat\'); %% make this file using createChannelMapFile.m	\r\n'
    
    config += '\r\n'
    
    config += 'ops.Nfilt               = %s;  %% number of clusters to use (2-4 times more than Nchan, should be a multiple of 32)\r\n' % config_ops['Nfilt'] 		
    config += 'ops.nNeighPC            = 4; %% visualization only (Phy): number of channnels to mask the PCs, leave empty to skip (12)\r\n'
    config += 'ops.nNeigh              = 4; %% visualization only (Phy): number of neighboring templates to retain projections of (16)\r\n'
    		
    config += '\r\n'
    
    config += '% options for channel whitening	\r\n'	
    config += 'ops.whitening           = \'%s\'; %% type of whitening (default \'full\', for \'noSpikes\' set options for spike detection below)\r\n' % config_ops['whitening']
    config += 'ops.nSkipCov            = %s; %% compute whitening matrix from every N-th batch (1)\r\n' % config_ops['nSkipCov']
    config += 'ops.whiteningRange      = %s; %% how many channels to whiten together (Inf for whole probe whitening, should be fine if Nchan<=32)\r\n' % config_ops['whiteningRange']

    config += '\r\n'
    		
    config += 'ops.criterionNoiseChannels = %s; %% fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info).\r\n' % config_ops['criterionNoiseChannels']
    
    config += '\r\n'
    
    config += '% other options for controlling the model and optimization\r\n'
    config += 'ops.Nrank               = %s;    %% matrix rank of spike template model (3)\r\n' % config_ops['Nrank']
    config += 'ops.nfullpasses         = %s;    %% number of complete passes through data during optimization (6)\r\n' % config_ops['nfullpasses']
    config += 'ops.maxFR               = %s;  %% maximum number of spikes to extract per batch (20000)\r\n' % config_ops['maxFR']	
    config += 'ops.fshigh              = %s;   %% frequency for high pass filtering\r\n' % config_ops['fshigh']
    config += 'ops.ntbuff              = %s;    %% samples of symmetrical buffer for whitening and spike detection	\r\n' % config_ops['ntbuff']
    config += 'ops.scaleproc           = %s;   %% int16 scaling of whitened data\r\n' % config_ops['scaleproc']
    config += 'ops.NT                  = 128*1024+ ops.ntbuff;%% this is the batch size (try decreasing if out of memory)\r\n'
    config += '%% for GPU should be multiple of 32 + ntbuff\r\n'
    		
    config += '% the following options can improve/deteriorate results.\r\n'		
    config += '% when multiple values are provided for an option, the first two are beginning and ending anneal values,\r\n'
    config += '% the third is the value used in the final pass.\r\n'
    config += 'ops.Th               = [%s %s %s];    %% threshold for detecting spikes on template-filtered data ([6 12 12])\r\n' % (config_ops['Th'][0],config_ops['Th'][1],config_ops['Th'][2])
    config += 'ops.lam              = [%s %s %s];   %% large means amplitudes are forced around the mean ([10 30 30])\r\n' % (config_ops['lam'][0],config_ops['lam'][1],config_ops['lam'][2])
    config += 'ops.nannealpasses    = %s;            %% should be less than nfullpasses (4)\r\n' % config_ops['nannealpasses']
    config += 'ops.momentum         = 1./[20 400];  %% start with high momentum and anneal (1./[20 1000])\r\n'
    config += 'ops.shuffle_clusters = %s;            %% allow merges and splits during optimization (1)\r\n' % config_ops['shuffle_clusters']
    config += 'ops.mergeT           = %s;           %% upper threshold for merging (.1)\r\n' % config_ops['mergeT']
    config += 'ops.splitT           = %s;           %% lower threshold for splitting (.1)\r\n' % config_ops['splitT']

    config += '\r\n'
    		
    config += '% options for initializing spikes from data\r\n'
    config += 'ops.initialize      = \'%s\';    %%\'fromData\' or \'no\'\r\n' % config_ops['initialize']
    config += 'ops.spkTh           = %s;      %% spike threshold in standard deviations (4)\r\n' % config_ops['spkTh']
    config += 'ops.loc_range       = [%s  %s];  %% ranges to detect peaks; plus/minus in time and channel ([3 1])\r\n' % (config_ops['loc_range'][0],config_ops['loc_range'][1])
    config += 'ops.long_range      = [%s  %s]; %% ranges to detect isolated peaks ([30 6])\r\n' % (config_ops['long_range'][0],config_ops['long_range'][1])
    config += 'ops.maskMaxChannels = %s;       %% how many channels to mask up/down ([5])\r\n' % config_ops['maskMaxChannels']
    config += 'ops.crit            = %s;     %% upper criterion for discarding spike repeates (0.65)\r\n' % config_ops['crit']
    config += 'ops.nFiltMax        = %s;   %% maximum "unique" spikes to consider (10000)\r\n' % config_ops['nFiltMax']
    		
    config += '\r\n'
    
    config += '%% load predefined principal components (visualization only (Phy): used for features)\r\n'
    config += 'dd                  = load(\'PCspikes2.mat\'); %% you might want to recompute this from your own data\r\n'
    config += 'ops.wPCA            = dd.Wi(:,1:7);   %% PCs\r\n'
    		
    config += '\r\n'
    
    config += '%% options for posthoc merges (under construction)\r\n'
    config += 'ops.fracse  = 0.1; %% binning step along discriminant axis for posthoc merges (in units of sd)\r\n'
    config += 'ops.epu     = Inf;\r\n'
    		
    config += '\r\n'
    
    config += 'ops.ForceMaxRAMforDat   = 20e9; %% maximum RAM the algorithm will try to use; on Windows it will autodetect.\r\n'

    config = config.encode()
    f = open(fpath,'wb')
    f.write(config)
    f.flush()
    f.close()

def create_defaults():
    
    ops = {}
    ops['GPU'] = str(1)
    ops['parfor'] = str(1)
    ops['verbose'] = str(1)
    ops['showfigures'] = str(1)
    ops['datatype'] = str('bin')
    ops['Nfilt'] = str(96)
    ops['whitening'] = str('full')
    ops['nSkipCov'] = str(1)
    ops['whiteningRange'] = str('Inf')
    ops['criterionNoiseChannels'] = str(0.2)
    ops['Nrank'] = str(3)
    ops['nfullpasses'] = str(6)
    ops['maxFR'] = str(20000)
    ops['fshigh'] = str(200)
    ops['ntbuff'] = str(64)
    ops['scaleproc'] = str(200)
    ops['Th'] = [str(15),str(20),str(20)]
    ops['lam'] = [str(3),str(5),str(7)]
    ops['nannealpasses'] = str(4)
    ops['shuffle_clusters'] = str(1)
    ops['mergeT'] = str(.1)
    ops['splitT'] = str(.1)
    ops['initialize'] = str('fromData')
    ops['spkTh'] = str(-4)
    ops['loc_range'] = [str(3),str(1)]
    ops['long_range'] = [str(30),str(6)]
    ops['maskMaxChannels'] = str(5)
    ops['crit'] = str(.65)
    ops['nFiltMax'] = str(10000)
    
    
    with open('default_config_ops.pickle','wb') as f:
        pickle.dump(ops,f,protocol=2)
        f.close()
                  
        
if __name__ == '__main__':
    
    create_defaults()