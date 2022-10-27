
savepath = strcat(fpath,'\kilofiles\');

% This part adds paths
%addpath(genpath('.\KiloSort')) % path to kilosort folder
%addpath(genpath('.\npy-matlab')) % path to npy-matlab scripts

% Run the configuration file
run(fullfile(savepath, 'config.m'))
ops.chanMap
% This part runs the normal Kilosort processing on the simulated data
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

%% AUTO MERGES 
%rez = merge_posthoc2(rez);

% save python results file for Phy
rezToPhy(rez, savepath);

%% save and clean up
% save matlab results file for future use
save(fullfile(savepath,  'rez.mat'), 'rez', '-v7.3');

% remove temporary file
delete(ops.fproc);
