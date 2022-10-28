# spike-suite

code for acquiring, sorting, and analyzing spike data with LED tracking

first make sure conda is installed. then install by running install.bat in this directory

run using run.bat

acquire: runs modified open-ephys gui along with red/green LED tracking -- requires an open-ephys acquisition system and overhead camera to be plugged in

sort: runs spike sorting using KiloSort on either neuralynx .NTT files or open-ephys .Spikes files, writes output to .NTT files for manual curation -- requires matlab and an NVIDIA GPU with CUDA

analyze: makes graphs and runs GLM classifier on clustered cells (either a .txt file for each cell containing spike times in microseconds, or a plexon offline sorter output .txt file containing cluster number, timestamp pairs). also requires either neuralynx .NVT file OR a tracking .txt file output from the 'Acquire' program

KiloSort in kilosort_control directory is modified from https://github.com/cortex-lab/KiloSort

plugin-GUI in oe_control directory is modified from https://github.com/open-ephys/plugin-GUI

only tested on 64-bit Windows 10

USE AT YOUR OWN RISK