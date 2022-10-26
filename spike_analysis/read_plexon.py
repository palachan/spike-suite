# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:52:05 2022

read plexon *offline sorter* output files

@author: plachanc
"""

import csv
import os
import shutil

def translate_plexon_cluster_file(trial,filename):
    
    fpath = trial + '/' + filename
    
    cluster_dict = {}
    
    with open(fpath,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            clust_num = int(row[0])
            
            if clust_num not in cluster_dict.keys():
                cluster_dict[clust_num] = []
            
            timestamp = 1000000. * float(row[1])
            
            cluster_dict[clust_num].append(int(timestamp))
            
            
    for clust_num in cluster_dict.keys():
        if clust_num > 0:
            if clust_num < 10:
                new_fpath = fpath[:len(fpath)-4] + '_SS_0%s.txt' % clust_num
            elif clust_num >= 10:
                new_fpath = fpath[:len(fpath)-4] + '_SS_%s.txt' % clust_num
            with open(new_fpath,'w',newline='') as f:
                writer = csv.writer(f)
                for timestamp in cluster_dict[clust_num]:
                    writer.writerow([timestamp])

    plexon_destination = trial + '/plexon'
    
    if not os.path.isdir(plexon_destination):
        os.makedirs(plexon_destination)
        
    shutil.move(fpath, plexon_destination + '/' + filename)
    