# -*- coding: utf-8 -*-
"""
Created on Sat Jul 07 12:19:47 2018

interpret results of cell classification/modeling

@author: Patrick
"""

import pickle
import os
import numpy as np
from scipy.stats import kruskal,ranksums
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['image.cmap'] = 'jet'
import numba as nb
import seaborn as sns
import pandas as pd
import copy
from itertools import chain
import shutil

import llmodel



def load_data(fname):
    ''' load pickled numpy arrays '''

    try:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    except:
        print('couldn\'t open data file! try again!!')
    
    return data

def calc_contribs(best_model,model_dict,data,best_params,single_ego=False):
    ''' calculate the effect on different measures of goodness-of-fit when adding
    or subtracting each variable '''
    
    #start a dict for contribs
    contribs = {}
    #note which variables we're looking at
    variables = [('allo',),('spatial',),('speed',),('ahv',),('theta',),('ego',)]
    
    #if the best model is a single-variable model... and not null...
    if len(best_model) == 1 and 'uniform' not in best_model:
        #calculate goodness-of-fit measures for the null model
        uniform_model_dict = llmodel.run_model(1., data, data, best_params, frozenset(('uniform',)),single_ego)
        #calculate difference in goodness-of-fit measures between the best model and the null
        #model -- these are the contributions for the single encoded variable
        contribs[best_model] = {}
        contribs[best_model]['ll'] = model_dict['ll'] - uniform_model_dict['ll']
        contribs[best_model]['llps'] = model_dict['llps'] - uniform_model_dict['llps']
        contribs[best_model]['explained_var'] = model_dict['explained_var'] - uniform_model_dict['explained_var']
        contribs[best_model]['corr_r'] = model_dict['corr_r'] - uniform_model_dict['corr_r']
        contribs[best_model]['pseudo_r2'] = model_dict['pseudo_r2'] - uniform_model_dict['pseudo_r2']
        
        #for each variable...
        for var in variables:
            #not including the one we just looked at...
            if frozenset(var) != best_model:
                #create a new model which contains the single encoded variable as well as
                #this new variable
                new_model = frozenset(chain(list(best_model),list(var)))
                #make the appropriate parameter dict
                new_params = copy.deepcopy(best_params)
                for key in new_params:
                    if key not in new_model:
                        new_params[key] = None
                #calc a new scale factor
                new_scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                #run the new model and collect the result
                new_model_dict = llmodel.run_model(new_scale_factor, data, data, new_params, new_model,single_ego)
                
                #calculate difference in goodness-of-fit measures between the new model and
                #the best model -- these are the contributions from the added variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
        
    #otherwise, if there are multiple variables in the best model...
    elif len(best_model) > 1:
        #for each variable in the whole list...
        for var in variables:
            #if this variable is included in the best model...
            if var[0] in best_model:
                #create a new model that includes all the variables in the best
                #model EXCEPT this one
                new_model = []
                for i in best_model:
                    if i != var[0]:
                        new_model.append(i)
                new_model = frozenset(new_model)
                #make an appropriate parameter dict
                new_params = copy.deepcopy(best_params)
                for key in new_params:
                    if key not in new_model:
                        new_params[key] = None
                #calculate the new scale factor
                new_scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                #run the new model and collect the result
                new_model_dict = llmodel.run_model(new_scale_factor, data, data, new_params, new_model,single_ego)
                #calculate difference in goodness-of-fit measures between the best model and
                #the new model -- these are the contributions from the subtracted variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = model_dict['ll'] - new_model_dict['ll']
                contribs[frozenset(var)]['llps'] = model_dict['llps'] - new_model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = model_dict['explained_var'] - new_model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = model_dict['corr_r'] - new_model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = model_dict['pseudo_r2'] - new_model_dict['pseudo_r2']
                
            #otherwise...
            else:
                #make a new model that adds the current variable to the best model
                new_model = frozenset(chain(list(best_model),list(var)))
                #make an appropriate parameter dict
                new_params = copy.deepcopy(best_params)
                for key in new_params:
                    if key not in new_model:
                        new_params[key] = None
                #calc the new scale factor
                new_scale_factor = llmodel.calc_scale_factor(new_params,data,new_model)
                #run the new model and collect the result
                new_model_dict = llmodel.run_model(new_scale_factor, data, data, new_params, new_model,single_ego)
                #calculate difference in goodness-of-fit measures between the new model and
                #the best model -- these are the contributions from the added variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
    
    #add the new stuff to the model dict
    model_dict['contribs'] = contribs
    model_dict['best_model'] = best_model
    
    #return the model dict
    return model_dict

def area_contribs(fdir):
    
    font_size = 1.4
#    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC']
    areas = ['PoS']
    variables = [('allo',),('spatial',),('ahv',),('speed',),('theta',),('ego',)]
    measures = ['corr_r','explained_var','llps','pseudo_r2']
    
    plt.style.use('default')
    plt.rcParams['image.cmap'] = 'jet'
    fig = plt.figure()
    
    ego_paired = {}
    area_dict = {}
    egoallo_dict = {}
    for a in areas:
        ego_paired[a] = {}
        egoallo_dict[a] = {}
        area_dict[a] = {}
        area_dict[a]['cell_count'] = 0.
        for var in variables:
            if var[0] != 'ego':
                ego_paired[a][var[0]] = 0.
            area_dict[a][var[0]] = {}
            for measure in measures:
                area_dict[a][var[0]][measure] = []
                egoallo_dict[a][measure] = {'ego':[],'allo':[],'colors':[]}
            area_dict[a][var[0]]['cell_count'] = 0.

    for animal in os.listdir(fdir):
        animaldir = fdir + '/' + animal
        print animaldir
        
        for area in os.listdir(animaldir):
            if area in areas:
                areadir = animaldir + '/' + area
                print areadir
                
                for trial in os.listdir(areadir):
                    trialdir = areadir + '/' + trial
                    
                    clusters = []
                    
                    for f in os.listdir(trialdir):
                        if f.startswith('TT') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                        elif f.startswith('ST') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                            
                    for cluster in clusters:
                        
                        fname = trialdir+'/corrected_best_model_%s.pickle' % cluster
                        print fname
                        model_dict = load_data(fname)
                        print model_dict['best_model']
                        
                        cluster_img_dir = trialdir + '/corrected_%s' % cluster
                        if not os.path.isdir(cluster_img_dir):
                            os.makedirs(cluster_img_dir)
                        else:
                            shutil.rmtree(cluster_img_dir)
                            os.makedirs(cluster_img_dir)
                            
                        print cluster_img_dir
                            
                        for var in variables:
                            if var[0] in model_dict['best_model']:
#                                print var[0]
                                if var[0] == 'ego':

                                    
                                    ego_params = model_dict['ego_params'] #.reshape((8,8,30))
                                    rayleighs = np.zeros((8,8))
                                    mean_angles = np.zeros((8,8))
                                    for i in range(8):
                                        for j in range(8):
                                            rayleighs[i][j],mean_angles[i][j]=rayleigh_r(np.arange(0,360,12),ego_params[i][j])                                                
#                                            rayleighs[i][j] = np.sum(ego_params[i][j])
                                            
                                    
                                            
                                    fig.clf()
                                    ax = fig.add_subplot(111)
                                    divider = make_axes_locatable(ax)
                                    cax = divider.append_axes('right', size='5%', pad=0.05)
                                    im = ax.imshow(rayleighs)
                                    fig.colorbar(im, cax=cax, orientation='vertical')
                                    fig.savefig('%s/ego_rayleighs' % cluster_img_dir,dpi=900)
                                    
                                    fig.clf()
                                    ax = fig.add_subplot(111)
                                    x,y=np.meshgrid(np.arange(8),np.arange(8))
                                    
                                    allocentrized_curves = allocentrize_ecd(ego_params)
                                    allo_rxs = np.zeros((8,8))
                                    allo_rys = np.zeros((8,8))
                                    
                                    for i in range(8):
                                        for j in range(8):
                                            r,allo_rxs[i][j],allo_rys[i][j], mean_angle = rayleigh_r(np.arange(0,360,12),allocentrized_curves[i][j],ego=True)
                                    
                                    plt.axis('equal')
                                    ax.quiver(x, y[::-1], allo_rxs, allo_rys)
                                    
#                                    fig2,axes = plt.subplots(8,8)
#                                    
#                                    max_param = np.max(allocentrized_curves)
#                                    for i in range(8):
#                                        for j in range(8):
#                                            axes[i,j].plot(allocentrized_curves[i][j])
#                                            axes[i,j].set_yticks([])
#                                            axes[i,j].set_xticks([])
#                                            axes[i,j].set_ylim([0,max_param])
#                                    plt.show()
                                    
                                    fig.savefig('%s/allocentrized' % cluster_img_dir,dpi=900)
                                    
                                    fig.clf()
                                    ax = fig.add_subplot(111)
                                    colormap = plt.get_cmap('hsv')
                                    norm = mplcolors.Normalize(vmin=0, vmax=360)
                                    divider = make_axes_locatable(ax)
                                    cax = divider.append_axes('right', size='5%', pad=0.05)
                                    im = ax.imshow(mean_angles,cmap=colormap,norm=norm)
                                    fig.colorbar(im, cax=cax, orientation='vertical')
                                    fig.savefig('%s/ego_angles' % cluster_img_dir,dpi=900)
                                    
                                    fig2,axes = plt.subplots(8,8)
                                    
                                    max_param = np.max(ego_params)
                                    for i in range(8):
                                        for j in range(8):
                                            axes[i,j].plot(ego_params[i][j])
                                            axes[i,j].set_yticks([])
                                            axes[i,j].set_xticks([])
                                            axes[i,j].set_ylim([0,max_param])
                                    fig2.savefig('%s/ego_plots' % cluster_img_dir,dpi=900)
                                    plt.close(fig2)
#                                
                                elif var[0] == 'spatial':
                                    fig.clf()
                                    ax = fig.add_subplot(111)
                                    ax.imshow(model_dict['spatial_params'][::-1])
                                    fig.savefig('%s/spatial' % cluster_img_dir)
#                                    
                                else:
                                    fig.clf()
                                    ax = fig.add_subplot(111)
                                    ax.plot(model_dict['%s_params' % var])
                                    ax.set_ylim([0,None])
                                    fig.savefig('%s/%s' % (cluster_img_dir,var[0]))
                                            
                        contrib_dict = model_dict['contribs']
                        
                        area_dict[area]['cell_count'] += 1.
                        
                        if len(contrib_dict.keys()) > 0:
                        
                            for var in variables:
                                
                                if var[0] in model_dict['best_model']:
                                    
                                    if 'ego' in model_dict['best_model'] and var[0] != 'ego':
                                        ego_paired[area][var[0]] += 1.
                                                    
                                    area_dict[area][var[0]]['cell_count'] += 1.
                                    
                                    for measure in measures:
                                        area_dict[area][var[0]][measure].append(contrib_dict[frozenset(var)][measure])
                                        
                            if 'allo' in model_dict['best_model'] or 'ego' in model_dict['best_model']:
                                
                                for measure in measures:
                                    
                                    egoallo_dict[area][measure]['allo'].append(contrib_dict[frozenset(('allo',))][measure])
                                    egoallo_dict[area][measure]['ego'].append(contrib_dict[frozenset(('ego',))][measure])
                                    print(measure)
                                    print(contrib_dict[frozenset(('ego',))][measure])
                                    
                                    if 'ego' not in model_dict['best_model']:
                                        egoallo_dict[area][measure]['colors'].append('r')
                                    elif 'allo' not in model_dict['best_model']:
                                        egoallo_dict[area][measure]['colors'].append('b')
                                    else:
                                        egoallo_dict[area][measure]['colors'].append('k')
#
#    base_dir = 'C:/Users/Jeffrey_Taube/Desktop/Patrick/egocentric figures/contribs/corrected'
#    pretty_vars = ['ahv','speed','ego','allo','theta','spatial']
#    order = ['ego','allo','spatial','theta','speed','ahv']
#    label_vars = ['Egocentric Bearing','Allocentric HD','Location','Theta','Speed','AHV']
#
#    for area in areas:
#                
#        image_dir = base_dir + '/' + area
#        if not os.path.isdir(image_dir):
#            os.makedirs(image_dir)
#            
#
#
#        for measure in measures:
#            
#            fig.clf()
#            ax = fig.add_subplot(111)
#            ax.scatter(egoallo_dict[area][measure]['ego'],egoallo_dict[area][measure]['allo'],c=egoallo_dict[area][measure]['colors'])
#            fig.savefig('%s/egoallo_%s.png' % (image_dir,measure),dpi=1200)
#            
#            mean_dict = {'value':[],'Variable':[]}
#   
#            for var in pretty_vars:
#                if len(area_dict[area][var][measure]) == 0:
#                    mean_dict['value'].append(np.nan)
#                    mean_dict['Variable'].append(var)
#                for i in range(len(area_dict[area][var][measure])):
#                    if measure == 'explained_var' or measure == 'pseudo_r2':
#                        mean_dict['value'].append(area_dict[area][var][measure][i]*100.)
#                    else:
#                        mean_dict['value'].append(area_dict[area][var][measure][i])
#                    mean_dict['Variable'].append(var)
#
#            fig.clf()
#            sns.set(font_scale=font_size)
#            mean_data = pd.DataFrame(mean_dict)
#            ax = sns.barplot(x='Variable',y='value',data=mean_data,ci=68,order=order)
#            
#            if measure == 'llps':
#                ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#                ax.set_title('Log Likelihood Increase - %s' % area)
#            elif measure == 'corr_r':
#                ax.set_ylabel('Correlation')
#                ax.set_title('Correlation Contribution - %s' % area)
#            elif measure == 'explained_var':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Explained Variance - %s' % area)
#            elif measure == 'pseudo_r2':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Poisson Explained Variance - %s' % area)
#            ax.set_xticklabels(label_vars)
#            ax.xaxis.label.set_visible(False)
#            sns_fig = ax.get_figure()
#            sns_fig.autofmt_xdate()
#            plt.tight_layout()
#            sns_fig.savefig('%s/%s.png' % (image_dir,measure),dpi=1200)
#                    
#            fig.clf()
#                
#            sns.set(font_scale=font_size)
#            ax = sns.boxplot(x='Variable',y='value',data=mean_data,order=order)
#
##            ax.set_xticklabels(pretty_vars)
#            if measure == 'llps':
#                ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#                ax.set_title('Log Likelihood Increase - %s' % area)
#            elif measure == 'corr_r':
#                ax.set_ylabel('Correlation')
#                ax.set_title('Correlation Contribution - %s' % area)
#            elif measure == 'explained_var':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Explained Variance - %s' % area)
#            elif measure == 'pseudo_r2':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Poisson Explained Variance - %s' % area)
#            ax.set_xticklabels(label_vars)
#            ax.xaxis.label.set_visible(False)
#            sns_fig = ax.get_figure()
#            sns_fig.autofmt_xdate()
#            plt.tight_layout()
#            sns_fig.savefig('%s/%s_boxplots.png' % (image_dir,measure),dpi=1200)
#                
#            fig.clf()
#            sns.set(font_scale=font_size)
#            ax = sns.stripplot(x='Variable',y='value',data=mean_data,jitter=True,order=order,zorder=1)
#            
#            median_width = 0.3
#
#            for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
#                var = text.get_text()  # "X" or "Y"
#        
#                # calculate the median value for all replicates of either X or Y
#                median_val = mean_data[mean_data['Variable']==var].value.median()
#        
#                # plot horizontal lines across the column, centered on the tick
#                ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
#                        lw=4, color='k',zorder=100)
#            
#            if measure == 'llps':
#                ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#                ax.set_title('Log Likelihood Increase - %s' % area)
#            elif measure == 'corr_r':
#                ax.set_ylabel('Correlation')
#                ax.set_title('Correlation Contribution - %s' % area)
#            elif measure == 'explained_var':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Explained Variance - %s' % area)
#            elif measure == 'pseudo_r2':
#                ax.set_ylabel('Percent Variance')
#                ax.set_title('Poisson Explained Variance - %s' % area)
#            ax.set_xticklabels(label_vars)
##            if measure == 'pseudo_r2':
##                ax.set_ylim([-10,45])
#            ax.xaxis.label.set_visible(False)
#
#            sns_fig = ax.get_figure()
#            sns_fig.autofmt_xdate()
#            
#            plt.tight_layout()
#            sns_fig.savefig('%s/%s_strip_plots.png' % (image_dir,measure),dpi=1200)
#            
#        fig.clf()        
#        percents = {'Percent':[],'Variable':[]}
#        for var in pretty_vars:
#            percents['Percent'].append(area_dict[area][var]['cell_count'] * 100./area_dict[area]['cell_count'])
#            percents['Variable'].append(var)
#            
#        percent_frame = pd.DataFrame(percents)
#        sns.set(font_scale=font_size)
#        ax = sns.barplot(x='Variable',y='Percent',data=percent_frame,order=order)
#        ax.set_title('Cell Classifications - %s (n=%d)' % (area, int(area_dict[area]['cell_count'])))
#        ax.set_ylim([0,100])
#        ax.set_xticklabels(label_vars)
#        ax.xaxis.label.set_visible(False)
#        sns_fig = ax.get_figure()
#        sns_fig.autofmt_xdate()
#        plt.tight_layout()
#        sns_fig.savefig('%s/percents.png' % image_dir,dpi=1200)
#        
#        fig.clf()
#        ego_pairs = {'Percent':[],'Variable':[]}
#        for var in pretty_vars:
#            if var != 'ego':
#                ego_pairs['Percent'].append(ego_paired[area][var] * 100./area_dict[area]['ego']['cell_count'])
#                ego_pairs['Variable'].append(var)
#            
#        pair_frame = pd.DataFrame(ego_pairs)
#        sns.set(font_scale=font_size)
#        ax = sns.barplot(x='Variable',y='Percent',data=pair_frame,order=order)
#        ax.set_title('Percent of Egocentric Cells - %s (n=%d)' % (area,int(area_dict[area]['ego']['cell_count'])))
#        ax.set_ylim([0,100])
#        ax.set_xticklabels(label_vars)
#        ax.xaxis.label.set_visible(False)
#        sns_fig = ax.get_figure()
#        sns_fig.autofmt_xdate()
#        plt.tight_layout()
#        sns_fig.savefig('%s/paired_percents.png' % image_dir,dpi=1200)
##
##        
#    for measure in measures:
#        mean_dict = {'value':[],'Area':[]}
#        for area in areas:
#            for i in range(len(area_dict[area]['ego'][measure])):
#                if measure == 'explained_var' or measure == 'pseudo_r2':
#                    mean_dict['value'].append(area_dict[area]['ego'][measure][i]*100.)
#                else:
#                    mean_dict['value'].append(area_dict[area]['ego'][measure][i])
#                mean_dict['Area'].append(area)
#                
#            
#        fig.clf()
#        mean_data = pd.DataFrame(mean_dict)
#        sns.set(font_scale=font_size)
#        ax = sns.barplot(x='Area',y='value',data=mean_data,ci=68)
#        
#        if measure == 'llps':
#            ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#            ax.set_title('Egocentric Log Likelihood Increase')
#        elif measure == 'corr_r':
#            ax.set_ylabel('Correlation')
#            ax.set_title('Egocentric Correlation Contribution')
#        elif measure == 'explained_var':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Explained Variance')
#        elif measure == 'pseudo_r2':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Poisson Explained Variance')
#        ax.xaxis.label.set_visible(False)
#        sns_fig = ax.get_figure()
#        sns_fig.autofmt_xdate()
#        plt.tight_layout()
#        sns_fig.savefig('%s/ego %s.png' % (base_dir,measure),dpi=1200)
#
#                
#        fig.clf()
#        sns.set(font_scale=font_size)
#        ax = sns.boxplot(x='Area',y='value',data=mean_data)
#        if measure == 'llps':
#            ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#            ax.set_title('Egocentric Log Likelihood Increase')
#        elif measure == 'corr_r':
#            ax.set_ylabel('Correlation')
#            ax.set_title('Egocentric Correlation Contribution')
#        elif measure == 'explained_var':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Explained Variance')
#        elif measure == 'pseudo_r2':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Poisson Explained Variance')
#        ax.xaxis.label.set_visible(False)
#        sns_fig = ax.get_figure()
#        sns_fig.autofmt_xdate()
#        plt.tight_layout()
#        sns_fig.savefig('%s/ego %s boxplots' % (base_dir,measure),dpi=900)
#        
#        fig.clf()
#        sns.set(font_scale=font_size)
#        ax = sns.stripplot(x='Area',y='value',data=mean_data,jitter=True,zorder=1)
#        
#        median_width = 0.3
#    
#        for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
#            area = text.get_text()  # "X" or "Y"
#    
#            # calculate the median value for all replicates of either X or Y
#            median_val = mean_data[mean_data['Area']==area].value.median()
#    
#            # plot horizontal lines across the column, centered on the tick
#            ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
#                    lw=4, color='k',zorder=100)
#        
#        if measure == 'llps':
#            ax.set_ylabel('Log Likelihood Increase (bits/spike)')
#            ax.set_title('Egocentric Log Likelihood Increase')
#        elif measure == 'corr_r':
#            ax.set_ylabel('Correlation')
#            ax.set_title('Egocentric Correlation Contribution')
#        elif measure == 'explained_var':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Explained Variance')
#        elif measure == 'pseudo_r2':
#            ax.set_ylabel('Percent Variance')
#            ax.set_title('Egocentric Poisson Explained Variance')
##        if measure == 'pseudo_r2':
##            ax.set_ylim([-10,18])
#        h,p = kruskal(area_dict['V2L']['ego'][measure],area_dict['PoS']['ego'][measure],
#                       area_dict['PaS']['ego'][measure],area_dict['deep MEC']['ego'][measure],
#                       area_dict['superficial MEC']['ego'][measure])
#
#        
#        print(measure + ':')
#        print('h = %f' % h)
#        print('p = %f' % p)
#        
#        if measure == 'pseudo_r2':
#            
#            for var in pretty_vars:
#                h,p = kruskal(area_dict['V2L'][var][measure],area_dict['PoS'][var][measure],
#                               area_dict['PaS'][var][measure],area_dict['deep MEC'][var][measure],
#                               area_dict['superficial MEC'][var][measure])
#                print('"""""""""""""""')
#                print(var)
#                print('h = %f' % h)
#                print('p = %f' % p)
#                print('"""""""""""""""')
#                
#                if p < .05:
#                    area_list = []
#                    for area in areas:
#                        area_list.append(area)
#                        for area2 in areas:
#                            if area2 not in area_list:
#                                t,p = ranksums(area_dict[area][var][measure],area_dict[area2][var][measure])
#                                print('%s - %s' % (area,area2))
#                                print('t=%f' % t)
#                                print('p=%f' % p)
#                                print(' ')
#                    print('""""""""""""""""')
#        
#        ax.xaxis.label.set_visible(False)
#        
#
#        sns_fig = ax.get_figure()
#        sns_fig.autofmt_xdate()
#        plt.tight_layout()
#        sns_fig.savefig('%s/ego %s strip_plots' % (base_dir,measure),dpi=1200)
#        
#        
#    percents = {'Percent':[],'Area':[]}
#    for area in areas:
#        percents['Percent'].append(area_dict[area]['ego']['cell_count'] * 100./area_dict[area]['cell_count'])
#        percents['Area'].append(area)
#        
#    percent_frame = pd.DataFrame(percents)
#    fig.clf()
#    sns.set(font_scale=font_size)
#    ax = sns.barplot(x='Area',y='Percent',data=percent_frame)
#    ax.xaxis.label.set_visible(False)
#    ax.set_title('Egocentric Classifications')
#    ax.set_ylim([0,100])
#    
#    sns_fig = ax.get_figure()
#    sns_fig.autofmt_xdate()
#    plt.tight_layout()
#    sns_fig.savefig('%s/ego_percents.png' % base_dir,dpi=1200)
##    
#    overall = {'Percent':[],'Area':[],'Variable':[]}
#    for area in areas:
#        for var in pretty_vars:
#            overall['Area'].append(area)
#            overall['Variable'].append(var)
#            overall['Percent'].append(np.int(area_dict[area][var]['cell_count'] * 100./area_dict[area]['cell_count']))
#    overall_frame = pd.DataFrame(overall)
#    
#    overall_ylabels = ['V2L (n=%d)' % int(area_dict['V2L']['cell_count']),'PoS (n=%d)' % int(area_dict['PoS']['cell_count']),'PaS (n=%d)' % int(area_dict['PaS']['cell_count']),'Deep MEC (n=%d)' % int(area_dict['deep MEC']['cell_count']),'Superficial MEC (n=%d)' % int(area_dict['superficial MEC']['cell_count'])]
#    
##    overall_table = overall_frame.groupby(['Area', 'Variable'], sort=False)
#    overall_table = pd.pivot_table(overall_frame,values='Percent',columns='Variable',index='Area',aggfunc=np.sum)
#    overall_table = overall_table.reindex_axis(order,axis=1).reindex_axis(areas,axis=0)
#    fig.clf()
#    sns.set(font_scale=font_size)
#    ax = sns.heatmap(overall_table,vmin=0,vmax=100,annot=True,cmap='Blues',cbar=False)
#    ax.set_xticklabels(label_vars)
#    ax.set_yticklabels(overall_ylabels)
#    ax.xaxis.label.set_visible(False)
#    ax.yaxis.label.set_visible(False)
#    
#    sns_fig = ax.get_figure()
#    sns_fig.autofmt_xdate()
#    plt.tight_layout()
#    sns_fig.savefig('%s/table.png' % base_dir,dpi=1200)
#
#    return area_dict

def nice_spreadsheet(fdir):
    
    font_size = 1.4
    areas = ['V2L','PoS','PaS','deep MEC','superficial MEC','PoR','V1B']
#    areas = ['PoS']
    variables = [('allo',),('spatial',),('ahv',),('speed',),('theta',),('ego',)]
    measures = ['corr_r','explained_var','llps','pseudo_r2']
    
    plt.style.use('default')
    plt.rcParams['image.cmap'] = 'jet'
    fig = plt.figure()
    
    ego_paired = {}
    area_dict = {}
    egoallo_dict = {}
    for a in areas:
        ego_paired[a] = {}
        egoallo_dict[a] = {}
        area_dict[a] = {}
        area_dict[a]['cell_count'] = 0.
        for var in variables:
            if var[0] != 'ego':
                ego_paired[a][var[0]] = 0.
            area_dict[a][var[0]] = {}
            for measure in measures:
                area_dict[a][var[0]][measure] = []
                egoallo_dict[a][measure] = {'ego':[],'allo':[],'colors':[]}
            area_dict[a][var[0]]['cell_count'] = 0.
            
    cell_labels = {}

    for animal in os.listdir(fdir):
        animaldir = fdir + '/' + animal
        print animaldir
        
        for area in os.listdir(animaldir):
            if area in areas and area == 'PaS' and animal == 'MM44':
                areadir = animaldir + '/' + area
                print areadir
                
                for trial in os.listdir(areadir):
                    cell_labels[trial] = {}
                    trialdir = areadir + '/' + trial
                    
                    clusters = []
                    
                    for f in os.listdir(trialdir):
                        if f.startswith('TT') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                        elif f.startswith('ST') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                            
                    for cluster in clusters:
                        
                        fname = trialdir+'/all_single_ego_corrected_best_model_%s.pickle' % cluster
                        print fname
                        try:
                            model_dict = load_data(fname)
                        except:
                            return cell_labels
                        print model_dict['best_model']
                        cell_labels[trial][cluster] = list(model_dict['best_model'])
                        
    return cell_labels

def ego_spots():
    
    locs = []
    directions = []
    rayleigh_list = []
    
    allo_rayleighs = []
    allo_directions = []
    
    def rand_jitter(arr):
        stdev = .01*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * stdev
    
    fdir = 'H:/Patrick'
    areas = ['PoS']
#    areas=['PoS']
    
    for animal in os.listdir(fdir):
        animaldir = fdir + '/' + animal
        print animaldir
        
        for area in os.listdir(animaldir):
            if area in areas:
                areadir = animaldir + '/' + area
                print areadir
                
                for trial in os.listdir(areadir):
                    trialdir = areadir + '/' + trial
                    
                    clusters = []
                    
                    for f in os.listdir(trialdir):
                        if f.startswith('TT') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                        elif f.startswith('ST') and f.endswith('.txt'):
                            clusters.append(f[:(len(f)-4)])
                            
                    for cluster in clusters:
                        
                        fname = trialdir+'/corrected_best_model_%s.pickle' % cluster

                        print fname
                        model_dict = load_data(fname)
                        print model_dict['best_model']
                        
                        cluster_img_dir = trialdir + '/corrected_%s' % cluster
                        if not os.path.isdir(cluster_img_dir):
                            os.makedirs(cluster_img_dir)
#                        else:
#                            shutil.rmtree(cluster_img_dir)
#                            os.makedirs(cluster_img_dir)
                            
                        print cluster_img_dir

                        model_dict = load_data(fname)
                        
#                        if 'allo' in model_dict['best_model'] and 'ego' in model_dict['best_model']:
                        if 'ego' in model_dict['best_model'] and 'allo' not in model_dict['best_model']:
                        
#                            allo_params = model_dict['allo_params']
#                            allo_rayleigh,allo_mean_angle = rayleigh_r(np.arange(0,360,12),allo_params)
#                            allo_rayleighs.append(allo_rayleigh)
#                            allo_directions.append(allo_mean_angle)
                            
                            ego_params = model_dict['ego_params'] #.reshape((8,8,30))
                            rayleighs = np.zeros((8,8))
                            mean_angles = np.zeros((8,8))
                            for i in range(8):
                                for j in range(8):
                                    rayleighs[i][j],mean_angles[i][j]=rayleigh_r(np.arange(0,360,12),ego_params[i][j])                                                
                        #            rayleighs[i][j] = np.sum(ego_params[i][j])
                                            
                        
                            highest_loc = np.where(rayleighs == np.max(rayleighs))
                            directions.append(mean_angles[highest_loc])
                            locs.append(highest_loc)
                            rayleigh_list.append(rayleighs[highest_loc])
                            
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    loc_array = np.zeros((len(locs),2)
#    for i in range(len(locs)):
#        loc_array[i][0] = locs[i][0]
#        loc_array[i][1] = locs[i][1]
    
#    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=np.max(rayleigh_list))
    
    locs_array = np.zeros((len(locs),2))
        
    for i in range(len(locs)):
        locs_array[i][0]=locs[i][0] * (1 + np.random.random() * .3 - .3)
        locs_array[i][1]=locs[i][1] * (1 + np.random.random() * .3 - .3)

    ax.scatter(locs_array[:,1],np.max(locs_array[:,0])-locs_array[:,0],c=rayleigh_list,norm=norm)
#    ax.set_xlim([0,9])
#    ax.set_ylim([0,9])
    fig.show()
    
#    fig2 = plt.figure()
#    ax = fig2.add_subplot(111)
#    norm = mplcolors.Normalize(vmin=0, vmax=np.max(allo_rayleighs))
#    ax.scatter(locs_array[:,1],np.max(locs_array[:,0])-locs_array[:,0],c=allo_rayleighs,norm=norm)
#    plt.show()
#    
#    colormap = plt.get_cmap('hsv')
#    fig3 = plt.figure()
#    ax = fig3.add_subplot(111)
#    norm = mplcolors.Normalize(vmin=0, vmax=360)
#    ax.scatter(locs_array[:,1],np.max(locs_array[:,0])-locs_array[:,0],c=allo_directions,cmap=colormap,norm=norm)
#    plt.show()
#    
    colormap = plt.get_cmap('hsv')
    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    ax.scatter(locs_array[:,1],np.max(locs_array[:,0])-locs_array[:,0],c=directions,cmap=colormap,norm=norm)
    plt.show()
#    
#    
#    fig1 = plt.figure()
#    ax = fig1.add_subplot(111)
#    ax.hist(np.array(allo_directions).flatten(),bins=15)
#    ax.set_title('allo')
#    plt.show()
    
    plt.figure()
    plt.hist(np.array(directions).flatten(),bins=15)
    plt.title('ego')
    plt.show()
    
    return directions

def one_ego_spot():
    
    locs = []
    directions = []

    fname = 'H:/Patrick/MM44/PoS/2014-08-21_14-12-24/corrected_best_model_TT2_SS_03.pickle'
    print fname
    model_dict = load_data(fname)
    print model_dict['best_model']

    model_dict = load_data(fname)
    
    if 'ego' in model_dict['best_model']:
    
        ego_params = model_dict['ego_params'] #.reshape((8,8,30))
        rayleighs = np.zeros((8,8))
        mean_angles = np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                rayleighs[i][j],mean_angles[i][j]=rayleigh_r(np.arange(0,360,12),ego_params[i][j])                                                
    #            rayleighs[i][j] = np.sum(ego_params[i][j])
                        
    
        highest_loc = np.where(rayleighs == np.max(rayleighs))
        directions.append(mean_angles[highest_loc])
        locs.append(highest_loc)
                            
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    loc_array = np.zeros((len(locs),2)
#    for i in range(len(locs)):
#        loc_array[i][0] = locs[i][0]
#        loc_array[i][1] = locs[i][1]
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    locs_array = np.zeros((len(locs),2))
        
    for i in range(len(locs)):
        locs_array[i][0]=locs[i][0] * (1 + np.random.random() * .3)
        locs_array[i][1]=locs[i][1] * (1 + np.random.random() * .3)
        
    
    ax.scatter(locs_array[:,1],9-locs_array[:,0],c=directions,cmap=colormap,norm=norm)
    ax.set_xlim([0,9])
    ax.set_ylim([0,9])
    fig.show()
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    
    ax.hist(np.array(directions).flatten(),bins=15)
    plt.show()
    
    return directions

def allocentrize_ecd(ego_params):

    thresh_dist = 100
    gr = 8
    hd_bins = 30
    
    #create arrays for x and y coords of spatial bins
    xcoords = np.arange(gr)
    ycoords = np.arange(gr)[::-1]
    #assign coordinates for x and y axes for each bin
    #(x counts up, y counts down)
#    for x in range(gr):
#        xcoords[x] = (np.float(x)/np.float(gr))*np.float(np.max(center_x)-np.min(center_x)) + np.float(np.min(center_x))
#        ycoords[x] = np.float(np.max(center_y)) - (np.float(x)/np.float(gr))*np.float((np.max(center_y)-np.min(center_y)))


    xgrid,ygrid = np.meshgrid(xcoords,ycoords)

    @nb.jit(nopython=True)
    def loop(ego_params,xcoords,ycoords,xgrid,ygrid,thresh_dist):

        allocentrized_curves = np.zeros((gr,gr,hd_bins))
        
        for i in range(gr):
            current_y = ycoords[i]
            for j in range(gr):
                current_x = xcoords[j]
                
                num_responsible = np.zeros(hd_bins)
                weights = np.ones((gr,gr))
                
                    
                dists = np.sqrt((xgrid - current_x)**2 + (ygrid - current_y)**2)

                for v in range(gr):
                    for w in range(gr):
                        
#                        if local_ego:
#                            if dists[v][w] > thresh_dist or dists[v][w] < 10:
#                                dists[v][w] = np.nan
                                
#                        else:
                        if dists[v][w] < 1:
                            dists[v][w] = np.nan
                                
#                if local_ego:
#                    weights = np.sin(np.deg2rad(90+((dists - (np.nanmin(dists)))/(np.nanmax(dists)-np.nanmin(dists)))*90))
                
                for v in range(gr):
                    for w in range(gr):
                        
                        if np.isnan(dists[v][w]):
                            weights[v][w] = 0

                
                for l in range(gr):
                    other_y = ycoords[l]
                    for m in range(gr):
      
                        other_x = xcoords[m] 
                        new_angle = np.rad2deg(np.arctan2((other_y-current_y),(other_x-current_x)))%360
    
                        for k in range(hd_bins):
                            hd_angle = k*360./float(hd_bins)
                            ecd_bin = np.int(((new_angle - hd_angle))/(360/hd_bins))
                            allocentrized_curves[i][j][k] += ego_params[l][m][ecd_bin] * weights[l][m]
                            num_responsible[k] += weights[l][m]
                            
#                allocentrized_curves[i][j] /= num_responsible
                
        return allocentrized_curves

    allocentrized_curves = loop(ego_params,xcoords,ycoords,xgrid,ygrid,thresh_dist)

#    import matplotlib.pyplot as plt
#    rayleighs = np.zeros((gr,gr))
#    hd_angles = np.arange(0,360,12)
#    plt.figure()
#    for i in range(gr):
#        for j in range(gr):
#            rayleighs[i][j],_ = rayleigh_r(hd_angles,allocentrized_curves[i][j])
#            
#    plt.imshow(rayleighs)
#    plt.show()
#

    return allocentrized_curves

def rayleigh_r(spike_angles,rates=None,ego=False):
    """finds rayleigh mean vector length for head direction curve"""
    
    #start vars for x and y rayleigh components
    rx = 0
    ry = 0
    
    #convert spike angles into x and y coordinates, sum up the results -- 
    #if firing rates are provided along with HD plot edges instead of spike angles,
    #do the same thing but with those
    if rates is None:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))
            ry += np.sin(np.deg2rad(spike_angles[i]))
    else:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))*rates[i]
            ry += np.sin(np.deg2rad(spike_angles[i]))*rates[i]

    #calculate average x and y values for vector coordinates
    if rates is None:
        if len(spike_angles) == 0:
            spike_angles.append(1)
        rx = rx/len(spike_angles)
        ry = ry/len(spike_angles)
    
    else:
        rx = rx/sum(rates)
        ry = ry/sum(rates)

    #calculate vector length
    r = np.sqrt(rx**2 + ry**2)
    
    #calculate the angle the vector points (rayleigh pfd)
    #piecewise because of trig limitations
    if rx == 0:
#        rx = 1
        mean_angle = 0
    elif rx > 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx))
    elif rx < 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
    try:
        if mean_angle < 0:
            mean_angle = mean_angle + 360
    except:
        mean_angle = 0
        
    if ego:
        return r,rx,ry, mean_angle
    else:
        return r, mean_angle