"""Common functions used for plotting in the project"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from pprint import pprint
from scipy.stats import zscore
from scipy import stats
from os.path import join as oj, isfile
from itertools import combinations
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.rcParams['hatch.linewidth'] = 3.0  # previous svg hatch linewidth


flierprops = dict(marker='+', markerfacecolor='red', markersize=12, markeredgecolor='red', linestyle='none')
cdict = {'red':   ((0.0, 0.0, 1.0),
                   (1.301/4, 0.7, 0.7),
                   (1.0, 0/255, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.301/4, 0.7, 0.7),
                   (1.0, 100/255, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.301/4,0.7,0.7),
                   (1.0, 1.0, 0.0))
        }
colorbar = LinearSegmentedColormap('pvalue', cdict)


def p_show(p):
    for i in range(2,15):
        if p > 1/10**i:
            return round(p, i+1)


def get_significance_label(p):
    if p > 0.05:
        return 'n.s.'
    if p > 0.01:
        return '*'
    if p > 0.001: 
        return '**'
    if p > 0.0001:
        return '***'
    return '****'


def plot_cnn_scores(file, category_col, score_col, x_param, ylabel=None, xlabel=None, save_name=None,
                   figsize=(12,10), fontsize=30, labelsize=30, ylim=None, facecolor='white', rotation=45,
                   title='', sig=True):
    """A function for plotting the CNN scores in whisker and box plot style grouped together by categorical
    feature.
    
    :param data : dataframe
        dataframe containing the data
    :param 
    """
    flierprops = dict(marker='+', markerfacecolor='red', markersize=12, markeredgecolor='red', linestyle='none')
    data = []
    scores = list(x_param.keys())
    values = [x_param[s] for s in scores]
    for score in scores:
        count = np.asarray(file[file[category_col]==score][score_col])
        print("{} count: {}".format(x_param[score], len(count)))
        count.shape = (-1,1)
        data.append(count[~np.isnan(count)])
        
    # box plot and dot plot
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    bp = ax.boxplot(data, flierprops=flierprops, showmeans=False, patch_artist=True)
    plt.setp(bp['boxes'], lw=3)
    plt.setp(bp['whiskers'], lw=3)
    plt.setp(bp['caps'], lw=3)
    
    for patch in bp['boxes']:
        patch.set(facecolor=facecolor)
        
    N = len(list(x_param.keys()))
    for i in range(N):
        plt.setp(bp['medians'][i], color='#D81B60', lw=3)
#         ax.scatter([i+1 for _ in data[i]], data[i], c='k')
        
    # t test - calculate p-values
    max_value = max([d.max() for d in data])
       
    if sig:
        for x in range(1, N):
            _, p = stats.ttest_ind(data[x-1], data[x], nan_policy='omit')
            # plot significance label
            x1, x2 = x+0.03, x+0.97   
            y, h, col = max_value*1.1, max_value*0.03, 'k'
            text = get_significance_label(p)
            if text != 'n.s.':
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=25)
        
    # calculate all the p-values
    # calculate the other p-values (non-adjacent groups
    for c in combinations(list(range(N)), 2):
        r, p = stats.ttest_ind(data[c[0]], data[c[1]], nan_policy='omit')
        print('{}, {}: p-value of {}'.format(values[c[0]], values[c[1]], p))

    if ylim != 'auto':
        ax.set_ylim([-max([d.max() for d in data])/10, max([d.max() for d in data])*1.3])
    _ = ax.set_xticklabels(list(x_param.values()))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=labelsize, size=0, width=2, rotation=rotation)
    ax.yaxis.set_tick_params(labelsize=labelsize, size=5, width=2)
    ax.set_title(title, fontsize=fontsize+2)
    
       
    if save_name is not None and not isfile(save_name):
        fig.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.pause(0.001)
    plt.show()


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='#D81B60')
    

def plot_grouped_box_plots(file1, file2, category_col, score_col, labels, scores=None, figsize=(12,10), fontsize=30,
                          ylabel=None, xlabel=None, xticks=None, save_name=None, facecolor=None, title='', labelsize=30):    
    flierprops = dict(marker='+', markerfacecolor='red', markersize=12, markeredgecolor='red', linestyle='none')
    
    data1 = []
    data2 = []
    if scores is None:
        scores = sorted(list(file1[category_col].unique()))
    
    for score in scores:        
        count1 = np.asarray(file1[file1[category_col] == score][score_col])
        count2 = np.asarray(file2[file2[category_col] == score][score_col])
        data1.append(list(count1))
        data2.append(list(count2))

    # t test - calculate p-values
    max_value = max(max([max(d) for d in data1]), max([max(d) for d in data2]))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
        
    # plot significance label
    for i, x in enumerate(range(0, len(scores) * 2, 2)):
        _, p = stats.ttest_ind(data1[i], data2[i], nan_policy='omit')
        x1, x2 = x-0.4, x+0.4
        y, h, col = max_value*1.1, max_value*0.01, 'k'
        text = get_significance_label(p)
        if text != 'n.s.':
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=25)

    bpl = plt.boxplot(data1, positions=np.array(range(len(data1)))*2.0-0.4, widths=0.6,
                     flierprops=flierprops, showmeans=False, patch_artist=True)


    bpr = plt.boxplot(data2, positions=np.array(range(len(data2)))*2.0+0.4, widths=0.6,
                     flierprops=flierprops, showmeans=False, patch_artist=True)
    
    for i in range(4):
        plt.setp(bpr['medians'][i], color='#D81B60', lw=3)
    plt.setp(bpl['boxes'], lw=3)
    plt.setp(bpl['whiskers'], lw=3)
    plt.setp(bpl['caps'], lw=3)
    for i in range(4):
        plt.setp(bpl['medians'][i], color='#D81B60', lw=3)
    plt.setp(bpr['boxes'], lw=3)
    plt.setp(bpr['whiskers'], lw=3)
    plt.setp(bpr['caps'], lw=3)
    for patch in bpr['boxes']:
        patch.set(facecolor=facecolor)
    for patch in bpl['boxes']:
        patch.set(facecolor=facecolor)
    for box in bpr['boxes']:
        box.set(hatch = '/')

    # create legend with hatch for second group
    a_val = 1
    circ1 = mpatches.Patch(facecolor=facecolor, alpha=a_val,label='Emory')
    circ2 = mpatches.Patch(facecolor=facecolor, alpha=a_val, hatch=r'//', label='Tang')
    ax.legend(handles= [circ1,circ2],loc=2, fontsize=20)
    
#     plt.legend(fontsize=20, loc='best')
    plt.xticks(range(0, len(scores) * 2, 2), scores, fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    ax.set_title(title, fontsize=fontsize+2)
    
    # find the max of data 1 and data 2
    max1 = max([max(d) for d in data1])
    max2 = max([max(d) for d in data2])
    _max = max([max1, max2])
    ax.set_ylim(-_max / 10, _max * 1.3)
    plt.xlim(-1, len(scores)*1.75)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    else:
        plt.xlabel(category_col, fontsize=30)
    if xticks is not None:
        _ = ax.set_xticklabels(xticks, fontsize=20, rotation=20)
        
        
    if save_name is not None and not isfile(save_name):
        fig.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.pause(0.001)
    plt.show()

    
# def compare_dfs(file1, file2, category_col, score_col, labels, scores=None,
#                           ylabel=None, xlabel=None, xticks=None, save_name=None):  
#     # in this case the category and score_col is the same for both dfs
#     data1 = []
#     data2 = []

#     if scores is None:
#         scores = sorted(list(file1[category_col].unique()))
    
#     for score in scores:        
#         count1 = np.asarray(file1[file1[category_col] == score][score_col])
#         count2 = np.asarray(file2[file2[category_col] == score][score_col])
#         data1.append(list(count1))
#         data2.append(list(count2))

#     # t test - calculate p-values
#     max_value = max(max([max(d) for d in data1]), max([max(d) for d in data2]))

#     fig = plt.figure(figsize=(12,10))
#     ax = fig.add_subplot(111)
        
#     # plot significance label
#     for i, x in enumerate(range(0, len(scores) * 2, 2)):
#         _, p = stats.ttest_ind(data1[i], data2[i], nan_policy='omit')
#         x1, x2 = x-0.4, x+0.4
#         y, h, col = max_value*1.1, max_value*0.03, 'k'
#         text = get_significance_label(p)
#         ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#         ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=25)

#     bpl = plt.boxplot(data1, positions=np.array(range(len(data1)))*2.0-0.4, widths=0.6,
#                      flierprops=flierprops, showmeans=True)

#     flierprops['markerfacecolor'] = '#2C7BB6'
#     flierprops['markeredgecolor'] = '#2C7BB6'
#     bpr = plt.boxplot(data2, positions=np.array(range(len(data2)))*2.0+0.4, widths=0.6,
#                      flierprops=flierprops, showmeans=True)

#     # draw temporary red and blue lines and use them to create a legend
#     # set the colors
#     set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
#     set_box_color(bpr, '#2C7BB6')
#     plt.plot([], c='#D7191C', label=labels[0])
#     plt.plot([], c='#2C7BB6', label=labels[1])
    
#     plt.legend(fontsize=20, loc='best')
#     plt.xticks(range(0, len(scores) * 2, 2), scores, fontsize=25)
#     plt.yticks(fontsize=25)
#     plt.xlim(-2, len(scores)*2)
#     if ylabel is not None:
#         plt.ylabel(ylabel, fontsize=30)
#     if xlabel is not None:
#         plt.xlabel(xlabel, fontsize=30)
#     else:
#         plt.xlabel(category_col, fontsize=30)
#     if xticks is not None:
#         _ = ax.set_xticklabels(xticks, fontsize=20)
        
#     if save_name is not None and not isfile(save_name):
#         fig.savefig(save_name, bbox_inches='tight', dpi=300)
#     plt.pause(0.001)
#     plt.show()