# plotlib.py
# functions and classes to plot data 
# Author: Alexander Sasse <alexander.sasse@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import sys,os
from scipy.linalg import svd
from functools import reduce
from matplotlib import cm
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib as mpl
import logomaker as lm
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from sklearn import linear_model
from .motif_analysis import align_compute_similarity_motifs, torch_compute_similarity_motifs
import matplotlib.patches as mpatches


def plot_losses(valfiles, modnames, percentages=False, combine_sets=False, logx=False, adjust_axis=False, savefig=None):
    if not isinstance(valfiles, list):
        valfiles = [valfiles]
    if not isinstance(modnames, list):
        modnames = [modnames]   
    if len(valfiles) != len(modnames):
        print('Number of validation files and model names do not match!')
        sys.exit(1) 
    if len(valfiles) > 10:
        print('Warning: more than 10 validation files provided, this might not be displayed correctly!')    


    losses = []
    for v, valf in enumerate(valfiles):
        print(valf)
        loss = np.genfromtxt(valf, dtype=float, delimiter='\t')
        nanmask = np.unique(np.where(~np.isnan(loss))[0])
        losses.append(loss[nanmask])
        if percentages:
            losses[-1][:, 0] = losses[-1][:, 0] / np.amax(losses[-1][:, 0])

    fig = plt.figure(figsize=(9, 9), dpi=180)
    cmap = np.append(cm.tab20b(np.arange(0, 20, 4)), cm.tab20c(np.arange(0, 20, 4)), axis=0)

    ax0 = fig.add_subplot(10, 1, 1)
    width = 0.4 if combine_sets else 0.8
    ax0.set_position([0.1, 0.78, width, min(1, (len(modnames) / 5)) * 0.17])
    ax0.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
    rows = min(len(modnames), 4)
    columns = int(len(modnames) / rows) + int(len(modnames) % rows > 0)
    ax0.set_xlim([-0.1, columns])
    ax0.set_ylim([-0.5, rows - 0.5])
    for m, name in enumerate(modnames):
        ax0.scatter([int(m / rows)], [rows - 1 - m % rows], color=cmap[m])
        ax0.text(int(m / rows) + 0.02 * columns, rows - 1 - m % rows, name, va='center', ha='left', fontsize=8)

    ax1 = fig.add_subplot(221)
    ax1.set_position([0.1, 0.45, 0.38, 0.32])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)
    for l, loss in enumerate(losses):
        ax1.plot(loss[:, 0], loss[:, 1], marker='.', ms=5, alpha=0.5, color=cmap[l])
    ax1.set_ylabel('Training Loss' if combine_sets else 'Training Loss (val)')
    if logx:
        ax1.set_xscale('symlog')
    ax1.grid()

    ax2 = fig.add_subplot(223)
    ax2.set_position([0.1, 0.1, 0.38, 0.32])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for l, loss in enumerate(losses):
        ax2.plot(loss[:, 0], loss[:, 2], marker='.', alpha=0.5, ms=5, color=cmap[l])
    ax2.set_ylabel('Validation Loss' if combine_sets else 'Validation Loss (val)')
    ax2.set_xlabel('% Total Epochs' if percentages else 'Epochs')
    ax2.grid()
    if logx:
        ax2.set_xscale('symlog')

    if combine_sets:
        ax3 = ax1
        marker = '2'
    else:
        ax3 = fig.add_subplot(222)
        ax3.set_position([0.52, 0.45, 0.38, 0.32])
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.tick_params(left=False, labelleft=False, right=True, labelright=True, labelbottom=False)
        marker = '.'
    for l, loss in enumerate(losses):
        ax3.plot(loss[:, 0], loss[:, 3], marker=marker, alpha=0.5, ms=5, color=cmap[l], label='Train')

    if not combine_sets:
        ax3.set_ylabel('Training Loss (train)')
        ax3.grid()
        if logx:
            ax3.set_xscale('symlog')
    else:
        ax3.legend()

    if combine_sets:
        ax4 = ax2
    else:
        ax4 = fig.add_subplot(224)
        ax4.set_position([0.52, 0.1, 0.38, 0.32])
        ax4.spines['left'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.tick_params(left=False, labelleft=False, right=True, labelright=True)

    for l, loss in enumerate(losses):
        ax4.plot(loss[:, 0], loss[:, 4], marker=marker, alpha=0.5, ms=5, color=cmap[l], label='Train')

    if not combine_sets:
        ax4.set_xlabel('% Total Epochs' if percentages else 'Epochs')
        ax4.set_ylabel('Validation Loss (train)')
        ax4.grid()
        if logx:
            ax4.set_xscale('symlog')
    else:
        ax4.legend()

    if adjust_axis:
        ax1.set_ylim([min(ax1.get_ylim()[0], ax3.get_ylim()[0]), max(ax1.get_ylim()[1], ax3.get_ylim()[1])])
        ax2.set_ylim([min(ax2.get_ylim()[0], ax4.get_ylim()[0]), max(ax2.get_ylim()[1], ax4.get_ylim()[1])])
        if not combine_sets:
            ax3.set_ylim([min(ax1.get_ylim()[0], ax3.get_ylim()[0]), max(ax1.get_ylim()[1], ax3.get_ylim()[1])])
            ax4.set_ylim([min(ax2.get_ylim()[0], ax4.get_ylim()[0]), max(ax2.get_ylim()[1], ax4.get_ylim()[1])])

    if savefig:
        fig.savefig(savefig, dpi=200, bbox_inches='tight')
    else:
        plt.show()

def _add_frames(att, locations, ax, color = 'k', cmap = None):
    '''
    Adds frames to a logo plot around per base attributions for visualization
    Parameters
    ----------
    att : numpy 2D array
        Attributions
    locations: list of tuples, or numpy array in that form
        Start and end positions of motif locations
    ax : 
        matplot subplot object
    
    Returns
    -------
    None
    '''
    if cmap is not None and isinstance(color, np.ndarray):
        if isinstance(cmap, str):
            # make it object to extract colors for each individual one
            cmap = cm.get_cmap(cmap)
        color = cmap(color)
    att = np.array(att)
    for l, loc in enumerate(locations):
        # Determine height of box
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = color)
        ax.plot(x, [maxa, maxa], c = color)
        ax.plot([x[0], x[0]] , [mina, maxa], c = color)
        ax.plot([x[1], x[1]] , [mina, maxa], c = color)


def plot_seqlogo(att, ax = None, ylabel = None, ylim = None, yticks = None, 
                 yticklabels = None,labelbottom = True, bottomticks = True, 
                 xticks = None, xticklabels = None, grid = False, 
                 basewidth = 0.4, channels = list('ACGT')):
    '''
    Creates logo plot for sequence attributions along positions
    Parameters
    ----------
    att : np.ndarray
        Sequence attributions of shape (N_positions, n_channels)
    '''
    
    if len(channels) != att.shape[-1]:
        if len(channels) == att.shape[-2]:
            att = np.transpose(att, axes = (1,0))
        else:
            raise ValueError(f"Shape mismatch: 'channels' length ({len(channels)}) does not match the last two dimensions of 'att' ({att.shape[-1]} or {att.shape[-2]}).")

    if ax is None:
        # open a pyplot figure
        fig = plt.figure(figsize = (np.shape(att)[0] * basewidth,
                                    np.shape(att)[1] * basewidth ) )
        ax = fig.add_subplot(111)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom = bottomticks, labelbottom = labelbottom)
    att = pd.DataFrame(att, columns = channels)
    lm.Logo(att, ax = ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if xticklabels is not None:
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(yticklabels)
        ax.set_yticklabels(yticklabels)
    if grid:
        ax.grid()
    return ax

    
def _plot_heatmap(arr, cmap = 'coolwarm', ylabel = None, grid = False, ratio=1,
            labelbottom = True, bottomticks = True, vlim = None, 
            add_text = False, yticklabels = None, xticklabels = None, 
            unit = 0.25, title = None, ax = None):
    '''
    Plot heatmap
    '''
    if ax is None:
        # open a pyplot figure
        fig = plt.figure(figsize = (np.shape(arr)[0] * unit,
                                    np.shape(arr)[1] * unit ) )
        ax = fig.add_subplot(111)
    
    if title is not None:
        ax.set_title(title)
        
    if vlim is None:
        vlim = np.amax(np.absolute(arr))
        vlim = [-vlim, vlim]
    
    ax.imshow(arr, aspect = 'auto', cmap = cmap, vmin = -vlim, vmax = vlim)
    if grid:
        ax.set_xticks(np.arange(0.5, np.shape(arr)[1], 1), minor = True)
        ax.set_yticks(np.arange(0.5, np.shape(arr)[0], 1), minor = True)
        ax.grid(which = 'minor')
    
    if add_text:
        for i in range(np.shape(arr)[0]):
            for j in range(np.shape(arr)[1]):
                ax.text(j,i,str(arr[i,j]), va = 'center', ha = 'center', fontsize = 6)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.tick_params(bottom = bottomticks, labelbottom = labelbottom)
    
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(list(yticklabels))
   
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(list(xticklabels))
    
    return ax

def _bar_plot(values, width = 0.8, color = 'lightsteelblue', unit = 1,
              ylim=None, xticklabels=None, yticks = None, yticklabels = None,
              ylabel = None, rotation = None, horizontal = False, ax = None):
    '''
    Barplot
    '''
    if ax is None:
        # open a pyplot figure
        if horizontal:
            fig = plt.figure(figsize = (3, len(values) * unit))
        else:
            fig = plt.figure(figsize = (len(values) * unit, 3) )
        ax = fig.add_subplot(111)
    
    if horizontal:
        ax.barh(np.arange(len(values)), values, height = width, color = color)  
    else:
        ax.bar(np.arange(len(values)), values, width = width, color = color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim is not None:
        if horizontal:
            ax.set_xlim(ylim)
        else:
            ax.set_ylim(ylim)
    
    if xticklabels is None:
        if horizontal:
            ax.tick_params(left = False, labelleft = False)
        else:
            ax.tick_params(bottom = False, labelbottom = False)
    else:
        if horizontal:
            ax.set_yticks(np.arange(len(xticklabels)))
            ax.set_yticklabels(xticklabels, rotation = rotation)
        else:
            ax.set_xticks(np.arange(len(xticklabels)))
            ax.set_xticklabels(xticklabels, rotation = rotation)

    if yticklabels is not None and yticks is not None:
        if horizontal:
            ax.set_xticks(yticks)
            ax.set_xticklabels(yticklabels)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
    elif yticks is not None:
        if horizontal:
            ax.set_xticks(yticks)
        else:
            ax.set_yticks(yticks)
    if ylabel is not None:
        if horizontal:
            ax.set_xlabel(ylabel)
        else:
            ax.set_ylabel(ylabel)

    return ax

def _generate_xticks(start, end, n):
    '''
    Generate xticks for sequences that are multiple of values below
    '''
    possible = np.concatenate([np.array([1,2,5,10])*10**i for i in range(-16,16)])
    steps=(end-start)/n
    steps = possible[np.argmin(np.absolute(possible - steps))]
    ticklabels = np.arange(start, end)
    ticks = np.where(ticklabels%steps == 0)[0]
    ticklabels = ticklabels[ticks]
    return ticks, ticklabels
    
def correct_sparse_attributions(att, n_channels = 4, l_seq = None):
    # Check if the attributions were stored in sparse coordinates and to
    # original size
    if np.shape(att)[-1] != n_channels:
        if (np.shape(att)[-1] != n_channels +1) and (np.shape(att)[-2] == n_channels +1):
            att = np.swapaxes(att, -1, -2)
        elif (np.shape(att)[-1] != n_channels +1) and (np.shape(att)[-2] != n_channels +1): 
            raise Warning(f'The shape of the attributions is {att.shape}, at axis=-1 does not match n_channels {n_channels} +1')
        if l_seq is None:
            l_seq = int(np.amax(att[..., -1]))+1
            print(f'l_seq not given, so working with max position {l_seq}')
        if att.ndim == 2:
            nshape = list(np.shape(att))
            nshape[-1] = nshape[-1] -1
            nshape[-2] = l_seq
            natt = np.zeros(nshape)
            natt[at[:,-1].astype(int)] = at[:,:n_channels]
        if att.ndim == 3:   
            nshape = list(np.shape(att))
            nshape[-1] = nshape[-1] -1
            nshape[-2] = l_seq
            natt = np.zeros(nshape)
            for a, at in enumerate(att):
                natt[a, at[:,-1].astype(int)] = at[:,:n_channels]
        if att.ndim == 4:   
            nshape = list(np.shape(att))
            nshape[-1] = nshape[-1] -1
            nshape[-2] = l_seq
            natt = np.zeros(nshape)
            for b, bat in enumerate(att):
                for a, at in enumerate(bat):
                    natt[b, a, at[:,-1].astype(int)] = at[:,:n_channels]
        att = natt
    else: 
        raise Warning(f'The shape of the attributions is {att.shape}, Axis -1 does not match n_channels {n_channels} +1')
    
    return att

def plot_attribution_maps(att, seq = None, motifs = None, experiments = None, 
                          vlim = None, unit = 0.15, ratio = 10, ylabel = None,
                          xtick_range = None, barplot = None, heatmap = False, 
                          center_attribution = False, channels = list('ACGT')):
    '''
    Plots single or multiple attribution maps above each other.
    If heatmap is True, heatmaps are added below the attribution. 
    Parameters
    ----------
    att : numpy array 
        Attributions of shape (N_seq, l_seq, channels)
    seq : numpy array
        One-hot encoded sequences for each attributions
        If given, the attributions will be multiplie with this and only show the
        attributions for the given sequence in attribution map
        Heatmaps are unaffeced
    motifs : list of lists wit tuples
        Locations of motifs in the sequence that will be framed
    vlim : tuple of floats
        vmin and vmax for heatmap colors
    unit : float
        Size of a single cell in the heatmap
    ratio : float
        Number of single cells used for height of attribution map
    ylabel : str
        Label on the y-axis
    xtick_range : tuple of int
        start and end of the sequence locations in the genome
    barplot : list of lists or np.arrays
        For a barplot on the side of the attribution maps that contain predicted or measured values
    heatmap: bool
        If true, heatmap with attributions will be added beneath attribution map
    center_attribution: bool
        Centers attributions for attribution map.
    exeriments : list of string
        Titles of seqeuence attributions
    '''
    #check correct input sizes
    if len(channels) != att.shape[-1]:
        if len(channels) == att.shape[-2]:
            att = np.transpose(att, axes = (0,2,1))
        else:
            raise ValueError(f"Shape mismatch: 'channels' length ({len(channels)}) does not match the last two dimensions of 'att' ({att.shape[-1]} or {att.shape[-2]}).")
    if seq is not None:
        if seq.shape != att.shape:
            if seq.shape[-1] == att.shape[-2] and seq.shape[-2] == att.shape[-1]:
                seq = np.transpose(seq, axes = (0,2,1))
            else:
                raise ValueError(f"Shape mismatch: 'channels' length ({len(channels)}) does not match the last two dimensions of 'seq' {seq.shape[-1]}.")

    if heatmap:
        if isinstance(heatmap, np.ndarray):
            if heatmap.shape == att.shape:
                ism = heatmap
        else:
            ism = np.copy(att)
    
    if center_attribution:
        att -= (np.mean(att, axis = -1))[...,None]
    
    if seq is not None:
        att = seq * att
    
    if experiments is None:
        experiments = np.arange(len(att), dtype = int).astype(str)
        
    if vlim is None:
        mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -1)))
        maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -1))
        attlim = [mina, maxa]
    else:
        attlim = vlim
    
    if xtick_range is not None:
        xticks, xticklabels = _generate_xticks(xtick_range[0], xtick_range[1], 7)
    else:
        xticks, xticklabels = None, None
    
    # multiplier in case heatmaps are plotted with logomaker
    _heat = (1+int(heatmap))
    
    fig = plt.figure(figsize = (unit*np.shape(att)[1], np.shape(att)[0] * _heat * ratio*unit), dpi = 50)
    
    axs = []
    for a, at in enumerate(att):
        ax = fig.add_subplot(len(att)*_heat, 1, 1+(a*_heat))
        ax.set_position([0.1, 0.1+(len(att)-1-(a*_heat))/len(att)/_heat*0.8, 0.8, 0.8*(1/len(att)/_heat)*0.8])
        axs.append(plot_seqlogo(at, ax = ax, ylabel = experiments[a], 
                                labelbottom = (a == len(att)-1) & (~heatmap), 
                                bottomticks = (a == len(att)-1)& (~heatmap), 
                                ylim = attlim, xticks = xticks, xticklabels = xticklabels, 
                                channels=channels
                                ))
        if motifs is not None:
            _add_frames(at, locations[a], ax, color = motifcolors)
        if heatmap:
            _vlim = np.amax(np.absolute(attlim))
            ax = fig.add_subplot(len(att)*_heat, 1, 2+(a*_heat))
            ax.set_position([0.1, 0.1+(len(att)-2-(a*_heat))/len(att)/_heat*0.8, 0.8, 0.8*(1/len(att)/_heat)*0.8])
            axs.append(_plot_heatmap(ism, ax = ax, ylim = attlim, labelbottom = (a == len(att)-1), bottomticks = (a == len(att)-1), xticks = xticks, xticklabels = xticklabels), cmap = 'coolwarm', ylabel = None, vlim = [-_vlim, _vlim])
            
    
    # This is for a barplot on the side of the sequence logo, that shows predicted and/or measured actibity
    if barplot is not None:
        ylim = [0, np.amax(barplot)]
        for b, bp in enumerate(barplot):
            ax = fig.add_subplot(len(barplot), len(barplot), len(barplot) + b)
            ax.set_position([0.9 + 2.5*0.8*(1/len(seq)), 0.1+(len(att)-1-b)/len(att)*0.8, 6*0.8*(1/len(seq)), 0.8*(1/len(att))*0.8])
            axs.append(_bar_plot(bp, ylim = ylim, ax = ax))

    return fig





def plot_single_pwm(pwm, log = False, showaxes = False, channels = list('ACGT'), ax = None):
    '''
    Plots single PWM, determines figsize based on length of pwm
    pwm : 
        shape=(length_logo, channels)
    '''
    if ax is None:
        fig = plt.figure(figsize = (np.shape(pwm)[0]*unit,np.shape(pwm)[1]*unit), dpi = 300)
        ax = fig.add_subplot(111)
    
    lim = [min(0, -np.ceil(np.around(np.amax(-np.sum(np.ma.masked_array(pwm, pwm >0),axis = 1)),2))), 
           np.ceil(np.around(np.amax(np.sum(np.ma.masked_array(pwm, pwm <0),axis = 1)),2))]
    
    if log:
        pwm = np.log2((pwm+1e-16)/0.25)
        pwm[pwm<0] = 0
        lim = [0,2]
    
    lm.Logo(pd.DataFrame(pwm, columns = channels), ax = ax, color_scheme = 'classic')
    ax.set_ylim(lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not showaxes:
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft = False, left = False, labelbottom = False, bottom = False)
    ax.set_yticks(lim)
    return ax

def reverse(pwm):
    return pwm[::-1][:,::-1]

def plot_pwms(pwm, log = False, showaxes = False, unit = 0.4, channels= list('ACGT'), offsets = None, revcomp_matrix = None, align_to = 0):
    '''
    Aligns and plots multiple pwms
    use align_to to determine to which pwm the others should be aligned
    set align_to to 'combine' to combine list of pwms and add combined motif
    at position 0
    '''
    if isinstance(pwm, list):
        if offsets is None:
            ifcont = True
            min_sim = 4
            for pw in pwm:
                min_sim = min(min_sim, np.shape(pw)[0])
                if (pw<0).any():
                    ifcont = False
            correlation, offsets, revcomp_matrix = torch_compute_similarity_motifs(pwm, pwm, exact=True, return_alignment = True, metric = 'correlation', min_sim = min_sim, infocont = ifcont, reverse_complement = revcomp_matrix is not None)
            offsets = offsets[:,align_to] # use offsets  from first pwm
            revcomp_matrix = revcomp_matrix[:,align_to] # use reverse complement assignment from first pwms
        else:
            if revcomp_matrix is None:
                revcomp_matrix = np.zeros(len(offsets))
                
        pwm_len=np.array([len(pw) for pw in pwm]) # array of pwm lengths
        
        # compute offsets for each pwm so that they will be put into an array,
        #so that all pwms will be aligned 
        offleft = abs(min(0,np.amin(offsets))) 
        offright = max(0,np.amax(offsets + pwm_len-np.shape(pwm[0])[0]))

        nshape = list(np.shape(pwm[0]))
        nshape[0] = nshape[0] + offleft + offright # total length that is 
        #needed to fit all pwms into region when aligned to each other
 
        fig = plt.figure(figsize = (unit*nshape[0], len(pwm) * nshape[1]*2*unit), dpi = 50)
        print((unit*nshape[0], len(pwm) * nshape[1]*unit), len(pwm), nshape)
        for p, pw in enumerate(pwm):
            ax = fig.add_subplot(len(pwm), 1, p + 1)
            if revcomp_matrix[p] == 1:
                pw = reverse(pw)
            # create empty array with nshape
            if not log:
                pw0 = np.zeros(nshape)
            else: 
                pw0 = np.ones(nshape)*0.25
            pw0[offleft + offsets[p]: len(pw) + offleft + offsets[p]] = pw
            pw = pw0
            plot_single_pwm(pw, log=log, showaxes = showaxes, channels = channels, ax = ax)
    else:
        
        fig = plt.figure(figsize = (np.shape(pwm)[0]*unit,3*np.shape(pwm)[1]*unit), dpi = 300)
        ax = fig.add_subplot(111)
        plot_single_pwm(pwm, log = log, showaxes = showaxes, channels = channels, ax = ax)
        
    return fig


def _check_symmetric_matrix(distmat):
    # Test if distmat is symmetric
    if np.shape(distmat)[0] != np.shape(distmat)[1]:
        print( 'Warning: not symmetric matrix: sort set to None if given')
        return False
    elif np.any(np.abs(distmat - distmat.T) > 1e-8):
        print( 'Warning: not symmetric matrix: sort set to None if given')
        return False
    else:
        return True

def _transform_similarity_to_distance(distmat):
    # checks if similarity matrix or distance matrix
    issimilarity = np.all(np.amax(distmat) == np.diag(distmat))
    heatmax, heatmin = np.amax(distmat), np.amin(distmat)
    simatrix = int(issimilarity) - (2.*int(issimilarity)-1.) * (distmat - heatmin)/(heatmax - heatmin)

    return simatrix

def plot_heatmap(heatmat, # matrix that is plotted with imshow
                 ydistmat = None, # matrix to compute sorty, default uses heatmat
                 xdistmat = None, # matrix to compute sortx, default uses hetamat
                 measurex = None, # if matrix is not a symmetric distance matrix
                 # measurex defines distannce metric to compute distances for linkage clustering 
                 measurey = None, # same as measurex just for y axic
                 sortx = None, # agglomerative clustering algorith used in likage, f.e average, or single
                 sorty = None, # same as above but for y axis
                 x_attributes = None, # additional heatmap with attributes of columns
                 y_attributes = None, # same as above for y axis
                 xattr_name = None, # names of attributes for columns
                 yattr_name = None, # names of attributes for rows
                 heatmapcolor = cm.BrBG_r, # color map of main matrix
                 xatt_color = None, # color map or list of colormaps for attributes
                 yatt_color = None, 
                 xatt_vlim = None, # vmin and vmas for xattributes, or list of vmin and vmax
                 yatt_vlim = None,
                 pwms = None, # pwms that are plotted with logomaker next to rows of matrix
                 infocont = True, # if True, the matrices will be plotted as information content
                 combine_cutx = 0., # NOT implemented, can be used to cut off 
                 # linkage tree at certain distance and reduce its resolution
                 combine_cuty = 0., 
                 color_cutx = 0., # cut off for coloring in linkage tree. 
                 color_cuty = 0., 
                 xdenline = None, # line drawn into linkage tree on x-axis
                 ydenline = None, 
                 plot_value = False, # if true the values are written into the cells of the matrix
                 vmin = None, # min color value 
                 vmax = None, 
                 grid = False, # if True, grey grid drawn around heatmap cells
                 xlabel = None, # label on x-axis
                 ylabel = None, # ylabel
                 xticklabels = None,
                 yticklabels  = None,
                 colormaplabel = None,
                 showdpi = None, # dpi value for plotting with plt.show()
                 dpi = None, # dpi value for savefig
                 figname = None, # if given, figure saved under this name
                 fmt = '.jpg', # format of saved figure
                 maxsize = 150, # largest size the figure can take along both axis
                 cellsize = 0.3, # size of a single cell in the heatmap
                 cellratio = 1., # ratio of cells y/x
                 noheatmap = False, # if True, only tree is plotted
                 row_distributions = None, # for each row in heatmap, add 
                 # a box or a bar plot with plot_distribution, 
                 row_distribution_kwargs = {} # kwargs fro plot_distribution
                 ):
    '''
    Plots a heatmap with tree on x and y
    Motifs can be added to the end of the tree
    Attributions of each column or row can be indicated by additoinal heatmap with different color code
    Other statistics, for example, barplot or boxplots can be added to the y-axis
    Heatmap can be blocked and only tree with motifs and other statistics can be shown
    TODO 
    Put dedrogram and pwm plot in function.
    '''
    
    # Determine size of heatmap
    if heatmat is None:
        Nx = 0
        Ny = np.shape(ydistmat)[0]
    else:
        Ny, Nx = np.shape(heatmat)[0], np.shape(heatmat)[1]
    # Use heatmat as default if xdistmat not specified
    if xdistmat is None:
        xdistmat = np.copy(heatmat)
    if ydistmat is None:
        ydistmat = np.copy(heatmat)
    # either provide similarity matrix as heatmap (measurex = None) or provide 
    # a similarity function from scipy.spatial.distance.pdist
    # If no measure is provided heatmap will be tested whether it is a distance
    # matrix and entries will be rescaled between 0,1 
    if not noheatmap:
        if measurex is not None:
            simatrixX = pdist(xdistmat.T, metric = measurex)
        elif xdistmat is not None:
            
            if not _check_symmetric_matrix(xdistmat):
                sortx = None
            
            if sortx is not None:        
                # checks if similarity matrix or distance matrix
                simatrixX = _transform_similarity_to_distance(xdistmat)
                simatrixX = simatrixX[np.triu_indices(len(simatrixX),1)]
        else:
            sortx = None
            simatrixX = None
                
    if measurey is not None:
        simatrixY = pdist(ydistmat, metric = measurey)
    
    elif ydistmat is not None:
            if not _check_symmetric_matrix(ydistmat):
                sorty = None
            
            if sorty is not None:        
                # checks if similarity matrix or distance matrix
                simatrixY = _transform_similarity_to_distance(ydistmat)
                simatrixY = simatrixY[np.triu_indices(len(simatrixY),1)]
    else:
        sorty = None
        simatrixY = None
    
    
    
    # Generate dendrogram for x and y
    #### NJ not yet included
    if sortx is not None and not noheatmap:
        Zx = linkage(simatrixX, sortx)
        #if combine_cutx > 0:
            #Zx = reduce_z(Zx, combine_cutx)

    if sorty is not None:
        Zy = linkage(simatrixY, sorty) 
        #if combine_cuty > 0:
            #Zy = reduce_z(Zy, combine_cuty)
    
    if not noheatmap and heatmat is not None:
        # Check if maxsize is exceeded and adjust parameters accordingly
        if cellsize*np.shape(heatmat)[1] > maxsize:
            xticklabels = None
            plot_value = False
            yattr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), 
                           maxsize/np.shape(heatmat)[1])
    
        if cellsize*np.shape(heatmat)[0] *cellratio > maxsize:
            yticklabels = None
            plot_value = False
            x_attr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), 
                           maxsize/np.shape(heatmat)[1])
    
    # Determine sizes for the elements that will have to be plotted in figure
    # Plan for extra space for attributes
    xextra = 0.
    if y_attributes is not None:
        y_attributes = np.array(y_attributes, dtype = object)
        xextra = np.shape(y_attributes)[1] + 0.25
    yextra = 0.
    if x_attributes is not None:
        x_attributes = np.array(x_attributes, dtype = object)
        yextra = np.shape(x_attributes)[0] + 0.25
    # Plan for extra space for dendrogram and pwms
    denx, deny, pwmsize, rowdistsize = 0, 0, 0, 0
    if sortx is not None and not noheatmap:
        denx = 3 + 0.25
    if sorty is not None:
        deny = 3+.25
    if pwms is not None:
        pwmsize = 3.25
    # Plan for extra space if row_distributions are added to heatmap
    if row_distributions is not None:
        rowdistsize = 6+ 0.25
    
    basesize = 0
    
    wfig = cellsize*(Nx+xextra+deny+pwmsize+rowdistsize+basesize)
    hfig = cellsize*cellratio*(Ny+yextra/cellratio+denx+basesize)
    
    fig = plt.figure(figsize = (wfig, hfig), dpi = showdpi)
    
    fullw = Nx+xextra+deny+pwmsize+rowdistsize+basesize
    fullh = Ny+yextra+denx+basesize
    # Determine all fractions of figure that will be assigned to each subplot
    left = 0.1
    bottom = 0.1
    width = 0.8
    height = 0.8
    before = width*(deny+pwmsize)/fullw
    after = width*(xextra+rowdistsize)/fullw
    beneath = height*yextra/fullh
    above = height*denx/fullh
    
    # Final width that heatmap will take
    wfac = width * Nx/fullw
    mfac = height * Ny/fullh
    
    if not noheatmap:
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_position([0.1+before,0.1+beneath, wfac, mfac])
        ax.tick_params(which = 'both', bottom = False, labelbottom = False,
                       left = False, labelleft = False)
    
    # plot dendrogram for x axis
    if sortx is not None and not noheatmap:
        axdenx = fig.add_subplot(711)
        axdenx.spines['top'].set_visible(False)
        axdenx.spines['right'].set_visible(False)
        axdenx.spines['bottom'].set_visible(False)
        axdenx.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdenx.set_position([0.1+before,0.9 - above, wfac, 
                             height*(denx-0.25)/fullh])
        dnx = dendrogram(Zx, ax = axdenx, no_labels = True, 
                         above_threshold_color = 'k', 
                         color_threshold = color_cutx, orientation = 'top')
        
        sortx = dnx['leaves']
        heatmat = heatmat[:, sortx]
        if x_attributes is not None:
            x_attributes = x_attributes[:, sortx]
            
        if xticklabels is not None:
            xticklabels = xticklabels[sortx]
            
        if xdenline is not None:
            axdenx.plot([0,len(heatmat[0])*10], [xdenline, xdenline], color = 'r')
    elif heatmat is not None:
        sortx = np.arange(Nx, dtype = int)
    
    sys.setrecursionlimit(100000)    
    
    if sorty is not None:
        axdeny = fig.add_subplot(171)
        axdeny.spines['top'].set_visible(False)
        axdeny.spines['right'].set_visible(False)
        axdeny.spines['left'].set_visible(False)
        axdeny.tick_params(which = 'both', left = False, labelleft = False)
        axdeny.set_position([0.1,0.1+beneath, width*(deny-0.25)/fullw, mfac])
        dny = dendrogram(Zy, ax = axdeny, no_labels = True, 
                         color_threshold = color_cuty, above_threshold_color = 'k',
                         orientation = 'left', get_leaves = True)
        sorty = dny['leaves']
        if heatmat is not None:
            heatmat = heatmat[sorty]
        #axdeny.set_yticks(axdeny.get_yticks()[1:])

        if y_attributes is not None:
            y_attributes = y_attributes[sorty]
            
        if yticklabels is not None:
            yticklabels = yticklabels[sorty]
        if ydenline is not None:
            axdeny.plot([ydenline, ydenline], [0,len(heatmat)*10], color = 'r')
    elif heatmat is not None:
        sorty = np.arange(len(heatmat), dtype = int)
    
    
    # Plot PWMs if given
    if pwms is not None:
        if infocont:
            pwm_min, pwm_max = 0,2
        else:
            pwm_min, pwm_max = 0, int(np.ceil(np.amax([np.amax(np.sum(np.ma.masked_less(pwm,0),axis = -1)) for pwm in pwms])))
        lenpwms = np.array([len(pwm) for pwm in pwms])
        maxlenpwms = np.amax(lenpwms)
        for s, si in enumerate(sorty[::-1]):
            axpwm = fig.add_subplot(len(sorty),1,s+1)
            axpwm.set_position([0.1+before-pwmsize*width/fullw, 
                                0.1+beneath+mfac-mfac*(s+0.9)/len(sorty),
                                (pwmsize-0.25)*width/fullw, 
                                mfac/len(sorty) *0.8])
            pwm = pwms[si]
            if infocont:
                pwm = np.log2((pwms[si]+1e-16)/0.25)
                pwm[pwm<0] = 0
            ppwm = np.zeros((maxlenpwms,4))
            ppwm[(maxlenpwms-lenpwms[si])//2:(maxlenpwms-lenpwms[si])//2+lenpwms[si]] = pwm
            lm.Logo(pd.DataFrame(ppwm, columns = list('ACGT')),
                           ax = axpwm, color_scheme = 'classic')
            axpwm.set_ylim([pwm_min, pwm_max])
            
            axpwm.spines['top'].set_visible(False)
            axpwm.spines['right'].set_visible(False)
            axpwm.spines['left'].set_visible(False)
            axpwm.tick_params(labelleft = False, labelbottom = False, bottom = False)
            if noheatmap and row_distributions is None and yticklabels is not None:
                axpwm.tick_params(labelleft = False, labelright = True, 
                                  labelbottom = False, bottom = False)
                axpwm.set_yticks([(pwm_max+pwm_min)/2])
                axpwm.set_yticklabels(yticklabels[[-s-1]])
    
    # Plot Heatmap
    if not noheatmap:
        if vmin is None:
            vmin = np.amin(heatmat)
        if vmax is None:
            vmax = np.amax(heatmat)
        
        ax.imshow(heatmat, aspect = 'auto', cmap = heatmapcolor, vmin = vmin, 
                  vmax = vmax, origin = 'lower')
        ax.set_yticks(np.arange(len(heatmat)))
        ax.set_xticks(np.arange(len(heatmat[0])))
       
        # add colorbar
        axcol = fig.add_subplot(999)  
        print(vmin, vmax)
        axcol.set_position([0.1+before+wfac+width*0.25/fullw, 
                            0.1+beneath+mfac+height*0.25/fullh, 
                            width*5/fullw, 
                            height*1/fullh])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True,
                          top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', 
                     cmap = heatmapcolor)
        axcol.set_xticks([0,101])
        
        colormapresolution = 1
        
        if colormaplabel is not None:
            axcol.set_xticklabels([colormaplabel[0], colormaplabel[-1]], rotation = 60)
        else:
            axcol.set_xticklabels([round(vmin,colormapresolution), round(vmax,colormapresolution)], rotation = 60)
        
            
        #Add text to heatmap if true
        if plot_value:
            # TODO add fuction to automate to scientific format, use 1 decimal
            # resolution in heatmap, add 10^i to colormap
            if np.amax(np.absolute(heatmat)) > 10:
                heattext = np.array(heatmat, dtype = int)
            elif np.amax(np.absolute(heatmat)) > 1:
                heattext = np.around(heatmat, 1)
            else:
                heattext = np.around(heatmat, 2)
            for c in range(len(heattext[0])):
                for d in range(len(heattext)):
                    ax.text(c,d,str(heattext[d,c]), color = 'k', ha = 'center', fontsize = 6)
        
        
        if grid:
            ax.set_yticks(np.arange(len(heatmat)+1)-0.5, minor = True)
            ax.set_xticks(np.arange(len(heatmat[0])+1)-0.5, minor = True)
            ax.grid(color = 'k', which = 'minor')

        # x_attributes are another heatmap that determines additiona features
        # of the columns
        if x_attributes is not None and not noheatmap:
            # transform the attributes into unique integers
            for x, xunique in enumerate(x_attributes):
                if xunique.dtype != float:
                    xunique = np.unique(xunique)
                    for s, xuni in enumerate(xunique):
                        x_attributes[x, x_attributes[x] == xuni] = s
            
            axatx = fig.add_subplot(717)
            axatx.spines['top'].set_visible(False)
            axatx.spines['bottom'].set_visible(False)
            axatx.spines['right'].set_visible(False)
            axatx.spines['left'].set_visible(False)
            axatx.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False, labelright = False)
            
            axatx.set_position([0.1+before,0.1, wfac, height*(yextra-0.25)/fullh])
            if isinstance(xatt_color,list):
                for xai, xac in enumerate(xatt_color):
                    mask = np.ones(np.shape(x_attributes))
                    mask[xai] = 0
                    mask = mask == 1
                    axatx.imshow(np.ma.masked_array(x_attributes,mask), aspect = 'auto', cmap = xac, vmin = np.amin(x_attributes[xai]), vmax = np.amax(x_attributes[xai]))
            else:
                axatx.imshow(x_attributes, aspect = 'auto', cmap = xatt_color)
            
            axatx.set_xticks(np.arange(len(heatmat[0])))        
            if xattr_name is not None:
                axatx.tick_params(labelright = True)
                axatx.set_yticks(np.arange(np.shape(x_attributes)[0]))
                axatx.set_yticklabels(xattr_name)
                
            # Determine which subplot gets the xlabels
            if xlabel is not None:
                axatx.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
                axatx.set_xlabel(xlabel)
            if xticklabels is not None:
                axatx.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
                axatx.set_xticklabels(xticklabels, rotation  = 90)
        
        elif xlabel is not None and not noheatmap:
            ax.set_xlabel(xlabel)
        elif xticklabels is not None and not noheatmap:
            ax.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
            ax.set_xticklabels(xticklabels, rotation = 90)
                
        
        if y_attributes is not None:
            # Make y attributes integer if they are not float or int
            for y, yunique in enumerate(y_attributes.T):
                if yunique.dtype != float and yunique.dtype != int:
                    yunique = np.unique(yunique)
                    for s, yuni in enumerate(yunique):
                        y_attributes[y_attributes[:,y] == yuni,y] = s
            
            axaty = fig.add_subplot(177)
            axaty.spines['top'].set_visible(False)
            axaty.spines['bottom'].set_visible(False)
            axaty.spines['right'].set_visible(False)
            axaty.spines['left'].set_visible(False)
            axaty.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
            
            axaty.set_position([0.1+before+wfac,0.1+beneath, width*(xextra-0.25)/fullw, mfac])
            if isinstance(yatt_color,list):
                if not (isinstance(yatt_vlim, list) and len(yatt_vlim) == len(yatt_color)):
                    yatt_vlim = [[None, None] for i in range(len(yatt_color))]
                for yai, yac in enumerate(yatt_color):
                    mask = np.ones(np.shape(y_attributes))
                    mask[:,yai] = 0
                    mask = mask == 1
                    axaty.imshow(np.ma.masked_array(y_attributes,mask), aspect = 'auto', cmap = yac, vmin = yatt_vlim[yai][0], vmax =yatt_vlim[yai][1],origin = 'lower')
            else:
                axaty.imshow(y_attributes, aspect = 'auto', cmap = yatt_color, origin = 'lower')
            if yattr_name is not None:
                axaty.tick_params(labeltop = True)
                axaty.set_xticks(np.arange(np.shape(y_attributes)[1]))
                axaty.set_xticklabels(yattr_name, rotation = 270)
            
            # Determine which subplot should have ticklabels
            axaty.set_yticks(np.arange(len(heatmat)))
            if ylabel is not None:
                axaty.tick_params(labelright = True)
                axaty.set_ylabel(ylabel)
            if yticklabels is not None:
                axaty.tick_params(labelright = True, right = True)
                #print('int', yticklabels)
                axaty.set_yticklabels(yticklabels)
        
        elif ylabel is not None and not noheatmap:
            ax.set_ylabel(ylabel)
        elif yticklabels is not None and not noheatmap:
            #print(yticklabels)
            ax.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
            ax.set_yticklabels(yticklabels)
    
    # If given, add box or barplot to right side of the plot
    if row_distributions is not None:
        if not isinstance(row_distributions, list) and not isinstance(row_distributions, np.ndarray):
            row_distributions = list(matrix)
        axdy = fig.add_subplot(188)
        axdy.spines['top'].set_visible(False)
        axdy.spines['right'].set_visible(False)
        axdy.tick_params(which = 'both', left = False, labelleft = False, labelright = True, right = True)
        
        if y_attributes is not None:
            dwidth = mfac*np.shape(y_attributes)[1]*cellsize/wfig++mfac*0.25*cellsize/wfig
        else:
            dwidth = 0
        
        axdy.set_position([0.1+before+wfac+width*(xextra-0.25)/fullw, 0.1+beneath, width*(rowdistsize-0.25)/fullw, mfac])
        if sorty is not None:
            yticklabels = yticklabels[np.argsort(sorty)]
        plot_distribution(row_distributions, yticklabels, vert = False, labelside = 'opposite', ax = axdy, sort = sorty, outname = None, **row_distribution_kwargs)
    

    if figname is not None:
        if not noheatmap:
            figname += '_heatmap'
        fig.savefig(figname+fmt, dpi = dpi, bbox_inches = 'tight')
        print( 'SAVED', figname+fmt, dpi)
    else:
        plt.show()
    plt.close()
    return sortx, sorty



def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def approximate_density(x, bins = 20, sliding_windows = 4, miny=None, maxy = None):
    '''
    Generates density for bins
    '''
    if miny is None:
        miny = np.amin(x)
    if maxy is None:
        maxy = np.amax(x)
    # Determine bin size
    bsize = (maxy-miny)/bins
    
    dens = np.zeros(len(x))
    dcount = np.zeros(len(x))
    for m in range(sliding_windows):
        # create bins
        bins1 = np.linspace(miny-(m+1)*bsize/(sliding_windows+1), maxy-(m+1)*bsize/(sliding_windows+1),bins + 1)
        bins2 = np.linspace(miny+(m+1)*bsize/(sliding_windows+1), maxy+(m+1)*bsize/(sliding_windows+1),bins + 1)
        # determine entries in each bin
        density= np.array([np.sum((x >= bins1[b]) * (x<bins1[b+1])) for b in range(len(bins1)-1)])
        density2= np.array([np.sum((x >= bins2[b]) * (x<bins2[b+1])) for b in range(len(bins2)-1)])
        # scale to max 1
        density = density/np.amax(density)
        density2 = density2/np.amax(density2)
        # assign desities to dens
        for b in range(bins):
            dens[(x >= bins1[b]) * (x<bins1[b+1])] += density[b]
            dens[(x >= bins2[b]) * (x<bins2[b+1])] += density[b]
            dcount[(x >= bins1[b]) * (x<bins1[b+1])] += 1
            dcount[(x >= bins2[b]) * (x<bins2[b+1])] += 1
    # take average over all bins and sliding_windows
    dens = dens/dcount
    
    return dens

def _simple_swarmplot(data, positions, vert = True, unit = 0.4, colormin = None, colormax = None, color = None, cmap = None, connect_swarm = False, scattersort = 'top', scatter_size = None, width = 1., scatter_alpha = 0.8, ax = None):
    '''
    Creates a simple swarmplot with scatter plot with control over all aspects
    in the swarm, such as size, color, connections between distributions
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        if vert:
            fig = plt.figure(figsize = (len(data)*0.4, 3))
        else:
            fig = plt.figure(figsize = (3, len(data)*0.4))
        ax = fig.add_subplot(111)
    
    if colormin is None and not isinstance(color, str):
        colormin = np.amin(color)
    if colormax is None and not isinstance(color, str):
        colormax = np.amax(color)
    
    if connect_swarm and len(np.shape(data)) > 1:
        xposses = []
        randomshift = np.random.random(len(data[0])) # usese the same random 
        # shifts on x for all distributions
    
    for i, set1 in enumerate(data):
        set1 = np.array(set1)
        if scattersort == 'top':
            setsort = np.argsort(set1)
        else:
            setsort = np.argsort(-set1)
        
        if color is None:
            cmap = cm.twilight
            sccolor = np.ones(len(setsort))*0.25
        elif isinstance(color, str):
            sccolor = np.array([color for ci in range(len(setsort))])
        else:
            sccolor = (color-colormin)/(colormax-colormin)
            
        if scatter_size is None:
            scsize = 0.2*np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
        elif isinstance(scatter_size, float) or isinstance(scatter_size, int):
            scsize = scatter_size * np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
        else:
            scsize = np.sqrt(scatter_size/3.)
            scsize = (((sizemax-sizemin)*(scsize - np.amin(scsize))/(np.amax(scsize) - np.amin(scsize))) + sizemin)
            scsize *= plt.rcParams['lines.markersize'] ** 2.
            
        
        dens = approximate_density(set1)
        if connect_swarm and len(np.shape(data)) > 1:
            randx = positions[i] + dens *width/2 * (randomshift-0.5)
        else:
            randx = positions[i] + dens * width * (np.random.random(len(setsort))-0.5) # + width/2 * simple_beeswarm(set1, nbins = 40) #
        
        if vert: 
            ax.scatter(randx[setsort], set1[setsort], cmap= cmap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
        else:
            ax.scatter(set1[setsort], randx[setsort], cmap= cmap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
        if connect_swarm and len(np.shape(data)) > 1:
            xposses.append(randx)
    
    if connect_swarm and len(np.shape(data)) > 1:
        xposses=np.array(xposses)
        for j, setj in enumerate(np.array(data).T):
            if vert:
                ax.plot(xposses[:,j], setj, color  = 'grey', alpha = 0.5, lw = 0.5)
            else:
                ax.plot(setj, xposses[:,j], color  = 'grey', alpha = 0.5, lw = 0.5)
    if return_fig:
        return fig
    
def _colorbar(cmap, ticklabels = None, vert = True, ratio = 3, tickpositions = None, ax = None):
    '''
    Generates heatmap for cmap
    TODO 
    Use in other functions as well. 
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        if vert:
            fig = plt.figure(figsize = (1,ratio))
        else:
            fig = plt.figure(figsize = (ratio, 1))
        ax = fig.add_subplot(111)
    if tickpositions is not None:
        if tickpositions == 'left':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = True, labelleft = True, right=False, labelright = False)
        if tickpositions == 'right':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = False, labelleft = False, right=True, labelright = True)
        if tickpositions == 'bottom':
            ax.tick_params(bottom = True, labelbottom = True, labeltop = False, top = False, left = False, labelleft = False, right=False, labelright = False)
        if tickpositions == 'top':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False, right=False, labelright = False)
        else:
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = False, labelleft = False, right=False, labelright = False)
            
    ax.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = cmap)
    if ticklabels is not None:
        ax.set_xticks(np.linspace(0,101, len(ticklabels)))
        if vert:
            ax.set_yticklabels(ticklabels)
        else:
            ax.set_xticklabels(ticklabels, rotation = 90)
    
    if return_fig:
        return fig

def plot_distribution(
    data, # list of lists with values for boxplots, can be list of list of lists 
    # if multiple boxplots for each position are plotted
    modnames, # names of the distributions
    vert = True, # if vertical or horizontal 
    labelside = 'default', # default is bottom for vertical and left for horizontal
    ax = None, # if ax is None a figure object is produced
    sort = None, # order of data points
    split = 1, # splits data int split parts and plots them split boxplots 
    # for each position
    legend_labels = None, # labels for the legend, legend can be produced if 
    # split> 1, for boxplots at the same position with different facecolors
    legend_above = True, # if legend should be placed above the plot
    # TODO: make function to choose legend positions above, below, upper left,
    # upper right
    xwidth = 0.6, # width of every position in figure
    height = 4, # height of figure
    width = 0.8, # width of boxplot
    show_mean = False, 
    showfliers = False, 
    showcaps = True, 
    facecolor = None, # color or list of colors if split > 1
    mediancolor = None, # color or list of colors if split > 1
    grid = True, 
    swarm = False, # adds swarmplot to boxplot
    barplot = False, # create barplot instead of boxplot, either use single
    # value input of function uses mean
    scatter_color = 'grey', 
    scatter_colormap = cm.jet, 
    scatter_alpha = 0.8, 
    scatter_size = 0.5, 
    connect_swarm = False, # can connect the dots in the swarm plot between distributions
    scattersort = 'top', # sorts scatter plot dots by value
    ylim = None, 
    ylabel = None, 
    sizemax = 2, # max size for scatters
    sizemin = 0.25, # min size for scatters
    colormin = None, 
    colormax = None, 
    dpi = 200, 
    savedpi = 200, 
    outname = None, # Name of figure, if given, figure will be saved
    fmt = 'jpg'):
    
    if sort is not None:
        positions = np.argsort(sort)
        modnames = np.array(modnames)[sort]
    else:
        positions = np.arange(len(data))
    
    fcolor = None # color for every boxplot at every position derived from
    # facecolors
    # Adjust parameters to split option
    if split > 1: # split boxplots will be plotted at each position
        if len(data) == split: # if data was given as list of lists for each split
            data = [m for ma in data for m in ma]
        
        if width * split >1: # adjust width of individual boxplots
            width = width/split
        
        # determine new positions of boxplots
        positions = []
        for s in range(split):
            if sort is None:
                positions.append(np.arange(int(len(data)/split)) + width*s - (split*width/2) + width/2)
            else:
                positions.append(np.argsort(sort) + width*s - (split*width/2) + width/2)
        positions = np.concatenate(positions)
        
        # create array with colors for each boxplot
        if isinstance(facecolor, list):
            if len(facecolor) == split:
                fcolor = [facecolor[c] for c in range(split) for j in range(int(len(data)/split))]
            else:
                fcolor = [facecolor[c] for c in range(len(data))]
        
        # same for median color
        if mediancolor is not None:
            if isinstance(mediancolor, list):
                if len(mediancolor) == split:
                    mediancolor = [mediancolor[c] for c in range(split) for j in range(int(len(data)/split))]
                else:
                    mediancolor = [mediancolor[c] for c in range(len(data))]
            else:
                mediancolor = [mediancolor for c in range(len(data))]
            
    # if median color is not None, need to replicate into list
    if mediancolor is not None:
        if not isinstance(mediancolor, list):
            mediancolor = [mediancolor for mc in range(len(data))]
    
    return_ax = False # if ax given: function returns manipulated subplot
    if ax is None:
        if vert:
            fig = plt.figure(figsize = (len(modnames)*xwidth, height), dpi = dpi)
        else:
            fig = plt.figure(figsize = (height, len(modnames)*xwidth), dpi = dpi)
        ax = fig.add_subplot(111)
        ax.set_position([0.1,0.1,0.8,0.8])
    else: 
        return_ax = True
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ylabel is not None:
        if vert:
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(ylabel)
        
    data = list(data)
    
    if swarm: # add fancy swarmplot with different option on top of boxplot
        _simple_swarmplot(data, positions, vert = vert, colormin = colormin, colormax = colormax, color = scatter_color, cmap = scatter_colormap, connect_swarm = connect_swarm, scattersort = scattersort, scatter_size = scatter_size, width = width*0.9, scatter_alpha = scatter_alpha, ax = ax)
            
        # generate colorbar
        if ((scatter_color is not None) and (not isinstance(scatter_color, str))):
            axcol = fig.add_subplot(911)
            axcol.set_position([0.6,0.925,0.3, 0.05])
            _colorbar(scatter_colormap, ticklabels = [round(colormin,1), round((colormin+colormax)/2,1), round(colormax,1)] , tickpositions = 'top', ax = axcol)
            
    
    if barplot:
        if fcolor is not None:
            barcolor = fcolor
        else:
            barcolor = 'grey'
        if vert:
            bplot = ax.bar(positions, np.mean(data,axis = 1), width = width*0.9, color = barcolor, linewidth = 1)
        else:
            if len(np.shape(data)) > 1:
                data = np.mean(data, axis = 1)
            bplot = ax.barh(positions, data, height = width*0.9, color = barcolor, linewidth = 1)
            ax.set_ylim([np.amin(positions)-0.5, np.amax(positions)+0.5])
        
        # create a legend()
        if isinstance(facecolor, list) and legend_labels is not None:
            handles = []
            for f, fcol in enumerate(facecolor):
                patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                handles.append(patch)
            ax.legend(handles = handles)
    else:
        if facecolor is None or fcolor is not None:
            boxplotcolor = (0,0,0,0)
        else:
            boxplotcolor = facelolor
        
        bplot = ax.boxplot(data, positions = positions, vert = vert, showcaps=showcaps, patch_artist = True, boxprops={'facecolor':boxplotcolor}, showfliers=showfliers, whiskerprops={'linewidth':1}, widths = width,zorder = 4)
    
        if fcolor is not None:
            for patch, color in zip(bplot['boxes'], fcolor):
                patch.set_facecolor(color)
                fc = patch.get_facecolor()
                patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
            # create legend()
            if isinstance(facecolor, list) and legend_labels is not None:
                handles = []
                for f, fcol in enumerate(facecolor):
                    patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                    handles.append(patch)
                if legend_above:
                    ax.legend(handles = handles,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3)
                else:
                    ax.legend(handles = handles)
                
                
        if mediancolor is not None:
            for mx, median in enumerate(bplot['medians']):
                median.set_color(mediancolor[mx])
    
    if ylim is not None:
        if vert:
            ax.set_ylim(ylim)
        else:
            ax.set_xlim(ylim)
    
    if show_mean:
        if vert:
            ax.plot(np.sort(positions), [np.mean(data[s]) for s in np.argsort(positions)], color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
        else:
            ax.plot([np.mean(data[s]) for s in np.argsort(positions)],np.sort(positions), color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
    
    if vert:
        if labelside =='opposite':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = True)
        ax.set_xticks(np.arange(len(modnames)))
        ax.set_xticklabels(modnames, rotation = 90)
        if grid:
            ax.grid(axis = 'y')
    else:
        if labelside =='opposite':
            ax.tick_params(left = False, labelleft = False, labelright = True)
        ax.set_yticks(np.arange(len(modnames)))
        ax.set_yticklabels(modnames)
        if grid:
            ax.grid(axis = 'x')
    
    if return_ax:
        return ax
    
    if outname is None:
        #fig.tight_layout()
        plt.show()
    else:
        fig.savefig(outname+'_distribution.'+fmt, dpi = savedpi, bbox_inches = 'tight')


def vulcano(fc, pv, figname = None, logfc = False, logpv = False, fccutoff=None, pvcutoff=None, annotate_significant = None, 
            colors = None, cmap = 'viridis', ax = None):
    '''
    Plot vulcano plot from fold change and p-values
    Parameters
    ----------
    fc : np.ndarray
        Fold changes
    pv : np.ndarray
        P-values
    xcutoff : float
        cutoff for significant fold changes
    ycotoff : float
        cutoff for significant pvalues
    annotate_significant: list
        names of data points, puts text next to significant data points
    colors = list
        list of colors for each data point
        
    '''
    if logfc:
        fc = np.log2(fc)
        if fccutoff is not None:
            fccutoff = np.log2(fccutoff)
    if logpv: 
        pv = -np.log10(pv)
        if pvcutoff is not None:
            pvcutoff = -np.log10(pvcutoff)
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figname, figsize = (4,4), dpi = 200)
        ax = fig.add_subplot(111)
    maskx = None
    masky = None
    if fccutoff is not None:
        maskx = np.absolute(fc) > fccutoff
    if pvcutoff is not None:
        masky = pv > pvcutoff
    
    if colors is not None:
        sort = np.argsort(colors)
        ap = ax.scatter(fc[sort], pv[sort], cmap = cmap, c = colors[sort], alpha = 0.6)
        fig.colorbar(ap, aspect = 2, pad = 0, anchor = (0,0.9), shrink = 0.15)
    else:
        if masky is not None and maskx is not None:
            ax.scatter(fc[~(maskx*masky)], pv[~(maskx*masky)], c = 'grey', alpha = 0.3)
            ax.scatter(fc[maskx*masky], pv[maskx*masky], c='maroon', alpha = 0.8)
        elif masky is not None:
            ax.scatter(fc[~masky], pv[~masky], c = 'grey', alpha = 0.3)
            ax.scatter(fc[masky], pv[masky], c='rosybrown', alpha = 0.8)
        elif maskx is not None:
            ax.scatter(fc[~maskx], pv[~maskx], c = 'grey', alpha = 0.3)
            ax.scatter(fc[maskx], pv[maskx], c='rosybrown', alpha = 0.8)
        else:
            ax.scatter(fc, pv, c = 'grey', alpha = 0.3)
            
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if figname is not None:
        ax.set_title(figname)
    ax.set_xlabel('Log2 Fold change')
    ax.set_ylabel('Log10 p-value')
    
    if annotate_significant is not None and masky is not None and maskx is not None:
        for s in np.where(masky*maskx)[0]:
            if fc[s] < 0:
                ax.text(fc[s], pv[s], text[s], ha = 'left', size = 8)
            else:
                ax.text(fc[s], pv[s], text[s],ha = 'right', size = 8)
    if return_fig:
        return fig




def plotHist(x, y = None, figsize = (3.5, 3.5), dpi = 150, 
             xcolor='navy', add_yaxis = False, add_median = False, xalpha= 0.5,
             ycolor = 'indigo', yalpha = 0.5, addcumulative = False, 
             bins = None, xlabel = None, title = None, logx = False, 
             logy = False, logdata = False, ax = None):
    '''
    Generates figure with histogram
    '''
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        axp = fig.add_subplot(111)
    else:
        axp = ax
    axp.spines['top'].set_visible(False)
    axp.spines['right'].set_visible(False)
    
    if logdata:
        x = np.log10(x+1)
    
    a,b,c = axp.hist(x, bins = bins, color = xcolor, alpha = xalpha)
    if y is not None:
        ay,by,cy = axp.hist(y, bins = bins, color = ycolor, alpha = yalpha)
    
    if addcumulative != False:
        axp2 = axp.twinx()
        axp2.spines['top'].set_visible(False)
        axp2.spines['left'].set_visible(False)
        axp2.tick_params(bottom = False, labelbottom = False)
        axp2.set_yticks([0.25,0.5,0.75,1])
        axp2.set_yticklabels([25,50,75,100])
        if addcumulative == 2:
            addcumulative = 1
            ag_,bg_,cg_ = axp2.hist(x, color = 'maroon', alpha = 1,
                                    density = True, bins = bins, 
                                    cumulative = -1, histtype = 'step')
        ag,bg,cg = axp2.hist(x, color = xcolor, alpha = 1, density = True,
                             bins = bins, cumulative = addcumulative, 
                             histtype = 'step')
        if y is not None:
            agy,bgy,cgy = axp2.hist(y, color = ycolor, alpha = 1, 
                                    density = True, bins = bins, 
                                    cumulative = addcumulative, histtype = 'step')
            
    if add_yaxis:
        print('yaxis',np.amax(a))
        axp.plot([0,0], [0, np.amax(a)], c = 'k', zorder = 5)
    if add_median:
        median = np.median(x)
        axp.plot([median, median], [0, np.amax(a)], c = 'firebrick', zorder = 5)
    if logx:
        if addcumulative:
            axp2.set_xscale('symlog')
        axp.set_xscale('symlog')
        
    if logy:
        axp.set_yscale('symlog')
    
    if xlabel is not None:
        axp.set_xlabel(xlabel)
    if title is not None:
        axp.set_title(title)
    return fig




def scatter3D(x,y,z, axis = True, color = None, cmap = None, 
              xlabel = None, ylabel = None, zlabel=None, alpha = 0.9, 
              diag = False):
    '''
    Generates figure with 3D scatter plot
    '''
    
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(111, projection='3d') #plt.axes(projection='3d')
    xlim = np.array([np.amin(x), np.amax(x)])
    ylim = np.array([np.amin(y), np.amax(y)])
    zlim = np.array([np.amin(z), np.amax(z)])
    lrat = 0.5
    if axis:
        # plot axis in there
        xlim0, ylim0, zlim0 = xlim * lrat, ylim * lrat, zlim * lrat
        ax.plot3D(xlim0,[0,0],[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],ylim0,[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],[0,0],zlim0, color = 'k', lw = 1)
    if diag:
        maxlim = np.array([xlim, ylim,zlim])*lrat
        maxlim = [np.amax(maxlim[:,0]), np.amin(maxlim[:,1])]
        ax.plot3D(maxlim, maxlim, maxlim, color = 'maroon', lw = 1)
    # plot a scatterplot in 3d
    ax.scatter3D(x, y, z, c=color, cmap=cmap, lw=0, alpha = alpha, s = 3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    lrat = 0.75
    ax.set_xlim(xlim*lrat)
    ax.set_xlim(ylim*lrat)
    ax.set_xlim(zlim*lrat)
    ax.view_init(elev=25, azim=-49)
    return fig



def plot_bars(x, width = 0.8, xticklabels=None, xlabel = None, ylabel=None, 
        ylim=None, color = None, figsize = (3.5,3.5), labels = None, 
        title = None, horizontal = False):
    
    """
    Plot bar plot figure 
    
    Parameter
    ---------
    x : list or np.array, shape = (n_bars,) or (n_bars, n_models), or 
    (n_bars, n_models, n_conditions)
        If three x of three dimensions, will generate a different bar plot for 
        the last dimension
    
    Return
    ------
    
    fig : matplotlib.pyplot.fig object
    
    TODO: 
        use with _bar_plot for individual barplots in subplots
    
    """

    x = np.array(x)
    positions = np.arange(np.shape(x)[0])
    xticks = np.copy(positions)
    
    if len(np.shape(x)) >1:
        n_models = np.shape(x)[1]
        bst = -width/2
        width = width/n_models
        shifts = [bst + width/2 + width *n for n in range(n_models)]
        positions = [positions+shift for shift in shifts]
        if color is None:
            color = [None for i in range(np.shape(x)[1])]
   
    if len(np.shape(x)) > 2:
        if horizontal: 
            fig = plt.figure(figsize = (figsize[0]* np.shape(x)[-1], 
                figsize[1]))
        else:
            fig = plt.figure(figsize = (figsize[0], 
                figsize[1] * np.shape(x)[-1]))
        
        for a in range(np.shape(x)[-1]):
            if horizontal:
                ax = fig.add_subplot(1, np.shape(x)[-1], a+1)
            else:
                ax = fig.add_subplot(np.shape(x)[-1], 1, a+1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p,a],  width = width, color = color[p], 
                        label = plotlabel)
            
            if labels is not None:
                ax.legend()
        
            ax.grid()
            
            if horizontal: 
                if a == 0:
                    if ylabel is not None:
                        ax.set_ylabel(ylabel)    
                if xticklabels is not None:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation = 60)
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                
            else:
                if a + 1 == np.shape(x)[-1]:
                    if xticklabels is not None:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, rotation = 60)
                    if xlabel is not None:
                        ax.set_xlabel(xlabel)
                else:
                    ax.tick_params(labelbottom = False)
            
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
            
            if ylim is not None:
                ax.set_ylim(ylim)
            
            if title is not None:
                ax.set_title(title[a])
        
        if horizontal:
            plt.subplots_adjust(wspace=0.2)
        else:
            plt.subplots_adjust(hspace=0.2)
    else:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(np.shape(x))>1:
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p], width = width, color = color[p], 
                        label = plotlabel)
        else:
            ax.bar(positions, x, width = width, color = color, label = label)
        if labels is not None:
            ax.legend()
    
        ax.grid()
        
        if xticklabels is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation = 60)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if ylim is not None:
            ax.set_ylim(ylim)
    
    return fig


def AssignDensity(X,Y, subsample = 3000, log = False):
    density = gaussian_kde(np.array([X,Y])[:, np.random.permutation(len(X))[:subsample]])(np.array([X,Y]))
    if log:
        density = np.log2(1+density)
    return density

def ContourMap(X,Y, log = False, levels = 8, lw = 0.5, color = 'k', ax = None):
    '''
    Creates contour map for distributions of X,Y
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = (3.8,3.8), dpi = dpi)
        ax.add_subplot(111)
        xlim, ylim = None, None
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
    density = AssignDensity(X,Y, log = log)
    ax.tricontour(X,Y, density, levels=levels, linewidths=lw, colors=color)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if return_fig:
        return fig

    

def BoxPlotfromBins(X, Y, start=None, end=None, bins=10, axis = 'x', ax = None, zorder = 0):
    '''
    Uses defined bins to create box plots along defined axis
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = (3.8,3.8), dpi = dpi)
        ax.add_subplot(111)
    if axis == 'x':
        vert = True
        maskdata = X
        valuedata = Y
    if axis == 'y':
        vert = False
        maskdata = Y
        valuedata = X
    
    if start is None:
        start = np.amin(maskdata)
    if end is None:
        end = np.amax(maskdata)
    if not isinstance(bins, np.ndarray):
        bins = np.linspace(start, end, bins+1)
    
    wticks = (bins[1:]+bins[:-1])/2
    boxes = [valuedata[(maskdata>=windows[n]) * (maskdata<=windows[n+1])] for n in range(len(bins)-1)]
    
    ax.boxplot(boxes, positions = wticks, vert = vert)
    if vert: 
        ax.set_xticks(wticks)
    else:
        ax.set_yticks(wticks)
    
    if return_fig:
        return fig

def scatterPlot(X, Y, title = None, xlabel = None, ylabel = None, include_fit = False,
                include_mainvar = False, diagonal = False, plot_axis = None, 
                boxplot_x = None, boxplot_y = None, contour = False, 
                pos_neg_contour = False, color=None, edgecolor = 'silver',
                cmap = None, sort_color = 'abshigh', color_density = False, 
                vlim = None, sizes = None, alpha = None, lw = None, yticklabels = None, 
                yticks = None, xticklabels = None, xticks = None, grid = False, 
                xlim = None, ylim =None, xscale = None, legend = False, add_text = None, 
                yscale = None, ax = None, dpi = 200):
    '''
    Creates fancy scatterplot with additional features
    
    Parameters
    ----------
    X : np.ndarray
    
    Y : np.ndarray
    
    include_fit : boolean
        If linear regression fit should be shown 
    include_mainvar: boolean
        If main variance vector should be shown
    diagonal : -1,0,1,2
        -1 creates negative diagonal, 2 creates both diagonals
    plot_axis : string
        x, y, both
    boxplot_x: int, or triplet
        Either n_bins or (start, end, n_bins). Creates boxplot for bins
        along x
    boxplot_y: int, or triplet
        Either n_bins or (start, end, n_bins). Creates boxplot for bins
        along_y
    contour : int
        creates contour with 'contour' layers and places it on top of scatter plot
    pos_neg_contour: 
        creates distinct contours for dots within 10% and 90% percentile of
        negative and positive values with 'contour' layers and places it on top of scatter plot
    color_density:
        colors scatters by gaussian density
    add_text: list of triplets
        x_cor, y_cor, string

    '''

    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = (3.8,3.8), dpi = dpi)
        ax = fig.add_subplot(111)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if boxplot_x:
        if isinstance(boxplot_x, int):
            start, end, bins = None, None, boxplot_x
        else:
            start, end, bins = boxplot_x
        BoxplotfromBins(X,Y, start = start, end = end, bins = bins, ax = ax, zorder = 1)
    
    if boxplot_y:
        if isinstance(boxplot_y, int):
            start, end, bins = None, None, boxplot_y
        else:
            start, end, bins = boxplot_y
        BoxplotfromBins(X,Y, start = start, end = end, bins = bins, ax = ax, zorder = 1, axis = 'y')
    
    if color_density:
        color = AssignDensity(X,Y, log = color_density == 'log')
    
    if vlim is None:
        vmin, vmax = None, None

    if color is not None:
        if isinstance(color, list) or isinstance(color, np.ndarray):
            if vlim is None:
                if np.amin(color) < 0:
                    vlim = np.amax(np.abs(color))
                    vmin, vmax = -vlim, vlim
                else:
                    vmin, vmax = 0, np.amax(color)

            if sort_color == 'high':
                sort = np.argsort(color)
            elif sort_color == 'low':
                sort = np.argsort(-color)
            elif sort_color == 'abshigh':
                sort = np.argsort(np.absolute(color))
            else:
                sort = np.arange(len(color))
            X, Y, color = X[sort], Y[sort], color[sort]

    if sizes is not None:
        if isinstance(sizes, int) or isinstance(sizes, float):
            sizes = np.ones(len(X)) * sizes
        if isinstance(color, list) or isinstance(color, np.ndarray):
            sizes = np.array(sizes)[sort]
    
    ax.scatter(X,Y, s = sizes, cmap = cmap, c=color, alpha = alpha, vmin = vmin, vmax = vmax, edgecolor = edgecolor, lw = lw, label = 'R:'+str(round(pearsonr(X,Y)[0],2)), zorder = 0)
    if legend:
        ax.legend()
        
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if grid:
        ax.grid(which='minor')
    
    limx, limy = ax.get_xlim(), ax.get_ylim()
    lim = [max(limx[0], limy[0]), min(limx[1], limy[1])]
    
    if diagonal:
        if diagonal == 2:
            ax.plot(lim, lim, color = 'dimgrey', ls = '-')
            ax.plot(lim, -lim, color = 'dimgrey', ls = '-')
        else:
            ax.plot(lim, lim, color = 'dimgrey', ls = '-')
            ax.plot(lim, diagonal * np.array(lim), color = 'dimgrey', ls = '-')
    if plot_axis: 
        if plot_axis == 'x' or plot_axis == 'both':
            ax.plot(limx, [0.,0], color = 'grey')
        if plot_axis == 'y' or plot_axis == 'both':
            ax.plot([0.,0], limy, color = 'grey')
    
    if contour:
        ContourMap(X,Y, ax = ax)

    elif pos_neg_contour and color is not None:
        contcolors = cm.get_cmap(cmap)(vlim)
        ContourMap(X[X>0],Y[X>0], ax = ax, color = contcolors[0])
        ContourMap(X[X<0],Y[X<0], ax = ax, color = contcolors[1])
        
    
    if include_fit:
        lr = linear_model.LinearRegression().fit(X.reshape(-1,1), Y)
        ax.plot(np.array(limx), lr.predict(np.array(limx).reshape(-1,1)), color = 'r', zorder = 1)

    if include_mainvar:
        centerx, centery = np.mean(X), np.mean(Y)
        maindir, u, v = np.linalg.svd(np.array([X-centerx, Y-centery]), full_matrices=False)
        maindir = maindir[:,0]
        slope = maindir[1]/maindir[0]
        bias = centery-slope*centerx
        ax.plot(np.array(limx), np.array(limx)*slope + bias, color = 'darkred', zorder = 1)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        if xticks is None:
            ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        if yticks is None:
            ax.set_yticks(yticklabels)
        ax.set_yticklabels(yticklabels)
    
    if xscale is not None:
        ax.set_xscale(xscale)
    
    if yscale is not None:
        ax.set_xscale(yscale)
        
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    if return_fig:
        return fig

def plot_scatter(X, Y, ax_size = 3.5, dpi = 100, titles=None, xlabel=None, ylabel=None, outname=None, include_fit=True, include_mainvar=False, **kwargs):
    """
    Receives list of Xs and Ys and creates a grid of scatter plots for them using scatterPlot.

    Parameters:
    ----------
    X : list
        list of X values for scatter plots.
    Y : list
        list of Y values for scatter plots.
    ax_size : float
        Size of each subplot.
    titles : list, optional
        Titles for each subplot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    outname : str, optional
        If provided, saves the figure to this file.
    include_fit : bool, optional
        Whether to include a linear regression fit in the scatter plots.
    include_mainvar : bool, optional
        Whether to include the main variance vector in the scatter plots.
    **kwargs : dict
        Additional keyword arguments passed to scatterPlot.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The created figure if `outname` is not provided.
    """
    
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")
    
    n = len(X)

    if n > 100:
        raise Exception(f'{n} examples is too large. Maximum allowed is 100.')

    x_col = int(np.sqrt(n))
    y_row = int(n / x_col) + int(n % x_col != 0)
    fig = plt.figure(figsize=(x_col * ax_size, y_row * ax_size), dpi=dpi)

    for e in range(n):
        ax = fig.add_subplot(y_row, x_col, e + 1)
        pcorr = pearsonr(X[e], Y[e])[0]
        scatterPlot(X[e], Y[e], ax=ax, include_fit=include_fit, include_mainvar=include_mainvar, **kwargs)

        if titles is not None:
            ax.set_title(titles[e])

    if xlabel is not None:
        fig.text(0.5, 0.05 - 0.05 / y_row, xlabel, ha='center')

    if ylabel is not None:
        fig.text(0.05 - 0.05 / x_col, 0.5, ylabel, va='center', rotation='vertical')

    return fig


def plot_lines(y, x = None, xticks = None, xticklabels = None, color = None,
               cmap = 'Set1', marker = None, ylabel = None, grid = False,
               legend_names = None, legend_outside = False, figsize = None,
               unit = 0.3, yscale = None, ax = None, ylim = None, plot_xaxis=True):
    
    '''
    Plots lines with markers
    Parameters
    ----------
    y : np.ndarray or list
        if list of lists each one will be plotted separately
    '''
    
    if not isinstance(y[0],list):
        y = [y]
    leny = [len(yi) for yi in y]
    
    legend = True
    if legend_names is None:
        legend = False
        legend_names = [None for i in range(len(y))]
    
    if x is None:
        x = [np.arange(len(yi)) for yi in y]
    
    if color is not None:
        if not isinstance(color,list):
            color = [color for c in range(len(y))]
    else:
        cmap = mpl.colormaps.get_cmap(cmap)
        color = [cmap(c) for c in np.linspace(0,1,len(y))]
    
    if marker is not None:
        if not isinstance(marker,list):
            marker = [marker for c in range(len(y))]
    
    if xticklabels is not None:
        if xticks is None:
            xticks = np.arange(len(xticklabels))
    elif x is not None:
        if xticks is None:
            xticks = np.unique(np.concatenate(x))
            if (np.diff(xticks) != np.diff(xticks)[0]).any(): 
                xticks = np.linspace(xticks[0], xticks[-1], 5)
    
    if figsize is None:
        if xticks is not None:
            figsize = ((xticks[-1]-xticks[0])*unit, 3.5)
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    for i, xi in enumerate(x):
        ax.plot(xi, y[i], color = color[i], marker = marker[i], ls = '-', lw = 1)
    
    if grid:
        ax.grid()
    
    if legend:
        handles = []
        for f, fcol in enumerate(color):
            patch = mpatches.Patch(color=fcol, label=legend_names[f])
            handles.append(patch)
        if legend_outside:
            ax.legend(handles = handles,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
        else:
            ax.legend(handles = handles)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation = 90)
    
    xlim = [np.amin(np.concatenate(x))-0.5, np.amax(np.concatenate(x))+0.5]
    if plot_xaxis: 
        ax.plot(xlim,[0,0], color = 'grey')
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if yscale is not None:
        ax.set_yscale(yscale)
   
    if return_fig:
        return fig


def piechart(percentages, labels = None, colors = None, cmap = 'tab10', cmap_range=[0,1], explode_size = None, explode_indices = None, labels_on_side = False, explode_color = None, ax = None):
    '''
    Plots piechart with some options
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = (3.,3.), dpi = 200)
        ax = fig.add_subplot(111)
    
    if labels is None:
        labels = np.arange(len(percentages)).astype(str)
        
    if colors is None:
        colors = plt.get_cmap(cmap)(np.linspace(cmap_range[0],cmap_range[1], len(percentages)))
        
    explode = None
    if explode_indices:
        explode = np.zeros(len(percentages))
        # Have Outside entries stick out
        if explode_size is None:
            explode_size = 0.1
        explode[explode_indices] = explode_size
        if explode_color is not None:
            if isinstance(explode_color, str) and not isinstance(colors[0], str):
                explode_color = mpl.colors.to_rgba(explode_color)
            colors[explode_indices] = explode_color
    
    if labels_on_side:
        wedges, texts = ax.pie(percentages, colors = colors, explode = explode) 
        bbox_props = dict(boxstyle="square,pad=0.", fc="w", ec=None, lw=0.)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)
    else:
        ax.pie(percentages, labels=labels, colors = colors, explode = explode)
    if return_fig:
        return fig
