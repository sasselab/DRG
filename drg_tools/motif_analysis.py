# motif_analysis.py
'''
Contains functions to extract, compare, modify and summarize sets of 2D arrays (motifs)
'''

import numpy as np
import sys, os
from scipy.stats import pearsonr 
import time
from joblib import Parallel, delayed
import torch 
import torch.nn.functional as F
from .stats_functions import correlation_to_pvalue

def reverse(ppm):
    '''
    Generates the reverse complement of a pwm
    '''
    rppm = np.copy(ppm)
    rppm = rppm[::-1][:,::-1]
    return rppm

def reverse_torch(ppm, clone =True):
    '''
    Generates the reverse complement of a pwm
    '''
    if clone:
        ppm = ppm.clone()
    return ppm.flip([-1,-2])

def determine_best_unique_matches(similarity):
    '''
    dependent best match. ppms in ppms_ref can only be assiged to one other.
    Parameters
    ----------
    similarity : similarity matrix
        i.e. the best matches are high values
    Returns
    -------
    bestmatch : np.ndarray
        indices of entris in axis 1 for entries in axis 0
    '''
    bestmatch = -np.ones(np.shape(similarity)[0], dtype = int)
    n_refs = np.arange(np.shape(similarity)[1], dtype = int)
    n_ppms = np.arange(np.shape(similarity)[0], dtype = int)
    asar = np.copy(similarity)
    while True:
        maxr = (int(np.argmax(asar)/np.shape(asar)[0]), np.argmax(asar)%np.shape(asar)[1])
        bestmatch[n_ppms[maxr[0]]] = n_refs[maxr[1]]
        asar = np.delete(asar, maxr[0], axis = 0)
        asar = np.delete(asar, maxr[1], axis = 1)
        n_refs = np.delete(n_refs, maxr[1])
        n_ppms = np.delete(n_ppms, maxr[0])
        if len(n_refs) == 0 or len(n_ppms) == 0:
            break
    return bestmatch

def padded_weight_conv1d(qpm, ppm, min_overlap, padding = 0.25, centered=False, standard=False, verbose = False):
    '''
    Adds padding to weights so that all of the qpm is covered and all positions
    that are part of either one motif are compared to each other.
    
    Parameters
    ----------
    qpm : 
        shape=(N_0,4,Length_0)
    ppm : 
        shape=(N_1,4,Length_1)
    min_overlap : 
        minimal overlap between qpm and ppm 
    padding : 
        value to fill padded areas
        
    '''
    cha = qpm.shape[-2]
    lq = qpm.shape[-1]
    lp = ppm.shape[-1]
    # padding for both motifs
    qpad = lp - min_overlap
    ppad = qpad - (lp -lq)
    qpmp = F.pad(qpm, (qpad, qpad), 'constant', padding)
    ppmp = F.pad(ppm, (ppad, ppad), 'constant', padding)
    lpp = ppmp.shape[-1]
    lqp = qpmp.shape[-1]
    # collect results from convolutions, inputs to convs have same length
    res = []
    for i in range(lqp-lp+1):
        # compute start and end for parts of motifs to compare
        qstart, qend = min(i,qpad), max(qpad+lq,i+lp)
        pstart, pend = ppad+min(0,(qpad-i)), max(lpp-ppad,lpp-i)
        with torch.no_grad():
            # Standardized values will be assigned to qpmp if not cloned
            qpmpin = qpmp[...,qstart:qend].detach().clone()
            ppmpin = ppmp[...,pstart: pend].detach().clone()
            if centered:
                qpmpin -= torch.mean(qpmpin,dim=(-1,-2),keepdim = True)
                ppmpin -= torch.mean(ppmpin,dim=(-1,-2),keepdim = True)
            if standard:
                # Use this for norm2 because is divided to every entry and summed
                # Need to cancel n basically
                qpmpin /= torch.sqrt(torch.mean(qpmpin**2,dim=(-1,-2),keepdim = True))
                ppmpin /= torch.sqrt(torch.mean(ppmpin**2,dim=(-1,-2),keepdim = True))
            plp = qend-qstart
            res.append(torch.conv1d(qpmpin, ppmpin)/(plp*cha))
    # concatenate along the length
    res = torch.cat(res,-1)
    return res


              
def torch_compute_similarity_motifs(ppms, ppms_ref, 
                                    metric = 'correlation', 
                                    min_sim = 4, 
                                    padding = 0.25, 
                                    infocont = False, 
                                    bk_freq = 0.25, 
                                    reverse_complement = False, 
                                    verbose = False, 
                                    device = 'cpu', 
                                    batchsize = 1024, 
                                    exact = True, 
                                    fill_logp_self = 127, 
                                    return_alignment = False, 
                                    sparse = False,
                                    ):
    '''
    Aligns PWMs and returns a correlation and p-value matrix for downstream analysis
    TODO: Deal with device
    
    Parameters
    ----------
    
    ppms : list or np.ndarray
        of motifs of shape (length, channels)
    ppms_ref : list or np.ndarray
        motif to compare ppms to
    fill_logp_self: 
        if one_half, diagonal elements will not be computed and just filled with 
        this value
    metric : 
        Can be correlation, cosine, correlation_pvalue, mse
    min_sim : int
        minimum bases that have to overlap in a comparison
    padding : float
        padding value, for nucleotides 0.25, representing uniform prob. that
        base is present
    reverse_complement: boolean
        or two arrays defining if an individual pwm in
        one of the sets should be compared with its reverse complement
    return_alignment : 
        returns matrices for aligmement, i.e. offsets and reverse complement
        assignment.
    exact : 
        includes regions with values for both matrices, not only the one that
        is used as the weight. If False, it is roughly 10 times faster but it
        underestimates the correlation because it does only include positions
        that are shared with weights and not overhanging positions from the 
        motif given as sequence.
    '''
    # TODO: automatically transpose the motifs if they are given as (4,length)

    # initialize numpy arrays that will be returned
    if return_alignment:
        # best offset to align matrices
        offsets = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
        # reverse complement matrix
        revcomp_matrix = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    
    # correlation matrix itself
    correlation = np.ones((len(ppms), len(ppms_ref)), dtype = np.float32)
    
    if metric == 'correlation_pvalue':
        # Number of positions to compute palues
        n_matrix = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int16)
    
    # Do this to avoid affecting original motifs
    ppms = [ppm.astype(np.float32) for ppm in ppms]
    ppms_ref = [ppm.astype(np.float32) for ppm in ppms_ref]
    
    if infocont: # transform ppms to information content, also transform padding
        if bk_freq is None:
            bk_freq = 1./pwm[0].shape[-1]
        padding = np.log2(padding/bk_freq)
        for p, ppm in enumerate(ppms):
            ppm =np.log2((ppm+1e-8)/bk_freq)
            ppm[ppm<0] = 0
            ppms[p] = ppm 
        for p, ppm in enumerate(ppms_ref):
            ppm =np.log2((ppm+1e-8)/bk_freq)
            ppm[ppm<0] = 0
            ppms_ref[p] = ppm
    
    ## Needs to happen with the padding being added!!!
    # Normalize pwms to compute correlation or cosine
    centered = False
    if 'correlation' in metric:
        centered = True
        if not exact:
            means = [np.mean(ppm) for ppm in ppms]
            means_ref = [np.mean(ppm) for ppm in ppms_ref]
            ppms = [ppm-np.mean(ppm) for ppm in ppms]
            ppms_ref = [ppm-np.mean(ppm) for ppm in ppms_ref]
            padding -= np.mean(np.append(means,means_ref))
            
    standard = False
    if metric == 'cosine' or 'correlation' in metric:
        standard = True
        if not exact:
            # If not exact, normalization is performed beforehand
            # padding needs to be addjusted as if it was in the motif after 
            # padding.
            std = [np.sqrt(np.mean(ppm**2)) for ppm in ppms]
            std_ref = [np.sqrt(np.mean(ppm**2)) for ppm in ppms_ref]
            ppms = [ppm/std[p] for p,ppm in enumerate(ppms)]
            ppms_ref = [ppm/std_ref[p] for p,ppm in enumerate(ppms_ref)]
            padding /= np.mean(np.append(std,std_ref))
    
    if metric not in ['correlation', 'correlation_pvalue', 'cosine', 'mse'] :
        raise ValueError(f'{metric} not implemented')
    
    # Reverse complement will be checked to see if computations with reverse 
    # complment should be included
    if isinstance(reverse_complement, bool):
        if reverse_complement == False:
            reverse_complement = [np.zeros(len(ppms), dtype = int), 
                                       np.zeros(len(ppms_ref), dtype = int)]
        elif reverse_complement == True:
            reverse_complement = [np.ones(len(ppms), dtype = int), 
                                       np.ones(len(ppms_ref), dtype = int)]
    elif len(reverse_complement) != 2:
        reverse_complement = [reverse_complement, 
                                       np.ones(len(ppms_ref), dtype = int)]
    rcmat = reverse_complement[0][:,None]*reverse_complement[1][None, :]
    # reverse complements will be compared if both motifs are assigned to occur
    # as reverse complement.
    
    # Determine the lenghth of input pwms to split them into different length tensors
    plen = np.array([len(p) for p in ppms])
    ppms_indeces = [np.where(plen == pl)[0] for pl in np.unique(plen)]
    ppms = [torch.tensor(np.array([ppms[pi] for pi in pin], dtype = float)).transpose(-1,-2) for pin in ppms_indeces]
    
    plen_ref = np.array([len(p) for p in ppms_ref])
    ppms_ref_indeces = [np.where(plen_ref == pl)[0] for pl in np.unique(plen_ref)]
    ppms_ref = [torch.tensor(np.array([ppms_ref[pi] for pi in pin], dtype = float)).transpose(-1,-2) for pin in ppms_ref_indeces]

    mscale = 1
    if metric == 'mse':
        # Euclidean distance is |x-y|_2 = sqrt(x*x + y*y -2xy)
        # Convolution returns xy, so we need to precompute xx and yy
        # Since xx and yy don't have any paddings, we will add a padding 
        # constant  times the used padding size for each combination
        # Precompute xx and yy:
        mscale = 2
        correlation[:] = 0
        for p, ppm in enumerate(ppms):
            res = torch.sum(ppm* ppm, dim = (-1,-2))
            
            correlation[ppms_indeces[p],:] += res.numpy()[:,None]
        for q, qpm in enumerate(ppms_ref):
            res = torch.sum(qpm* qpm, dim = (-1,-2))
            correlation[:,ppms_ref_indeces[q]] += res.numpy()[None,:]
        
            
    # Compare sets of ppms against each other
    for p, ppm in enumerate(ppms):
        lp, plp, cha = ppm.shape[0], ppm.shape[-1], ppm.shape[-2]
        for q, qpm in enumerate(ppms_ref):
            lq, plq = qpm.shape[0], qpm.shape[-1]
            # if not exact, use conv1d with longer pwm
            res, resrev = [], []
            if exact:
                pad = plp-min_sim
                with torch.no_grad():
                    for b in range(0,lq, batchsize):
                        res.append(padded_weight_conv1d(qpm[b:b+batchsize], ppm, min_sim, padding = padding, centered = centered, standard = standard).transpose(0,1))
                        if rcmat[ppms_indeces[p]][:,ppms_ref_indeces[q]].any(): # check for reverse complement for any combination
                            resrev.append(padded_weight_conv1d(qpm[b:b+batchsize], reverse_torch(ppm), min_sim, padding = padding, centered = centered, standard = standard).transpose(0,1))
                        else:
                            resrev = None
            else:
                    
                if plp >= plq: # always fun the longer pwm as weights
                    pad = plp-min_sim
                    with torch.no_grad():
                        for b in range(0,lq, batchsize):
                            qpmp = F.pad(qpm[b:b+batchsize], (pad, pad), 'constant', padding)
                            res.append(torch.conv1d(qpmp, ppm).transpose(0,1)/plp/cha)
                            if rcmat[ppms_indeces[p]][:,ppms_ref_indeces[q]].any():
                                resrev.append(torch.conv1d(qpmp, reverse_torch(ppm)).transpose(0,1)/plp/cha)
                            else:
                                resrev = None
                else:
                    pad = plq-min_sim
                    with torch.no_grad():
                        for b in range(0,lq, batchsize):
                            ppmp = F.pad(ppm, (pad, pad), 'constant', padding)
                            res.append(torch.conv1d(ppmp, qpm[b:b+batchsize])/plq/cha)
                            if rcmat[ppms_indeces[p]][:,ppms_ref_indeces[q]].any():
                                resrev.append(torch.conv1d(ppmp, reverse_torch(qpm[b:b+batchsize]))/plq/cha)
                            else:
                                resrev = None
            
            res = torch.cat(res, 1)
            if resrev is not None:
                resrev = torch.cat(resrev, 1)
            
            # get the position of the best match            
            bmax, best = torch.max(res, dim = -1)
            bmax, best = bmax.cpu().numpy(), best.cpu().numpy()
            if resrev is not None:
                # if reverse was done, check if there's a better alignment
                # in the reverse matrix
                bmaxrev, bestrev = torch.max(resrev, dim=-1)
                bmaxrev, bestrev = bmaxrev.cpu().numpy(), bestrev.cpu().numpy()
                # if there is a better alignment, use the revese
                mask = rcmat[ppms_indeces[p]][:,ppms_ref_indeces[q]] & (bmaxrev > bmax)
                best[mask==1] = bestrev[mask == 1]
                bmax[mask==1] = bmaxrev[mask == 1]
           
            # calculate offsets of ppm to qpm
            off = best - pad
            # if qpm was used as weight, need to turn around offsets for ppm to qpm
            if plp<plq and not exact:
                off = -off
                if resrev is not None:
                    # if reverse complement was used on qpm as weight
                    # offset needs to measured from other direction
                    # to represent offset of ppm with reverse complement
                    off[mask==1] = plq-plp-off[mask==1]
            
            # Give values to output arrays
            for j,i in enumerate(ppms_indeces[p]):
                # Number of positions is distance between offset of ppm to qpm
                # to the end.
                # Distance computed as end_pos-start_pos
                # = max(plp+off, plq) - min(0,off)
                ns = (np.amax([plp+off[j], np.ones(len(off[j]))*plq], axis = 0)-np.amin([np.zeros(len(off[j])),off[j]],axis =0)) *cha
                if metric == 'correlation_pvalue':
                    n_matrix[i,ppms_ref_indeces[q]] = ns
                if metric == 'mse':
                    # add the missing parts to x*x and y*y for the padded areas
                    correlation[i,ppms_ref_indeces[q]] += padding**2 * ((ns-plp*cha) + (ns-plq*cha))
                    # devide through ns because bmax is already divided 
                    correlation[i,ppms_ref_indeces[q]] /= ns
                correlation[i,ppms_ref_indeces[q]] += -mscale*bmax[j]
                
                if return_alignment:
                    offsets[i,ppms_ref_indeces[q]] = off[j]
                    if resrev is not None:
                        revcomp_matrix[i,ppms_ref_indeces[q]] = mask[j]
    
    # compute pvalues with t-distribution, make
    if metric == 'correlation_pvalue':
        pvalue = correlation_to_pvalue(1.-correlation, n_matrix) + 10**-(fill_logp_self)
        # since correlations can also have good p-values, transform into log pvalue
        correlation = np.sign(1.-correlation) * -np.log10(pvalue)
        # multiply by the sign of the correlation and transform back
        correlation = 10**(-correlation)
    
    if return_alignment:
        return correlation, offsets, revcomp_matrix
    else:
        return correlation


def assign_leftout_to_cluster(tripletclusters, checkmat, linkage, distance_threshold):
    '''
    Assign data points that were left out of origional clustering to one of 
    the clusters
    Parameters
    ----------
    tripletclusters :
        cluster numbers of triplets, 3 repeats of clusternumber
    checkmat : 
        contains the distance matrix between triplets with cluster
        assignment and left out data points
        shape = (3, n_cluster, n_left_datapoints)
    linkage :
        linkage function
    
    '''
    
    clusters_left = -np.ones(checkmat.shape[-1], dtype = int)
    
    for c in range(checkmat.shape[-1]):
        if c%1000==0:
            print(c)
        if linkage == 'complete':
            poscl = np.where(np.sum(checkmat[...,c] < distance_threshold, axis = 0) == 3)[0]
            if len(poscl) > 0:
                poscl = poscl[np.argmin(np.mean(checkmat[:, poscl,c], axis = 0))]
                clusters_left[c] = tripletclusters[3*poscl]
        if linkage == 'single':
            amins = np.amin(checkmat[...,c],axis=0)
            poscl = np.argmin(amins)
            if amins[poscl] <= distance_threshold:
                clusters_left[c] = tripletclusters[3*poscl]
        if linkage == 'average':
            ameans = np.mean(checkmat[...,c],axis=0)
            poscl = np.argmin(ameans)
            if ameans[poscl] <= distance_threshold:
                clusters_left[c] = tripletclusters[3*poscl]

    return clusters_left

def complement_clustering(clusters, pwmnames, pwm_set, similarities, pwmnames_left, pwm_left, randmix):
    '''
    Wrapper to save space in main. 
    1) Computes distancce matrix between triplets from assigned clusters and
    left out data points.
    2) Assigns left out data points based on distance cut off and linkage 
    function to the already existing clusters
    3) If data points could not be assigned, performs another clustering of
    those independently
    4) Combines cluster sets and returns names, clusters and pwms back to
    original order
    
    '''
    triplets, tripclusters = _determine_triplets(clusters, similarities) # three members of the cluster to which left out data points will be measured
    corr_left = torch_compute_similarity_motifs(pwm_set[triplets], pwm_left, metric = args.distance_metric, return_alignment = False, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = True)
    
    checkmat = corr_left.reshape(3,-1,corr_left.shape[-1])
    
    clusters_left = assign_leftout_to_cluster(tripclusters, checkmat, args.linkage, args.distance_threshold)
    
    print(len(clusters_left)-len(np.where(clusters_left == -1)[0]), 'added to assigned clusters')
    
    if len(np.where(clusters_left == -1)[0]) > 1 and len(np.where(clusters_left == -1)[0]) <= args.approximate_cluster_on:
        print(f'Reclustering of {len(np.where(clusters_left == -1)[0])} clusters')
        corr_left, ofs_left, revcomp_matrix_left = torch_compute_similarity_motifs(pwm_left[clusters_left == -1], pwm_left[clusters_left == -1], metric = args.distance_metric, return_alignment = True, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = args.approximate_distance)

        clustering = AgglomerativeClustering(n_clusters = None, metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(corr_left)
        clusters_left[clusters_left == -1] = np.amax(clusters) + clustering.labels_

    resort = np.argsort(randmix)
    clusters = np.append(clusters, clusters_left)[resort]
    pwmnames = np.append(pwmnames, pwmnames_left)[resort]
    pwm_set = np.array(list(pwm_set)+list(pwm_left), dtype = object)[resort]

    return clusters, pwmnames, pwm_set

def _get_third(csim, consider = 60):
    argmin = np.argsort(csim.flatten())[:consider]
    argmin = argmin//csim.shape[-1], argmin%csim.shape[-1]
    simtothird = csim[argmin[0]] + csim[argmin[1]]
    third = np.argmin(simtothird, axis = 1)
    best = [argmin[0], argmin[1], third]
    bestdist = csim[best[0], best[1]] + csim[best[2], best[1]] + csim[best[0], best[2]]
    best = np.array(best)[:,np.argmin(bestdist)]
    return best

def _determine_triplets(clusters, similarity):
    '''
    Determine three data points in a cluster that are the furthest apart from
    each other. Use these three data points to determine if another data point
    should be assigned to that cluster.
    
    '''
    uclusters = np.unique(clusters)
    bests = []
    for u, uc in enumerate(uclusters):
        mask = np.where(clusters == uc)[0]
        csim = similarity[mask][:,mask]
        best = _get_third(csim)
        bests.append(np.array(mask)[best])
        
    return np.concatenate(bests), np.repeat(uclusters,3)





def align_compute_similarity_motifs(ppms, ppms_ref, fill_logp_self = 1000, min_sim = 5, padding = 0.25,
                                     infocont = False, bk_freq = 0.25, 
                                     non_zero_elements = False, reverse_complement
                                     = False, njobs = 1, verbose = False):
    '''
    Wrapper function for _align_compute_similarity_motifs that uses joblib to 
    parallize the computation of the similarity matrix
    
    Parameters
    ----------
    
    ppms : list or np.ndarray
        of motifs of shape (length, channels)
    ppms_ref : list or np.ndarray
        motif to compare ppms to
    fill_logp_self: 
        if ppms and ppms_ref are the same, diagonal elements will not be computed and just filled with 
        this value
    min_sim : int
        minimum bases that have to overlap in a comparison
    padding : float
        padding value, for nucleotides 0.25, representing uniform prob. that
        base is present
    non_zero_elements: boolean
        if True, only compare overlapping elements between pwms, or in other words
        mask everyting that is zero one of the motifs
    reverse_complement: boolean
        or two arrays defining if an individual pwm in
        one of the sets should be compared with its reverse complement
    njobs: int
        Number of processors to perform computation
    
    Returns
    -------
    correlation : 
        matrix with correlation distance between ppms and ppms_ref
    log_pvalues : 
        matrix with -log10 p values for the correlation matrix
    offsets : 
        best offset between motif in ppms and motif in ppms_ref to align motifs
    revcomp_matrix: 
        binary matrix determines if pwms was best aligned when in reverse complement
    '''
    
    for p, ppm in enumerate(ppms):
        if np.isnan(ppm).any():
            raise ValueError(f'nan in ppm {ppm}')

    for p, ppm in enumerate(ppms_ref):
        if np.isnan(ppm).any():
            raise ValueError(f'nan in ppm_ref {ppm}')


    
    # reverse_complement array determines if one should also compare the reverse complement of the pwms to all other pwms
    if isinstance(reverse_complement, bool):
        if reverse_complement == False:
            reverse_complement = np.array([np.zeros(len(ppms), dtype = int), 
                                       np.zeros(len(ppms_ref), dtype = int)])
        elif reverse_complement == True:
            reverse_complement = np.array([np.ones(len(ppms), dtype = int), 
                                       np.ones(len(ppms_ref), dtype = int)])
    elif len(reverse_complement) != 2:
        reverse_complement = np.array([reverse_complement, 
                                       np.ones(len(ppms_ref), dtype = int)])
    
    # check if pwm set is the same or different
    l_in, l_ref = len(ppms), len(ppms_ref)
    is_the_same = False
    if l_in == l_ref:
        all_the_same = []
        for p in range(len(ppms)):
            all_the_same.append(np.array_equal(ppms[p], ppms_ref[p]))
        is_the_same = np.array(all_the_same).all()
    
    one_half = is_the_same
    
    if njobs == 1:
        correlation, log_pvalues, offsets, revcomp_matrix, _ctrl = _align_compute_similarity_motifs(ppms, ppms_ref, one_half = one_half, fill_logp_self = fill_logp_self, min_sim = min_sim, infocont = infocont,bk_freq = bk_freq, reverse_complement=reverse_complement, verbose = verbose)
    else:
        correlation = 2*np.ones((l_in, l_ref), dtype = np.float32)
        log_pvalues = np.zeros((l_in, l_ref), dtype = np.float32)
        offsets = 100*np.ones((l_in, l_ref), dtype = np.int16)
        revcomp_matrix = -np.ones((l_in, l_ref), dtype = np.int8)
        
        spacesi = np.linspace(0,len(ppms), njobs + 1, dtype = int)
        spacesj = np.linspace(0,len(ppms_ref), njobs*int(one_half)+1,
                              dtype = int)
    
        if verbose:
            print('Computation split into', spacesi, spacesj)
        if one_half:
            results = Parallel(n_jobs=njobs)(delayed(_align_compute_similarity_motifs)
                                             (ppms[spacesi[i]:spacesi[i+1]], 
                                              ppms_ref[spacesj[j]:spacesj[j+1]], 
                                              one_half = (i == j) & one_half, 
                                              fill_logp_self = fill_logp_self, 
                                              min_sim = min_sim, 
                                              infocont = infocont, 
                                              bk_freq = bk_freq,
                                              reverse_complement = [reverse_complement[0][spacesi[i]:spacesi[i+1]], 
                                                                    reverse_complement[1][spacesj[j]:spacesj[j+1]]],
                                              ctrl = (i,j),
                                              verbose = verbose
                                              ) 
                                              for i in range(0, njobs) 
                                              for j in range(i*int(one_half), (njobs-1)*int(one_half)+1)
                                             )
                                                                                                       
            for res in results:
                idx = res[-1]
                
                correlation[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[0]
                log_pvalues[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[1]
                revcomp_matrix[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[3]
                
                if idx[0] == idx[1]:
                    offsets[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[2]
                else:
                    correlation[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[0].T
                    log_pvalues[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[1].T
                    revcomp_matrix[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[3].T
                    offsets[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[2][0]
                    offsets[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[2][1].T
                    
    return correlation, log_pvalues, offsets, revcomp_matrix

# Previously compare_ppms
def _align_compute_similarity_motifs(ppms, ppms_ref, one_half = False, 
                                     fill_logp_self = 1000, min_sim = 5, padding = 0.25,
                                     infocont = False, bk_freq = 0.25, 
                                     non_zero_elements = False, reverse_complement
                                     = False, ctrl = None, verbose = False):
    '''
    Aligns PWMs and returns a correlation and p-value matrix for downstream analysis
    
    Parameters
    ----------
    
    ppms : list or np.ndarray
        of motifs of shape (length, channels)
    ppms_ref : list or np.ndarray
        motif to compare ppms to
    one_half: boolean
        if ppms and ppms_ref are equal, we only need to compute one half of the pairs
    fill_logp_self: 
        if one_half, diagonal elements will not be computed and just filled with 
        this value
    min_sim : int
        minimum bases that have to overlap in a comparison
    padding : float
        padding value, for nucleotides 0.25, representing uniform prob. that
        base is present
    non_zero_elements: boolean
        if True, only compare overlapping elements between pwms, or in other words
        mask everyting that is zero one of the motifs
    reverse_complement: boolean
        or two arrays defining if an individual pwm in
        one of the sets should be compared with its reverse complement
    ctrl: tuple
        if not None, return this tuple for joblib
    
    TODO
        Replace the individual alignments with torch.conv1d:
        Normalize values in pwms, so that dot product is correlation
        Either make all PWMs the same length with padding, or split into
        groups by size and use the larger one as weights. 
        Issue: Some parts of the pwms that is used as sequence will not be 
        considered in this correlation. 
    
    '''
    
    # reverse_complement array determines if one should also compare the reverse complement of the pwms to all other pwms
    if isinstance(reverse_complement, bool):
        if reverse_complement == False:
            reverse_complement = np.array([np.zeros(len(ppms), dtype = int), 
                                       np.zeros(len(ppms_ref), dtype = int)])
        elif reverse_complement == True:
            reverse_complement = np.array([np.ones(len(ppms), dtype = int), 
                                       np.ones(len(ppms_ref), dtype = int)])
    elif len(reverse_complement) != 2:
        reverse_complement = np.array([reverse_complement, 
                                       np.ones(len(ppms_ref), dtype = int)])
    
    if ctrl is not None and verbose:
        print('Computing', ctrl, 'part of matrix')
    # measure lengths of motifs
    motif_len, motif_len_ref = [np.shape(ppm)[0] for ppm in ppms], [np.shape(ppm)[0] for ppm in ppms_ref]
    # alignment offsets that will be saved
    offsets = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    if not one_half and ctrl is not None:
        offdiagoffsets = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    # log_pvalues of correlation that will be saved
    log_pvalues = np.zeros((len(ppms), len(ppms_ref)), dtype = np.float32)
    # correlation matrix itself
    correlation = np.zeros((len(ppms), len(ppms_ref)), dtype = np.float32)*2
    # reverse complement matrix
    revcomp_matrix = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    
    # whether to measure correlation of frequencies (pfms) of information content (pwms)
    if infocont:
        padding = max(0,np.log2(padding/bk_freq)) # padding needs to be adjusted if information content is chosen
    t0 = time.time()
    for p, ppm0 in enumerate(ppms):
        # just to check the speed of the algorithm
        if p> 0 and p% 25 == 0 and verbose:
            print(p, round(time.time() - t0,3))
            t0 = time.time()
        
        if one_half: # if one_half then ppms and ppms_ref have to be identical 
            p_start = p+1
        else:
            p_start = 0
        # normalize pfm to pwm
        if infocont:
            ppm0 = np.log2((ppm0+1e-8)/bk_freq)
            ppm0[ppm0<0] = 0
        
        for q in range(p_start, len(ppms_ref)):
            qpm = ppms_ref[q]
            if infocont:
                qpm = np.log2((qpm+1e-8)/bk_freq)
                qpm[qpm<0] = 0
            pvals = []
            pearcors = []
            offs = []
            rvcmp = []
            
            for rc in range(reverse_complement[0][p] * reverse_complement[1][q] +1):
                if rc == 1:
                    ppm = reverse(ppm0)
                else:
                    ppm = np.copy(ppm0)
                
                for i in range(min(0,-motif_len[p]+min_sim), motif_len_ref[q]-min(motif_len[p],min_sim) + 1):
                    if padding is not None:
                        refppm, testppm = np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding, np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding
                        
                        refppm[-min(0,-motif_len[p]+min_sim):motif_len_ref[q]-min(0,-motif_len[p]+min_sim)] = qpm
                        
                        testppm[i-min(0,-motif_len[p]+min_sim):i-min(0,-motif_len[p]+min_sim)+motif_len[p]] = ppm
            
                        mask = np.sum(testppm+refppm != 2*padding ,axis = 1) > 0
                        refppm, testppm = refppm[mask], testppm[mask]
                    else:
                        refppm = qpm[max(i,0):min(motif_len_ref[q],motif_len[p]+i)]
                        testppm = ppm[max(0,-i): min(motif_len[p],motif_len_ref[q]-i)]
                    
                    if non_zero_elements:
                        nonzmask = (testppm != 0) | (refppm != 0)
                        peacor, pval = pearsonr(testppm[nonzmask].flatten(), refppm[nonzmask].flatten())
                    else:
                        peacor, pval = pearsonr(testppm.flatten(), refppm.flatten())

                    pvals.append(np.sign(peacor)*-np.log10(pval))
                    pearcors.append(peacor)
                    offs.append(i)
                    rvcmp.append(rc)
            
            maxp = np.argmax(pvals)
            log_pvalues[p,q] = pvals[maxp]
            offsets[p,q] = offs[maxp]
            correlation[p,q] = 1.-pearcors[maxp]
            revcomp_matrix[p,q] = rvcmp[maxp]

            if one_half:
                log_pvalues[q,p] = pvals[maxp]
                if rvcmp[maxp]:
                    offsets[q,p] = motif_len[p] - motif_len_ref[q] +offs[maxp] #### Need to adjust when reverse complement
                else:
                    offsets[q,p] = -offs[maxp]
                correlation[q,p] = 1.-pearcors[maxp]
                revcomp_matrix[q,p] = rvcmp[maxp]
            elif not one_half and ctrl is not None:
                if rvcmp[maxp]:
                    offdiagoffsets[p,q] = motif_len[p] - motif_len_ref[q] +offs[maxp] #### Need to adjust when reverse complement
                else:
                    offdiagoffsets[p,q] =-offs[maxp]
                    
    log_pvalues[np.isinf(log_pvalues)] = fill_logp_self
    
    if one_half:
        np.fill_diagonal(log_pvalues, fill_logp_self)
    
    if ctrl is not None and not one_half:
        offsets = [offsets, offdiagoffsets]
        
    return correlation, log_pvalues, offsets, revcomp_matrix, ctrl



def pfm2iupac(pwms, bk_freq = None):
    '''
    Translates position frequency matrix to iupac annotation
    Parameters
    ----------
    pwms: 
        list of pfms of shape=(l,4) or (4,l), representing A,C,G,T in this order
    bk_freq:
        cut_off when to consider a chance for base to appear position
    '''
    hash = {'A':16, 'C':8, 'G':4, 'T':2}
    dictionary = {'A':16, 'C':8, 'G':4, 'T':2, 'R':20, 'Y':10, 'S':12, 'W':18, 'K':6, 'M':24, 'B':14, 'D':22, 'H':26, 'V':28, 'N':0}
    
    res = dict((v,k) for k,v in dictionary.items()) # reverse of dictionary
    n_nts = len(pwms[0][0])
    
    if bk_freq is None:
        bk_freq = (1./float(n_nts))*np.ones(n_nts)
    else:
        bk_freq = bk_freq*np.ones(n_nts)
    # determine the axis with the channels
    shapes = []
    for pwm in pwms:
        shapes.append(np.shape(pwm))
    maxshape = np.amax(shapes, axis = 0)
    axis = np.where(maxshape == 4)[0][0]
   
    motifs = []
    for pwm in pwms:
        if axis == 0:
            pwm = pwm.T
        m = ''
        for p in pwm:
            score = 0 # score to look up ipac in 'res'
            for i in range(len(p)):
                if p[i] > bk_freq[i]: # only consider base p[i] if over bk_freq
                    score += list(hash.values())[i]
            m += res[score]
        motifs.append(m)
    return np.array(motifs)
          




def combine_pwms(pwms, clusters, similarity, offsets, orientation, norm = 'max',
                 remove_low = 0.075, minlen = 3):
    
    '''
    Combine set of pwms based on their cluster assignments
    Need also precomputed offsets and orientation to each other 
    (i.e. reverse or forward)
    
    Parameters
    ----------
    
    pwms : 
        motifs of shape = (L,4)
    clusters : 
        cluster number for each pwm 
    similarity: 
        Similarity matrix, the large values are better
    offsets : 
        matrix with offsets of pwms at axis 0 to pwms at axis 1
    orientation: 
        orientation of pwms at axis 0 when aliged to pwm at axis 1, 0 forward
        1 for reverse complement
    norm : 
        determines average 
    remove_low :
        removes regions with lwo coverage from alignment
    minlen : 
        minimum length of resulting motifs
    '''
    
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    unclusters = unclusters[unclusters>=0]
    comb_pwms = []
    for u in unclusters:
        mask = np.where(clusters == u)[0]

        sim = similarity[mask][:,mask]
        offsetcluster = offsets[mask][:,mask]
        orient = orientation[mask][:,mask]
        pwmscluster = [pwms[m] for m in mask]
        clusterlen = lenpwms[mask]
        
        sort = np.argsort(-np.sum(sim, axis = 1))
        sim = sim[sort][:, sort]
        offsetcluster = offsetcluster[sort][:, sort]
        orient = orient[sort][:, sort]
        pwmscluster = [pwmscluster[s] for s in sort]
        clusterlen = clusterlen[sort]

        centerpiece = pwmscluster[0]
        
        centeroffsets = offsetcluster.T[0] # offset to the pwm that is most 
        # similar to all others
        centerorient = orient.T[0]
        seed = _join_pwms(pwmscluster, clusterlen, centeroffsets, centerorient,
                          norm = norm, remove_low = remove_low, minlen = minlen)
                
        comb_pwms.append(seed)
    return comb_pwms


def _join_pwms(pwmscluster, clusterlen, centeroffsets, centerorient, norm = 'max', remove_low = 0.05, minlen = 3):
    '''
    Joins PWMS in pwmcluster to the pwm in position 0. 
    '''
    
    if len(pwmscluster) > 1:
        centerpiece = pwmscluster[0]
        
        lenmat = len(centerpiece)+max(0,np.amax(centeroffsets+clusterlen-len(centerpiece))) - min(0,np.amin(centeroffsets))
        
        seed = np.zeros((lenmat,len(pwmscluster[0][0])))
        seedcount = np.zeros(len(seed))
        
        center = - min(0,np.amin(centeroffsets))
        
        centerrev = center + len(centerpiece)
        
        for o in range(len(pwmscluster)):
            pwm0 = pwmscluster[o]
            if centerorient[o] == 1:
                pwm0 = reverse(pwm0)
            
            seed[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += pwm0.astype(float)
            seedcount[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += 1
            
        if norm == 'mean':
            seed = seed/seedcount[:,None]

    else:
        seed = pwmscluster[0]
        seedcount = np.ones(len(seed))

    if norm == 'max':
        seed = seed/np.amax(seedcount)
    elif norm == 'sum':
        seed = seed/np.sum(np.absolute(seed),axis = 1)[:, None]

    if remove_low > 0:
        sumseed = np.sum(np.absolute(seed), axis = 1)
        edges = np.where(sumseed/np.amax(sumseed)>remove_low)[0]
        if len(edges) > 0:
            if edges[-1]+1-edges[0]>= minlen:
                seed = seed[edges[0]:edges[-1]+1]
            else:
                seed[:] = 0.
        else:
            seed[:] = 0.

    return seed



def find_motifs(ref_attribution, cut, max_gap = 1, min_motif_size = 4, smooth_gap = True):
    ''' 
    Determine locations of motifs in attribution profiles at reference
    Parameters
    ----------
    ref_attribution : 
        array of shape = (length,) with the attributions at the reference
    cut : 
        cutoff for calling a base to be significant, and in a motif
    max_gap: 
        maximal allowed gap size
    min_motif_size: 
        minimal size of motifs to be added to motif set
    
    Returns
    -------
        list of lists that contain every position which is considered to be part of a motif. 
        
    '''
    
    def gap_call(i, msi, ref_attribution, cut, avgwindow = 0):
        '''
        If smooth_gap:
        Define more relaxed rule on what to call a gap based on the average
        around the base within avgwindow that is not significant
        
        Parameters: 
        ----------
        i : 
            position in ref_attribution
        msi : 
            supposed sign of attribution at i
        ref_attribution:
            array with attributions at reference
        cut : 
            cut off for significance
        avgwindow : 
            window to consider for average to check if a position is still part of motif
        '''
        return msi *np.mean(ref_attribution[max(0,i-avgwindow): min(len(ref_attribution),i+avgwindow+1)]) < cut
    
    aloc = np.absolute(ref_attribution)>cut
    sign = np.sign(ref_attribution) # get sign of effects
    lra = len(ref_attribution)
    motiflocs = []

    gap = max_gap +1 # gapsize count
    msi = 1 # sign of motif
    potloc = [] # potential location of motif
    i = 0 
    while i < lra:
        if aloc[i]: # if location significant
            if len(potloc) == 0: # if potloc just started
                msi = np.copy(sign[i]) # determine which sign the entire motif should have
            
            if sign[i] == msi: # check if base has same sign as rest of motif
                potloc.append(i)
                gap = 0
            
            elif gap_call(i, msi, ref_attribution, cut, avgwindow = max_gap * int(smooth_gap)): 
                # if the average with gapsize around the location is smaller than the cut, 
                # only then count as gap, otherwise just count as nothing
                gap += 1
                
                if gap > max_gap: # check that gap is still smaller than maximum gap size
                    # if this brought the gapsize over the max gapsize but the
                    # motif is long enough, then add to motifs
                    if len(potloc) >= min_motif_size: 
                        motiflocs.append(potloc)
                    # restart where the gap started so that motifs with 
                    # different direction following directly on other motifs can be counted
                    if len(potloc) > 0: 
                        i -= gap
                    # reset gap and poloc
                    gap = max_gap + 1
                    potloc = []
        
        elif gap_call(i, msi, ref_attribution, cut, avgwindow = max_gap * int(smooth_gap)):
            gap +=1
            if gap > max_gap:
                if len(potloc) >= min_motif_size:
                    motiflocs.append(potloc)
                    
                if len(potloc) > 0:
                    i -= gap
                gap = max_gap + 1
                potloc = []
        i += 1
    
    if len(potloc) >= min_motif_size:
        motiflocs.append(potloc)
    return motiflocs




    
    
    
    
    
    
