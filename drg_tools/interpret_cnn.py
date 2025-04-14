# interpret_cnn.py

'''
functions to interpret the kernels of models
and to create sequence attributions from the model

'''

import numpy as np
import sys, os
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
from functools import reduce 

from .modules import func_dict, func_dict_single, SoftmaxNorm, EXPmax
from .sequence_utils import generate_random_onehot
from .motif_analysis import pfm2iupac
from .stats_functions import correlation, mse
from .model_training import batched_predict

def kernel_to_ppm(kernels, kernel_bias = None, bk_freq = None):
    '''
    manipulate kernel values to generate pfms
    '''
    n_kernel, n_input, l_kernel = np.shape(kernels)
    if kernel_bias is not None:
        kernels += kernel_bias[:,None,None]/(n_input*l_kernel)
    # strechting factor to increase impact of large parameter values
    # alternatively use: kernels *= l_kernel
    kernels /= np.std(kernels, axis = (-1,-2))[...,None, None]
    if bk_freq is None:
        bk_freq = np.ones(n_input)*np.log2(1./float(n_input))
    elif isinstance(bk_freq, float) or isinstance(bk_freq, int):
        bk_freq = np.ones(n_input)*np.log2(1./float(bk_freq))
    kernels += bk_freq[None,:,None]
    ppms = 2.**kernels
    ppms = ppms/np.amax(np.sum(ppms, axis = 1),axis = -1)[:,None, None] #[:,None, :]
    return ppms

def kernels_to_pwms_from_seqlets(weights, seqlet_set, maxact, biases = None, activation_func = None, zscore = False, device = 'cpu', batchsize = None):
    if batchsize is not None:
        pwms = []
        for i in range(0, len(weights), batchsize):
            if biases is not None:
                bias = biases[i:i+batchsize]
            else:
                bias = None
            
            seqactivations = kernels_seqactivations_from_seqlets(weights[i:i+batchsize], seqlet_set, biases = bias, activation_func=activation_func, device = device)
            pwm = pwms_from_seqs(seqlet_set, seqactivations, maxact, z_score = zscore)
        
            pwms.append(pwm)
        pwms = np.concatenate(pwms, axis = 0)
    else:
        seqactivations = kernels_seqactivations_from_seqlets(weights, seqlet_set, biases = biases, activation_func=activation_func, device = device)
        pwms = pwms_from_seqs(seqlet_set, seqactivations, maxact, z_score = zscore)
    pwms = np.transpose(pwms, axes = (0,2,1))
    return pwms
    

def kernels_seqactivations_from_seqlets(weights, seqlet_set, biases = None, activation_func = None, device = 'cpu'):
    '''
    Uses torch conv1d to compute kernel activations
    '''
    with torch.no_grad():
        weights, biases, seqlet_set = torch.Tensor(weights).to(device), torch.Tensor(biases).to(device), torch.Tensor(seqlet_set).to(device)
        #seqactivations0 = torch.sum(weights[:,None]*seqlet_set[None,...], dim = (2,3))
        seqactivations = torch.nn.functional.conv1d(seqlet_set, weights)
        seqactivations = seqactivations.squeeze(-1).transpose(0,1)
        if biases is not None:
            seqactivations += biases[:,None]
        if activation_func is not None:
            seqactivations = activation_func(seqactivations)
    seqactivations = seqactivations.cpu().detach().numpy()
    return seqactivations




def pwms_from_seqs(ohseqs, activations, cut, z_score = True, pseudo = 0.025):
    '''
    multiplies the one hot encoded sequences with the activations
    Selects only sequences with activations higher than cut 
    Creates Position frequency matrix from sequences
    '''
    # scale kernel activations between 0 and 1
    minact = np.amin(activations, axis = 1)
    activations = activations - minact[:,None]
    maxact = np.amax(activations, axis = 1)
    activations = activations/maxact[:,None]
    
    if z_score:
        # instead of rescaling the activations, rescale the cut parameter
        cut = (np.mean(activations, axis = 1)+cut*np.std(activations,axis =1))[:, None]
    seqs = np.where(activations >= cut)
    pwms = []
    for a, act in enumerate(activations):
        mask = seqs[1][seqs[0]==a]
        chseq = act[mask][:,None,None]*ohseqs[mask]
        # pseudo count represents an additional sequence that is filled with pseudo
        chseq += pseudo/len(chseq)
        pwms.append(np.sum(chseq, axis = 0)/np.sum(chseq, axis = (0,1))[None, :])
        # check distribution
        #bins = np.linspace(0,1,21)
        #plt.hist(act, bins = bins)
        #plt.hist(act[seqs[1][seqs[0]==a]],bins = bins, color='orange',zorder = 1)
        #plt.show()
    return np.array(pwms)
 


# IDEA: Use this to propagate importance through network and determine
# for individual nodes and then interpret them.

def parameter_importance_score(ypred, y, coef_, inputs):
    '''
    Perform zscore test for each parameter as in linear regressio
    '''
    invcovardiag = np.diagonal(np.pinv(np.dot(inputs.T, inputs),rcond = 1e-8, hermetian = True))
    loss = np.mean((ypred-y)**2, axis = 0)
    var_b = loss[None,:]*(invcovardiag)[:,None]
    sd_b = np.sqrt(var_b)
    z_scores = coef_ * sd_b
    z_scores = np.nan_to_num(z_scores)
    return z_scores




def correct_by_mean(grad, channel_axis=-2):
    '''
    Corrects the gradient by subtracting the mean of the gradient
    along the channel axis
    Parameters
    ----------
    grad : np.ndarray, or torch.Tensor
        Gradient to be corrected
    channel_axis : int
        Axis along which the mean is computed
        This is the channel of the nucleotides
    Returns 
    -------
    grad : np.ndarray, or torch.Tensor
        Corrected gradient
    '''
    if isinstance(grad, torch.Tensor):
        return grad - torch.mean(grad, dim = channel_axis, keepdim = True)
    if isinstance(grad, np.ndarray):
        return grad - np.mean(grad, axis = channel_axis, keepdims = True)  
    

def ism(x, model, tracks=None, output = 'ism_attributions', batchsize = None,
        device = None):
    '''
    Generates attributions via in silico saturated mutagenesis
    
    Parameters
    ----------
    x : np.ndarray, list of np.ndarray, torch.Tensor or list of torch.Tensor
        Can either be one hot encoded sequences of shape = (N_seq, 4, L_seq) or
        list of paired sequences, for example RNA and paired DNA sequences
    model : torch.Module 
        CNN model to make predictions with
    tracks : list of int 
        Indeces for selected tracks
        None will use all tracks
        If tracks is list of lists, it will use cnn_average_wrapper to create the average
        of the tracks in each list
    output : str
        'attributions' will return the ISM attributions
        'ism' will return differences between reference and alternative sequences
        'predictions' will return the predictions of the model in the same shape
    batchsize : int 
        Batch size for the model, if None, tries to use model.batchsize
        If that doesn't exit, it will use 1
        
    Returns
    -------
    ismout : numpy.ndarray
        If x is a list of n_type arrays or tensors
        the output will be of shape = (n_type, N_seqs, tracks, L_seq, channels)
        If x is a single array or tensor, the output will be of shape = (N_seqs, tracks, L_seq, channels)

    '''
    
    if tracks is not None:
        if isinstance(tracks, (np.integer,int)):
            tracks = [tracks]
        # If the tracks are given as list of lists, the wrapper will return averages
        # of the tracks in each list
        # If the tracks are given as list of numpy arrays, the wrapper will return averages
        if isinstance(tracks[0], list) or isinstance(tracks[0], np.ndarray):
            model = cnn_average_wrapper(model, tracks)
            tracks = [i for i in range(len(tracks))]
    
    if isinstance(x, list):
        if not isinstance(x[0], torch.Tensor):
            x = [torch.Tensor(xi) for xi in x]
    else:
        if not isinstance(x, torch.Tensor):
           x = torch.Tensor(x)

    # Predict values for original sequences, 
    # model.predict() returns numpy arrays and acts with torch.no_grad
    reference_pred = batched_predict(model, x)
    if isinstance(reference_pred, list):
        reference_pred = np.concatenate(reference_pred, axis = 1)
    
    if tracks is None:
        tracks = np.arange(np.shape(reference_pred)[1])
    
    reference_pred = reference_pred[..., tracks]
    if output == 'predictions':
        offset = reference_pred
    else:
        offset = np.zeros_like(reference_pred)

    if batchsize is None:
        if 'batchsize' in model.__dict__:
            batchsize = model.batchsize
        else: # This could be slow
            batchsize = 1

    # chekk if device is None, then use the device of the model
    if device is None:
        if 'device' in model.__dict__:
            device = model.device
        else:
            device = 'cpu'
    model.to(device)

    if isinstance(x, list):
        Nin = len(x)
        ismout = []
        for n in range(Nin):
            xi = x[n]
            size = list(np.shape(xi))
            size += [len(tracks)]
            ismout0 = np.zeros(size)
            for i, si in enumerate(xi):
                isnot = torch.where(si == 0)
                # Needs cloning otherwise no changes can be made to si
                xalt = torch.clone(si.expand([len(isnot[0])] + list(si.size())))
                # change every base at at time
                for j in range(len(isnot[0])):
                    xalt[j,:,isnot[1][j]] = 0
                    xalt[j,isnot[0][j],isnot[1][j]] = 1
                xin = [] # prepare xin sequence pairs
                for k in range(Nin):
                    if k != n: 
                        xin.append(torch.clone(x[k][i].expand([len(isnot[0])] + list(si.size()))))
                    else:
                        xin.append(xalt)
                altpred = batched_predict(model,xin, device = device)
                if isinstance(altpred,list):
                    altpred = np.concatenate(altpred,axis =1)
                altpred = altpred[...,tracks]
                # assign delta values to locoation of changes base
                for j in range(len(isnot[0])):
                    ismout0[i, isnot[0][j], isnot[1][j]] = altpred[j] - reference_pred[i] + offset[i]
            ismout.append(np.swapaxes(np.swapaxes(ismout0, 1, -1), -1,-2))
            if 'attributions' in output:
                ismout[-1] = correct_by_mean(ismout[-1])
    else:
        Nin = 1
        size = list(np.shape(x))
        size += [len(tracks)]
        ismout = np.zeros(size)
        start = time.time()
        for i, si in enumerate(x):
            isnot = torch.where(si == 0)
            # Needs cloning otherwise no changes can be made to si
            xalt = torch.clone(si.expand([len(isnot[0])] + list(si.size())))
            for j in range(len(isnot[0])):
                xalt[j,:,isnot[1][j]] = 0
                xalt[j,isnot[0][j],isnot[1][j]] = 1
            altpred = batched_predict(model,xalt, device = device)
            if isinstance(altpred,list):
                altpred = np.concatenate(altpred,axis =1)
            altpred = altpred[...,tracks]
            for j in range(len(isnot[0])):
                ismout[i, isnot[0][j], isnot[1][j]] = altpred[j] - reference_pred[i] + offset[i]
        end = time.time()
        print('ISM time for', ismout.shape, len(tracks),  end-start)
        ismout = np.swapaxes(np.swapaxes(ismout, 1, -1), -1,-2)
        if 'attributions' in output:
            ismout = correct_by_mean(ismout)
        
    return ismout
                                                     

class cnn_average_wrapper(torch.nn.Module):
    '''
    Wrapper for cnn get attributions for average of tracks
    
    Parameters
    ----------
    model : pytorch.nn.module
        The cnn_multi object
    tracks : list of lists
        tuples contain all the track indices that are combined to averages
    '''
    def __init__(self, model, tracks):
        super(cnn_average_wrapper, self).__init__()
        self.shift_sequence = model.shift_sequence
        self.random_shift = model.random_shift
        self.model = model
        self.tracks = tracks
        self.Nt = len(tracks)
        
    def forward(self, X, **kwargs):
        pred = self.model(X)
        if isinstance(pred, list):
            pred = torch.hstack(pred)

        npred = []
        for tup in self.tracks:
            npred.append(pred[:,tup].mean(dim = 1,keepdim = True))
        pred = torch.hstack(npred)
        return pred
    # helper function to call batched_predict for wrapper with model.predict()
    def predict(self, X, device = None, enable_grad = False, batchsize = None, 
                shift_sequence = None, random_shift = False):
        if device is None:
            if hasattr(self.model, 'device'):
                device = self.model.device
        if batchsize is None:
            if hasattr(self.model, 'batchsize'):
                batchsize = self.model.batchsize
            else:
                batchsize = 1
        if shift_sequence is None:
            if 'shift_sequence' in self.model.__dict__:
                shift_sequence = self.model.shift_sequence
        if random_shift is None:
            if 'random_shift' in self.model.__dict__:
                random_shift = self.model.random_shift
        predout = batched_predict(self, X, device = device, batchsize = batchsize, 
                                  shift_sequence = shift_sequence, 
                                  random_shift = random_shift, 
                                  enable_grad = enable_grad)
        return predout

def only_keep_top_dense(grad, x, top, dilute_to = 5, dilute_type = 'weighted'):
    '''
    Only keep the top attributions along all tracks
    with their position
    Parameters
    ----------  
    grad : np.ndarray
        Attributions for all four bases
    x : np.ndarray
        One hot encoded sequence
    top : int
        Number of positions to keep with attributions
        Reverse function will set all other positions to zero because they
        did not pass the threshold
    dilute_to: int NOT IMPLEMENTED
        Adds weights of surrunding base withing a windw of dilute_to to all bases
        This will return also bases with small contribution in vicinity of motif
        NOT IMPLEMENTED
    '''
    ngrashape = list(np.shape(grad))
    ngrashape[-2] += 1 # add dimension to channel for position
    ngrashape[-1] = top # reduce length to of attributions to top
    ngra = np.zeros(ngrashape)
    xshape = x.shape
    if len(xshape) == 3 and len(ngrashape) == 3:
        try: 
            x = np.squeeze(x, axis = 0)
        except ValueError:
            raise ValueError(f"Squeezing failed of one-hot sequence array shape = {x.shape} for attributions of shape {grad.shape}")
    
    if dilute_to is not None:
        if isinstance(dilute_to, (int, float, np.integer)):
            # Create a kernel for averaging
            dilute_to = int(dilute_to)
            dilute_to += int(dilute_to%2 == 0) # make it odd number so that position is in the center
            if dilute_type == 'weighted':
                kernel = np.append(np.arange(1,dilute_to//2 +2),np.arange(1,dilute_to//2 +1)[::-1])
                kernel = kernel/np.sum(kernel)
            elif dilute_type == 'constant':
                kernel = np.ones(dilute_to) / dilute_to

        elif isinstance(dilute_to, np.ndarray): # give it kernel with different distribution if wanted, e.g.
            # more weight on center
            if dilute_to.ndim == 1:  
                kernel = np.copy(dilute_to)
                dilute_to = len(dilute_to)

    if len(xshape) == 3 and len(ngrashape) == 4:
        for i in range(xshape[0]):
            for t in range(np.shape(grad)[1]): # iterate over tracks
                # looking for largest at reference
                graatref = np.sum(np.absolute(grad[i][t]*x[i]), axis = -2)
                if dilute_to is not None:
                    graatref = np.convolve(np.pad(graatref, dilute_to//2, mode= 'reflect'), kernel, mode='valid')
                lcn = np.argsort(-graatref,axis = -1)
                lcn = np.sort(lcn[:top])
                # add the position of the top attributions as a fifth channel
                ngra[i][t] = np.append(grad[i][t][...,lcn], lcn[None, :], axis = -2)
                
    elif len(xshape) == 2 and len(ngrashape) == 3:
        for t in range(np.shape(grad)[0]): # iterate over tracks
            # looking for largest at reference
            graatref = np.sum(np.absolute(grad[t]*x), axis = -2)
            if dilute_to is not None:
                graatref = np.convolve(np.pad(graatref, dilute_to//2, mode= 'reflect'), kernel, mode='valid')
            lcn = np.argsort(-graatref,axis = -1)
            lcn = np.sort(lcn[:top])
            # add the position of the top attributions as a fifth channel
            ngra[t] = np.append(grad[t][...,lcn], lcn[None, :], axis = -2)
            
    return ngra


def takegrad(x, model, tracks=None, ensemble = 1, output = 'attributions', top=None,
             device = None):
    '''
    Returns the gradient of the model with respect to the individual bases
    
    Parameters
    ----------
    x : np.ndarray, list of np.ndarray, torch.Tensor or list of torch.Tensor
        Can either be one hot encoded sequences of shape = (N_seq, 4, L_seq) or
        list of paired sequences, for example RNA and paired DNA sequences
    model : torch.Module 
        CNN model to make predictions with
    ensemble : int
        If > 1, sets model to train to use drop-out as approximation for 
        different models to get attributions from their ensemble
        Then averages over attributions from all ensembles to improve them
    tracks : list of int 
        Indeces for selected tracks
        None will use all tracks
        If tracks is list of lists, it will use cnn_average_wrapper to create the average
        of the tracks in each list
    output : str
        'attributions' will return the corrected gradient subtracted by the mean
        'grad' will return the gradient without correction
        'tism' will return approximated ism values
    top : int 
        Determines how many positions will be kept in final attribution array
        adds another channel to the array with position of the attribution in 
        the full array
    Returns
    -------
    grad : numpy.ndarray or torch tesnsor
        If x is a list of n_type arrays or tensors
        the output will be of shape = (n_type, N_seqs, tracks, L_seq, channels)
        If x is a single array or tensor, the output will be of shape = (N_seqs, tracks, L_seq, channels)

    '''

    if tracks is not None:
        if isinstance(tracks, (np.integer, int)):
            tracks = [tracks]
        if isinstance(tracks[0], list) or isinstance(tracks[0], np.ndarray):
            model = cnn_average_wrapper(model, tracks)
            tracks = [i for i in range(len(tracks))]
    
    if isinstance(x, list):
        x = [torch.Tensor(xi) for xi in x]
    else:
        x = torch.Tensor(x)
    
    if ensemble > 1:
        model.train()
    
    # chekk if device is None, then use the device of the model
    if device is None:
        if 'device' in model.__dict__:
            device = model.device
        else:
            device = 'cpu'
    model.to(device)

    grad = []    
    if isinstance(x,list):
        Nin = len(x)
        grad = [[] for n in range(Nin)]
        for i in range(x[0].size(dim = 0)):
            xi = []
            for n in range(Nin):
                xij = torch.clone(x[n][[i]])
                xij.requires_grad = True
                xi.append(xij)
            gra = [[[] for e in range(ensemble)] for n in range(Nin)]
            # Iterate over the number of ensembles, if 1, this is just a 
            # technical difference
            for e in range(ensemble):
                # we use predict here because it handles padding and batching
                pred = batched_predict(model, xi, enable_grad = True, device=device)
                if isinstance(pred, list):
                    pred = torch.cat(pred, axis = 1)
                if tracks is None:
                    tracks = np.arange(np.shape(pred)[1])
                for t, tr in enumerate(tracks):
                    pred[0, tr].backward(retain_graph = True)
                    for n, xij in enumerate(xi):
                        gr = xij.grad.clone().cpu().numpy()
                        gra[n][e].append(gr)
                        xij.grad.zero_()
            for n in range(Nin):
                # take mean over ensemble
                gra[n] = np.mean(gra[n],axis = 0)
            
            # Perform corrections to gradient
            for n in range(Nin):
                # concatenate tracks
                gra[n] = np.concatenate(gra[n],axis = 0)
                if 'attributions' in output:
                    gra[n] = correct_by_mean(gra[n])
                if top is None and 'ism' in output:
                    # subtract the gradient at the reference sequence to approximate ism
                    gra[n] = gra[n] - np.sum(gra[n]*x[n][i].numpy(), axis = -2, keepdims = True)
                # Only keep top attributions along all tracks
                if top is not None:
                    gra[n] = only_keep_top_dense(gra[n], xi[n].squeeze(0).detach().cpu().numpy(), top)
                grad[n].append(gra[n])
                
        for n in range(Nin):
            grad[n] = np.array(grad[n])

    else:
        start = time.time()
        for i in range(x.size(dim = 0)):
            xi = torch.clone(x[[i]])
            xi.requires_grad = True
            # need the list of list for ensemble
            gra = [[] for e in range(ensemble)]
            for e in range(ensemble):
                pred = batched_predict(model, xi, enable_grad = True, device=device)
                if isinstance(pred, list):
                    pred = torch.cat(pred, axis = 1)
                if tracks is None:
                    tracks = np.arange(np.shape(pred)[1])
                for t, tr in enumerate(tracks):
                    pred[0, tr].backward(retain_graph = True)
                    gr = xi.grad.clone().cpu().numpy()
                    gra[e].append(gr)
                    xi.grad.zero_()
            # take mean over ensemble
            gra = np.mean(gra, axis = 0)
            gra = np.concatenate(gra,axis = 0)
            if 'attributions' in output:
                gra = correct_by_mean(gra)
            if top is None and 'ism' in output:
                # subtract the gradient at the reference sequence to approximate ism
                gra = gra - np.sum(gra*x[i].numpy(), axis = -2, keepdims = True)
            # Only keep top attributions along all tracks
            if top is not None:
                gra = only_keep_top_dense(gra, xi.squeeze(0).detach().cpu().numpy(), top)
                
            grad.append(gra)
        grad = np.array(grad)
        end = time.time()
        print('TISM time for', grad.shape, len(tracks),  end-start)
    return grad




def correct_deeplift(dlvec, freq, channel_axis = -2):
    '''
    Compute hypothetical attributions from multipliers to a baseline frequency
    '''
    if len(np.shape(freq)) < len(np.shape(dlvec)):
        freq = np.ones_like(dlvec) * freq[:,None] 
    # for each base, we have to "take a step" from the frequency in baseline to 1.
    # For example to get from 0.25 in baseline to 1, we have to take steps of 0.75.
    # This whole procedure is similar to ISM, where we remove the reference base and set it to zero, so take a step of -1, and then go a step of one in the direction of the alternative base. DeepLift returns the regression coefficients, which need to be scaled to the steps that are taken to arrive at the base we're looking at.
    negmul = np.sum(dlvec * freq, axis = channel_axis, keepdims = True)
    # this is the step that we take to get from the baseline to the reference base
    dlvec = dlvec - negmul
    return dlvec



class cnn_multi_deeplift_wrapper(torch.nn.Module):
    '''
    Wrapper for cnn_multi to use tangermeme's deepshap on one of the inputs
    
    Parameters
    ----------
    model : pytorch.nn.module
        The cnn_multi object
    N : int 
        Number of sequences that the model takes as input
    n : int 
        The index of the sequence that will be investigated with the wrapper
        
    '''
    def __init__(self, model, N = 2, n = 0):
        super(cnn_multi_deeplift_wrapper, self).__init__()
        self.model = model
        self.N = N
        self.n = n
    
    def forward(self, X, args):
        '''
        X : torch.Tensor 
            is the sequence of interest
        arg : list of torch.Tensors 
            contains other sequences in order as given originally 
        '''
        x = []
        t = 0
        for i in range(self.N):
            if i == self.n:
                x.append(X)
                t = 1
            else:
                x.append(args[i-t])
        return self.model(x)



from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear, _maxpool
from tangermeme.ersatz import dinucleotide_shuffle, shuffle

def deeplift(x, model, tracks, baseline = None, batchsize = None, 
             output = 'attributions', raw_outputs = True, 
             hypothetical = False, shuffle_function = 'dinucleotide_shuffle', 
             top=None, device = None):
    
    '''
    Uses tangermeme's deep_lift_shap to generate attributions for single sequence, 
    multi-sequence, single output, and multi-output models.
    Attributions can come in many different flavors, depending on the given
    parameters.
    
    Parameters
    ----------
    x : np.ndarray, list of np.ndarray
        Can either be one hot encoded sequences of shape = (N_seq, 4, L_seq) or
        list of paired sequences, for example RNA and paired DNA sequences
    model : torch.Module 
        CNN model to make predictions with
    tracks : list of int 
        Indeces for selected tracks
        None will use all tracks
        If tracks is list of lists, it will use cnn_average_wrapper to create the average
        of the tracks in each list
    baseline: 
        can either be array of random baseline sequences, baseline frequencies,
        or integer. Integer will use tangermemes shuffle or dinucleotide_shuffle
        in tangermeme 
    batchsize : int 
    
    output : str
        'multipliers' will return the multipliers to the baseline frequencies
        'attributions' will return the attributions corrected with the baseline frequencies to "hypothetical" attributions
    raw_outputs : bool
        tangermeme function returns multipliers to every baseline
        Default should be true
    hypothetical:
        tangermeme function returns its hypothetical attributions, which are 
        derived from each baseline sequence individually and then averaged. 
        Note, that tangermeme also multiplies by x-0 automatically when this
        is set to true
    Returns
    -------
    
    TODO
    Does now work with multi-sequence input CNN yet, something wrong with batch_norm mean.
    '''

    
    if isinstance(x, list):
        Nin = len(x)
        Ni = x[0].shape[0]
        x = [torch.Tensor(xi) for xi in x]
    else:
        Ni = x.shape[0]
        x = torch.Tensor(x)
        
    model.eval()
    
    # chekk if device is None, then use the device of the model
    if device is None:
        if 'device' in model.__dict__:
            device = model.device
        else:
            device = 'cpu'
    model.to(device)
        
    # batch size will be taken from model
    if batchsize is None:
        if 'batchsize' in model.__dict__:
            batchsize = model.batchsize
        else:
            batchsize = 1
    
    n_shuffles = None
    # if baseline is an integer, it will be used as the number of shuffles, and
    # the shuffle function will be used as the baseline
    # if baseline is an array, it will be used as the baseline frequencies
    if isinstance(baseline, int):
        n_shuffles = baseline
        if shuffle_function == 'dinucleotide_shuffle':
            baseline = dinucleotide_shuffle
        if shuffle_function == 'shuffle':
            baseline = shuffle
        else:
            baseline = shuffle_function
    # create baseline frequencies if not given        
    elif baseline is None:
        if isinstance(x, list):
            baseline = []
            for xi in x:
                basefreq = 1./xi.size(-2)
                baseline.append(torch.ones_like(xi[:batchsize])*basefreq)
        else:
            basefreq = 1./x.size(-2)
            baseline = torch.ones_like(x[:batchsize])*basefreq

    # make baseline a tensor and check if sizes match the input
    # if they don't, expand the baseline to the size of the input
    if baseline is not None:
        if isinstance(x, list):
            if isinstance(baseline, list):
                baseline = [torch.Tensor(bl) for bl in baseline] 
            if isinstance(baseline, np.ndarray):
                baseline = [torch.Tensor(baseline) for i in range(Nin)]
            if isinstance(baseline, list):
                if baseline[0][:batchsize].size() != x[0][:batchsize].size():
                    # always need to expand baseline to number of data points to 
                    baseline = [bl.unsqueeze(0).expand((x[b].size(0),) + tuple(bl.size())) for b, bl in enumerate(baseline)]
                    # if only frequencies were given, need to expand it along the length of the sequences
                    if baseline[0][:batchsize].size() != x[0][:batchsize].size():
                        baseline = [bl.unsqueeze(-1).expand(tuple(bl.size())+(x[b].size(-1),)) for b, bl in enumerate(baseline)]
                baseline = [bl.unsqueeze(1) for bl in baseline]
        else:
            if isinstance(baseline, np.ndarray):
                baseline = torch.Tensor(baseline)
            if isinstance(baseline, torch.Tensor):
                if baseline[:batchsize].size() != x[:batchsize].size():
                    # always need to expand baseline to number of data points to 
                    baseline = baseline.unsqueeze(0).expand((x.size(0),) + tuple(baseline.size()))
                    # if only frequencies were given, need to expand it along the length of the sequences
                    if baseline.size() != x.size():
                        baseline = baseline.unsqueeze(-1).expand(tuple(baseline.size())+(x.size(-1),))
                baseline = baseline.unsqueeze(1)
    
    # when shift_sequence in my model is not None, inputs will be padded
    if hasattr(model, 'shift_sequence'):
        shift_sequence = model.shift_sequence
        if model.shift_sequence is not None:
            if n_shuffles is not None: # if dinucleotide shuffle is used, we need to pad with binary values only
                if isinstance(x, list):
                    x = [torch.nn.functional.pad(xi, (model.shift_sequence, model.shift_sequence), mode = 'circular') for xi in x]
                else:
                    x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'circular')
            else:
                if isinstance(x, list):
                    x = [torch.nn.functional.pad(xi, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25) for xi in x]
                    baseline = [torch.nn.functional.pad(bl, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25) for bl in baseline]
                else:
                    x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
                    baseline = torch.nn.functional.pad(baseline, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
    else:
        shift_sequence = None

    if tracks is not None:
        if isinstance(tracks, (np.integer, int)):
            tracks = [tracks]
        if isinstance(tracks[0], list) or isinstance(tracks[0], np.ndarray):
            model = cnn_average_wrapper(model, tracks)
            tracks = [i for i in range(len(tracks))]
    
    start = time.time()
    if isinstance(x, list):
        grad = []
        for n in range(Nin):
            # use this to get deeplift of sequence input n in the input list
            model = cnn_multi_deeplift_wrapper(model, N = Nin, n = n)
            gra = []
            for b in range(0, Ni, batchsize):
                for t, tr in enumerate(tracks):
                    xi = x[n][b:b+batchsize]
                    # might have to make this a tensor to work with tangermeme
                    # otherwise it might think that these are distinct additional
                    # inputs
                    bs = min(batchsize, Ni-b)
                    if n_shuffles is not None:
                        bl = baseline
                    else:
                        bl = baseline[n][:bs]
                    # additional inputs given to tangermeme
                    args = [xj[b:b+batchsize] for j, xj in enumerate(x) if j != n] 
                    
                    gr = deep_lift_shap(model, xi[b:b+bs], args = args, target = tr, 
                                        references = bl, n_shuffles = n_shuffles, 
                                        device=device, raw_outputs = raw_outputs, 
                                        hypothetical = hypothetical, 
                                        additional_nonlinear_ops = {SoftmaxNorm: _nonlinear, EXPmax : _nonlinear}
                                        ).cpu().detach().numpy()
                    if raw_outputs:
                        gr = np.mean(gr, axis = 1)
                    if shift_sequence is not None:
                        gr = gr[..., shift_sequence: np.shape(grad)[-1]-shift_sequence]
                        bl = bl[..., shift_sequence: np.shape(baseline)[-1]-shift_sequence]
                    # correct the deeplift output
                    if not hypothetical and output == 'attributions':
                        # correct the deeplift output
                        gr = correct_deeplift(gr, bl.detach().squeeze(1).numpy())
                    
                    # Only keep top attributions along all tracks
                    if top is not None:
                        # add dimension to channel for position
                        gr = only_keep_top_dense(gr, xi[b:b+bs].detach().cpu().numpy(), top)
                    
                    gra.append(gr)
            gra = np.concatenate(gra, axis = 0)
            # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
            gra = np.transpose(gra, axes = (1,0,2,3))
            # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
            grad.append(gra)
        
    else:
        grad = []
        for b in range(0, Ni, batchsize):
            gra = []
            for t, tr in enumerate(tracks):
                # if you want to represent certain custom functions with replacement functions use:
                # additional_nonlinear_ops = {SoftmaxNorm: _nonlinear}
                bs = min(batchsize, Ni-b)
                bl = baseline[:bs]
                gr = deep_lift_shap(model, x[b:b+bs], target = tr, 
                                    references = bl, n_shuffles = n_shuffles, device=device, 
                                    raw_outputs = raw_outputs, hypothetical = hypothetical, 
                                    additional_nonlinear_ops = {SoftmaxNorm: _nonlinear, EXPmax : _nonlinear, nn.GELU : _nonlinear}
                                    ).cpu().detach().numpy()
                if raw_outputs:
                    gr = np.mean(gr, axis = 1)
                #deltas = (predx[:,[tr]] - predbase[:,[tr]]) - torch.sum(gr * (x.unsqueeze(1)-baseline), dim = (-1,-2))
                if shift_sequence is not None:
                    gr = gr[..., shift_sequence: np.shape(gr)[-1]-shift_sequence]
                    bl = bl[..., shift_sequence: np.shape(baseline)[-1]-shift_sequence]
                # correct the deeplift output
                if not hypothetical and output == 'attributions':
                    # correct the deeplift output
                    gr = correct_deeplift(gr, bl.detach().squeeze(1).numpy())
                # Only keep top attributions along all tracks
                if top is not None:
                    # add dimension to channel for position
                    gr = only_keep_top_dense(gr, x[b:b+bs].detach().cpu().numpy(), top)
                
                gra.append(gr)
            grad.append(gra)
        grad = np.concatenate(grad, axis = 0)
        # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
        grad = np.transpose(grad, axes = (1,0,2,3))
        # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    
    end = time.time()
    print('Deeplift time for', grad.shape, len(tracks),  end-start)
    return grad  

from captum.attr import DeepLift, DeepLiftShap, GradientShap, IntegratedGradients
# Does not work well for all models: 
# Needs GELU manually added to list of nonlinear functions in source code for captum.deeplift and deepliftshap

# TODO: implement multi-sequence feature.
# additional_forward_args can be used for multi-seq model, similar to tangermeme
def captum_sequence_attributions(x, model, tracks = None, attribution_method = 'IntegratedGradients', 
                                 basefreq = None, batchsize = None, output = 'attributions', 
                                 shift_sequence = 0):
    '''
    Currently only supports single seq input model
    
    Parameters
    ----------
    x : np.ndarray, list of np.ndarray
        Can either be one hot encoded sequences of shape = (N_seq, 4, L_seq) or
        list of paired sequences, for example RNA and paired DNA sequences
    model : torch.Module 
        CNN model to make predictions with
    tracks : list of int 
        Indeces for selected tracks
        None will use all tracks
        If tracks is list of lists, it will use cnn_average_wrapper to create the average
        of the tracks in each list
    basefreq: 
        background frequency of bases across genome, 0.25 is default
    batchsize : int 
    
    output : str
        'multipliers' will return the multipliers to the baseline frequencies
        'attributions' will return the attributions corrected with the baseline frequencies to "hypothetical" attributions
    '''
    if attribution_method.lower() == 'deepliftshap':
        dl = DeepLiftShap(model.eval(), multiply_by_inputs=False)
    elif 'deeplift' in attribution_method.lower():
        dl = DeepLift(model.eval(), multiply_by_inputs=False, eps=1e-6)
    elif  attribution_method.lower() == 'gradientshap':
        GradientShap(model.eval(), multiply_by_inputs = False)
    elif  attribution_method.lower() == 'integratedgradients':
        dl = IntegratedGradients(model.eval(), multiply_by_inputs = False)
    
    x = torch.Tensor(x)
    if tracks is None: 
        Ntracks = batched_predict(model, x[[0]]).size(-1)
        tracks = np.arange(Ntracks, dtype = int)
    elif isinstance(tracks, (int, np.integer, float)):
        tracks = [int(tracks)]
    
    # batch size will be taken from model
    if batchsize is None:
        if hasattr(model, 'batchsize'):
            batchsize = model.batchsize
        else:
            batchsize = 1
    
    fnum = x.size(-2)
    if basefreq is None:
        basefreq = 1./x.size(-2)
    
    if isinstance(basefreq, float):
        basefreq = torch.ones(fnum) * basefreq
    
    # basefreq needs to be an array of length fnum by now, can also be provided as an array
    baseline = torch.ones_like(x)*basefreq[:,None]
    # Version were each sequence is tested against a random set of permutations is not implemented yet and requires way more resources
    # when shift_sequence in my model is not None, inputs will be padded
    if hasattr(model, 'shift_sequence'):
        if model.shift_sequence is not None:
            shift_sequence = model.shift_sequence
            x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
            baseline = torch.nn.functional.pad(baseline, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
        
    grad = []
    deltas = [] # Deltas determine if the completeness is fulfilled, i.e. that the sum 
    # multipliers times one-hot encoded sequence is equal to diff predictins.
    # perform deeplift independently for each output track
    start = time.time()
    for t, tr in enumerate(tracks):
        gr = []
        delt = []
        # use batchsize to avoid memory issues
        for b in range(0,x.size(dim = 0), batchsize):
            xb = x[b:b+batchsize].to(model.device)
            bb = baseline[b:b+batchsize].to(model.device)
            attributions, delta = dl.attribute(xb, bb, target=int(tr), return_convergence_delta=True)
            delt.append(delta.cpu().detach().numpy())
            gr.append(attributions.cpu().detach().numpy())
        grad.append(np.concatenate(gr, axis = 0))
        delt = np.concatenate(delt)
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(np.array(grad), axes = (1,0,2,3))
    grad = grad[..., shift_sequence: np.shape(grad)[-1]-shift_sequence]
    # correct the deeplift output
    if output == 'attributions':
        grad = correct_deeplift(grad, basefreq.detach().numpy())
    print(attribution_method, np.shape(grad), time.time()-start)
    return grad           






def extract_kernelweights_from_state_dict(state_dict, kernel_layer_name = 'convolutions.conv1d', full_name = False, concatenate = True):
    '''
    iterates through state_dict and returns parameters of convolutional kernels
    
    Returns
    -------
    weights, biases, kernel_names
    '''
    
    weights = []
    biases = []
    motifnames = []

    # if multi sequence input, generate additional names for kernels to
    # determine from which cnn they came
    unique_names = None
    if full_name == False:
        kernelmats = []
        for namep in state_dict:
            if kernel_layer_name in namep and namep.rsplit('.',1)[-1] == 'weight': 
                kernelmats.append(namep.split('.'))
        if len(kernelmats) > 1:
            unique_names = []
            for k in range(len(kernelmats)):
                kernelnames = kernelmats[k:k+1] + kernelmats[:k] + kernelmats[k+1:]
                unique_name = reduce(np.setdiff1d, kernelnames)
                if len(unique_name) > 0:
                    unique_names.append(unique_name[0])
                else:
                    unique_names.append('CNN'+str(k))
    
    i =0
    if full_name:
        for namep in state_dict:
            if kernel_layer_name == namep.replace('.weight', ''):
                kernelweight = state_dict[namep].detach().cpu().numpy()
                weights.append(kernelweight)
                if namep.rsplit('.',1)[0]+'.bias' in state_dict:
                    bias = state_dict[namep.rsplit('.',1)[0]+'.bias'].detach().cpu().numpy()
                    biases.append(bias)
                else:
                    biases.append(np.zeros(len(kernelweight)))
                motifnames.append(np.array(['filter'+str(j) for j in range(len(kernelweight))]))

    else:
        for namep in state_dict:
            if kernel_layer_name in namep:
                add = ''
                if namep.rsplit('.',1)[-1] == 'weight':
                    kernelweight = state_dict[namep].detach().cpu().numpy()
                    weights.append(kernelweight)
                    if unique_names is not None:
                        add = '_'+unique_names[i]
                    motifnames.append(np.array(['filter'+str(j)+add for j in range(len(kernelweight))]))
                    i += 1
                    if namep.rsplit('.',1)[0]+'.bias' in state_dict:
                        bias = state_dict[namep.rsplit('.',1)[0]+'.bias'].detach().cpu().numpy()
                        biases.append(bias)
                    else:
                        biases.append(np.zeros(len(kernelweight)))
    if concatenate:
        weights, biases, motifnames = np.concatenate(weights,axis = 0), np.concatenate(biases, axis = 0), np.concatenate(motifnames, axis = 0)
    return weights, biases, motifnames
    

def kernel_assessment(model, X, Y, testclasses = None, stppm= None, Nppm = None, num_mean = 200000, nullify = 'mean', device = None, **extract_kernel_kwargs):
    '''
    Computes kernel effect, and various kernel importances per track from
    setting the kernel activation after the convolution to the mean of 
    'num_mean' random sequences.
    
    Parameters
    ---------
    
    testclasses: numpy.array
        Class label for each track, for examplel B-cells
    stppm : int 
        kernel index to start with
    Nppm : int 
        Number of kernels to run 
    num_mean : int 
        Number of random sequences for mean estimation
    nullify: str 
        Method to nullify kernel activation (mean, median, zero)
    
    Returns
    -------
    ppms : np.ndarray
        ppms generated with scaling and softmax from kernel parameters
    importance : np.2darray
        change in correlation across data points for each experiment shape = (kernelxexperiments)
    mseimportance : np.2darray
        change in mse across data points for each experiment shape = (kernelxexperiments)
    effect : np.2darray
        weighted change in prediction, prediction is weighted by mse of each gene, if mse is small then not much of an effect, shape = (kernelxexperiments)
    abseffect : np.2darray
        sum of difference of predictions shape = (kernelxexperiments). This is the main matrix we're working with because it determines the direction and effect size of the kernel
    genetraincorr : dict for each testclass with array of shape (N_points,)
        data point wise correlation to prediction across testclasses for full model (same as in pnt_corr*.dat files)
    geneimportance : dict for each testclass with array of shape (N_kernel, N_points)
        difference in data point wise correlation to prediction across testclasses between full model and nullified kernel 
    
    '''
    
    weights, biases, motifnames = extract_kernelweights_from_state_dict(model.state_dict(), concatenate = False, **extract_kernel_kwargs)
    # iterate over weights
    importance = [] # change in correlation across data points for each experiment shape = (kernelxexperiments)
    mseimportance = [] # change in mse across data points for each experiment shape = (kernelxexperiments)
    effect = [] # weighted change in prediction, prediction is weighted by mse of each gene, if mse is small then not much of an effect, shape = (kernelxexperiments)
    abseffect = [] # absoltue difference of predictions shape = (kernelxexperiments)
    genetraincorr = {}
    if testclasses is None:
        testclasses = np.zeros(np.shape(Y)[1]).astype(str)
    # generate genewise kernel impact matrix for all test classes
    geneimportance = {}
    if device is None:
        device = model.device
   
    # use random sequences to compute mean activations AND pwms from sequence alignments
    seq_seq = generate_random_onehot(model.l_kernels, num_mean) # generate 200000 random sequences with length l_kernels
    i =0
    islist = isinstance(X, list) # check if several or only one input
    # compute unmasked performances
    # predict original training predictions
    Ypredtrain = model.predict(X)
    # compute the correlation between data points for each output track
    traincorr = correlation(Ypredtrain,Y, axis = 0)
    trainmse = mse(Ypredtrain, Y, axis = 0)
    # compute correlation for each training gene
    for t, testclass in enumerate(np.unique(testclasses)):
        consider = np.where(testclasses == testclass)[0]
        if len(consider) > 1:
            genetraincorr[testclass] = correlation(Ypredtrain[:,consider],Y[:,consider], axis = 1)
    
    for i, motifname in enumerate(motifnames):
        kernelweight = weights[i]
        bias = biases[i]
        if stppm is not None: 
            kernelweight = kernelweight[stppm : stppm+Nppm]
            bias = bias[stppm : stppm+Nppm]
            motifname = motifname[stppm : stppm+Nppm]
        else:
            stppm, Nppm = 0, len(motifname)
        
        seqactivations = kernels_seqactivations_from_seqlets(kernelweight, seq_seq, biases = bias, activation_func = None, device = device)
        if nullify == 'mean':
            motifmeans = np.mean(seqactivations, axis = 1).astype(float)
        if nullify == 'median':
            motifmeans = np.median(seqactivations, axis = 1).astype(float)
        if nullify == 'zero':
            motifmeans = np.zeros(seqactivations.shape[0])
        
        t0 = time.time()
        for m in range(len(motifname)):
            # make predictions from models with meaned activations from kernels
            if m%10==0:
                print(m+stppm, '/' , stppm+Nppm, round(time.time()-t0,2))
                t0 = time.time()
            if islist:
                mask = [i,m+stppm]
            else:
                mask = m+stppm
            
            Ymask = model.predict(X, mask = mask, mask_value = motifmeans[m])
            # importance as difference in correlation between full model to train data and masked model to train data
            importance.append(np.around(correlation(Ymask,Y, axis = 0) - traincorr,3))
            mseimportance.append(np.around(mse(Ymask,Y, axis = 0) - trainmse,6))
            # effect for a track shows the direction that the kernel causes on average over all genes
            # it is weighted by how much it changes the mse of each gene
            effect.append(np.around(np.sum((Ypredtrain-Ymask)**3/np.sum((Ypredtrain-Ymask)**2,axis = 1)[:,None],axis = 0),6))
            abseffect.append(np.around(np.sum(Ypredtrain-Ymask,axis = 0),6))
            # compute the importance of the kernel for every gene
            for t, testclass in enumerate(np.unique(testclasses)):
                consider = np.where(testclasses == testclass)[0]
                if len(consider) > 1:
                    if testclass in geneimportance:
                        geneimportance[testclass].append(np.around(correlation(Ymask[:,consider],Y[:,consider], axis = 1) - genetraincorr[testclass],3))
                    else:
                        geneimportance[testclass] = [np.around(correlation(Ymask[:,consider],Y[:,consider], axis = 1) - genetraincorr[testclass],3)]
   
    motifnames = np.concatenate(motifnames)
    weights = np.concatenate(weights, axis = 0)
    biases = np.concatenate(biases, axis = 0)
    
    importance = np.array(importance)
    mseimportance = np.array(mseimportance)
    effect = np.array(effect)
    abseffect = np.array(abseffect)
    for k in geneimportance:
        geneimportance[k] = np.array(geneimportance[k])
    # create ppms directly from kernel matrix
    ppms = np.around(kernel_to_ppm(weights[:,:,:], kernel_bias =biases),3)
    
    return ppms, motifnames, importance, mseimportance, effect, abseffect, geneimportance, genetraincorr
        
     

def indiv_network_contribution(model, X, Y, testclasses, outname, names):
    # pick a random set of input sequences to compute mean output of individual networks
    randomseqs = np.random.permutation(len(X[0]))[:7000]
    # predict original predictions to compare losses to 
    Ypredtrain = model.predict(X)
    Ymask = []
    netbsize = 10 # batchsize for computing the mean
    for i in range(len(X)):
        # get mean representations for individual networks
        mean_rep = []
        for r in range(0,len(randomseqs), netbsize):
            rand = randomseqs[r:r+netbsize]
            mrep = model.forward([torch.Tensor(x[rand]).to(model.device) for x in X], location = '0', cnn = i)
            if isinstance(mrep,list):
                print(mrep[0].size(), len(mrep))
                print('--test_individual_network : Does not work with direct, difference or any analytic connection')
                sys.exit()
            mrep = mrep.detach().cpu().numpy()
            mean_rep.append(mrep)
        mean_rep = np.concatenate(mean_rep, axis = 0)
        #print(mean_rep[0], mean_rep[1], mean_rep[2])
        #print(np.std(mean_rep,axis = 0))
        mean_rep = np.mean(mean_rep,axis = 0)
        # predict values with masked networks
        Ymask.append(model.predict(X, mask = i, mask_value = torch.Tensor(mean_rep).to(model.device)))
        #print(i, mean_rep)
    
    for tclass in np.unique(testclasses):
        # go through all the different classes, for example cell types
        consider = np.where(testclasses == tclass)[0]
        genecont = []
        header= ['MSEtoReal', 'CorrtoReal']
        # compute the performance for this class for the full predictions
        trainmse = mse(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
        traincorr = correlation(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
        #print(tclass)
        #print(traincorr)
        for i in range(len(X)):
            header.append('CNN'+str(i)+'DeltaMSE')
            header.append('CNN'+str(i)+'DeltaCorr')
            #header.append('CNN'+str(i)+'DeltaAvgYpred')
            # compare the performance of the masked predictions with the full predicdtions
            MSEdif = mse(Ymask[i][:,consider],Y[:,consider],axis = 1) - trainmse
            genecont.append(MSEdif)
            Corrdif = correlation(Ymask[i][:,consider],Y[:,consider],axis = 1) - traincorr
            genecont.append(Corrdif)
            #DYdiff = np.mean(Ymask[i][:,consider] - Ypredtrain[:,consider], axis - 1)
            
        np.savetxt(outname+'_netatt'+tclass+'.dat', np.concatenate([[names],np.around([trainmse,traincorr],4), np.around(np.array(genecont),6)],axis = 0).T, header = 'Gene '+' '.join(np.array(header)), fmt = '%s')




