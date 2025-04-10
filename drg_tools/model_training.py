# model_training.py 

'''
Functions for model training and batched predictions
TODO: 
- solve resetting optimizer after gradient explosion
- introduce masks into training

'''


import torch
import numpy as np
import sys, os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from collections import OrderedDict
from torch import Tensor
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .model_utils import MyDataset, get_device
from .modules import loss_dict, func_dict, final_convolution
from .torch_regression import torch_Regression
import time






def create_sets(n_samples, folds, fold, seed=1010, Yclass=None, genenames=None):
    """
    Create random train, test, and validation sets using permutations.
    Samples can be split into different classes, ensuring each set has the same fraction of each class.

    Parameters:
    ----------
    n_samples : int
        Total number of samples.
    folds : int or str
        Number of folds for cross-validation or a file path containing predefined sets.
    fold : int
        The fold index to use for testing.
    seed : int, optional
        Random seed for reproducibility (default is 1010).
    Yclass : np.ndarray, optional
        Class labels for the samples. If None, all samples are treated as a single class.
    genenames : np.ndarray, optional
        Names of the samples, required if `folds` is a file.

    Returns:
    -------
    trainset : np.ndarray
        Indices of the training set.
    testset : np.ndarray
        Indices of the test set.
    valset : np.ndarray
        Indices of the validation set.
    """
    if isinstance(folds, int):
        # Handle random splitting with cross-validation
        if Yclass is None:
            Yclass = np.ones(n_samples)  # Treat all samples as a single class

        np.random.seed(seed)
        permut = np.random.permutation(n_samples)
        classes = np.unique(Yclass)
        testset, valset = [], []

        valfold = (fold - 1) % folds  # Validation fold index

        for cla in classes:
            inclass = Yclass[permut] == cla
            totinclass = np.sum(inclass)

            # Calculate start and end indices for test and validation sets
            test_start = (fold * totinclass) // folds
            test_end = ((fold + 1) * totinclass) // folds
            val_start = (valfold * totinclass) // folds
            val_end = ((valfold + 1) * totinclass) // folds

            testset.append(permut[inclass][test_start:test_end])
            valset.append(permut[inclass][val_start:val_end])

        testset = np.concatenate(testset)
        valset = np.concatenate(valset)
        trainset = np.delete(np.arange(n_samples), np.append(testset, valset))

    elif os.path.isfile(folds):
        # Handle predefined sets from a file
        if genenames is None:
            raise ValueError("genenames must be provided when folds is a file.")

        with open(folds, 'r') as f:
            lines = [line.strip() for line in f if not line.startswith('#')]

        sets = [np.where(np.isin(genenames, line.split()))[0] for line in lines]
        testset = sets[(fold + 1) % len(sets)]
        valset = sets[fold]
        trainset = np.delete(np.arange(len(genenames)), np.append(testset, valset))

    else:
        raise ValueError("folds must be an integer or a valid file path.")

    return trainset, testset, valset


# Kernel initialization with statistical motif enrichment or Lasso regression
def kernel_hotstart(n_kernels, l_kernels, X, Y, kmervalues = 'Lasso', XYval = None, verbose = True, alpha = 1.):
    # minimal occurance of motif in sequence
    n_minimal = 200.
    # maximal length of initiated patterns is capped to 7, l_kernels or statistical number of different motifs
    maxlen = min(7,min(int(np.log(np.shape(X)[0]*np.shape(X)[-1]/n_minimal)/np.log(4))+1, l_kernels))
    print('Hotstart with', maxlen, 'mers')
    # Generate k-merlist from onehot encoding, and count occurrance, return occurance matrix
    kmernumbers, allkmers = kmer_count(X, maxlen)
    
    if XYval is not None:
        Xval, Yval = XYval[0], XYval[1]
        kmernumbersval, allkmers = kmer_count(Xval, maxlen, allkmers = allkmers)
    
    zscores = np.zeros((len(allkmers), np.shape(Y)[-1]))
    if kmervalues == 'Lasso':
        kmernumbers = (kmernumbers-np.mean(kmernumbers,axis = 0))/np.std(kmernumbers, axis = 0)
        # Start regularization for Lasso
        while True:
            print('Hotstart with Lasso', alpha, "Classes left", len(np.where(np.sum(zscores, axis = 0) == 0)[0]))
            model = Lasso(fit_intercept = True, alpha = alpha, warm_start = False, max_iter = 250)
            model.fit(kmernumbers, Y)
            modpred = model.predict(kmernumbers)
            if len(np.shape(modpred)) == 1:
                modpred = modpred.reshape(-1,1)
            print('Predicted correlation', ' '.join(np.around(np.diagonal(cdist(modpred.T, Y.T, 'correlation')),3).astype(str)))
            coef_ = model.coef_
            if len(np.shape(coef_)) ==1:
                coef_ = coef_.reshape(1,-1)
            ndetected = []
            for yi, ycoef in enumerate(np.absolute(coef_)):
                ndetected.append(int(np.sum(ycoef > 0)))
                if np.sum(ycoef != 0) > n_kernels and np.sum(zscores[:,yi]) == 0:
                    zscores[:, yi] = ycoef*alpha
            print('Ndtected', ' '.join(np.array(ndetected).astype(str)))
            if (np.sum(zscores, axis = 0) == 0).any():
                alpha = alpha/2.
            else:
                break
      
    zscores = np.amax(zscores,axis = 1)
    chosenk = np.argsort(-zscores)[:n_kernels]
    select_kmers = allkmers[chosenk]
    if verbose:
        print('Initialized kernels', select_kmers)
        theomodel = LinearRegression(fit_intercept = True).fit(kmernumbers[:, chosenk], Y)
        theopred = theomodel.predict(kmernumbers[:, chosenk])
        if len(np.shape(theopred)) == 1:
            theopred = theopred.reshape(-1,1)
        print('Final predicted correlation with chosenk', np.diagonal(cdist(theopred.T, Y.T, 'correlation')))
        
        if XYval is not None:
            theopred = theomodel.predict(kmernumbersval[:, chosenk])
            if len(np.shape(theopred)) == 1:
                theopred = theopred.reshape(-1,1)
            print('Final validation correlation with chosenk', np.diagonal(cdist(theopred.T, Yval.T, 'correlation')))
    # Return pwms for top k-mers
    hotpwms = np.array([pwm_from_kmer(km, l_pwm = l_kernels) for km in select_kmers])
    return hotpwms



# Generates a list of pwms with targetlen from a pwm with different or same length
# for example, creates two pwms of length 7 from pwm of length 8
def pwmset(pwm, targetlen, shift_long = True):
    if np.shape(pwm)[-1] == targetlen:
        return pwm[None, :]
    elif np.shape(pwm)[-1] > targetlen:
        if shift_long:
            npwms = []
            for l in range(np.shape(pwm)[-1]-targetlen+1):
                npwm = np.zeros((np.shape(pwm)[0], targetlen))
                npwm[:] = pwm[:, l:l+targetlen]
                npwms.append(npwm)
        else:
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            l = int((np.shape(pwm)[-1]-targetlen)/2)
            npwm[:] = pwm[:, l:l+targetlen]
            npwms = [npwm]
        return np.array(npwms)
    elif np.shape(pwm)[-1] < targetlen:
        if shift_long:
            npwms = []
            for l in range(targetlen - np.shape(pwm)[-1]+1):
                npwm = np.zeros((np.shape(pwm)[0], targetlen))
                npwm[:,l:l+np.shape(pwm)[-1]] = pwm[:]
                npwms.append(npwm)
        else:
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            l = int((targetlen-np.shape(pwm)[-1])/2)
            npwm[:,l:l+np.shape(pwm)[-1]] = pwm[:]
            npwms = [npwm]
        return np.array(npwms)
        
# Scans onehot encoded numpy array for pwms
# TODO implement pytorch version - needs only be done once before model training 
def pwm_scan(sequences, pwms, targetlen = None, activation = 'max', motif_cutoff = None, set_to = 0., verbose = False):
    # if pwms are longer than the targeted kernel size then its unclear if we should the convolution with the right side of the pwm or the left side. Each would create  a different positional pattern. Therefore we take the mean over both options all options of pwms with the target length.
    # If Pwms are smaller than the target len we use all the options of padded pwms to create scanning pattern
    if targetlen is None:
        targetlen = np.amax([len(pqm.T) for pqm in pwms])
    outscan = np.zeros((np.shape(sequences)[0], len(pwms), np.shape(sequences)[-1]-targetlen+1))
    if verbose:
        print('Scanning', len(sequences), 'sequences with', len(pwms), 'PWMs')
    for p, pwm in enumerate(pwms):
        if p%10 == 0 and verbose:
            print(p)
        setps = pwmset(pwm, targetlen = targetlen)
        for l in range(np.shape(sequences)[-1]-targetlen+1):
            setscans = np.sum(sequences[:,None,:,l:l+targetlen] * setps[None, :, :, :], axis = (-1, -2))
            # max activation across all shorter subpwms
            if activation == 'max':
                outscan[:,p, l] = np.amax(setscans, axis = -1)
            elif activation == 'mean':
                outscan[:,p, l] = np.mean(setscans, axis = -1)
    # pwms also assign values to partial fits of the sequence, to remove these partial fits one can 
    if motif_cutoff is not None:
        outscan[outscan < motif_cutoff] = set_to
    return outscan

# Might be better to use torch module instead
def l1_loss(w):
    return torch.abs(w).mean()
    
def L2_loss(w):
    return torch.square(w).mean()
  
def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)
    
def load_model(model, PATH, device):
    # avoids memory error on gpu: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
    state_dict = torch.load(PATH, map_location = 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)


class hook_grads():
    '''
    Unclear for what to use here
    '''
    def __init__(self):
        self.grads = {}
    def save_grad(self,name):
        def hook(grad):
            self.grads[name] = grad
        return hook

def fit_model(model, X, Y, XYval = None, sample_weights = None, loss_function = 'MSE', 
              validation_loss = None, loss_weights = 1, val_loss_weights = 1, batchsize = None, 
              device = 'cpu', optimizer = 'Adam', optim_params = None, optim_weight_decay = None, 
              verbose = True, lr = 0.001, kernel_lr = None, hot_start = False, hot_alpha = 0.01, 
              warm_start = False, outname = 'Fitmodel', adjust_lr = 'F', patience = 25, 
              init_adjust = True, reduce_lr_after = 1, keepmodel = False, load_previous = True, 
              write_steps = 1, checkval = True, writeloss = True, init_epochs = 250, epochs = 1000, 
              l1reg_last = 0, l2reg_last = 0, l1_kernel= 0, reverse_sign = False, shift_back = None, 
              random_shift = False, smooth_onehot = 0, multiple_input = False, restart = False, 
              masks = None, nmasks = None, augment_representation = None, aug_kernel_size = None, 
              aug_conv_layers = 1, aug_loss_masked = True, aug_loss= None, aug_loss_mix = None, 
              aug_lr = None, warmup_function = 'Exp', warm_up_lr_factor = 4., warm_up_epochs = 7, 
              sequence_warmup = False, min_sequence_warmup=200, sequence_warmup_start = 1, 
              sequence_warmup_factor = 2., sequence_warmup_location='c', 
              trainvariability_cut = 0.3, trainvariability_cutn = 25, valtrain_ratio = 0., 
              finetuning = True, finetuning_patience = 3, finetuning_rounds = 5, 
              finetuning_rate = 0.2, **kwargs):
    
    
    # Default parameters for each optimizer
    if optim_params is None:
        if optimizer == 'SGD':
            optim_params = 0
        elif optimizer == 'NAG':
            optim_params = 0.7 # momentum
        elif optimizer == 'Adagrad':
            optim_params = 0. # lr_decay rate
        elif optimizer == 'Adadelta':
            optim_params = 0.9 # coefficient for running average
        elif optimizer in ['Adam','NAdam','AdamW', 'Amsgrad']:
            optim_params = (0.9,0.999)
        else:
            optim_params = None
    
    if optim_weight_decay == None:
        if optimizer == 'AdamW':
            optim_weight_decay = 0.01
        else:
            optim_weight_decay = 0.
        
    
    if model.outname is not None:
        outname = model.outname
        
    if shift_back is not None:
        if isinstance(shift_back, int):
            if shift_back > 0:
                shift_back = np.arange(0,shift_back+1, dtype = int)
            else:
                shift_back = None
        else:
            shift_back = np.unique(np.append([0],shift_back))
    
    # Assignment of loss function
    if loss_function is None:
        loss_function = 'MSE'
        
    if isinstance(loss_function, list):
        loss_func = [loss_dict[loss_f] for loss_f in loss_function]
    elif isinstance(loss_function, str):
        if isinstance(model.n_classes, list):
            loss_func = [loss_dict[loss_function] for lff in model.n_classes]
        else:
            loss_func = loss_dict[loss_function]
    elif type(loss_function) == nn.Module:
        loss_func = loss_function
    
    if isinstance(loss_func, list):
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights for i in range(len(loss_func))]
    
    # Loss function for validation set to determine stop criterion
    if validation_loss is None:
        validation_loss = loss_function
    
    if isinstance(validation_loss, list):
        val_loss = [loss_dict[loss_f] for loss_f in validation_loss]
    elif isinstance(validation_loss, str):
        if isinstance(model.n_classes, list):
            val_loss = [loss_dict[validation_loss] for lff in model.n_classes]
        else:
            val_loss = loss_dict[validation_loss]
    elif type(validation_loss) == nn.Module:
        val_loss = validation_loss

    if isinstance(val_loss, list):
        if not isinstance(val_loss_weights, list):
            val_loss_weights = [val_loss_weights for i in range(len(val_loss))]
        for v, valloss in enumerate(val_loss):
            if valloss is None:
                val_loss_weights[v] = 0
    
    seq_size = np.shape(X)[-1]
    
    aug_convolution = aug_kernel_size
    seqmaskcut = None
    if augment_representation is not None:
        # find out size of output at "augment_representation" to determine padding for final_convolution()
        testrep = model.forward(torch.Tensor(X[:1]), location = augment_representation)
        rep_size = testrep.size(dim = -1)
        rep_dim = testrep.size(dim = -2)
        seq_size = np.shape(X)[-1]
        aug_convolution = nn.Sequential(final_convolution(rep_dim, 4, aug_kernel_size, strides = 1, n_convolutions = aug_conv_layers), nn.Softmax(dim = -2))
        aug_convolution.to(device)
        max_windowsize = min(rep_size, seq_size) # For some representation one may not be able to reconstruct masked windows along the entire sequence
        seqmaskcut = int(np.ceil((seq_size - max_windowsize)/2))
        # sequences values will only be masked within these values
        seqmaskcut = [seqmaskcut, seqmaskcut + max_windowsize]
        # remove masks that mask values outside the window that can be covered
        maskcover = np.where(masks[:,0])
        # find masks that have values outside seqmaskcut
        delmask = np.unique(maskcover[0][(maskcover[1]<seqmaskcut[0]) * (maskcover[1]>=seqmaskcut[1])])
        masks = masks[~np.isin(np.arange(len(masks)),delmask)]
        aug_loss = loss_dict[aug_loss]
        if aug_lr is None:
            aug_lr = lr
        
    
    # if the model uses fixed pwms that are concatenated with the input before pooled
    fixed_kernels = None
    if 'fixed_kernels' in model.__dict__:
        fixed_kernels = model.__dict__['fixed_kernels']
        l_kernels = model.__dict__['l_kernels']
        motif_cutoff = model.__dict__['motif_cutoff']
    
    last_layertensor = None
    if l1reg_last > 0 or l2reg_last > 0:
        # Determine the name of the parameter tensors in the last layer that need to be regularized
        last_layertensor = []
        for param_tensor in model.state_dict():
            if 'classifier' in str(param_tensor) and 'weight' in str(param_tensor):
                last_layertensor.append(str(param_tensor))

    
    kernel_layertensor = None
    if l1_kernel > 0:
        kernel_layertensor = []
        for param_tensor in model.state_dict():
            if 'convolutions.weight' in str(param_tensor):
                kernel_layertensor.append(str(param_tensor))
        
    # If pwms are provided, scan the sequences with them to avoid reoccuring scanning
    pwm_out = None
    if fixed_kernels is not None:
        # generate maxpooled feature list with given kernels
        pwm_out = pwm_scan(X, fixed_kernels, targetlen = l_kernels, verbose = verbose, motif_cutoff = motif_cutoff)
        pwm_out = torch.Tensor(pwm_out)
    
    
    # XYval represents a validation set on which the performance for the stop criterion is measured
    pwm_outval = None
    if XYval is None:
        # If XYval is None, the training data will be used
        Xval, Yval = X, Y
        if fixed_kernels is None:
            pwm_outval = pwm_out
    else:
        Xval, Yval = XYval[0], XYval[1]
        if fixed_kernels is not None:
            pwm_outval = pwm_scan(Xval, fixed_kernels, targetlen = l_kernels, verbose = verbose, motif_cutoff = motif_cutoff) 
            pwm_outval = torch.Tensor(pwm_outval)
    
    val_weights = None
    if sample_weights is not None:
        val_weights = torch.ones(np.shape(Xval)[0])
    
    if sample_weights is not None:
        # If weighted outputs then the loss needs to normalized by the sum of weights
        trainsize = np.sum(sample_weights)
    else:
        if multiple_input:
            trainsize = float(len(X[0]))
        else:
            trainsize = float(len(X))
    if multiple_input:
        valsize = float(len(Xval[0]))
    else:
        valsize = float(len(Xval))
    
    if multiple_input and not isinstance(sequence_warmup_location,list):
        sequence_warmup_location = [sequence_warmup_location for sq in range(len(X))]
    
    if multiple_input:
        X = [torch.Tensor(x) for x in X] # transform to torch tensor
        Xval = [torch.Tensor(xval) for xval in Xval] # transform to torch tensor
    else:
        X = torch.Tensor(X) # transform to torch tensor
        Xval = torch.Tensor(Xval) # transform to torch tensor
    if isinstance(Y, list):
        trainlen, vallen = len(Y[0]), len(Yval[0])
    else:
        trainlen, vallen = len(Y), len(Yval)
    
    if isinstance(Y, list):
        Y = [torch.Tensor(y) for y in Y]
        Yval = [torch.Tensor(yval) for yval in Yval]
    else:
        Y = torch.Tensor(Y)
        Yval = torch.Tensor(Yval)
    
    my_dataset = MyDataset(X, Y, axis = int(multiple_input), yaxis = isinstance(Y,list)) # create your datset
    if batchsize is None:
        if multiple_input:
            batchsize = len(X[0])
        else:
            batchsize = len(X)
    
    droplast = False
    mindata = 10 # minimum left data points for last batch to not be dropped
    if trainlen%batchsize < mindata:
        droplast = True
    dataloader = DataLoader(my_dataset, batch_size = batchsize, shuffle = True, drop_last = droplast)
    
    my_val_dataset = MyDataset(Xval, Yval, axis = int(multiple_input), yaxis = isinstance(Yval,list)) # create your datset
    val_batchsize = int(min(batchsize,vallen)) # largest batchsize for validation set is 250 to save memory on gpu
    
    vdroplast = False
    if vallen%val_batchsize < mindata:
        vdroplast = True
    val_dataloader = DataLoader(my_val_dataset, batch_size = val_batchsize, shuffle = True, drop_last = vdroplast)
    
    if batchsize < mindata and loss_function in ['Correlationclass', 'MSECorrelation']:
        print(loss_function, 'NOT RECOMMENDED WITH BATCHSIZE <', mindata)
    if val_batchsize < mindata and validation_loss in ['Correlationclass', 'MSECorrelation']:
        print(validation_loss, 'NOT RECOMMENDED WITH BATCHSIZE <', mindata)
    
    if warm_start:
        ### Implement training on shortenend/zero-flanked sequence
        print('Warm start')
        tmodel = torch_Regression(alpha = 0., fit_intercept = False, loss_function = 'MSE', logistic = 'Linear', penalty = 'none', kernel_length = model.l_kernels, n_kernels = model.num_kernels, kernel_function = model.kernel_function, pooling = 'Max', pooling_length = None, alpha_kernels = 0., is_zero = 0, epochs = 100, lr = 0.1, optimizer = 'SGD', optim_params = None, batchsize = batchsize, device = device, seed = model.seed, verbose = verbose, patience = 5, adjust_lr = 0.3, outname = None, write_model_params = False, norm = False)
        tmodel.fit(X, Y,XYval = [Xval, Yval], sample_weights = sample_weights)
        Ypred = torch.Tensor(tmodel.predict(X))
        Yvalpred = torch.Tensor(tmodel.predict(Xval))
        print('Warmstart valitrain', val_loss(Ypred.to(device),Y).item()/trainsize)
        print('Warmstart valival', val_loss(Yvalpred.to(device), Yval).item()/valsize)
        hotpwms = tmodel.kernels_#*tmodel.coef_.T[...,None]
        
        #if kernel_bias:
        #    hotpwms += tmodel.intercept_ #nn.Parameter(torch.ones_like(convolutions.bias)*1e-16)
        model.convolutions.weight = nn.Parameter(torch.Tensor(hotpwms))
    # Hot start initializes the kernels with onehot encoded pwms from Lasso regression
    elif hot_start:
        ### REPLACE with 1-d convolutional neural network or introduce another variable to do that
        hotpwms = kernel_hotstart(model.num_kernels, model.l_kernels, X.numpy(), Y.numpy(), XYval = [Xval.numpy(), Yval.numpy()], alpha = hot_alpha)
        maxweights = np.sum(hotpwms, axis = (-1,-2))
        model.convolutions.weight = nn.Parameter(torch.Tensor(hotpwms))
        if model.kernel_bias:
            model.convolutions.bias = nn.Parameter(torch.ones_like(model.convolutions.bias)*-np.amin(maxweights)*0.5)
        if model.kernel_thresholding > 0:
            model.modelstart.Kernelthresh.bias = nn.Parameter(torch.ones_like(model.modelstart.Kernelthresh.bias)*-np.amin(maxweights)*0.5)
        
        
    
    # Send model parameter to GPU
    model.to(device)
    
    # Collect the sizes and names of all parameters of the model layers 
    layersizes = []
    layernames = []
    i = 0
    for param_tensor in model.state_dict():
        if verbose: 
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        layname = param_tensor.strip('.weight').strip('.bias')
        if layname in layernames:
            layersizes[layernames.index(layname)] += np.prod(model.state_dict()[param_tensor].size())
        else:
            layernames.append(layname)
            layersizes.append(np.prod(model.state_dict()[param_tensor].size()))
    
    # Increasing or decreasing learning rates from kernel_lr if learning_rate set to "gradual"
    # Parameter size adjusted lr with Size or Antisize
    if adjust_lr == 'LogGradual':
        a_lrs = 10**np.linspace(np.log10(kernel_lr), np.log10(lr), len(layernames))
    elif adjust_lr == 'Gradual':
        a_lrs = np.linspace(kernel_lr, lr, len(layernames))
    elif adjust_lr == 'Size':
        a_lrs = lr * np.array(layersizes)/np.amax(layersizes)
    elif adjust_lr == 'Antisize':
        a_lrs = lr/(np.array(layersizes)/np.amin(layersizes))
    else:
        a_lrs = lr*np.ones(len(layersizes))
        if kernel_lr is not None:
            for a, lan in enumerate(layernames):
                if 'convolutions' in lan:
                    a_lrs[a] = kernel_lr
    
    # Give all the lrs to a list of dictionaries
    a_dict = []
    a_lrs0 = []
    for param_tensor, tensor in model.named_parameters():
        layname = param_tensor.strip('.weight').strip('.bias')
        if verbose:
            print(param_tensor, a_lrs[layernames.index(layname)])
        a_dict.append({'params':tensor, 'lr':a_lrs[layernames.index(layname)]})
        a_lrs0.append(a_lrs[layernames.index(layname)])
   
    # initialize learning rate for aug_convolution
    if augment_representation is not None:
        for param_tensor, tensor in aug_convolution.named_parameters():
            if verbose:
                print(param_tensor, aug_lr)
            a_dict.append({'params':tensor, 'lr':aug_lr})
    
    
    # Give this dictionary to the optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(a_dict, lr=lr, weight_decay= optim_weight_decay, momentum=optim_params)
    elif optimizer == 'NAG':
        optimizer = optim.SGD(a_dict, lr=lr, weight_decay= optim_weight_decay, momentum=optim_params, nesterov = True)
    elif optimizer == 'Adagrad':
        optimizer = optim.Adagrad(a_dict, lr=lr, weight_decay= optim_weight_decay, lr_decay = optim_params)
    elif optimizer == 'Adadelta':
        optimizer = optim.Adadelta(a_dict, lr=lr, weight_decay= optim_weight_decay, rho=optim_params)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(a_dict, lr=lr, weight_decay= optim_weight_decay, betas=optim_params)
    elif optimizer == 'NAdam':
        optimizer = optim.NAdam(a_dict, lr=lr, weight_decay= optim_weight_decay, betas=optim_params, momentum_decay = 4e-3)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(a_dict, lr=lr, weight_decay= optim_weight_decay, betas=optim_params)
    elif optimizer == 'Amsgrad':
        optimizer = optim.AdamW(a_dict, lr=lr, weight_decay= optim_weight_decay, betas=optim_params, amsgrad = True)
    else:
        print(optimizer, ' optimizer not allowed')
    
    if warmup_function == 'Linear':
        linwarm_scale = []
        for a, adict in enumerate(a_dict):
            linwarm_scale.append( (adict['lr'] * (1.- 1./(warm_up_lr_factor**warm_up_epochs)))/warm_up_epochs)
    for a, adict in enumerate(a_dict):
        a_dict[a]['lr'] = adict['lr'] * 1./(warm_up_lr_factor**warm_up_epochs)
    
    print('Warm up from', a_dict[a]['lr'])
    
    # Compute losses at the beginning with randomly initialized model
    lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = shift_back, random_shift = random_shift,smooth_onehot = 0,multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut, zero_flanks = 0, zero_flank_loc = sequence_warmup_location)
    # sequence warm up with: zero_flanks = max(int(sequence_warmup)*int((seq_size-min_sequence_warmup*sequence_warmup_factor**max(0,e-sequence_warmup_start))/2),0)
    
    
    beginning_loss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back,random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut, zero_flanks = int(sequence_warmup)*int((seq_size-min_sequence_warmup)/2), zero_flank_loc = sequence_warmup_location)
    
    saveloss = [lossval, loss2, lossorigval, beginning_loss, 0]
    best_lrs = np.copy(a_lrs0)
    save_model(model, outname+'_params0.pth')
    save_model(model, outname+'_parameter.pth')   
    
    roundto = 6
    trainlossbefore = beginning_loss
    trainlossincreased = 0
    if verbose:
        print('Train_loss(val), Val_loss(val), Train_loss(train), Val_loss(train)')
    if writeloss:
        writebeginning = str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(beginning_loss,roundto))+'\t'+str(round(loss2,roundto))
        save_losses(outname+'_loss.txt', 0, writebeginning)
    if verbose:
        print(0, writebeginning)
    stopcriterion = stop_criterion(checkval, patience, valtrain_ratio = valtrain_ratio)
    early_stop, stopexp = stopcriterion(0, lossval, loss2)
    
    # Start epochs and updates
    restarted = 0
    e = 0
    been_larger = 1
    time0 = time.time()
    while True:
        trainloss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, True, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = optimizer, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back, random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, augment_representation = augment_representation, aug_layer = aug_convolution, aug_loss_masked = aug_loss_masked, aug_loss = aug_loss, aug_loss_mix = aug_loss_mix, seqmaskcut = seqmaskcut, zero_flanks = max(int(sequence_warmup)*int((seq_size-min_sequence_warmup*sequence_warmup_factor**max(0,e-sequence_warmup_start))/2),0), zero_flank_loc = sequence_warmup_location)
        
        model.eval() # Sets model to evaluation mode which is important for batch normalization over all training mean and for dropout to be zero
        e += 1
        if e <= warm_up_epochs:
            for a, adict in enumerate(a_dict):
                if warmup_function == 'Linear':
                    a_dict[a]['lr'] = adict['lr'] + linwarm_scale[a]
                else:
                    a_dict[a]['lr'] = adict['lr'] * warm_up_lr_factor
            print("Warming up lr", a_dict[a]['lr'])
                
        lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = shift_back, random_shift = random_shift, smooth_onehot = 0, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
        
        
        
        if e%write_steps == 0 or e < init_epochs:
            if writeloss:
                save_losses(outname+'_loss.txt', e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
            if verbose:
                print(e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto))+'\t'+str(round(time.time()-time0,1)))
            
        model.train() # Set model modules back to training mode
        
        early_stop, stopexp = stopcriterion(e, lossval, loss2)
        
        #print(trainloss, beginning_loss, e, init_epochs, init_adjust)
        if (np.isnan(lossval) or np.isnan(loss2) or np.isnan(lossorigval) or np.isnan(trainloss)) or ((been_larger >= reduce_lr_after) and (trainloss > beginning_loss)):
            if init_adjust and e > init_epochs:
                # reduces learning rate if training loss goes up actually
                # need something learnable for each layer during training
                restarted += 1
                if restarted > 15:
                    break
                save_losses(outname+'_loss.txt', 0, writebeginning)
                load_model(model, outname+'_params0.pth',device)
                e = 0
                lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, smooth_onehot = 0, shift_back = shift_back, random_shift = random_shift, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
                early_stop, stopexp = stopcriterion(0, lossval, loss2)
                
                trainlossbefore = np.copy(beginning_loss)
                trainlossincreased = 0
                
                for a, adict in enumerate(a_dict):
                    a_lrs0[a] = a_lrs0[a] * 0.25
                    if warmup_function == 'Linear':
                        linwarm_scale[a] *= 0.25
                    a_dict[a]['lr'] = a_lrs0[a] * 1./(warm_up_lr_factor**warm_up_epochs)
                print('Learning rate reduced', restarted,  lossorigval, lossval, a_dict[-1]['lr']*warm_up_lr_factor**warm_up_epochs)
                been_larger = 1
        
            elif trainloss > beginning_loss:
                been_larger += 1
            else:
                been_larger = 1
        
        elif (e > warm_up_epochs) and (trainvariability_cutn != 0) and ((trainloss - trainlossbefore)/(1e-8+beginning_loss-trainloss) > trainvariability_cut or trainlossincreased > trainvariability_cutn):
            for a, adict in enumerate(a_dict):
                a_dict[a]['lr'] = adict['lr'] *0.5
            print('Learning rate reduced because train increased over fraction of distance to beginning', round((trainloss - trainlossbefore)/(beginning_loss-trainloss),3), a_dict[-1]['lr'], trainlossincreased)
            print('Parameters NOT reloaded', 'increase "trainlossincreased" or "trainvariability_cut" to avoid')
            trainlossincreased = 0
            trainlossbefore = np.copy(trainloss)
        
        elif trainloss > trainlossbefore:
            trainlossincreased +=1
        
        elif trainloss < trainlossbefore:
            trainlossincreased = 0
            trainlossbefore = np.copy(trainloss)
        
        if e == epochs or (early_stop and (e > init_epochs)):
            # stop after reaching maximum epoch number
            
            # restart training with random parameters in all layers besides the learned kernels in the first layer --> Idea of transfer learning or of trying random restarts. 
            if restart:
                load_model(model, outname+'_parameter.pth',device)
                if verbose:
                    print("Loaded model from", outname+'_parameter.pth', e - saveloss[-1], 'steps ago with loss', saveloss[0])
                restarted =0
                for mname, layer in model.named_modules():
                    if hasattr(layer, 'reset_parameters') and 'convolutions' not in mname.split('.'):
                        layer.reset_parameters()
                        if verbose:
                            print('Reseted', mname)
                save_model(model, outname+'_params0.pth')
                beginning_loss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back, random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut, zero_flanks = max(int(sequence_warmup)*int((seq_size-min_sequence_warmup*sequence_warmup_factor**max(0,e-sequence_warmup_start))/2),0), zero_flank_loc = sequence_warmup_location)
                lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, smooth_onehot = 0, shift_back = shift_back, random_shift = random_shift, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
                
                for a, lr0 in enumerate(a_lrs0):
                    a_dict[a]['lr'] = a_lrs0[a]
                    
                trainlossbefore = beginning_loss
                early_stop, stopexp = stopcriterion(0, lossval, loss2)
                # set restart to False so that the training is stopped next time the model converges. 
                restart = False
                epochs = epochs + e
                
                if verbose:
                    print('New start with pretrained first layer', str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
                
            else:
                if early_stop and verbose:
                    # Early stopping if validation loss is not improved for patience epochs
                    if verbose:
                        print(e, lossval, stopexp)
                
                if lossval > saveloss[0] and load_previous:
                    # Load better model if it was created in an earlier epoch
                    load_model(model, outname+'_parameter.pth',device)
                    if verbose:
                        print("Loaded model from", outname+'_parameter.pth', e - saveloss[-1], 'steps ago with loss', saveloss[0])
                    
                    if finetuning:
                        print('Fine tuning best performing model from step', saveloss[-1])
                        e = saveloss[-1]
                        ftunecount = 0
                        ftuneround = 0
                        # run a few more steps with lowering learning rate from the best performing model and the learning rate at this point
                        for a, blr in enumerate(best_lrs):
                            a_dict[a]['lr'] = blr * finetuning_rate
                        print("Learning rate reduced for finetuning to", a_dict[a]['lr'])
                        while True:
                            # Run update with training data 
                            trainloss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, True, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = optimizer, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back, random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, augment_representation = augment_representation, aug_layer = aug_convolution, aug_loss_masked = aug_loss_masked, aug_loss = aug_loss, aug_loss_mix = aug_loss_mix, seqmaskcut = seqmaskcut)
        
                            model.eval() # Sets model to evaluation mode which is important for batch normalization over all training mean and for dropout to be zero
                            e += 1
                    
                            lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, loss_weights = loss_weights, val_loss = val_loss, val_loss_weights = val_loss_weights, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = shift_back, random_shift = random_shift, smooth_onehot = 0, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)

                            if e%write_steps == 0:
                                if writeloss:
                                    save_losses(outname+'_loss.txt', e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
                                if verbose:
                                    print(e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto))+'\t'+str(round(time.time()-time0,1)))
                                
                            model.train() # Set model modules back to training mode
                            
                            if lossval >= saveloss[0]:
                                ftunecount += 1
                            else:
                                saveloss = [lossval, loss2, lossorigval, trainloss, e]
                                save_model(model, outname+'_parameter.pth')
                            
                            if ftunecount > finetuning_patience:
                                ftuneround += 1
                                ftunecount = 0
                                load_model(model, outname+'_parameter.pth',device)
                                for a, blr in enumerate(a_dict):
                                    a_dict[a]['lr'] = blr['lr'] * finetuning_rate
                                print("Learning rate reduced for finetuning to", a_dict[a]['lr'], 'in round', ftuneround)
                                
                                
                            if ftuneround >= finetuning_rounds or e > epochs:
                                # Load better model if it was created in an earlier epoch
                                load_model(model, outname+'_parameter.pth',device)
                                if verbose:
                                    print("Loaded model from", outname+'_parameter.pth', e - saveloss[-1], 'steps ago with loss', saveloss[0])
                                break
                            
                        

                else:
                    saveloss = [lossval, loss2, lossorigval, trainloss, e]
                os.remove(outname+'_params0.pth')
                break
                    
        elif not early_stop:
            # if early stopping is False, save loss and parameters
            if (~np.isnan(lossval) and lossval < saveloss[0]) or (~np.isnan(lossval) and np.isnan(saveloss[0])):
                for a, adict in enumerate(a_dict):
                    best_lrs[a] = a_dict[a]['lr']
                saveloss = [lossval, loss2, lossorigval, trainloss, e]
                save_model(model, outname+'_parameter.pth')
    
    if not keepmodel:
        os.remove(outname+'_parameter.pth')
    return saveloss


def save_losses(PATH, i, lo):
    if i == 0:
        obj = open(PATH, 'w')
    else:
        obj = open(PATH, 'a')
    obj.write(str(i)+'\t'+str(lo)+'\n')
    



class stop_criterion(nn.Module):
    def __init__(self, checkval, patience, valtrain_ratio = 0.2):
        super(stop_criterion, self).__init__()
        self.best_loss = None
        self.steps_since = 0
        self.checkval = checkval
        self.patience = patience
        self.valtrain_ratio = valtrain_ratio # stops when valloss/trainloss <= valtrain_ratio
        if valtrain_ratio is None:
            self.valtrain_ratio = 0
        
    def forward(self, i, lossi, lossj):
        # if loss gets nan it's not producing meaningful losses
        if np.isnan(lossi) or np.isnan(lossj):
            return True, 'Loss nan'
        
        if self.checkval:
            if i == 0 or self.best_loss is None:
                self.best_loss = lossi
                self.steps_since = 0
            else:
                if lossi < self.best_loss:
                    self.best_loss = lossi
                    self.steps_since = 0
                else:
                    self.steps_since +=1
                if self.steps_since == self.patience:
                    return True, 'Loss has not decreased since '+str(self.patience)+' steps'
                elif lossj/lossi <= self.valtrain_ratio:
                    return True, 'Loss has not decreased since '+str(self.patience)+' steps and trainloss/valloss is '+str(round(lossj/lossi,2))
        return False, None



def reverse_inoutsign(x):
    x = torch.cat([x, -x], dim = 0)
    return x

# For sequence warm up: only keep the most central 'c' or most left 'l' or most right 'r' sequence parts
def zero_sequence_flanks(x, flanks = 0, flankloc = 'c'):
    if flankloc == 'c' or flankloc == 'l':
        x[...,:flanks*(1+int(flankloc == 'l'))] = 0
    if flankloc == 'c' or flankloc == 'r':
        x[...,-1-flanks*(1+int(flankloc == 'r')):] = 0
    return x

# if sequences use different padding on one side, create a copy of the sequence that is aligned to the other side
def realign_sequence(X, end = True, start = False):
    ltensor = X.size(dim = -1)
    if end: 
        Xmirrorend = torch.zeros_like(X)
    if start:
        Xmirrorstart = torch.zeros_like(X)
    
    for i, x in enumerate(X):
        pos_seqs = torch.where(x == 1)[-1]
        start_of_seq, end_of_seq = pos_seqs.min(), pos_seqs.max()+1
        if end:
            Xmirrorend[i, :,  ltensor-end_of_seq+start_of_seq:] = x[:,start_of_seq:end_of_seq]
        if start: 
            Xmirrorstart[i, :,  :end_of_seq-start_of_seq] = x[:,start_of_seq:end_of_seq]
    if end:
        X = torch.cat([X, Xmirrorend], axis = 0)
    if end:
        X = torch.cat([X, Xmirrorstart], axis = 0)
    return X

# make version so that nothing gets lost and so that array with shifts can be given    
def shift_sequences(sample_x, shift_back, maxshift = None, just_pad = False, mode = 'constant', value = 0.25, random_shift = False): # 'constant', 'reflect', 'replicate' or 'circular'
    if maxshift is None:
        maxshift = np.amax(shift_back)
    sample_xs = [] #nn.functional.pad(sample_x, (maxshift, maxshift), mode = mode, value = value)]
    if random_shift and not just_pad:
        sbs = np.random.choice(shift_back, int(random_shift))
        for sb in sbs:
            sample_xs.append(nn.functional.pad(sample_x, (maxshift-sb, maxshift+sb), mode = mode, value = value))
            sample_xs.append(nn.functional.pad(sample_x, (maxshift+sb, maxshift-sb), mode = mode, value = value))
    elif not just_pad:
        for sb in shift_back:
            sample_xs.append(nn.functional.pad(sample_x, (maxshift-sb, maxshift+sb), mode = mode, value = value))
            sample_xs.append(nn.functional.pad(sample_x, (maxshift+sb, maxshift-sb), mode = mode, value = value))
    else:
        sample_xs = [nn.functional.pad(sample_x, (maxshift, maxshift), mode = mode, value = value)]
    sample_x = torch.cat(sample_xs, dim = 0)
    return sample_x

def smooth_onehotfunc(sample_x, ns = 2):
    sample_x = torch.cat([sample_x, torch.sign(torch.sum(sample_x, dim =1).repeat((ns,1))).unsqueeze(1)*(sample_x.absolute().repeat((ns,1,1)) - torch.rand(ns*sample_x.size(dim=0),ns*sample_x.size(dim=1),ns*sample_x.size(dim=2))*0.5).absolute()], dim = 0)
    return sample_x

def mask_sequence(sample_x, masks, nmasks, mask_val = 0.25):
    nsamp = sample_x.size(dim=0)
    cmasks = masks[np.random.choice(len(masks), nmasks * nsamp)]
    sample_x = sample_x.repeat(nmasks, 1, 1)
    sample_x[cmasks] = mask_val
    return sample_x, cmasks


# execute one epoch with training or validation set. Training set takes gradient but validation set computes loss without gradient
def excute_epoch(model, dataloader, loss_func, pwm_out, normsize, take_grad, device, loss_weights = 1, val_loss = None, val_loss_weights = 1, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = None, reverse_sign = False, shift_back = None, random_shift = False, smooth_onehot = 0, multiple_input = False, masks = None, nmasks = None, augment_representation = None, aug_layer = None, aug_loss_masked = True, aug_loss = None, aug_loss_mix = None, seqmaskcut = None, zero_flanks =0, zero_flank_loc = 'c'):
    
    if val_loss is None:
        val_loss = loss_func
    
    trainloss = 0.
    validatloss = 0.
    
        
    multiple_output = False
    yclassestrain = model.n_classes
    yclassesval = model.n_classes
    if isinstance(yclassestrain, list):
        yclassestrain = sum(yclassestrain)
        multiple_output = True
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights for i in yclassestrain]
        if not isinstance(val_loss_weights, list):
            val_loss_weights = [val_loss_weights for i in yclassesval]
        yclassesval = np.sum(np.array(yclassesval)[np.array(val_loss_weights) != 0])
        
    # if size of validation set different from batchsize, then do val_all
    if val_all is not None:
        if multiple_output:
            Ypred_all = [torch.empty(valall.size()) for valall in val_all]
            allindex = torch.zeros(val_all[0].size(dim = 0))
        else:   
            Ypred_all = torch.empty(val_all.size())
            allindex = torch.zeros(val_all.size(dim = 0))
    
    if 'l_out' in model.__dict__:
        if model.l_out is not None:
            yclassesval *= model.l_out
            yclassestrain *= model.l_out
    
    tsize = 1
    augsize = None
    if nmasks is not None and augment_representation is not None:
        augsize = tsize * nmasks
    elif nmasks is not None and seqmaskcut is None:
        tsize = tsize * nmasks
    
    if shift_back is not None:
        if random_shift:
            tsize = tsize*int(random_shift)*2
        else:
            tsize = tsize*2*len(shift_back)
    if reverse_sign:
    # all operations are also performed in the negative space, i.e. 1 => -1 in one hot encoding and y = 4 => y = -4
    # shoud only be used if no relu is used after first convolution, otherwise it's confusing why this should help. Idea is basically to learn a symmetric function.
        tsize = tsize*2
    if smooth_onehot > 0:
        tsize = tsize *(smooth_onehot +1)
    
    
    for sample_x, sample_y, index in dataloader:
        if nmasks is not None:
            if augment_representation is not None and take_grad:
                sample_xaug, sample_mask = mask_sequence(sample_x,masks, nmasks)
                sample_xaug = sample_xaug[:,:,seqmaskcut[0]:seqmaskcut[1]]
                sample_mask = sample_mask[:,:,seqmaskcut[0]:seqmaskcut[1]]
                sample_yaug = torch.cat(nmasks*[sample_x[:,:,seqmaskcut[0]:seqmaskcut[1]]])
                sample_xaug = sample_xaug.to(device)
                sample_yaug = sample_yaug.to(device)
                sample_mask = sample_mask.to(device)
            elif seqmaskcut is None: 
                sample_x, sample_mask = mask_sequence(sample_x,masks, nmasks)
                if multiple_output:
                    sample_y = [torch.cat(nmasks*[s_y]) for s_y in sample_y]
                else:
                    sample_y = torch.cat(nmasks*[sample_y])
        
        if pwm_out is None:
            saddx = None
        else:
            saddx = pwm_out[index]
        
        
        if zero_flanks > 0:
            if multiple_input:
                sample_x = [zero_sequence_flanks(sam_x, flanks = zero_flanks, flankloc = zero_flank_loc[s]) for s, sam_x in enumerate(sample_x)]
            else:
                sample_x = zero_sequence_flanks(sample_x, flanks = zero_flanks, flankloc = zero_flank_loc)
        
        
        # Add samples that are shifted by 1 to 'shift_back' positions
        if shift_back is not None:
            if multiple_input:
                sample_x = [shift_sequences(sam_x, shift_back, random_shift = random_shift) for sam_x in sample_x]
            else:
                sample_x = shift_sequences(sample_x, shift_back, random_shift = random_shift)
            if random_shift:
                if multiple_output:
                    sample_y = [torch.cat(2*int(random_shift)*[s_y]) for s_y in sample_y]
                else:
                    sample_y = torch.cat(2*int(random_shift)*[sample_y])
            else:
                if multiple_output:
                    sample_y = [torch.cat(2*int(shift_back)*[s_y]) for s_y in sample_y]
                else:
                    sample_y = torch.cat(2*int(shift_back)*[sample_y])
            if saddx is not None:
                if random_shift:
                    saddx = saddx.repeat(3, 1, 1)
                else:
                    saddx = saddx.repeat(len(shift_back)*2, 1, 1)
        
        # Add samples with reverse sign in X and Y
        if reverse_sign:
            if multiple_input:
                sample_x = [reverse_inoutsign(sam_x) for sam_x in sample_x]
            else:
                sample_x = reverse_inoutsign(sample_x)
            if multiple_output:
                sample_y = [reverse_inoutsign(s_y) for s_y in sample_y]
            else:
                sample_y = reverse_inoutsign(sample_y)
            if saddx is not None:
                saddx = reverse_inoutsign(saddx)
        
        if smooth_onehot > 0:
            if multiple_input:
                sample_x = [smooth_onehotfunc(sam_x, ns = smooth_onehot) for sam_x in sample_x]
            else:
                sample_x = smooth_onehotfunc(sample_x, ns = smooth_onehot)
            if multiple_output:
                sample_y = [torch.cat((smooth_onehot+1)*[s_y]) for s_y in sample_y]
            else:
                sample_y = torch.cat((smooth_onehot+1)* [sample_y])
            if saddx is not None:
                saddx = saddx.repeat(smooth_onehot +1, 1, 1)
            
        if saddx is not None:
            if multiple_input:
                saddx = [sax.to(device) for sax in saddx]
            else:
                saddx = saddx.to(device)
        if multiple_input:
            sample_x = [sam_x.to(device) for sam_x in sample_x]
        else:
            sample_x= sample_x.to(device)
        if multiple_output: 
            sample_y = [s_y.to(device) for s_y in sample_y]
        else:
            sample_y = sample_y.to(device)
        
        if take_grad:
            optimizer.zero_grad()
            Ypred = model.forward(sample_x, xadd = saddx)
            
            if multiple_output:
                loss = []
                for l, lossfunc in enumerate(loss_func):
                    loss.append(loss_weights[l]*lossfunc(Ypred[l], sample_y[l]))
                loss = torch.cat(loss, dim = 1)
            else:
                loss = loss_func(Ypred, sample_y)
            if sample_weights is not None:
                loss = loss*sample_weights[index][:,None]
            
            loss = torch.sum(loss)
            trainloss += float(loss.item())
            
            if l1reg_last > 0 and last_layertensor is not None:
                # NEED to use named_parameters instead of model.state_dict because state_dict does not come with gradient
                for param_name, tensor in model.named_parameters():
                    if param_name in last_layertensor:
                        loss += l1reg_last * l1_loss(tensor)
            
            if l2reg_last > 0 and last_layertensor is not None:
                for param_name, tensor in model.named_parameters():
                    if param_name in last_layertensor:
                        loss += l2reg_last * l2_loss(tensor)
                    
            if l1_kernel > 0 and kernel_layertensor is not None:
                for param_name, tensor in model.named_parameters():
                    if param_name in kernel_layertensor:
                        loss += l1_kernel * l1_loss(tensor)
            if augment_representation is not None:
                aug_rep = model.forward(sample_xaug, location = augment_representation)
                aug_rep = aug_layer(aug_rep)
                if aug_loss_masked:
                    aug_rep = aug_rep[sample_mask].reshape(1,4,-1)
                    sample_yaug = sample_yaug[sample_mask].reshape(1,4,-1)
                loss += aug_loss_mix * torch.sum(aug_loss(aug_rep, sample_yaug))
            
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                Ypred = model.forward(sample_x, xadd = saddx)
                if multiple_output:
                    loss = []
                    for l, lossfunc in enumerate(loss_func):
                        loss.append(loss_weights[l]*lossfunc(Ypred[l], sample_y[l]))
                    loss = torch.cat(loss, dim = 1)
                else:
                    loss = loss_func(Ypred, sample_y)
                if sample_weights is not None:
                    loss = loss*sample_weights[index][:,None]
                    
                loss = torch.sum(loss)
                trainloss += float(loss.item())
            
        # Sending some to the cpu might take up a good amout of time
        # Should not happen for losses but for Correlationclass, or MSECorrelation
        if val_all is not None:
            allindex[index] = 1
            if multiple_output:
                for iy, ypred in enumerate(Ypred):
                    # :len(index)] so that only predictions for the original data points are looked at. 
                    ## Missing a model that averages over predictions from original and shifted or modified sequence input
                    Ypred_all[iy][index] = ypred[:len(index)].detach().cpu() 
                    
            else:
                Ypred_all[index] = Ypred[:len(index)].detach().cpu() 
        else:
            with torch.no_grad():
                if multiple_output:
                    for iy, ypred in enumerate(Ypred):
                        if val_loss[iy] is not None:
                            validatloss += val_loss_weights[iy]*float(torch.sum(val_loss[iy](ypred, sample_y[iy])).item())
                else:
                    validatloss += float(torch.sum(val_loss(Ypred, sample_y)).item())
                
    if val_all is not None:
        allindex = allindex == 1
        with torch.no_grad():
            if multiple_output:
                for iy, ypred in enumerate(Ypred_all):
                    if val_loss[iy] is not None:
                        validatloss += val_loss_weights[iy]*float(torch.sum(val_loss[iy](ypred[allindex], val_all[iy][allindex])).item())
            else:
                validatloss = float(torch.sum(val_loss(Ypred_all[allindex], val_all[allindex])).item())
            
    trainloss /= (normsize*tsize)*yclassestrain
    validatloss /= normsize*yclassesval
    return trainloss, validatloss


# The prediction after training are performed on the cpu
def batched_predict(model, X, pwm_out = None, mask = None, mask_value = 0, 
                    device = 'cpu', batchsize = None, shift_sequence = None, 
                    random_shift = True, enable_grad = False, location = 'None'):
    """
    Function to predict with a trained model. The model is set to evaluation mode and the input is moved to the device.
    The function uses no_grad to avoid computation of gradient and graph. If enable_grad is set to True, the function will compute the gradient of the input.
    The function will return the prediction of the model.
    Parameters
    ----------
    model : nn.Module
        The trained model.
    X : torch.Tensor or numpy.ndarray
        The input tensor.
    pwm_out : torch.Tensor
        The additional input tensor, with precomputed pwm activations
    mask : (int, or list of int)
        The mask indexes define the kernel in the first layer whose activations should be set to mask_value
    mask_value : float
        The value to set the activations to.
    device : str    
        The device to move the model and input to.
    batchsize : int
        The batch size to use for prediction. If None, the function will use the default batch size of the model.
    shift_sequence : int or list of int
        Whether the sequence was shifted during training
        If random_shift is True, we just add padding to account for this.
        If random_shift is False, we create all possible shifts and average predictions
    random_shift : bool
        Whether to pad or average over shifts.
    enable_grad : bool
        Whether to enable computation of the gradient of the input.
    location : str
        The location determines pre-defined locations of outputs from layers in the forwards function.
        These outputs can be used as embeddings for the input sequence.
    Returns
    -------
    predout : numpy.ndarray or torch.Tensor
        The prediction of the model.
        Or the embedings if location is not None.
        returns torch.Tensor if enable_grad is True, else numpy.ndarray
    """
    if shift_sequence is None:
        if hasattr(model, 'shift_sequence'):
            shift_sequence = model.shift_sequence
    
    # If shift sequence was used during training, we need to use padding
    if shift_sequence is not None:
        if isinstance(shift_sequence, int):
            if shift_sequence > 0:
                # if random_shift then we just pad with max number of shifts
                if not random_shift:
                    # else we pad with all the shifts and average over them
                    shift_sequence = np.arange(1,shift_sequence+1, dtype = int)
            else:
                shift_sequence = None
    
    # set model to evaluation mode
    model.eval()

    # chekk if device is None, then use the device of the model
    if device is None:
        if 'device' in model.__dict__:
            device = model.device
        else:
            device = 'cpu'
    model.to(device)
    
    # Use no_grad to avoid computation of gradient and graph
    # Use enable grad for computation of gradient of input
    if enable_grad:
        _grad = torch.enable_grad()
    else:
        _grad = torch.no_grad()
    
    with _grad:
        # if X is a list, then we need to convert it to a tensor
        islist = False
        if isinstance(X, list):
            islist = True
            if not isinstance(X[0], torch.Tensor):
                X = [torch.Tensor(x) for x in X]
            dsize = X[0].size(dim = 0)
        else:
            if not isinstance(X, torch.Tensor):
                X = torch.Tensor(X)
            dsize = X.size(dim = 0)
        
        if pwm_out is not None:
            if isinstance(pwm_out, list):
                pwm_out = [torch.Tensor(x) for x in pwm_out]
                psize = pwm_out[0].size(dim = 0)
            else:
                pwm_out = torch.Tensor(pwm_out)
                psize = pwm_out.size(dim = 0)
            if psize != dsize:
                raise ValueError('The size of the input and the pwm_out must be the same.')
                sys.exit()
            if pwm_out.size(dim = -1) != X.size(dim = -1):
                raise ValueError('The size of the input and the pwm_out must be the same.')
                sys.exit()
            

        if batchsize is None:
            batchsize = dsize
        if batchsize > dsize:
            batchsize = dsize
  
  
        predout = []
        for i in range(0, dsize, batchsize):
            # prepare precomputed pwm activations input for model predictions
            if pwm_out is not None:
                pwm_outin = pwm_out[i: i+batchsize]
                if shift_sequence is not None:
                    pwm_outin = shift_sequences(pwm_outin, shift_sequence, just_pad = random_shift)
                pwm_outin = pwm_outin.to(device)
            else:
                pwm_outin = None
            
            # prepare batch for model predictions and send to devide
            if islist:
                xin = [x[i:i+batchsize] for x in X]
                xsize = xin[0].size(dim = 0)
                if shift_sequence is not None:
                    xin = [shift_sequences(x, shift_sequence, just_pad = random_shift) for x in xin]
                xin = [x.to(device) for x in xin]
            
            else:
                xin = X[i:i+batchsize]
                xsize = xin.size(dim = 0)
                if shift_sequence is not None:
                    xin = shift_sequences(xin, shift_sequence, just_pad = random_shift)
                xin = xin.to(device)

            # model the input with the model
            fpred = model.forward(xin, xadd = pwm_outin, mask = mask,mask_value = mask_value, location = location)
            
            # if model returns list for multiple data matrices, just concatenate them
            if isinstance(fpred, list):
                fpred = torch.cat(fpred, axis = 1)
            # take them mean over all shifts 
            # if random_shift is True, we just pad with max number of shifts
            if shift_sequence is not None:
                fpred = fpred.reshape(-1,xsize,fpred.size(-1)).mean(dim=0)
            # if not used for gradient of input, then make numpy array
            if not enable_grad:
                fpred = fpred.detach().cpu().numpy()
            
            predout.append(fpred)

        # concatenate collected batches
        if enable_grad:
            predout = torch.cat(predout, axis = 0)
        else:   
            predout = np.concatenate(predout, axis = 0)
            
    return predout

