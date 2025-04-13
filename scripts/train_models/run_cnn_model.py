
import sys, os 
import numpy as np
import torch.nn as nn
import torch

from drg_tools.io_utils import readin, check, numbertype, isfloat, create_outname, write_meme_file, separate_sys, inputkwargs_from_string, add_name_from_dict, get_index_from_string, readin_motif_files
from drg_tools.model_training import create_sets
from drg_tools.data_processing import manipulate_input
from drg_tools.model_output import print_averages, save_performance
from drg_tools.plotlib import plot_scatter
from drg_tools.model_output import add_params_to_outname
from drg_tools.motif_analysis import pfm2iupac
from drg_tools.interpret_cnn import takegrad, ism, deeplift, indiv_network_contribution, kernel_assessment, kernels_to_pwms_from_seqlets,  kernel_to_ppm,extract_kernelweights_from_state_dict, captum_sequence_attributions
from drg_tools.cnn_model import cnn
from drg_tools.model_utils import get_device, load_parameters




if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    # Delimiter for values in output file
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    # Whether to assign a different regional channel for each input file
    aregion = True
    if '--regionless' in sys.argv:
        aregion = False
    
    # Whether sequences in each region should be aligned to right side, for example if sequences have different lengths and there could be a positional signal at the end of sequences
    mirror = False
    if '--realign_input' in sys.argv:
        mirror = True
    
    # Whether to combine input files into a single input or keep them split as input for multiple networks
    combinput = True

        
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True, assign_region = aregion, mirrorx = mirror, combinex = combinput)
    
    if '--load_output_track_names' in sys.argv and experiments is None:
        track_names = sys.argv[sys.argv.index('--load_output_track_names')+1]
        if ',' in track_names:
            experiments = [np.genfromtxt(tr, dtype = str) for tr in track_names.split(',')]
        else:
            experiments = np.genfromtxt(tr, str)

    # make X with 8 rows for ACGTACGT to capture the reverse complement
    reverse_complement = False
    if '--reverse_complement' in sys.argv:
        reverse_complement = True
        
    if '--testrandom' in sys.argv:
        trand = int(sys.argv[sys.argv.index('--testrandom')+1])
        mask = np.random.permutation(len(X))[:trand]
        X, names = X[mask], names[mask]
        if Y is not None:
            if isinstance(Y, list):
                Y = [y[mask] for y in Y]
            else:
                Y = Y[mask]
    
    if '--select_sample' in sys.argv:
        sel = sys.argv[sys.argv.index('--select_sample')+1]
        if ',' in sel:
            sel = sel.split(',')
        else:
            sel = [sel]
        mask = np.isin(names, sel)
        if np.sum(mask) < 1:
            print('Selected list names do not match the names in the data')
            sys.exit()
        X, names = X[mask], names[mask]
        print('List selected', np.shape(X))
        if Y is not None:
            if isinstance(Y, list):
                Y = [y[mask] for y in Y]
            else:
                Y = Y[mask]
    
    if '--select_list' in sys.argv:
        sel = np.genfromtxt(sys.argv[sys.argv.index('--select_list')+1], dtype = str)
        mask = np.isin(names, sel)
        if np.sum(mask) < 1:
            print('Selected list names do not match the names in the data')
            sys.exit()
        X, names = X[mask], names[mask]
        print('List selected', np.shape(X))
        if Y is not None:
            if isinstance(Y, list):
                Y = [y[mask] for y in Y]
            else:
                Y = Y[mask]
    # select output tracks to refine network or only train on a specific track from the beginning
    select_track = None
    if '--select_tracks' in sys.argv:
        selfile = sys.argv[sys.argv.index('--select_tracks')+1]
        if ',' in selfile:
            sel = selfile.split(',')
        elif os.path.isfile(selfile):
            sel = np.genfromtxt(sys.argv[sys.argv.index('--select_tracks')+1], dtype = str)
        else:
            sel = [selfile]
        if isinstance(experiments, list):
            select_track = []
            for e, exp in enumerate(experiments):
                select_track.append(np.isin(exp, sel))
                if np.sum(select_track[-1]) < 1:
                    print('Selected list of tracks do not match the names in the data')
                    #sys.exit()
        else:
            select_track = np.isin(experiments, sel)
            if np.sum(select_track) < 1:
                print('Selected list of tracks do not match the names in the data')
                sys.exit()
        
        if not '--load_parameters' in sys.argv:
            if isinstance(experiments, list):
                for e, exp in enumerate(experiments):
                    experiments[e] = exp[select_track[e]]
                if isinstance(Y, list):
                    Y = [y[:, select_track[yi]] for yi, y in enumerate(Y)]
            else:
                Y = Y[:, select_track]
                experiments = experiments[select_track]
                
        
    # Don't use with multiple Y
    if '--remove_allzero' in sys.argv and Y is not None:
        mask = np.sum(Y==0, axis = 1) != np.shape(Y)[1]
        Y = Y[mask]
        X, names = X[mask], names[mask]
        print('removed zeros', np.shape(X), np.shape(Y))
    
    if '--remove_allones' in sys.argv and Y is not None:
        mask = np.sum(Y==1, axis = 1) != np.shape(Y)[1]
        Y = Y[mask]
        X, names = X[mask], names[mask]
        print('removed ones', np.shape(X), np.shape(Y))
        
    # Dont use with multiple Y
    if '--adjust_allzerovar' in sys.argv and Y is not None:
        mask = np.std(Y, axis = 1) == 0
        rand = np.random.normal(loc = 0, scale = 1e-4, size = np.shape(Y[mask]))
        Y[mask] += rand
        
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = create_outname(inp, inputfile, lword = '')
    
    if ',' in outputfile:
        outputfiles = outputfile.split(',')
        outputfile = outputfiles[0]
        for inp in outputfiles[1:]:
            outputfile = create_outname(inp, outputfile, lword = '')
    
    outname = create_outname(inputfile, outputfile) 
    if '--regionless' in sys.argv:
        outname += '-rgls'
    if '--realign_input' in sys.argv:
        outname += 'mirx'
    if '--reverse_complement' in sys.argv:
        outname += 'rcomp'
    
    # Parameter dictionary for initializing the cnn
    params = {}
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]
    
    # Add class names if Y is a list and has the same names in different files
    fileclass = None
    if '--add_fileclasses' in sys.argv:
        addclasses = sys.argv[sys.argv.index('--add_fileclasses')+1].split(',')
    elif isinstance(Y, list):
        addclasses = [str(i) for i in range(len(experiments))]
    if isinstance(Y, list):
        fileclass = []
        for e, exps in enumerate(experiments):
            fileclass.append([addclasses[e] for i in range(len(exps))])
        #fileclass = np.concatenate(fileclass)
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname')+1]
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    # reads in file that had locations and type of mutations, currently not able to handle mirror=True or multiple inputs, needs to be adjusted to common vcf file format
    weights = None
    if '--mutation_file' in sys.argv:
        mutfile = sys.argv[sys.argv.index('--mutation_file')+1]
        X,Y,names,experiments,weights = read_mutationfile(mutfile,X,Y,names,experiments)
        if sys.argv[sys.argv.index('--mutation_file')+2] != 'weighted':
            weights = None
        outname = create_outname(mutfile, outname+'.dat', lword = 'with')
        print(outname)

    if '--crossvalidation' in sys.argv:
        folds = check(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        else:
            # IF NO SIGNIFICANT DATA POINTS DEFINED, USE CUTOFF TO DEFINE MORE IMPORTANT DATA POINTS    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            if isinstance(Y, list):
                Yclass = (np.sum(np.absolute(np.concatenate(Y,axis =1))>=cutoff, axis = 1) > 0).astype(int)
            else:
                Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
        cvs = True
    elif '--predictnew' in sys.argv:
        cvs = False
        trainset, testset, valset = [], np.arange(len(X), dtype = int), []
    else:
        raise Warning(f'--crossvalidation not defined in sys.argv, {sys.argv}')
        folds, fold, Yclass = 10, 0, None
        cvs = True
    
    if cvs:
        if isinstance(folds,int):
            outname += '-cv'+str(folds)+'-'+str(fold)
        else:
            outname += '-cv'+str(int(cutoff))+'-'+str(fold)
        trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
        print('Train', len(trainset))
        print('Test', len(testset))
        print('Val', len(valset))
    
    if '--maketestvalset' in sys.argv:
        trainset = np.append(trainset, testset)
        testset = np.copy(valset)
        print('Set testset to validation set')
    
    if '--downsample' in sys.argv:
        df = float(sys.argv[sys.argv.index('--downsample')+1])
        ltra, lvl = len(trainset), len(valset)
        trainset = np.random.permutation(trainset)[:int(len(trainset)*df)]
        print('from train', ltra, 'to', len(trainset))
        valset = np.random.permutation(valset)[:int(len(valset)*df)]
        print('from val', lvl, 'to', len(valset))
        
    if '--RANDOMIZE' in sys.argv:
        # randomly permute data and see if model can learn anything
        outname +='_RNDM'
        permute = np.permute(len(X))
        if isinstance(Y, list):
            Y = [y[permute] for y in Y]
        else:
            Y = Y[permute]
        
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        if isinstance(Y, list):
            outnorm, Y = [], []
            for y in Y:
                outnorm.append(np.sqrt(np.sum(y*y, axis = 1))[:, None])
                Y.append(y/outnorm[-1])
        else:
            outnorm = np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
            Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outc'
        if isinstance(y, list):
            outnorm, Y = [], []
            for y in Y:
                outnorm.append(np.sqrt(np.sum(y*y, axis = 0)))
                Y.append(y/outnorm[-1])
        else:
            outnorm = np.sqrt(np.sum(Y*Y, axis = 0))
            Y = Y/outnorm
    
    pfms = None
    pwmusage = 'none'
    if '--list_of_pwms' in sys.argv:
        list_of_pwms = sys.argv[sys.argv.index('--list_of_pwms')+1]
        pwmusage = sys.argv[sys.argv.index('--list_of_pwms')+2]
        
        psam = False
        if '--psam' in sys.argv:
            psam = True
        infcont = False
        if '--infocont' in sys.argv:
            infcont = True
        
        pwmnameset = np.array(os.path.splitext(os.path.split(list_of_pwms)[1])[0].split('_'))
        
        outname += '_'.join(pwmnameset[~np.isin(pwmnameset, outname.split('_'))]) + 'ps'+str(psam)[0]+'ic'+str(infcont)[0]
        motcut = None
        
        if '--motif_cutoff' in sys.argv: # makes the binding score binary
            motcut = float(sys.argv[sys.argv.index('--motif_cutoff')+1])
            outname += 'mc'+str(motcut)
        
        pfms, rbpnames, nts = readin_motif_files(list_of_pwms)
        pwms = rescale_pwm(pfms, psam = psam, infcont = infcont, norm = True)
        if pwmusage != 'initialize': 
            params['fixed_kernels'] = pwms
            params['motif_cutoff'] = motcut
        # elif initialize: pwms are trimmmed to the kernel length and used as initialization for the kernels

    
    
    if '--cnn' in sys.argv:
        parameterfile = False
        train_model = True
        parameters = sys.argv[sys.argv.index('--cnn')+1]
        params['device'] = get_device()
        if '+' in parameters:
            parameters = parameters.split('+')
            if isinstance(Y, list):
                params['n_classes'] = [np.shape(y)[-1] for y in Y]
            else:
                params['n_classes'] = np.shape(Y)[-1] 
        
        elif os.path.isfile(parameters):
            parameterfile = parameters.replace('model_params.dat', 'parameter.pth')
            obj = open(parameters,'r').readlines()
            parameters = []
            for l, line in enumerate(obj):
                if line[0] != '_' and line[:7] != 'outname':
                    parameters.append(line.strip().replace(' ', ''))
                if line[:7] == 'outname':
                    outname += 'from'+os.path.split(line.strip().split(' : ')[1])[1]
                    #outname += 'from'+os.path.split(parameterfile.replace('_parameter.pth', ''))[1]
            params['add_outname'] = False
            params['generate_paramfile'] = False
            train_model = False
            if ':' in sys.argv[sys.argv.index('--cnn')+2] or '=' in sys.argv[sys.argv.index('--cnn')+2]:
                if '+' in sys.argv[sys.argv.index('--cnn')+2]:
                    for a in sys.argv[sys.argv.index('--cnn')+2].split('+'):
                        parameters.append(a)
                else:
                    parameters.append(sys.argv[sys.argv.index('--cnn')+2])
        else:
            parameters = [parameters]
            if isinstance(Y, list):
                params['n_classes'] = [np.shape(y)[-1] for y in Y]
            else:
                params['n_classes'] = np.shape(Y)[-1]    
        params['outname'] = outname
        for p in parameters:
            if ':' in p and '=' in p:
                p = p.split('=',1)
            elif ':' in p:
                p = p.split(':',1)
            elif '=' in p:
                p = p.split('=',1)
            params[p[0]] = check(p[1])
        
        
        print('Device', params['device'])
        params['n_features'], params['l_seqs'], params['reverse_complement'] = np.shape(X)[-2], np.shape(X)[-1], reverse_complement
        model = cnn(**params)
        
    if weights is not None:
        weights = weights[trainset]
            
    
    translate_dict = None
    exclude_dict = []
    allow_reduction = False
    include = False
    if '--load_parameters' in sys.argv:
        # loads model parameters into state_dict, but only if the names of the parameters in the loaded path are the same as in the current model
        parameterfile = separate_sys(sys.argv[sys.argv.index('--load_parameters')+1])[0]
        translate_params = sys.argv[sys.argv.index('--load_parameters')+2] # potential dictionary with names from current model : to names in loaded models
        exclude_params = sys.argv[sys.argv.index('--load_parameters')+3] # if include False (default), list with names from current model that should not be replaced even if they are found in parameterfile.
        # if include True: then this is the exclusive list of model parameters that should be replaced
        include = sys.argv[sys.argv.index('--load_parameters')+4] == 'True' # if include then exclude list is list of parameters that will be included
        allow_reduction = sys.argv[sys.argv.index('--load_parameters')+5] == 'True' # Reduction can be used if number of kernels in current model differs from number of kernels in loaded models. 

        if ":" in translate_params:
            if ',' in translate_params:
                translate_params = translate_params.split(',')
            else:
                translate_params = [translate_params]
            translate_dict = {}
            for tp in translate_params:
                translate_dict[tp.split(":")[0]] = tp.split(":")[1]
        
        if '.' in exclude_params:
            if ',' in exclude_params:
                exclude_dict = exclude_params.split(',')
            else:
                exclude_dict = [exclude_params]
    
    # Can be used if training might interrupted and training should be restarted automatically. 
    if '--loadifexist' in sys.argv and os.path.isfile(model.outname+'_parameter.pth'):
        parameterfile = model.outname+'_parameter.pth'
        if len(sys.argv) > sys.argv.index('--loadifexist')+1:
            if '=' in sys.argv[sys.argv.index('--loadifexist')+1]:
                if '+' in sys.argv[sys.argv.index('--loadifexist')+1]:
                    adjpar = sys.argv[sys.argv.index('--loadifexist')+1].split('+')
                else:
                    adjpar = [sys.argv[sys.argv.index('--loadifexist')+1]]
            for p in adjpar:
                p = p.split('=',1)
                model.__dict__[p[0]] = check(p[1])
                    
            
            
        
    # parameterfile is loaded into the model
    if parameterfile:
        load_parameters(model, parameterfile, translate_dict = translate_dict, exclude = exclude_dict, include = include, allow_reduction = allow_reduction)
        if model.generate_paramfile:
            obj = open(model.outname+'_model_params.dat', 'a')
            obj.write('param_file : '+ os.path.split(parameterfile)[1] +'\n')
            obj.close()
    
    # Only use with one output file
    if select_track is not None and parameterfile:
        
        if isinstance(experiments,list):
            n_classes = []
            for m, mask in enumerate(select_track):
                # if loaded model with '--cnn' has already reduced tracks, don't perform reduction
                if np.sum(mask) != model.classifier[m].classifier.Linear.weight.size(0):
                    model.classifier[m].classifier.Linear.weight = nn.Parameter(model.classifier[m].classifier.Linear.weight[mask])
                    model.classifier[m].classifier.Linear.bias = nn.Parameter(model.classifier[m].classifier.Linear.bias[mask])
                    n_classes.append(np.sum(mask))
                else:
                    n_classes.append(np.sum(mask))
                
            model.n_classes = n_classes
            
        else:
            if np.sum(select_track) != model.classifier.classifier.Linear.weight.size(0):
                model.classifier.classifier.Linear.weight = nn.Parameter(model.classifier.classifier.Linear.weight[select_track])
                model.classifier.classifier.Linear.bias = nn.Parameter(model.classifier.classifier.Linear.bias[select_track])
                model.n_classes = len(select_track)
        
        if model.generate_paramfile:
            lines = open(model.outname+'_model_params.dat', 'r').readlines()
            obj = open(model.outname+'_model_params.dat', 'w')
            for line in lines:
                if line[:len('n_classes')] == 'n_classes':
                    obj.write('n_classes : '+str(np.array(model.n_classes, dtype = str))+'\n')
                else:
                    obj.write(line)
            obj.close()
        
        if '--load_parameters' in sys.argv:
            if isinstance(experiments, list):
                for e, exp in enumerate(experiments):
                    experiments[e] = exp[select_track[e]]
                    if fileclass is not None:
                        fileclass[e] = np.array(fileclass[e])[select_track[e]]
                    print('From total', len(select_track[e]), len(experiments[e]), '('+','.join(experiments[e])+')', 'tracks selected', e)
                if isinstance(Y, list):
                    Y = [y[:, select_track[yi]] for yi, y in enumerate(Y)]
            else:
                Y = Y[:, select_track]
                experiments = experiments[select_track]
                print('From total', len(select_track), len(experiments), '('+','.join(experiments)+')', 'tracks selected')
        
    # pre-loaded pwms are loaded into the model as convolutional weights
    if pwmusage == 'initialize':
        init_kernels = [pwmset(pwm.T, params['l_kernels'], shift_long = False) for pwm in pwms]
        init_kernels = np.concatenate(init_kernels, axis = 0)
        print('INIT kernels', len(init_kernels))
        if model.generate_paramfile:
            obj = open(model.outname+'_model_params.dat', 'a')
            obj.write('param_file : '+ os.path.split(list_of_pwms)[1] +'\n')
            obj.close()
        init_kernels = OrderedDict({ 'convolutions.conv1d.weight': torch.Tensor(init_kernels[np.argsort(-np.sum(init_kernels, axis = (1,2)))])})
        if model.kernel_bias is not None:
            init_kernels['convolutions.conv1d.bias'] = torch.zeros(model.num_kernels)
        load_parameters(model, init_kernels, allow_reduction = True)
    
    
    if train_model:
        if isinstance(Y, list):
            Ytraintomodel, Yvaltomodel = [y[trainset] for y in Y], [y[valset] for y in Y]
        else:
            Ytraintomodel, Yvaltomodel = Y[trainset], Y[valset]
        model.fit(X[trainset], Ytraintomodel, XYval = [X[valset], Yvaltomodel], sample_weights = weights)
    
    Y_pred = model.predict(X[testset])
    
    if isinstance(Y,list):
        # Y_pred returns concatenated list
        Y = np.concatenate(Y, axis = 1)
    if isinstance(experiments, list):
        experiments = np.concatenate(experiments)
        
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    if model.outname is not None and model.outname != outname:
        outname = model.outname
    print(outname)
    
    if '--save_embedding' in sys.argv:
        emb_layer = sys.argv[sys.argv.index('--save_embedding')+1]
        Y_emb = model.predict(X[testset], location = emb_layer)
        np.savez_compressed(outname+'_embed'+emb_layer+'.npz', names = names[testset], values = Y_emb)
        
    meanclasses = None
    if '--average_outclasses' in sys.argv:
        meanclasses = np.genfromtxt(sys.argv[sys.argv.index('--average_outclasses')+1], dtype = str)
        tsort = []
        for exp in experiments:
            tsort.append(list(meanclasses[:,0]).index(exp))
        meanclasses = meanclasses[tsort][:,1]
    
    
    if Y is not None:
            
        # If we split on different classes, such as p_values and foldchanges but train on both, one can split the assessment between them
        if '--split_outclasses' in sys.argv:
            testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
            tsort = []
            for exp in experiments:
                tsort.append(list(testclasses[:,0]).index(exp))
            testclasses = testclasses[tsort][:,1]
        else:
            testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
        
        if fileclass is not None:
            fileclass = np.concatenate(fileclass)
            experiments = experiments.astype('<U200')
            testclasses = testclasses.astype('<U200')
            if meanclasses is not None:
                meanclasses = meanclasses.astype('<U200')
            unexp, unexp_n = np.unique(experiments, return_counts = True)
            expname_add_class = (unexp_n > 1).any() # Determine automatically of fileclasses should be added to experiment names
            add_to_meanclass = False
            if meanclasses is not None:
                unmeanclass = np.unique(meanclasses)
                for umnc in unmeanclass:
                    mcmask = meanclasses == umnc
                    if len(np.unique(fileclass[mcmask]))>1:
                        add_to_meanclass = True
            for tclass in np.unique(testclasses):
                mask = np.where(testclasses == tclass)[0]
                if len(np.unique(fileclass[mask])) > 1:
                    for m in mask:
                        testclasses[m] = testclasses[m] + fileclass[m]
                        if expname_add_class:
                            experiments[m] = experiments[m] + '_' + fileclass[m]
                        if add_to_meanclass:
                            meanclasses[m] = meanclasses[m] + fileclass[m]
        
        
        # Sometimes we're not interested in the reconstruction for all genes in the training set and we can define a set of genes that we're most interested in
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            tlen = len(testset)
            tsetmask = np.isin(names[testset], siggenes)
            testset = testset[tsetmask]
            Y_pred = Y_pred[tsetmask]
            print('Testset length reduced to significant genes from', tlen, 'to', len(testset))

        # USE: --aurocaverage --auprcaverage --mseaverage --correlationaverage
        print_averages(Y_pred, Y[testset], testclasses, sys.argv)
        # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_perpoint '--save_mse_perpoint --save_auroc_perpoint --save_auprc_perpoint --save_topdowncorrelation_perclass
       
        save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], outname, sys.argv, compare_random = True, meanclasses = meanclasses)
    
        # Generate PWMs from activation of sequences
        # Generate global importance for every output track
        # Generate mean direction of effect from kernel
        # Generate global importance of kernel for each gene and testclasses
        if '--kernel_analysis' in sys.argv:
            # stppm determines the start index of the ppms that should be looked at and Nppm determines the number of ppms for which the operations is performed. only compute the correlation for the pwms from stppm up to stppm + Nppm
            
            kerkwargs = {}
            addppmname = ''
            if len(sys.argv) > sys.argv.index('--kernel_analysis')+1:
                # define stppm : first kernel to be collected, 
                # Nppm : number of kernels to be selected, 
                # nullify : mean, median, or zero
                # kernel_layer_name: name to detect kernel layer in state_dict
                # creates dict from '=' seperated item in string, items are separated by '+'
                kerkwargs = inputkwargs_from_string(sys.argv[sys.argv.index('--kernel_analysis')+1], definer='=', separater = '+')
                addppmname = add_name_from_dict(kerkwargs, cutkey = 2, cutitem = 3, keysep = '_')
                
            ppms, motifnames, importance, mseimportance, effect, abseffect, geneimportance, genetraincorr = kernel_assessment(model, X, Y, testclasses = testclasses, **kerkwargs)
            
            # Generate PPMs for kernels
            min_nonrand_freq = 0.3
            iupacmotifs = pfm2iupac(ppms, bk_freq = min_nonrand_freq)
            
            write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms.meme')
            
            print(np.shape(motifnames), np.shape(iupacmotifs), np.shape(importance))
            np.savetxt(outname+'_kernattrack'+addppmname+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), importance], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments)) # change in correlation for each class
            np.savetxt(outname+'_kernmsetrack'+addppmname+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), mseimportance], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments)) # change in mse for each class
            np.savetxt(outname+'_kernimpact'+addppmname+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), effect], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments)) # weighted effect by the change in mse for each gene
            np.savetxt(outname+'_kerneffct'+addppmname+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), abseffect], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments)) # directly computed effect for each cell type
            
            # summarize the correlation change of all genes that are predicted better than corrcut
            corrcut = 0.3 # minimum prediction correlation of data points to 
            # be considered
            if 'corrcut' in kerkwargs:
                corrcut = kerkwargs['corrcut']
            # correlation is returned as distance 1-R
            ccut = 1.-corrcut
            
            uclasses, nclasses = np.unique(testclasses, return_counts = True)
            # shape(geneimportance) = (testclasses, kernels, genes)
            meangeneimp = []
            for t, testclass in enumerate(np.unique(testclasses)):
                consider = np.where(testclasses == testclass)[0]
                if len(consider) > 1:
                    meangeneimp.append(np.mean(geneimportance[testclass][:,genetraincorr[testclass] <= ccut], axis = 1))
            meangeneimp = np.around(np.array(meangeneimp).T,6)
            # this is a measure on how important the kernel is for differences
            # tracks within a group, for example different B-cells
            np.savetxt(outname+'_kernattmeantestclass'+str(corrcut)+addppmname+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), meangeneimp], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(np.unique(testclasses)))
            # save all the impacts of all kernels on all genes in all testclasses
            if '--save_pointwise_kernel_correlation_importance' in sys.argv:
                np.savez_compressed(outname+'_kernattperpnt'+addppmname+'.npz', motifnames = motifnames, importance = geneimportance, base_correlation=genetraincorr, classes = uclasses, nclasses=nclasses, names = names)
    
    
    if '--save_predictions' in sys.argv:
        print('SAVED', outname+'_pred.npz', np.shape(Y_pred))
        #np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), fmt = '%s')
        np.savez_compressed(outname+'_pred.npz', names = names[testset], values = Y_pred, columns = experiments)
    
    
    if '--save_kernel_filters' in sys.argv:
        weights, biases, motifnames = extract_kernelweights_from_state_dict(model.state_dict(), kernel_layer_name = 'convolutions.conv1d')
        write_meme_file(weights, motifnames, 'ACGT', outname+'_kernelweights.meme', biases = biases)
    
    if '--sequence_attributions' in sys.argv:
        attribution_type = sys.argv[sys.argv.index('--sequence_attributions')+1]
        selected_tracks = sys.argv[sys.argv.index('--sequence_attributions')+2]
        track_indices = get_index_from_string(selected_tracks, experiments, delimiter = ',')
        
        kwargs = {}
        addppmname = ''
        if len(sys.argv) > sys.argv.index('--sequence_attributions')+2:
            kwargs = inputkwargs_from_string(sys.argv[sys.argv.index('--sequence_attributions')+1], definer='=', separater = '+')
            addppmname = add_name_from_dict(kwargs, cutkey = 2, cutitem = 3, keysep = '_')
                
        topattributions = None
        if '--topattributions' in sys.argv:
            topattributions = int(sys.argv[sys.argv.index('--topattributions')+1])
        if '--seqattribution_name' in sys.argv:
            selected_tracks = sys.argv[sys.argv.index('--gradname')+1]
        
        if meanclasses is not None:
            track_indices_classes = meanclasses[track_indices]
            ntrack_indices = []
            unique_track_indices_classes = np.unique(track_indices_classes)
            for trincl in unique_track_indices_classes:
                ntrack_indices.append(track_indices[track_indices_classes == trincl])
            track_indices = ntrack_indices
            
        if attribution_type == 'ism':
            attarray = ism(X[testset], model, track_indices)
        elif attribution_type == 'grad':
            attarray = takegrad(X[testset], model, track_indices, top = topattributions)
        elif attribution_type == 'deepshap': 
            attarray = deeplift(X[testset], model, track_indices, top = topattributions)
        else:
            attarray = captum_sequence_attributions(X[testset], model, track_indices, attribution_method = attribution_type)
        
        if experiments is not None and meanclasses is None:
            track_indices = experiments[track_indices]
        elif meanclasses is not None:
            track_indices = unique_track_indices_classes
        
        np.savez_compressed(outname + '_'+attribution_type+addppmname+selected_tracks.replace(',', '-') + '.npz', names = names[testset], values = attarray, experiments = track_indices)
        print('Attributions saved as', outname + '_'+attribution_type+addppmname+selected_tracks.replace(',', '-') + '.npz')

    
    
    








