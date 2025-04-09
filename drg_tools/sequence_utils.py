# sequence_utils.py
'''
Contains functions to manipulate sequences or their onehot encodings
Most functions work with numpy arrays

'''
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def split_seqs(seqs, size):
    '''
    splits arrays along the first axis and strings into k-mers of size
    # TODO inlude to k-mer search and integrate axis for arrays
    '''
    seqlets = []
    for s, seq in enumerate(seqs):
        lseq = len(seq)
        for l in range(lseq - size + 1):
            seqlets.append(seq[l:l+size])
    return np.array(seqlets)

def avgpool(x,window):
    lx = np.shape(x)[-1]
    if lx%window!=0:
        xtend = [int(np.floor((lx%window)/2)), int(np.ceil((lx%window)/2))]
        x = np.pad(x, pad_width = [[0,0],[0,0],xtend])
    lx = np.shape(x)[-1]
    xavg = np.zeros(list(np.shape(x)[:-1])+[int(lx/window)])
    for i in range(int(lx/window)):
        xavg[..., i] = np.mean(x[..., i*window:(i+1)*window], axis = -1)
    return xavg
    

def realign(X):
    '''
    Shifts a onehot encoded sequence with zeros on one end,
    f.e. transcripts of different length, to right side of 
    the array
    '''
    end_of_seq = np.sum(X,axis = (1,2)).astype(int)
    Xmirror = []
    lx = np.shape(X)[-2]
    for s, es in enumerate(end_of_seq):
        xmir = np.zeros(np.shape(X[s]), dtype = np.int8)
        xmir[lx-es:] = X[s,:es]
        Xmirror.append(xmir)
    return np.append(X, np.array(Xmirror), axis = -2)

def generate_random_onehot(lseq, nseq):
    seqs = np.zeros((nseq,4,lseq))
    pos = np.random.randint(0,4,lseq*nseq)
    pos0 = (np.arange(lseq*nseq,dtype=int)/lseq).astype(int)
    pos1 = np.arange(lseq*nseq,dtype=int)%lseq
    seqs[pos0,pos,pos1] = 1
    return seqs

def seqlen(arrayofseqs):
    '''
    Returns array with length of strings in arrayofseqs
    '''
    return np.array([len(seq) for seq in arrayofseqs])

def check_addonehot(onehotregion, shapeohvec1, selen):
    # check conditions if one-hot encoded regions can be added
    addonehot = False
    if onehotregion is not None:
        if np.shape(onehotregion)[1] == shapeohvec1 :
            addonehot = True
        else:
            print("number of genetic regions do not match sequences")
            sys.exit()
    return addonehot





def seq_onehot(sequence, nucs = 'ACGT', case_sensitive = False):
    '''
    Generates a one-hot encoding of a sequence
    '''
    if not case_sensitive:
        sequence = sequence.upper()
    if isinstance(sequence, str):
        sequence = np.array(list(sequence))
    elif isinstance(sequence, list):
        sequence = np.array(sequence)
    ohvec = sequence[:, None] == np.array(list(nucs))
    return ohvec

# generates one-hot encoding by comparing arrays
def quick_onehot(sequences, nucs = 'ACGT', wildcard = None, onehotregion = None, region_names = None, align = 'left'):
    selen = seqlen(sequences)
    nucs = np.array(list(nucs))
    if align == 'bidirectional':
        mlenseqs = 2*np.amax(selen) + 20
    else:
        mlenseqs = np.amax(selen) 
    ohvec = np.zeros((len(sequences),mlenseqs, len(nucs)), dtype = np.int8)
    
    if wildcard is not None:
        nucs.append(wildcard)
    nucs = np.array(nucs)
    # check conditions if one-hot encoded regions can be added
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)

    if align == 'left' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][:len(sequence)] = np.array(list(sequence))[:, None] == nucs
    if align == 'right' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][-len(sequence):] = np.array(list(sequence))[:, None] == nucs
    if align == 'center':
        for s, sequence in enumerate(sequences):
            ohvec[s][(mlenseqs-len(sequence))//2:(mlenseqs-len(sequence))//2+len(sequence)] = np.array(list(sequence))[:, None] == nucs
    ohvec = ohvec.astype(np.int8)
    
    if addonehot:
        ohvec = np.append(ohvec, onehotregion, axis = -1)
        nucs = np.append(nucs, region_names)
        print(nucs)
    return ohvec, nucs

def append_reverse_complement(X):
    '''
    Reverse complement with X[::-1][:,::-1] only works if order of bases
    is ACGT, or TGCA.
    '''
    X = np.append(X, X[::-1][:,::-1], axis = 1)
    return X

# given a pwm, generate a kmer sequence for largest frequencies along the length of the pwm   
def kmer_from_pwm(pwm, nts = 'ACGT', axis = None):
    nts = np.array(list('ACGT'))
    if axis is None:
        axis = np.where(np.array(np.shape(pwm)) == len(nts))[0][-1]
    if axis == 0:
        pwm = pwm[:,np.sum(np.absolute(pwm).T,axis = 1)>0]
    else:
        pwm = pwm[np.sum(np.absolute(pwm),axis = 1)>0]
    kmer = ''.join(nts[np.argmax(pwm, axis = axis)])
    return kmer

# given a kmer sequence, generate a one-hot encoded pwm and normalize to 1.
def pwm_from_kmer(kmer, nts = 'ACGT', l_pwm = None):
    pwm = np.array(list(kmer))[None, :] == np.array(list(nts))[:,None]
    if l_pwm is None:
        pwm = pwm.astype(float)/np.sum(pwm)
    else:
        npwm = np.zeros((np.shape(pwm)[0], l_pwm))
        npwm[:, int((l_pwm-np.shape(pwm)[1])/2):int((l_pwm-np.shape(pwm)[1])/2)+np.shape(pwm)[1]] = pwm.astype(float)/np.sum(pwm)
        pwm = npwm
    return pwm

# Counts k-mers of lenght l_kmer in onehot encoded sequence
# K-mers that are looked for can be given as allkmers
def kmer_count(onehot, l_kmer, allkmers = None):
    kmerlist = []
    for s, seq in enumerate(onehot):
        seq = seq.T
        seq = kmer_from_pwm(seq)
        seqlist = []
        for p in range(len(seq)-l_kmer+1):
            seqlist.append(seq[p:p+l_kmer])
        kmerlist.append(seqlist)
    if allkmers is None:
        allkmers = np.unique(np.concatenate(kmerlist))
    kmernumbers = np.zeros((len(onehot), len(allkmers)))
    for s, slist in enumerate(kmerlist):
        slist, n = np.unique(slist, return_counts = True)
        kmernumbers[s,np.isin(allkmers, slist)] = n[np.isin(slist, allkmers)]
    return kmernumbers, allkmers



def add_sing(arr, sing):
    outarr = []
    for ar in arr:
        for si in sing:
            outarr.append(ar+si)
    return outarr

def def_region(onehotregkmer):
    region, regionnum = np.unique(np.nonzero(onehotregkmer)[-1], return_counts = True)
    region = region[np.argmax(regionnum)]
    return region
    


def make_kmer_representation(sequences, kmertype, kmerlength, gapsize = 0, kmers = None, nucleotides = 'ACGT', onehotregion = None, datatype = np.int8, mprocessing = False, num_cores = 1):
    genkmers = False
    if kmers is None:
        genkmers = True
        kmers = list(nucleotides)
    
    # check conditions if one-hot encoded regions can be added
    selen = seqlen(sequences)
    mlenseqs = np.amax(selen) 
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)
    ohvec = 0
    
    # regular kmer counts of lenght kmerlength
    if kmertype == 'regular':
        
        if genkmers:
            for i in range(kmerlength-1):
                kmers = add_sing(kmers, list(nucleotides))
        
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength,kmers, features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        featlist.append(seq[k:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            else:
                def findkmer(seq, kmerlength,kmers, features, s):
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        featlist.append(seq[k:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec,r in results:
                features[r] = rvec
            
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if not addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        featlist.append(seq[k:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
            
                else:
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        rfeatlist.append(seq[k:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                
    # all kmers of length 2 to kmerlength
    elif kmertype == 'decreasing':
        if genkmers:
            kmerlist = []
            for i in range(kmerlength-1):
                kmers = add_sing(kmers, list(nucleotides))
                if i + 2 >= gapsize:
                    kmerlist.append(kmers)
            
            kmers = list(np.concatenate(kmerlist))
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            featlist.append(seq[k:k+kl])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]   
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist = []
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            region = def_region(onehotregion[s, k:k+kl])
                            featlist.append(seq[k:k+kl]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if addonehot:
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            region = def_region(onehotregion[s, k:k+kl])
                            featlist.append(seq[k:k+kl]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            featlist.append(seq[k:k+kl])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
        
    elif kmertype == 'gapped':
        if genkmers:
            for i in range(kmerlength-1-gapsize):
                kmers = add_sing(kmers, list(nucleotides))
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]        
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist= []
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            region = def_region(np.append(onehotregion[s, k:k+g], onehotregion[s, k+g+gapsize:k+kmerlength], axis = 0))
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            region = def_region(np.append(onehotregion[s, k:k+g], onehotregion[s, k+g+gapsize:k+kmerlength], axis = 0))
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                
    elif kmertype == 'mismatch':
        # generate kmerlist
        if genkmers:
            print( 'Generate kmer')
            
            for i in range(kmerlength-1-gapsize):
                kmers = add_sing(kmers, list(nucleotides))
            rkmerfeat = np.copy(kmers)
            kmers = []
            for g in range(gapsize):
                gappedkmerfeat = []
                for kmer in rkmerfeat:
                    gkmer = []
                    for i in range(1,len(kmer)):
                        gkmer.append(kmer[:i]+'X'+kmer[i:])
                    gappedkmerfeat.append(gkmer)
                rkmerfeat = np.unique(np.concatenate(gappedkmerfeat))
            kmers = list(rkmerfeat)
            print( 'kmers done ...')
    
        # generate masks for wildcard elements:
        mask = np.zeros(kmerlength) > 0.
        masks = [mask]
        for g in range(gapsize):
            gapmasks = []
            for mask in masks:
                for i in range(1,len(mask)-1):
                    if mask[i] == False:
                        comask = np.copy(mask)
                        comask[i] = True
                        gapmasks.append(comask)
            masks = list(np.unique(gapmasks, axis = 0))
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer))
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[ np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]  
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer)+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[ np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(np.array(list(sequences[i])), kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                seq = np.array(list(seq))
                if addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer)+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for k in range(len(seq)-kmerlength+1):
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer))
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
        
    return features, kmers


def pwmset(pwms, targetlen):
    '''
    Puts the pwms in the middle of an array with length targetlen
    '''
    def largerpwm(pwm, targetlen):
        npwm = np.zeros((np.shape(pwm)[0], targetlen), dtype = np.float32)
        npwm[:, int((targetlen - np.shape(pwm)[1])/2):int((targetlen - np.shape(pwm)[1])/2)+np.shape(pwm)[1]] = pwm
        return npwm
    
    return np.array([largerpwm(pwm,targetlen) for pwm in pwms])
        

def pwm_scan(sequences, pwms, targetlen = None, pooling_type = 'Max', pooling_size = None, pooling_steps = None, motif_cutoff = None, set_to = 0., verbose = False, maxscale = False):
    '''
    Transforms one-hot encoded sequences into PWM activations
    TODO switch torch convolutions to do the scanning
    Can handle PWMs of different sizes. 
    '''
    
    
    if targetlen is None:
        pwmlen = np.array([len(pqm.T) for pqm in pwms])
        targetlen = np.amax(pwmlen)
    scanlen = np.shape(sequences)[-1]-targetlen+1
    
    if pooling_type == 'Max':
        def combfunc(sca ,axis = -1):
            return np.amax(sca, axis = axis)
    if pooling_type == 'Mean':
        def combfunc(sca ,axis = -1):
            return np.mean(sca, axis = axis)
    
    if pooling_size is None:
        pooling_size = scanlen
        pooling_steps = scanlen
        
    steps = int((scanlen - pooling_size)/pooling_steps) + 1 + int((scanlen - pooling_size)%pooling_steps > 0)
    
    setps = pwmset(pwms, targetlen)
    outscan = np.zeros((np.shape(sequences)[0],np.shape(setps)[0], steps), dtype = np.float32)
    
    if verbose:
        print('Scanning', len(sequences), 'sequences with', len(pwms), 'PWMs with pooling', pooling_size, pooling_steps)
   
    i = 0
    s = 0
    outscanmed = np.zeros((np.shape(sequences)[0],np.shape(setps)[0], pooling_size), dtype = np.float32)
    for l in range(scanlen):
        outscanmed[:, :, i] = np.sum(sequences[:,None,:,l:l+targetlen] * setps[None, :, :, :], axis = (-1, -2))
        i += 1
        if i == pooling_size:
            if verbose:
                print(l)
            outscan[:, :, s] = combfunc(outscanmed)
            s +=1
            i -= pooling_steps
            outscanmed[:, :, :pooling_size-pooling_steps] = outscanmed[:, :, pooling_steps:]

    if maxscale:
        outscan = outscan/np.amax(outscan, axis = (0,2))[None,:, None]
    if motif_cutoff is not None:
        outscan[outscan < motif_cutoff] = set_to
    return outscan
    
