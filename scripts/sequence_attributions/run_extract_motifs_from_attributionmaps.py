import numpy as np
import sys, os
import glob

from drg_tools.io_utils import write_meme_file, get_indx
from drg_tools.motif_analysis import find_motifs

if __name__ == '__main__': 
    
    npzfile = sys.argv[1] # 
    npz = np.load(npzfile, allow_pickle = True)
    names, values, experiments = npz['names'], npz['values'], npz['experiments']
    outname = os.path.splitext(npzfile)[0]
    
    values = np.transpose(values, axes = (0,1,3,2))
    
    if '--select_tracks' in sys.argv:
        tracks = sys.argv[sys.argv.index('--select_tracks')+1]
        outname += tracks
        tracks = get_indx(tracks, experiments, islist = True)
        experiments=experiments[tracks]
        values = values[:,tracks]
    
    
    seq = np.load(sys.argv[2], allow_pickle = True)
    seqs, snames = seq['seqfeatures'], seq['genenames']
    
    sort = np.argsort(names)[np.isin(np.sort(names), snames)]
    names, values = names[sort], values[sort]
    sort = np.argsort(snames)[np.isin(np.sort(snames), names)]
    snames, seqs = snames[sort], seqs[sort]
    
    
    
    # Check if the attributions were stored in sparse coordinates and to
    # original size
    returned = False
    if np.shape(values)[-1] != np.shape(seqs)[-1]:
        returned = True
        nshape = list(np.shape(values))
        nshape[-1] = np.shape(seqs)[-1]
        nshape[-2] = np.shape(seqs)[-2]
        natt = np.zeros(nshape, dtype = np.float32)
        for a, at in enumerate(values):
            for b, bt in enumerate(at):
                natt[a,b, bt[:,-1].astype(int)] = bt[:, :nshape[-1]]
        values = natt
        print(np.shape(values))
    

    
    cut = float(sys.argv[3]) # cutoff for significant positions
    maxgap = int(sys.argv[4]) # maximum gap in motifs
    minsig = int(sys.argv[5]) # minimum number of signifianct bases in motif
    norm = sys.argv[6] # normalization to detect significant bases
    
    outname += '_'+norm+'motifs'+str(cut)+'_'+str(maxgap)+'_'+str(minsig)
    print(outname)
    if returned:
        # Estimate the variation of positions without attributions
        # TODO: Plot absolute attributions against rank
        # Derive correct function for behavior and replace
        mask = np.repeat(np.sum(np.abs(values),axis=-1)[...,None],4, axis =-1) == 0
        stdmask = np.sqrt(np.mean(np.ma.masked_array(values, mask)**2, axis = (-1,-2)))
        fvalues = np.copy(values)
        for s, stdm in enumerate(stdmask):
            for t, ttdm in enumerate(stdm):
                fvalues[s,t] = ttdm/2
        std = np.sqrt(np.mean(fvalues**2, axis = (-1,-2)))
    else:
        std = np.sqrt(np.mean(values**2, axis = (-1,-2)))
    
    if norm == 'condition':
        std = np.mean(std, axis = 0)[None,:, None]
    elif norm == 'seq':
        std = np.mean(std, axis = 1)[:,None, None]
    elif norm == 'global':
        std = np.mean(std)
    elif norm == 'std':
        std = np.float64(sys.argv[7])
    else:
        std = np.array(1.)
    print('std', std, type(std))
    
    refatt = np.sum(values*seqs[:,None,:,:], axis = -1)
    stats = refatt/std
    
    if '--normpwms' in sys.argv and norm not in ['global', 'seq', 'condition']:
        print('normpwms') # normalize values by std of individual sequence
        std = np.mean(np.sqrt(np.mean(values**2, axis = (-1,-2))))
    
    values = values/std[...,None]
    
    obj = open(outname+'.txt', 'w')
    pwms, pwmnames = [], []
    for n, name in enumerate(names):
        for e, exp in enumerate(experiments):
            motiflocs = find_motifs(stats[n,e], cut, maxgap, minsig)
            for m, ml in enumerate(motiflocs):
                ml = np.array(ml)
                seqname = name+'_'+str(exp)+'_'+str(ml[0])+'-'+str(ml[-1])
                # compute mean, max, loc
                mean = np.mean(refatt[n,e,ml])
                maxs = np.argmax(np.abs(refatt[n,e,ml]))
                maxs = refatt[n,e,ml[maxs]]
                obj.write(seqname+' '+str(round(mean,3))+' '+str(round(maxs,3))+' '+','.join(ml.astype(str))+'\n')
                pwmnames.append(seqname)
                pwms.append(values[n,e,ml[0]:ml[-1]+1]*np.sign(mean))
    
    print(f'{len(pwmnames)} extracted pwms from attributions of shape', np.shape(refatt))
    pwms = np.array(pwms, dtype = object)
    print(np.shape(pwms))
    np.savez_compressed(outname+'.npz', pwms = pwms, pwmnames = pwmnames)
            
    
    
    
    






    
