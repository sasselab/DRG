import numpy as np
import sys, os
import pickle

from drg_tools.io_utils import readgtf, readgenomefasta, reverse_complement_seqstring



if __name__ == '__main__':
        
    chrfold = sys.argv[1] # folder with all chromosomes in it
    infofile = sys.argv[2] # gtffile that contains all the locations for the regions

    outdir = os.path.split(sys.argv[2])[0]
    if outdir != '':
        outdir = outdir+'/'
    outname = outdir+sys.argv[1].strip('/').split('/')[-1]+os.path.splitext(os.path.split(sys.argv[2].strip('.gz'))[1])[0]

    # bedfile order:
    #chr start end name number_exon strand
    if os.path.splitext(infofile)[-1][:4] == '.bed':
        ifile = np.genfromtxt(infofile, dtype = str,ndmin = 2)
    elif os.path.splitext(infofile)[-1] == '.csv' or os.path.splitext(infofile)[-1] == '.txt':
        # This is a special file that we sometimes used with Name, chr, start, end in it.
        ifile = np.genfromtxt(infofile, dtype = str)
        ifile = ifile[:,[1,2,3,0]]
    else:
        ifile = readgtf(infofile)
    
    #print('Data format', np.shape(ifile))


    if '--generate_transcripts' in sys.argv:
        mask =ifile[:,4] == 'exon'
        ifile = ifile[mask]
        outname += '.trscrpt'

    elif '--from_tss' in sys.argv:
        mask =ifile[:,4] == 'gene'
        ifile = ifile[mask]
        outname += '.tss'

    elif '--seqtype' in sys.argv:
        seqtype = sys.argv[sys.argv.index('--seqtype')+1]
        mask =ifile[:,4] == seqtype
        ifile = ifile[mask]
        outname += '.'+seqtype
        print('After selecting seqtype', np.shape(ifile))

    if '--genetype' in sys.argv:
        genetype = sys.argv[sys.argv.index('--genetype')+1]
        mask =ifile[:,6] == genetype
        ifile = ifile[mask]
        outname += '.'+genetype
        print('After selecting genetype', np.shape(ifile))

    if '--usegeneid' in sys.argv:
        names = ifile[:,-1]
    elif '--usegeneid_noversion' in sys.argv:
        vnames = ifile[:,-1]
        names = []
        for v, vn in enumerate(vnames):
            if '.' in vn:
                vn = vn.split('.')[0]
            names.append(vn)
        names = np.array(names)
    else:
        names = ifile[:,3]


    offset = 0
    if '--oneindex' in sys.argv:
        # USE this if the bed file counts from 1-X and not from 0 to X
        offset = 1

    flank = 0
    if '--add_flanks' in sys.argv:
        flank = int(sys.argv[sys.argv.index('--add_flanks')+1])
        outname += 'flank'+str(flank)

    extend = False
    if '--extend_to_length' in sys.argv:
        extend = int(sys.argv[sys.argv.index('--extend_to_length')+1])
        outname += str(extend)

    if '--list' in sys.argv:
        thelist = np.genfromtxt(sys.argv[sys.argv.index('--list')+1], dtype = str)
        mask = np.isin(names, thelist)
        names, ifile = names[mask], ifile[mask]
        print(len(names), 'left')

    # iterate over all chromosomes in the infofile
    uchroms = np.unique(ifile[:,0])
    #print(uchroms)

    #print(outname)
    pos_info = {}

    # generate fasta file with sequences
    outfasta = open(outname+'.fasta', 'w')
    for u, uchr in enumerate(uchroms):
        if os.path.isfile(chrfold.rstrip('/')+'/'+uchr+'.fa.gz'):
            #print('Read',chrfold.rstrip('/')+'/'+uchr+'.fa.gz') 
            chrfasta = readgenomefasta(chrfold.rstrip('/')+'/'+uchr+'.fa.gz')
            print('Length', len(chrfasta))
            chrextra = 'N' * flank
            chrfasta = chrextra + chrfasta + chrextra
            chrmask = ifile[:,0] == uchr
            print('Generate seq for '+uchr)
            unames, indx = np.unique(names[chrmask], return_index = True)
            unames = unames[np.argsort(indx)]
            print('Locations in chr', len(unames))
            
            for n, na in enumerate(unames):
                chrmask = np.where(names == na)[0]
                
                if '--generate_transcripts' in sys.argv:
                    extseq = ''
                    for e in chrmask:
                        ext= chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                        if ifile[e, 3] == '-':
                            ext = reverse_complement_seqstring(ext)
                            extseq = ext + extseq
                        else:
                            extseq = extseq + ext
                    outfasta.write('>'+na+'\n'+extseq+'\n')
                
                elif '--from_tss' in sys.argv:
                    e = chrmask[0]
                    
                    if ifile[e, 5] == '-':
                        extseq = chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                        extseq = reverse_complement_seqstring(extseq)
                        outfasta.write('>'+na+'\n'+extseq+'\n')
                    elif ifile[e, 5] == '+':
                        extseq = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                        outfasta.write('>'+na+'\n'+extseq+'\n')
                    elif ifile[e, 5] == '.':
                        extseqf = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                        extseqb= chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                        extseqb = reverse_complement_seqstring(extseq)
                        outfasta.write('>'+na+'\n'+extseqf+'\n'+'>'+na+'.rev\n'+extseqb+'\n')
                else:
                    e = chrmask[0]
                    start_region, end_region = int(ifile[e, 1]),int(ifile[e,2])  
                    before, after = 0, 0
                    if extend:
                        before = (extend - (end_region-start_region)) // 2
                        after = (extend - (end_region-start_region)) // 2 + (extend - (end_region-start_region)) % 2
                        
                    extseq = chrfasta[start_region-before-offset-flank+flank:end_region + after-offset+flank+flank]
                    outfasta.write('>'+na+'\n'+extseq+'\n')
                    
                    if '--save_pos_info' in sys.argv:
                        pos_info[na]=np.array([uchr,start_region-before-offset-flank+flank,end_region + after-offset+flank+flank])
                
            if '--save_pos_info' in sys.argv:
                with open(f"{outname}_pos_info.pkl", "wb") as file:
                    pickle.dump(pos_info, file)

        else:
            print(chrfold.rstrip('/')+'/'+uchr+'.fa.gz', 'not found')
            
        


    
