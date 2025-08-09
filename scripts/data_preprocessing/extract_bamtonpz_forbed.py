#Uses bam and bed file to compute the coverage 
# within the regions in the bed file. 
# Can shift the reads or fragments and count individual 
# ends or over the entire fragment. 
# Returns npz file with 'names' of the regions from bed file, 
# 'counts' within that region. 
# Counts are of shape (names,1,bins_in_window) 
# to concatenate along axis one later.

# USE: python atac_counts.py sorted.GN.Thio.PC.bam ImmGenATAC1219.peak_matched.txt 
# --shift 4,-4 --outdir BPprofiles/ --countmode 53


import numpy as np
import sys, os
import pysam
import time

from drg_tools.io_utils import isint
from drg_tools.sequence_utils import avgpool

def readbed(bedfile, offset = 0):
    bf = open(bedfile, 'r').readlines()
    bedfile = {'names':[], 'contigs':[], 'positions': []}
    for l, line in enumerate(bf):
        line = line.strip().split()
        bedfile['names'].append(line[0])
        bedfile['contigs'].append(line[1])
        bedfile['positions'].append([int(line[2])-offset,int(line[3])-offset])
    bedfile['names'] = np.array(bedfile['names'])
    bedfile['contigs'] = np.array(bedfile['contigs'])
    bedfile['positions'] = np.array(bedfile['positions'])
    return bedfile


def get_total_reads(bamfile):
    total = 0
    obj = pysam.idxstats(bamfile).split('\n')
    print(obj)
    for l, line in enumerate(obj):
        line = line.split('\t')
        if line[0][:3] == 'chr': # and len(line[0]) <=5 :
            total += int(line[2])
    print('Total reads', total)
        
# generate arrays with indexes of all covered bases per chromosome
def chromcoverage(bedfile):
    indchrom = np.unique(bedfile['contigs'])
    coverage = {}
    for ic in indchrom:
        regions = np.where(bedfile['contigs'] == ic)[0]
        covered = np.unique(np.concatenate([np.arange(x[0], x[1]) for x in bedfile['positions'][regions]]))
        coverage[ic] = covered
    return coverage


if __name__ == '__main__':

    bamfile = pysam.AlignmentFile(sys.argv[1], 'rb')
    get_total_reads(sys.argv[1])

    offset = 0
    if '--oneindex' in sys.argv:
        offset = 1 # some location files start with 1, instead of 0. So 1 becomes 0 and so on. 
    bedfile = readbed(sys.argv[2], offset = offset)

    outname = os.path.splitext(sys.argv[2])[0]+'_in_'+os.path.splitext(os.path.split(sys.argv[1])[1])[0]

    #open bp array for each region in bedfile
    bedlen = [x[1] - x[0] for x in bedfile['positions']]
    seqlen = np.amax(bedlen)

    if '--extendbed' in sys.argv:
        exten = sys.argv[sys.argv.index('--extendbed')+1]
        outname += 'sl'+exten
        extend = int((int(exten) - seqlen) /2)
        print('Extend sequence by 2X', extend)
        seqlen = 2*extend + seqlen
        bedfile['positions'][:,0] -= extend
        bedfile['positions'][:,1] += extend
        bedfile['positions'][:,0][bedfile['positions'][:,0]<0] = 0
            


    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]

    rmode = 'fragment'
    if '--readmode' in sys.argv:
        # either 'fragment' based on 'read' based
        rmode = sys.argv[sys.argv.index('--readmode')+1]
        outname += rmode

    cmode = '53'
    if '--countmode' in sys.argv:
        # either 5 prime or 3' end or 53 or entire sequence, entirefraction distributes one weight along the fragment
        cmode = sys.argv[sys.argv.index('--countmode')+1]
        outname += cmode
        
    maxlen = None
    if '--maxlen' in sys.argv:
        maxlen = int(sys.argv[sys.argv.index('--maxlen')+1])
        outname += 'max'+str(maxlen)

    strand = None
    if '--strand' in sys.argv:
        strand = sys.argv[sys.argv.index('--strand')+1]
        outname += strand

    shift = None
    shift = np.array([4,-4])
    if '--shift' in sys.argv:
        shift = np.array(sys.argv[sys.argv.index('--shift')+1].split(','), dtype = int)
        outname += 'sh'.join(shift.astype(str))

    blacklisted = None
    if '--blacklist' in sys.argv:
        blacklisted = readbed(sys.argv[sys.argv.index('--blacklist')+1])
        outname += 'black'+os.path.splitext(os.path.split(sys.argv[sys.argv.index('--blacklist')+1])[1])[0]
        blacklisted = chromcoverage(blacklisted)
        
    dtype = np.int16
    if cmode == 'entirefraction':
        dtype = np.float32
    bedcounts = np.zeros((len(bedfile['names']), seqlen), dtype = dtype)

    print('Bed regions', len(bedfile['names']))
    i = 0 # number of bedfile regions 
    j = 0 # number of total counts
    if maxlen is None:
        offset = 1000 # offset is number of bp to look left and right of the region with fetch
    else:
        offset = maxlen + 1

    t0 = time.time()
    for c, chrom in enumerate(np.unique(bedfile['contigs'])):
        # Check if chromosome exists in bamfile and get length
        try:
            lenchrom = bamfile.get_reference_length(chrom)
            contained = True
        except:
            contained = False
        
        if contained:
            print(chrom, lenchrom)
            # get all positions in bedfile that are in chromosome
            mask = np.where(bedfile['contigs'] == chrom)[0]
            # iterate over these positions in the bedfile
            for m in mask:
                if i %10000 == 0:
                    print(i, 'time', round(time.time() -t0,1))
                i += 1
                bst, ben = bedfile['positions'][m]
                # fetch all reads that are within
                reads = bamfile.fetch(contig = chrom, start = max(0,bst-offset), stop = min(ben+offset, lenchrom))
    
                for read in reads:
                    # if we're interested in fragments, we only need to use the forward reads to determine start and end of the fragment and just ignore the backward reads to save time
                    if rmode == 'fragment' and read.is_proper_pair: # and read.query_name not in usedreads:
                        tl = read.template_length
                        st = read.reference_start
                        en = read.reference_end
                        if read.is_reverse:
                            st = en+tl
                        else:
                            en = st+tl
                        
                        #check if length of fragment too long
                        winlen = True
                        if maxlen is not None:
                            if maxlen < tl:
                                winlen = False
                        if winlen:
                            # shift start and end 
                            if shift is not None:
                                st, en = st + shift[0], en + shift[1]
                                
                            # check if the start and end of a fragement are in the bed area
                            stin, enin = (st >= bst) & (st<ben), (en <= ben)&(en>=bst)
                            if stin or enin:
                                
                                # check if start or end fall into a blacklisted area
                                nobl = True
                                if blacklisted is None:
                                    nobl = True
                                elif st not in blacklisted[chrom] and en not in blacklisted[chrom]:
                                    nobl = False
                                
                                if nobl:
                                    if cmode == '53':
                                        # start and end of fragment are counted
                                        if stin:
                                            bedcounts[m][st-bst] += 1
                                            j += 1
                                        if enin: 
                                            bedcounts[m][en-bst-1] += 1
                                            j += 1
                                    elif cmode == 'entire':
                                        # entire fragment region gets a count
                                        if stin and enin:
                                            bedcounts[m][st-bst:en-bst] += 1
                                        elif stin:
                                            bedcounts[m][st-bst:] += 1
                                        elif enin: 
                                            bedcounts[m][:en-bst] += 1
                                        j += 1
                                    elif cmode == 'entirefraction':
                                        # entire fragment region gets a count
                                        if stin and enin:
                                            bedcounts[m][st-bst:en-bst] += 1./tl
                                        elif stin:
                                            bedcounts[m][st-bst:] += 1./tl
                                        elif enin: 
                                            bedcounts[m][:en-bst] += 1./tl
                                        j += 1
                                    elif cmode == '5':
                                        if read.is_reverse and enin:
                                            bedcounts[m][en-bst-1] += 1
                                        elif stin:
                                            bedcounts[m][st-bst] += 1
                                        j += 1
                                    elif cmode == '3':
                                        if read.is_reverse and stin:
                                            bedcounts[m][st-bst] += 1
                                        elif enin:
                                            bedcounts[m][en-bst-1] += 1
                                        j += 1
                    


    print('Number of counts sequences per region', i, j, np.around(j/i,1))

    if '--meancounts' in sys.argv:
        # Takes the mean over 'mean_window' bps.
        mean_window= sys.argv[sys.argv.index('--meancounts')+1]
        if isint(mean_window):
            bedcounts = avgpool(bedcounts, int(mean_window))
        else:
            bedcounts = np.mean(bedcounts, axis = -1).reshape(-1,1)
        outname += 'avg'+str(mean_window)

    np.savez_compressed(outname+'.npz', names = np.array(bedfile['names']), counts = bedcounts[:,None, :])


