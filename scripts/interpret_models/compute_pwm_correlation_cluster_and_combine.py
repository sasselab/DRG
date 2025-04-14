# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering
import argparse
 
from drg_tools.io_utils import readin_motif_files, write_meme_file
from drg_tools.motif_analysis import pfm2iupac, combine_pwms, torch_compute_similarity_motifs, complement_clustering


'''
Internal functions that are only needed for clustering with subset and approximating clusters
'''

def combine_pwms_separately(pwm_set, clusters):
    '''
    Compute offsets and distances for each cluster separately to avoid memory
    issues
    '''
    clusterpwms = [] 
    for c, uc in enumerate(np.unique(clusters)):
        if uc >=0:
            mask = clusters == uc
            corr_left, ofs_left, revcomp_matrix_left = torch_compute_similarity_motifs(pwm_set[mask], pwm_set[mask], metric = args.distance_metric, return_alignment = True, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = args.approximate_distance)
            clusterpwms.append(combine_pwms(pwm_set[mask], clusters[mask], 1-corr_left, ofs_left, revcomp_matrix_left)[0])
    return clusterpwms


'''
TODO: 
Give connectivity graph to agglomerative clustering, f.e. from correlation of effects

'''





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='cluster_seqlets',
                    description='Aligns and computes Pearson correlation distance between all seqlets, then uses agglomerative clustering to determine groups, and combines seqlets into common motif')
    parser.add_argument('pwmfile', type=str, 
                        help='This can be a meme, a txt, or npz file with pwms and pwmnames, OR the .npz file with the stats from previous clustering')
    parser.add_argument('linkage', type=str, 
                        help='Linkage type for agglomerative clustering, or file with clusters from previous computation to generate combined PWMs')
    parser.add_argument('--distance_threshold', type=float, 
                        help='Clustering distance cut-off to form groups', default = None)
    parser.add_argument('--n_clusters', type=float, 
                        help='If given, clustering motifs into N clusters instead of using distance threshold', default = None)
    parser.add_argument('--outname', type=str, default = None)
    parser.add_argument('--infocont', action='store_true')
    parser.add_argument('--distance_metric', default = 'correlation', 
                        help = 'Either correlation, cosine, mse, or correlation_pvalue')
    parser.add_argument('--clusternames', action='store_true', 
                        help = 'If True combines original names with ; to long name for meme file. By default returns identifier as name')
    parser.add_argument('--save_stats', action='store_true')
    parser.add_argument('--reverse_complement', action='store_true', 
                        help = 'Determines if reverse complement will be compared as well')
    parser.add_argument('--approximate_distance', action='store_false', 
                        help = 'Uses regular torch conv1d to compute distance, ignores overhanging parts of shorter motif. Generally underestimates correlation between two motifs')
    parser.add_argument('--approximate_cluster_on', default = None, type = int, 
                        help='Define number of random motifs on which clustering will be performed, while rest will be assiged to best matching centroid of these clusters. Should be used if memory is too small for large distance matrix')
    
    parser.add_argument('--reduce_by_name', action='store_true', 
                        help='Combines seqlets with same name')
    parser.add_argument('--seqname_delimiter', type=str, default = '_',
                        help='Delimiter to split motif name on')
    parser.add_argument('--seqname_inclusion', type=int, default = 2,
                        help='Number of strings that belong to sequences name after split by delimiter')
    
    parser.add_argument('--min_overlap', type = int, default = 4)
    
    args = parser.parse_args()
    
    if args.outname is None:
        outname = os.path.splitext(args.pwmfile)[0]+'ms'+str(args.min_overlap)
        if args.infocont:
            outname+='ic'
    else:
        outname = args.outname
    
    infmt= os.path.splitext(args.pwmfile)[1]
    
    # Determine if file contains only pwms, or also statistics that were saved 
    # in previous run
    isstatsfile = False
    if infmt == '.npz':
        pf = np.load(args.pwmfile, allow_pickle = True)
        pfiles = pf.files
        if ('correlation' in pfiles) and ('offsets' in pfiles):
            isstatsfile = True
    ofs = None
    if isstatsfile:
        pwmnames =pf['pwmnames']
        correlation = pf['correlation']
        ofs = pf['offsets']
        pwm_set = pf['pwms']
        revcomp_matrix = pf['revcomp_matrix']
        if args.outname is None:
            outname = outname.split('_stats')[0]
       
    else:
        pwm_set, pwmnames, nts = readin_motif_files(args.pwmfile)
        
        if not os.path.isfile(args.linkage):
        
            if args.reduce_by_name:
                # Reduce seqlets if they have the same name and the same 
                seq_names = np.array([args.seqname_delimiter.join(np.array(mn.split(args.seqname_delimiter, args.seqname_inclusion)[:args.seqname_inclusion]))+'_'+mn.rsplit('_',1)[-1] for mn in pwmnames])
                seqlet_names = np.copy(pwmnames)
                pwmnames, reduce_index, reverse_index = np.unique(seq_names, return_index = True, return_inverse = True)
                print(f'Reduced seqlets to {len(pwmnames)} from {len(seqlet_names)} by their names', args.seqname_delimiter, args.seqname_inclusion)
                
                npwm_set = []
                for pwmn in pwmnames:
                    mask = np.where(seq_names == pwmn)[0]
                    npwm_set.append(np.mean(pwm_set[mask], axis = 0))
                    #corrpwm = np.corrcoef(np.array(list(pwm_set[mask])).reshape(len(mask), -1), np.array(list(pwm_set[mask])).reshape(len(mask), -1))
                    #if (corrpwm < 0.95).any():
                        #print(np.amin(corrpwm))
                pwm_set = np.array(npwm_set, dtype = object)
            
            if args.approximate_cluster_on is not None: # Approximation of clusters
                # cluster only subset and assign left over seqlets to assigned 
                # clusters based on the similarity to the three most distant 
                # points in the cluster. Reduced memory by only computing 
                # similarity to three points per cluster.
                np.random.seed(0)
                if args.outname is None:
                    outname += 'aprx'+str(args.approximate_cluster_on)
                rand_mix = np.random.permutation(len(pwm_set))
                rand_set,rand_left = rand_mix[:args.approximate_cluster_on], rand_mix[args.approximate_cluster_on:]
                
                pwm_left = pwm_set[rand_left]
                pwmnames_left = pwmnames[rand_left]
                pwm_set = pwm_set[rand_set]
                pwmnames = pwmnames[rand_set]
            
            # Align and compute correlatoin between seqlets using torch conv1d.
            correlation, ofs, revcomp_matrix= torch_compute_similarity_motifs(pwm_set, pwm_set, metric = args.distance_metric, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = args.approximate_distance, return_alignment = True)
            
            # Save computed statistics for later
            if args.save_stats and args.approximate_cluster_on is None:
                np.savez_compressed(outname+'_stats.npz', pwmnames = pwmnames, correlation = correlation, offsets = ofs, pwms = pwm_set, revcomp_matrix = revcomp_matrix)
    
    # Linkage defines the linkage function for clustering, or can be a file
    # with cluster assigments. In this case, distance_treshold is ignored and
    # only combine_pwms is executed.
    
    if os.path.isfile(args.linkage):
        if args.outname is None:
            outname = os.path.splitext(args.linkage)[0]
        clusters = np.genfromtxt(args.linkage, dtype = str)
        if not np.array_equal(pwmnames, clusters[:,0]):
            sort = []
            for p, pwn in enumerate(pwmnames):
                sort.append(list(clusters[:,0]).index(pwn))
            clusters = clusters[sort]
        clusters = clusters[:,1].astype(int)
        
        if ofs is None:
            clusterpwms = combine_pwms_separately(pwm_set, clusters)
        else:
            clusterpwms = combine_pwms(pwm_set, clusters, 1-correlation, ofs, revcomp_matrix)
    else: # If clusters not given, perform clustering
        
        outname += '_cld'+args.linkage
        if args.n_clusters:
            outname += 'N'+str(args.n_clusters)
        else:
            outname += str(args.distance_threshold)
        if '_' in args.distance_metric:
            for dm in args.distance_metric.split('_'):
                outname += dm[:3] 
        else:
            outname += args.distance_metric[:3]
            
        clustering = AgglomerativeClustering(n_clusters = args.n_clusters, metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(correlation)
        
        clusters = clustering.labels_
        
        if args.approximate_cluster_on is not None:
            
            clusters, pwmnames, pwm_set = complement_clustering(clusters, pwmnames, pwm_set, 1-correlation, pwmnames_left, pwm_left, rand_mix)
            
            clusterpwms = combine_pwms_separately(pwm_set, clusters)
        
        else:
           
            clusterpwms = combine_pwms(pwm_set, clusters, 1.-correlation, ofs, revcomp_matrix)
        
        if args.reduce_by_name:
            clusters = clusters[reverse_index]
            pwmnames = seqlet_names
        
        np.savetxt(outname + '.txt', np.array([pwmnames,clusters]).T, fmt = '%s')
        print(outname + '.txt')
        
    print(len(pwm_set), 'form', len(np.unique(clusters)), 'clusters')
        
    if args.clusternames:
        clusternames = [str(i) for i in np.unique(clusters) if i >= 0]
    else:
        clusternames = [';'.join(np.array(pwmnames)[clusters == i]) for i in np.unique(clusters)]
    
    clusterpwms = [np.around(cp,3) for cp in clusterpwms]
    write_meme_file(clusterpwms, clusternames, 'ACGT', outname +'pfms.meme', )
    
    uclust, nclust = np.unique(clusters, return_counts = True)
    uclust, nclust = uclust[uclust > -1], nclust[uclust > -1]
    nhist, yhist = np.unique(nclust, return_counts = True)
    for n, nh in enumerate(nhist):
        print(nh, yhist[n])


    
    
    
    

    
    
    
    
    
    
    
    
    
