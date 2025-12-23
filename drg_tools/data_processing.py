# data_processing.py

'''
Contains functions to modify the content of data matrices

'''

import sys, os 
import numpy as np
import umap
from scipy.linalg import svd
import sklearn.manifold as skm
from sklearn.decomposition import TruncatedSVD, NMF, PCA

def _groupings(tlen, groupsizes, kfold):
    groups = []
    csize = []
    avail = np.arange(len(groupsizes), dtype = int)
    while True:
        if len(avail) < 1 or len(csize) == kfold:
            break
        seed = np.random.choice(avail)
        group = np.array([seed])
        avail = avail[~np.isin(avail, group)]
        gdist = abs(tlen-np.sum(groupsizes[group]))
        while True:
            if len(avail) < 1:
                groups.append(group)
                csize.append(int(np.sum(groupsizes[group])))
                break
            ngr = avail.reshape(-1,1)
            egr = np.repeat(group, len(ngr)).reshape(len(group), len(ngr)).T
            pgr = np.append(egr, ngr, axis = 1)
            #print(gdist, len(avail), len(group))
            pdist = np.abs(tlen-np.sum(groupsizes[pgr],axis = 1))
            if (pdist < gdist).any():
                mgr = np.argmin(pdist)
                group = pgr[mgr]
                gdist = pdist[mgr]
                avail = avail[~np.isin(avail, group)]
            else:
                groups.append(group)
                csize.append(int(np.sum(groupsizes[group])))
                break
    return groups, np.array(csize), np.mean(np.abs(np.array(csize) - tlen))


def generatetesttrain(names, groups, outname=None, kfold = 10, seed = 42):
    '''
    uses random sampling of groups to select combination of groups, so that 
    they are close to equal size.
    '''
    np.random.seed(seed)
    ugroups, ugroupsize = np.unique(groups, return_counts = True)
    #print(ugroups, ugroupsize)
    n = len(names)
    st = int(n/kfold)
    cdist = st
    for i in range(10000): #sampel 10,000 random possibile combinations
        cgroups, cgroupsizes, msize = _groupings(st, ugroupsize, kfold)
        #print(cgroups, cgroupsizes, msize)
        if msize < cdist:
            combgroups = cgroups
            combsize = cgroupsizes
            cdist = np.copy(msize)
    print('Best split', cdist)

    if outname is None:
        fold_ids = []
        for j, grp in enumerate(combgroups):
            print(j, ugroups[grp], np.sum(ugroupsize[grp]) - st)
            test = names[np.isin(groups, ugroups[grp])]
            fold_ids.append(test)
        return fold_ids
    else:
        obj=open(outname, 'w')
        for j, grp in enumerate(combgroups):
            print(j, ugroups[grp], np.sum(ugroupsize[grp]) - st)
            test = names[np.isin(groups, ugroups[grp])]
            obj.write('# Set_'+str(j)+'\n' + ' '.join(test)+'\n')

def reduce_dim(X, red, var, center = True):
    '''
    Reduces the dimension of a matrix of size N_data, d_features
    '''
    if center:
        Xmean = np.mean(X, axis = 0)
        X = X - Xmean
    if 'standard' in red:
        X /= np.std(X, axis = 0)
    U, S, Vh = svd(X, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
    print(np.shape(U))
    # check if singular values are in correct order
    if not np.array_equal(np.argsort(-S), np.arange(len(S))):
        print('Singular values not sorted')
    if var > 1:
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
    else: 
        Sv = S**2
        SV = np.cumsum(Sv)/np.sum(Sv)
        var = np.where(SV < var)[0][-1] + 1
        print('Selected', var, 'singular values')
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
        print(SV[:var])
    if 'normed' not in red:
        U = U * S
    return U


class embedding():
    def __init__(self, norm = 'standard', normvar = 0.9, algorithm = 'umap', n_components = 2, metric = 'euclidean', **kwargs):
        self.normvar = normvar
        self.algorithm = algorithm
        self.center = True
        self.norm = norm
        if self.algorithm == 'NMF':
            self.center = False
            self.norm = 'none'
        self.mean = None
        self.std = None
        self.xfit = None
        self.n_components = n_components
        self.metric = 'euclidean'
        self.kwargs = kwargs
        
    def fit(self,x):
        print(self.normvar, self.algorithm)
        if self.normvar != 1:
            x = reduce_dim(x, self.norm, self.normvar, center = self.center)
        
        if self.algorithm == 'svd' or self.algorithm == 'SVD':
            self.decoder = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        elif self.algorithm == 'NMF':
            self.decoder = NMF(n_components=self.n_components, init = 'nndsvda')
        elif self.algorithm == 'pca' or self.algorithm == 'PCA':
            self.decoder = PCA(n_components=self.n_components)
        elif self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            print('Fit TSNE')
            self.decoder = skm.TSNE(n_components=self.n_components, metric = self.metric, **self.kwargs)
            # perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='deprecated'
        elif self.algorithm == 'umap' or self.algorithm == 'UMAP':
            print('Fit UMAP')
            self.decoder = umap.UMAP(n_components=self.n_components, metric = self.metric, **self.kwargs)
            # n_neighbors=15, n_components=2, metric='euclidean', metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True, n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None)
        if self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            return self.decoder.fit_transform(x)
        
        self.decoder.fit(x)
        print('Done fit')
        return self.transform(x)
    
    def transform(self,x):
        return self.decoder.transform(x)



def getcentroids(labels, distmat):
    '''
    Use distance matrix and group labels to determine the centroid of the group
    '''
    
    clust = np.unique(labels)
    print(len(clust), 'centroids in', len(labels))
    centroids = []
    maxdist = []
    for c, cl in enumerate(clust):
        mask = np.where(labels == cl)[0]
        mdist = distmat[mask][:,mask]
        argmin = np.argmin(np.sum(mdist, axis = 1))
        cent = mask[argmin]
        centroids.append(cent)
        maxdist.append(np.amax(mdist[argmin]))
    
    return np.array(centroids), np.array(maxdist)
    

def determine_cluster(X, distance, clustering, clustpar, cparams, maxsize = 10000):
    '''
    Cluster X on distance metric with clustering algorithm
    Only cluster maxsize data points, and aligns other data points to cluster centroids
    TODO
    Currently only agglomerative clustering implemented
    
    Parameters
    ----------
    clustpar : int, float
        if float, clustpar is distance threshold
        if int, clustpar is number of clusters
    cparams : dictionary
        with parameters for clustering algorithm
    maxsize : int 
        max number of data points for which distance matrix should be computed
    '''
    if type(clustpar) == int:
        n_clusters = clustpar
        distance_threshold = None
    else:
        n_clusters = None
        distance_threshold = clustpar
    
    Xold = None
    if len(X) > maxsize:
        mask = np.zeros(len(X)) # select random set of maxsize data points for clustering
        mask[np.random.permutation(len(X))[:maxsize]] = 1
        mask = mask == 1
        Xold = np.copy(X) # copy original X for assigning rest of data points later
        X= X[mask] # compute distance matrix only for maxsize data points
    distmat = cdist(X, X, distance)
    
    if clustering == 'agglomerative' or clustering == 'Agglomerative' or clustering == 'AgglomerativeClustering':
        ca = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', distance_threshold = distance_threshold, **cparams)
    ca.fit(distmat)
    labels = ca.labels_
    
    # possible changes:
        # add option to compare to the mean of clusters instead of centroid
        # add option to compare data points to other clusters if they were set to -1 because of cluster specific threshold.
    if Xold is not None:
        centroids, distcentroid = getcentroids(labels, distmat) # determine the centroids of all clusters to assign leftover data points by measuring the distance to them
        adjust = np.median(distcentroid[distcentroid != 0])# adjust centroid dists for single clusters to the median of distances of other clusters
        distcentroid[distcentroid == 0] = adjust # distcentroid contains the max distance of the centroid to other data points in the same cluster
        ndist = cdist(Xold, X[centroids], distance)
        argmins = np.argmin(ndist, axis = 1) # determine which centroid is closest for all data points
        amins = np.amin(ndist, axis = 1)
        nlabels = -np.ones(len(ndist), dtype = int) # new labels for all data points, intialized as -1
        nlabels = labels[centroids][argmins] # quickly assign all data points the closest cluster
        for l, lab in enumerate(np.unique(labels)): # set cluster labels back to -1 if distance to centroid does not fulfill distance threshold
            mask = np.where(nlabels == lab)[0]
            nlabels[mask[amins[mask] > distcentroid[l]]] = -1
        #nlabels[amins > distance_threshold] = -1 # alternatively one could simply check if it is closer than the original cut-off to the centroid. That would make it less likely that all other data points are within that distance
        print(int(np.sum(nlabels == -1)), 'not assigned out of', len(nlabels))
        labels = nlabels
    return labels


def manipulate_input(X, features, sysargv):
    if '--select_features' in sysargv:
        selected_feat = np.genfromtxt(sysargv[sysargv.index('--select_features')+1], dtype = str)
        if len(features[0]) == len(selected_feat[0]):
            featmask = np.isin(features, selected_feat)
        elif len(features[0]) < len(selected_feat[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        elif len(features[0]) > len(select_features[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        features, X = np.array(features)[featmask], X[:, featmask]
        print('X reduced to', np.shape(X))
        outname+= '_featsel'+str(len(X[0]))
        
        
    if '--centerfeature' in sysargv:
        outname += '-cenfeat'
        X = X - np.mean(X, axis = 0)
    
    elif '--centerdata' in sysargv:
        outname += '-cendata'
        X = X - np.mean(X, axis = 1)[:, None]


    if '--norm2feature' in sysargv:
        outname += '-n2feat'
        norm = np.sqrt(np.sum(X*X, axis = 0))
        norm[norm == 0] = 1.
        X = X/norm
        
    elif '--norm1feature' in sysargv:
        outname += '-n1feat'
        norm =np.sum(np.absolute(X), axis = 0)
        norm[norm == 0] = 1.
        X = X/norm
    
    if '--norm2data' in sysargv:
        outname += '-n2data'
        norm =np.sqrt(np.sum(X*X, axis = 1))[:, None] 
        X = X/norm
        
    elif '--norm1data' in sysargv:
        outname += '-n1data'
        X = X/np.sum(np.absolute(X), axis = 1)[:,None]
        
    return X, features


