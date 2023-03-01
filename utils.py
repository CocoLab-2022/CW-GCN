import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from Graph import Graph

import scipy.io as scio
import scipy

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_noisy_data(source_name, target_name, data_folder=None):
    if data_folder is None:
        data_folder = './data/'

    source_file = data_folder + source_name + '.mat'
    target_file = data_folder + target_name + '.mat'

    data = scio.loadmat(source_file)
    
    xs = data['attrb']
    ys = data['group']
    adj = data['network']
     

    selected_idx = np.random.choice(len(ys), int(0.*len(ys)), replace=False)
    #print(selected_idx) 
    adj_ori = adj 
    
    for i in range(selected_idx.shape[0]):
       for j in range(adj.shape[1]):
             if adj_ori[selected_idx[i],j]==1:
                adj[selected_idx[i],j]=0
                adj[j,selected_idx[i]]=0
				
	
    for i in range(selected_idx.shape[0]):
           a=adj[selected_idx[i],:]
           #a= np.where( a> 0, a, 0)
           sum_0 = np.sum(a)
           selected = np.random.choice(adj.shape[1], int(100), replace=False)
           for j in range(selected.shape[0]):

               adj[selected_idx[i],selected[j]]=1
               adj[selected[j],selected_idx[i]]=1
    


    selected_idx_2 = np.random.choice(len(ys), int(0.1*len(ys)), replace=False)

    permuted_idx = np.random.permutation(xs.shape[1])


    for i in range(selected_idx_2.shape[0]):
          x=xs[selected_idx_2[i]]
          x=np.array(x).reshape(1,len(x))

          x=scipy.stats.cauchy.rvs(loc=0, scale=1, size=xs.shape[1])
          xs[selected_idx_2[i]] =x

    adj = sp.coo_matrix(adj)
    xs = sp.coo_matrix(xs)
    
    idx_train = range(len(ys))
    idx_val = range(len(ys),len(ys))

    train_mask = sample_mask(idx_train, ys.shape[0])
    val_mask = sample_mask(idx_val, ys.shape[0])

    y_train = np.zeros(ys.shape)
    y_val = np.zeros(ys.shape)
    y_train[train_mask, :] = ys[train_mask, :]
    y_val[val_mask, :] = ys[val_mask, :]


    data = scio.loadmat(target_file)
    
    xt = data['attrb']
    yt = data['group']
    adj_target = data['network']
    adj_target = sp.coo_matrix(adj_target)
    xt = sp.coo_matrix(xt)
    

    idx_target = range(len(yt))

    target_mask = sample_mask(idx_target, yt.shape[0])

    y_target = np.zeros(yt.shape)
    y_target[target_mask, :] = yt[target_mask, :]

    return  adj,  xs, y_train, y_val, train_mask, val_mask, adj_target, xt, y_target, target_mask


def load_clean_data(source_name, target_name, data_folder=None):
    if data_folder is None:
        data_folder = './data/'

    source_file = data_folder + source_name + '.mat'
    target_file = data_folder + target_name + '.mat'

    data = scio.loadmat(source_file)

    xs = data['attrb']
    ys = data['group']
    adj = data['network']
    adj = sp.coo_matrix(adj)
    xs = sp.coo_matrix(xs)

    idx_train = range(len(ys))
    idx_val = range(len(ys), len(ys))

    train_mask = sample_mask(idx_train, ys.shape[0])
    val_mask = sample_mask(idx_val, ys.shape[0])

    y_train = np.zeros(ys.shape)
    y_val = np.zeros(ys.shape)
    y_train[train_mask, :] = ys[train_mask, :]
    y_val[val_mask, :] = ys[val_mask, :]

    data = scio.loadmat(target_file)

    xt = data['attrb']
    yt = data['group']
    adj_target = data['network']
    adj_target = sp.coo_matrix(adj_target)
    xt = sp.coo_matrix(xt)

    idx_target = range(len(yt))

    target_mask = sample_mask(idx_target, yt.shape[0])

    y_target = np.zeros(yt.shape)
    y_target[target_mask, :] = yt[target_mask, :]

    return adj, xs, y_train, y_val, train_mask, val_mask, adj_target, xt, y_target, target_mask


def load_data_1(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    seed = 30
    np.random.seed(seed)
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj_clean = adj    

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+300)
    

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

     
    selected_idx = np.random.choice(len(y), int(0.*len(y)), replace=False)
    #print(selected_idx) 
    A=adj.todense()
    G=nx.from_dict_of_lists(graph)
    
    for i in range(selected_idx.shape[0]):
       for j in range(selected_idx[i],labels.shape[0]):
             if A[selected_idx[i],j]==1:
                G.remove_edge(selected_idx[i],j)
  
    adj_reduce=nx.adjacency_matrix(G)


    for i in range(int(1*selected_idx.shape[0])):
       
       x=A[selected_idx[i],:]
       #print(x)
       y_ind=np.where(x==0)
       y_ind=y_ind[1]
       #print(y_ind)

       selected = np.random.choice(y_ind, int(50), replace=False)
       #selected = np.random.choice(labels.shape[0], int(500), replace=False)
                       

       for j in range(selected.shape[0]):
         
           G.add_edge(selected_idx[i], selected[j])
       
       x=np.zeros((1,np.size(x,1)),dtype=float)
       #print(x.shape)
       x[:,selected]=1

       A[selected_idx[i],:]=x
      
    #adj=sp.csr_matrix(A)
    adj=nx.adjacency_matrix(G)
    #print(adj)    
    
    
    print(features.shape)
    #permuted_idx_num = np.random.permutation(len(idx_train))
    selected_idx_2 = np.random.choice(len(y), int(0.*len(y)), replace=False)
    permuted_idx = np.random.permutation(features.shape[1])



    for i in range(selected_idx_2.shape[0]):
          x=features[selected_idx_2[i]]
          x=x.todense()
          #print(i)
           
          selected_idx_idx = np.random.choice(permuted_idx, 1000, replace=False)
          x[:,selected_idx_idx]=np.random.normal(scale=1e0,size=(1,1000))#1-x[:,selected_idx_idx]
          features[selected_idx_2[i]] =sp.csr_matrix(x)
          #print( features[selected_idx_2[i]])
    
       

    return  adj, adj_reduce, adj_clean, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask,  placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_feed_dict_target(features,support,labels,labels_mask,mean_s,std_s, mean_s_2,std_s_2, embedding_1, embedding_2, placeholders):
    """Construct feed dictionary."""
    # A=adj.todense()
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['support_orig']: A})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['mean_s']: mean_s})
    feed_dict.update({placeholders['std_s']: std_s})
    feed_dict.update({placeholders['mean_s_2']: mean_s_2})
    feed_dict.update({placeholders['std_s_2']: std_s_2})
    feed_dict.update({placeholders['embedding_1']: embedding_1})
    feed_dict.update({placeholders['embedding_2']: embedding_2})

    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
