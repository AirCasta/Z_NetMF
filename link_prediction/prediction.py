#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 12:18
# @Author  : Yuchen Sun
# @FileName: prediction.py
# @Project: PyCharm

import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.manifold import spectral_embedding
import scipy.sparse as sparse
import pickle
import time
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import pandas as pd

from link_prediction.netmf import approximate_normalized_graph_laplacian, approximate_deepwalk_matrix, \
    svd_deepwalk_matrix_lp, svd_deepwalk_matrix, get_biased_matrix
from link_prediction.node2vec import Graph
from link_prediction.train_split import mask_test_edges

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "L1": lambda a, b: np.abs(a - b),
    "L2": lambda a, b: np.abs(a - b) ** 2,
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_prediction_model(edges_emb, edges_label):
    lr = LogisticRegression()
    lr.fit(edges_emb, edges_label)
    return lr


# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=True):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score

def to_tuple(edges):
    edges_list = edges.tolist()  # convert to nested list
    edges_list = [tuple(node_pair) for node_pair in edges_list]  # convert node-pairs to tuples
    return edges_list

# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    train_edges_list = to_tuple(train_edges)
    train_edges_false_list = to_tuple(train_edges_false)
    test_edges_list = to_tuple(test_edges)
    test_edges_false_list = to_tuple(test_edges_false)
    val_edges_list = to_tuple(val_edges)
    val_edges_false_list = to_tuple(val_edges_false)
    return (train_edges_list + train_edges_false_list + test_edges_list +
            test_edges_false_list + val_edges_list + val_edges_false_list)

def get_correlation(edges, edges_false, corr_mat):
    pos_corrs = []
    for edge in edges:
        pos_corrs.append(corr_mat[edge[0], edge[1]])
    neg_corrs = []
    for edge in edges_false:
        neg_corrs.append(corr_mat[edge[0], edge[1]])
    corrs = np.concatenate([np.array(pos_corrs).reshape(-1,1), np.array(neg_corrs).reshape(-1,1)])
    labels = np.concatenate([np.ones(len(edges)), np.zeros(len(edges_false))])
    return corrs, labels

def train_lr(train_test_split, corr_matrix):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    train_edges_corr, train_edges_label = get_correlation(train_edges, train_edges_false, corr_matrix)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_edges_corr, val_edges_label = get_correlation(val_edges, val_edges_false, corr_matrix)
    test_edges_corr, test_edges_label = get_correlation(test_edges, test_edges_false, corr_matrix)

    classifier = get_prediction_model(train_edges_corr, train_edges_label)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = classifier.predict(val_edges_corr)
    test_preds = classifier.predict(test_edges_corr)

    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_roc = roc_auc_score(val_edges_label, val_preds)
        val_avg = average_precision_score(val_edges_label, val_preds)
    else:
        val_roc = None
        val_avg = None

    test_roc = roc_auc_score(test_edges_label, test_preds)
    test_avg = average_precision_score(test_edges_label, test_preds)
    return val_roc, val_avg, test_roc, test_avg


def common_neighbor_scores(g_train,train_test_split):
    if g_train.is_directed():  # Only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    start_time = time.time()
    cn_scores = {}

    # Calculate scores
    cn_matrix = np.zeros(adj_train.shape)
    for u, v in get_ebunch(train_test_split):  # (u, v) = node indices, p = Adamic-Adar index
        cn = len(list(nx.common_neighbors(g_train, u, v)))
        cn_matrix[u][v] = cn
        cn_matrix[v][u] = cn  # make sure it's symmetric
    cn_matrix = cn_matrix / cn_matrix.max()  # Normalize matrix

    runtime = time.time() - start_time
    # cn_roc, cn_ap = get_roc_score(test_edges, test_edges_false, cn_matrix)
    val_roc, val_avg, test_roc, test_avg = train_lr(train_test_split, cn_matrix)

    cn_scores['test_roc'] = test_roc
    cn_scores['test_ap'] = test_avg
    cn_scores['runtime'] = runtime
    return cn_scores

# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def adamic_adar_scores(g_train, train_test_split):
    if g_train.is_directed():  # Only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    start_time = time.time()
    aa_scores = {}

    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train, ebunch=get_ebunch(
            train_test_split)):  # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p  # make sure it's symmetric
    aa_matrix = aa_matrix / aa_matrix.max()  # Normalize matrix

    runtime = time.time() - start_time
    # aa_roc, aa_ap = get_roc_score(test_edges, test_edges_false, aa_matrix)
    val_roc, val_avg, test_roc, test_avg = train_lr(train_test_split, aa_matrix)
    aa_scores['test_roc'] = test_roc
    aa_scores['test_ap'] = test_avg
    aa_scores['runtime'] = runtime
    return aa_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def jaccard_coefficient_scores(g_train, train_test_split):
    if g_train.is_directed():  # Jaccard coef only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    start_time = time.time()
    jc_scores = {}

    # Calculate scores
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train, ebunch=get_ebunch(
            train_test_split)):  # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p  # make sure it's symmetric
    # print('max:', jc_matrix.max())
    jc_matrix = jc_matrix / jc_matrix.max()  # Normalize matrix

    runtime = time.time() - start_time
    # jc_roc, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)
    val_roc, val_avg, test_roc, test_avg = train_lr(train_test_split, jc_matrix)

    jc_scores['test_roc'] = test_roc
    jc_scores['test_ap'] = test_avg
    jc_scores['runtime'] = runtime
    return jc_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def preferential_attachment_scores(g_train, train_test_split):
    if g_train.is_directed():  # Only defined for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    start_time = time.time()
    pa_scores = {}

    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train, ebunch=get_ebunch(
            train_test_split)):  # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p  # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max()  # Normalize matrix

    runtime = time.time() - start_time
    # pa_roc, pa_ap = get_roc_score(test_edges, test_edges_false, pa_matrix)
    val_roc, val_avg, test_roc, test_avg = train_lr(train_test_split, pa_matrix)

    pa_scores['test_roc'] = test_roc
    pa_scores['test_ap'] = test_avg
    pa_scores['runtime'] = runtime
    return pa_scores

# Input: train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def spectral_clustering_scores(train_test_split, random_state = 0):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    sc_score = {}

    # spectral clustering
    spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    train_edges_corr, train_edges_label = get_correlation(train_edges, train_edges_false, sc_score_matrix)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_edges_corr, val_edges_label = get_correlation(val_edges, val_edges_false, sc_score_matrix)
    test_edges_corr, test_edges_label = get_correlation(test_edges, test_edges_false, sc_score_matrix)

    classifier = get_prediction_model(train_edges_corr, train_edges_label)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = classifier.predict(val_edges_corr)
    test_preds = classifier.predict(test_edges_corr)

    run_time = time.time() - start_time

    if len(val_edges) > 0 and len(val_edges_false) > 0:
        sc_val_roc = roc_auc_score(val_edges_label, val_preds)
        sc_val_avg = average_precision_score(val_edges_label, val_preds)
    else:
        sc_val_roc = None
        sc_val_avg = None

    sc_test_roc = roc_auc_score(test_edges_label, test_preds)
    sc_test_avg = average_precision_score(test_edges_label, test_preds)


    run_time = time.time()-start_time
    # sc_test_roc, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    # sc_val_roc, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    sc_score['test_roc'] = sc_test_roc
    sc_score['test_ap'] = sc_test_avg
    sc_score['val_roc'] = sc_val_roc
    sc_score['val_ap'] = sc_val_avg
    sc_score['run_time'] = run_time
    return sc_score

def get_edge_embedding(edge_list, emb_mat, edge_function):
    embs = []
    for edge in edge_list:
        n1 = edge[0]
        n2 = edge[1]
        emb1 = emb_mat[n1]
        emb2 = emb_mat[n2]
        edge_emb = edge_function(emb1, emb2)
        embs.append(edge_emb)

    return embs

def get_X_y(edges, edges_false, emb_mat, edge_function):
    pos_emb = get_edge_embedding(edges, emb_mat, edge_functions[edge_function])
    neg_emb = get_edge_embedding(edges_false, emb_mat, edge_functions[edge_function])
    edges_emb = np.concatenate([pos_emb, neg_emb])
    edges_label = np.concatenate([np.ones(len(edges)), np.zeros(len(edges_false))])
    return edges_emb, edges_label

# Input: original adj_sparse, train_test_split (from mask_test_edges), n2v hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def node2vec_scores(g_train, train_test_split, P = 1,
    Q = 1,
    WINDOW_SIZE = 10, 
    NUM_WALKS = 10, 
    WALK_LENGTH = 80, 
    DIMENSIONS = 128, 
    DIRECTED = False, 
    WORKERS = 8, 
    ITER = 1, 
    edge_score_funcs=["hadamard",], # computing methods of prediction scores
    verbose=1,):

    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    start_time = time.time()
    if verbose >= 1:
        print('Preprocessing graph for node2vec')
    g_n2v = Graph(g_train, DIRECTED, P, Q)
    g_n2v.preprocess_transition_probs()

    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
    emb_embedding = model.wv
    emb_list = []
    for node_index in range(adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_embedding[node_str]
        emb_list.append(node_emb)
    emb_mat = np.vstack(emb_list)
    res = []
    for edge_score_func in edge_score_funcs:
        train_edges_emb, train_edges_label = get_X_y(train_edges, train_edges_false, emb_mat, edge_score_func)
        if len(val_edges)>0 and len(val_edges_false)>0:
            val_edges_emb, val_edges_label = get_X_y(val_edges, val_edges_false, emb_mat, edge_score_func)
        test_edges_emb, test_edges_label = get_X_y(test_edges, test_edges_false, emb_mat, edge_score_func)

        classifier = get_prediction_model(train_edges_emb, train_edges_label)
        if len(val_edges)>0 and len(val_edges_false)>0:
            val_preds = classifier.predict(val_edges_emb)
        test_preds = classifier.predict(test_edges_emb)

        run_time = time.time() - start_time

        if len(val_edges)>0 and len(val_edges_false)>0:
            n2v_val_roc = roc_auc_score(val_edges_label, val_preds)
            n2v_val_avg = average_precision_score(val_edges_label, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_avg = None

        n2v_test_roc = roc_auc_score(test_edges_label, test_preds)
        n2v_test_avg = average_precision_score(test_edges_label, test_preds)

        n2v_scores = {}

        n2v_scores['test_roc'] = n2v_test_roc
        n2v_scores['test_ap'] = n2v_test_avg
        n2v_scores['val_roc'] = n2v_val_roc
        n2v_scores['val_ap'] = n2v_val_avg
        n2v_scores['runtime'] = run_time
        res.append(n2v_scores)
    return edge_score_funcs, res

# Input: original adj_sparse, train_test_split (from mask_test_edges), znm hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
# This is z_netmf_t
def znetmf_scores(
        g_train, 
        train_test_split,
        RANK = 256,  # Approximating Matrix decomposition dimension
        DIMENSIONS = 128,  # Embedding dimension
        WINDOW_SIZE = 10, # Context size for optimization
        NEGATIVE = 1.0,  # Negative sample
        Z = 1.0,  # Bias parameter
        edge_score_funcs=["hadamard",],
        verbose=1,
        emb_side='left',
):
    if g_train.is_directed():
        DIRECTED = True
    P = 1.0
    Q = 1.0
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split  # Unpack train-test split

    start_time = time.time()
    # load adjacency matrix
    vol = float(adj_train.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}, keep top-k eigenpairs, k is RANK
    evals, D_rt_invU = approximate_normalized_graph_laplacian(adj_train, rank=RANK, which="LA")
    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, Z, D_rt_invU,
                                                      window=WINDOW_SIZE,
                                                      vol=vol, b=NEGATIVE)
    # factorize deepwalk matrix with SVD
    if emb_side == 'right':
    	emb_matrix = svd_deepwalk_matrix_lp(deepwalk_matrix, dim=DIMENSIONS)
    else:
        emb_matrix = svd_deepwalk_matrix(deepwalk_matrix, dim=DIMENSIONS)
    res = []
    for edge_score_func in edge_score_funcs:
        train_edge_embs, train_edge_labels = get_X_y(train_edges, train_edges_false, emb_matrix, edge_score_func)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_edge_embs, val_edge_labels = get_X_y(val_edges, val_edges_false, emb_matrix, edge_score_func)
        test_edge_embs, test_edge_labels = get_X_y(test_edges, test_edges_false, emb_matrix, edge_score_func)

        # Train logistic regression classifier on train-set edge embeddings
        classifier = get_prediction_model(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = classifier.predict(val_edge_embs)
        test_preds = classifier.predict(test_edge_embs)

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            znm_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # znm_val_avg = average_precision_score(val_edge_labels, val_preds)
        else:
            znm_val_roc = None
            znm_val_avg = None

        # znm_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # znm_test_avg = average_precision_score(test_edge_labels, test_preds)

        # Record scores
        znm_scores = {}

        # znm_scores['test_roc'] = znm_test_roc
        # znm_scores['test_ap'] = znm_test_avg
        znm_scores['val_roc'] = znm_val_roc
        # znm_scores['val_ap'] = znm_val_avg
        znm_scores['runtime'] = runtime
        res.append(znm_scores)
    return edge_score_funcs, res

# Input: original adj_sparse, train_test_split (from mask_test_edges), znm hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
#  This is z_netmf_g
def nodemf_scores(
        g_train, 
        train_test_split,
        RANK = 256,  # Approximating Matrix decomposition dimension
        DIMENSIONS = 128,  # Embedding dimension
        WINDOW_SIZE = 10, # Context size for optimization
        NEGATIVE = 1.0,  # Negative sample
        P = 1.0,
        Q = 1.0,
        edge_score_funcs=["hadamard",],
        verbose=1,
        emb_side='left',
        Z = 1.0,
):
    if g_train.is_directed():
        DIRECTED = True
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split  # Unpack train-test split

    start_time = time.time()
    # load adjacency matrix
    if P != 1 or Q != 1:
        adj_train = get_biased_matrix(adj_train, P, Q)
    vol = float(adj_train.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}, keep top-k eigenpairs, k is RANK
    evals, D_rt_invU = approximate_normalized_graph_laplacian(adj_train, rank=RANK, which="LA")
    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, Z, D_rt_invU,
                                                      window=WINDOW_SIZE,
                                                      vol=vol, b=NEGATIVE)
    # factorize deepwalk matrix with SVD
    if emb_side == 'right':
    	emb_matrix = svd_deepwalk_matrix_lp(deepwalk_matrix, dim=DIMENSIONS)
    else:
        emb_matrix = svd_deepwalk_matrix(deepwalk_matrix, dim=DIMENSIONS)
    
    res = []
    for edge_score_func in edge_score_funcs:
        train_edge_embs, train_edge_labels = get_X_y(train_edges, train_edges_false, emb_matrix, edge_score_func)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_edge_embs, val_edge_labels = get_X_y(val_edges, val_edges_false, emb_matrix, edge_score_func)
        test_edge_embs, test_edge_labels = get_X_y(test_edges, test_edges_false, emb_matrix, edge_score_func)

        # Train logistic regression classifier on train-set edge embeddings
        classifier = get_prediction_model(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = classifier.predict(val_edge_embs)
        test_preds = classifier.predict(test_edge_embs)

        # runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            nnm_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # nnm_val_avg = average_precision_score(val_edge_labels, val_preds)
        else:
            nnm_val_roc = None
            # nnm_val_avg = None

        # nnm_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # nnm_test_avg = average_precision_score(test_edge_labels, test_preds)

        # Record scores
        nnm_scores = {}

        # nnm_scores['test_roc'] = nnm_test_roc
        # nnm_scores['test_ap'] = nnm_test_avg
        nnm_scores['val_roc'] = nnm_val_roc
        # nnm_scores['val_ap'] = nnm_val_avg
        # nnm_scores['runtime'] = runtime
        res.append(nnm_scores)
    return edge_score_funcs, res

def calculate_scores(adj_mat, feat=None, test_frac=.1, val_frac=.05, random_seed=0, verbose=1, train_test_file=None):
    np.random.seed(random_seed)

    # scores dictionary
    lp_scores = {}

    ### ---------- PREPROCESSING ---------- ###
    # train_test_split = ts.mask_test_edges(adj_mat, test_frac=test_frac, val_frac=val_frac)
    try:  # If found existing train-test split, use that file
        with open(train_test_file, 'rb') as f:
            train_test_split = pickle.load(f)
            print('Train-test split file is existed!')
    except:  # Else, generate train-test split on the fly
        print('Generating train-test split...')
        train_test_split = mask_test_edges(adj_mat, test_frac=test_frac, val_frac=val_frac)
        with open(train_test_file, 'wb') as f:
            pickle.dump(train_test_split,f)

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    g_train = nx.Graph(adj_train)
    if verbose >= 1:
        print("Total nodes:", adj_mat.shape[0])
        print("Total edges:", int(adj_mat.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
        print("Training edges (positive):", len(train_edges))
        print("Training edges (negative):", len(train_edges_false))
        print("Test edges (positive):", len(test_edges))
        print("Test edges (negative):", len(test_edges_false))
        print('')
        print("------------------------------------------------------")

    ### ---------- LINK PREDICTION BASELINES ---------- ###
    # # common neighbors
    # cn_scores = common_neighbor_scores(g_train, train_test_split)
    # lp_scores['cn'] = cn_scores
    # if verbose >= 1:
    #     print('')
    #     print('common neighbors Test ROC score: ', str(cn_scores['test_roc']))
    #     print('common neighbors Test AP score: ', str(cn_scores['test_ap']))

    # # Adamic-Adar
    # aa_scores = adamic_adar_scores(g_train, train_test_split)
    # lp_scores['aa'] = aa_scores
    # if verbose >= 1:
    #     print('')
    #     print('Adamic-Adar Test ROC score: ', str(aa_scores['test_roc']))
    #     print('Adamic-Adar Test AP score: ', str(aa_scores['test_ap']))

    # # Jaccard Coefficient
    # jc_scores = jaccard_coefficient_scores(g_train, train_test_split)
    # lp_scores['jc'] = jc_scores
    # if verbose >= 1:
    #     print('')
    #     print('Jaccard Coefficient Test ROC score: ', str(jc_scores['test_roc']))
    #     print('Jaccard Coefficient Test AP score: ', str(jc_scores['test_ap']))

    # # Preferential Attachment
    # pa_scores = preferential_attachment_scores(g_train, train_test_split)
    # lp_scores['pa'] = pa_scores
    # if verbose >= 1:
    #     print('')
    #     print('Preferential Attachment Test ROC score: ', str(pa_scores['test_roc']))
    #     print('Preferential Attachment Test AP score: ', str(pa_scores['test_ap']))

    ### ---------- SPECTRAL CLUSTERING ---------- ###
    # sc_scores = spectral_clustering_scores(train_test_split)
    # lp_scores['sc'] = sc_scores
    # if verbose >= 1:
    #     print('')
    #     print('Spectral Clustering Validation ROC score: ', str(sc_scores['val_roc']))
    #     print('Spectral Clustering Validation AP score: ', str(sc_scores['val_ap']))
    #     print('Spectral Clustering Test ROC score: ', str(sc_scores['test_roc']))
    #     print('Spectral Clustering Test AP score: ', str(sc_scores['test_ap']))

    modes = ["hadamard","L1","L2"]
    paras_str = ''
    ### ---------- NODE2VEC ---------- ###
    # # When p = q = 1, Node2Vec degrades into DeepWalk
    # P = 1  # p
    # Q = 1  # q
    # WINDOW_SIZE = 10  # context length
    # NUM_WALKS = 10  # Random walk times from each node
    # WALK_LENGTH = 80  # sequence length
    # DIMENSIONS = 128  # Embedding dimensions
    # DIRECTED = False  # Is directed?
    # WORKERS = 8  # Random workers
    # ITER = 1  # SGD iteration times

    # paras = [0.5, 1, 2]
    # paras_str += 'z_netmf_paras' + ','.join([str(i) for i in paras])
    # for P in paras:
    #     for Q in paras:
    #         modes, n2v_edge_emb_scores = node2vec_scores(g_train, train_test_split,
    #                                               P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED,
    #                                               WORKERS, ITER,
    #                                               modes,
    #                                               verbose)
    #         for i in range(len(modes)):
    #             mode = modes[i]
    #             mode1 = mode + '_' + str(P) + '_' + str(Q)
    #             lp_scores['n2v_' + mode1] = n2v_edge_emb_scores[i]

    #             if verbose >= 1:
    #                 print('')
    #                 print('node2vec (' + mode1 + ') Validation ROC score: ', str(n2v_edge_emb_scores[i]['val_roc']))
    #                 print('node2vec (' + mode1 + ') Validation AP score: ', str(n2v_edge_emb_scores[i]['val_ap']))
    #                 print('node2vec (' + mode1 + ') Test ROC score: ', str(n2v_edge_emb_scores[i]['test_roc']))
    #                 print('node2vec (' + mode1 + ') Test AP score: ', str(n2v_edge_emb_scores[i]['test_ap']))
    #                 print('')

    # ### ---------- Z-NetMF-t ---------- ###
    # RANK = 256
    # DIMENSIONS = 128
    # WINDOW_SIZE = 10
    # NEGATIVE = 1.0
    # Z = 1.0
    # paras = [0.4, 0.8, 1.2, 1.6, 1.7, 1.9, 2.2, 2.5, 2.8, 3.0, 3.5]
    # paras_str += 'z_netmf_paras' + ','.join([str(i) for i in paras])+'\t'
    # df = pd.DataFrame([], columns=['mode', 'paras', 'val_roc'])
    # val_roc_min = 1
    # val_roc_max = 0
    # for Z in paras:
    #     modes, znm_edge_emb_scores = znetmf_scores(g_train, train_test_split, RANK, DIMENSIONS,
    #                                         WINDOW_SIZE, NEGATIVE, Z, modes, verbose, 'right')
    #
    #     for i in range(len(modes)):
    #         mode = modes[i]
    #         mode1 = mode + '_' + str(Z)
    #         lp_scores['znm_' + mode1] = znm_edge_emb_scores[i]
    #         if znm_edge_emb_scores[i]['val_roc'] < val_roc_min:
    #             val_roc_min = znm_edge_emb_scores[i]['val_roc']
    #         if znm_edge_emb_scores[i]['val_roc'] > val_roc_max:
    #             val_roc_max = znm_edge_emb_scores[i]['val_roc']
    #         dff = pd.DataFrame([[mode, Z, znm_edge_emb_scores[i]['val_roc']]], columns=['mode', 'paras', 'val_roc'])
    #         df = pd.concat([df, dff])
    #
    #
    #
    #         if verbose >= 1:
    #             print('')
    #             print('znetmf (' + mode1 + ') Validation ROC score: ', str(znm_edge_emb_scores[i]['val_roc']))
    #             # print('znetmf (' + mode1 + ') Validation AP score: ', str(znm_edge_emb_scores[i]['val_ap']))
    #             # print('znetmf (' + mode1 + ') Test ROC score: ', str(znm_edge_emb_scores[i]['test_roc']))
    #             # print('znetmf (' + mode1 + ') Test AP score: ', str(znm_edge_emb_scores[i]['test_ap']))

    ### ---------- Z-NetMF-g ---------- ###
    # RANK = 256
    # DIMENSIONS = 128
    # WINDOW_SIZE = 10
    # NEGATIVE = 1.0
    # P = 1.0
    # Q = 1.0
    # paras = [0.5, 1, 2]
    # paras_str += 'z_netmf_paras' + ','.join([str(i) for i in paras])
    # df = pd.DataFrame([], columns=['mode', 'paras', 'val_roc'])
    # val_roc_min = 1
    # val_roc_max = 0
    # for P in paras:
    #     for Q in paras:
    #         modes, nnm_edge_emb_scores = nodemf_scores(g_train, train_test_split, RANK, DIMENSIONS,
    #                                             WINDOW_SIZE, NEGATIVE, P, Q, modes, verbose, 'right')
    #         for i in range(len(modes)):
    #             mode = modes[i]
    #             mode1 = mode + '_' + str(P) + '_' + str(Q)
    #             lp_scores['nnm_' + mode1] = nnm_edge_emb_scores[i]
    #
    #             if nnm_edge_emb_scores[i]['val_roc'] < val_roc_min:
    #                 val_roc_min = nnm_edge_emb_scores[i]['val_roc']
    #             if nnm_edge_emb_scores[i]['val_roc'] > val_roc_max:
    #                 val_roc_max = nnm_edge_emb_scores[i]['val_roc']
    #             dff = pd.DataFrame([[mode, str(P) + ',' + str(Q), nnm_edge_emb_scores[i]['val_roc']]], columns=['mode', 'paras', 'val_roc'])
    #             df = pd.concat([df, dff])
    #
    #             if verbose >= 1:
    #                 print('')
    #                 print('nodemf (' + mode1 + ') Validation ROC score: ', str(nnm_edge_emb_scores[i]['val_roc']))
    #                 # print('nodemf (' + mode1 + ') Validation AP score: ', str(nnm_edge_emb_scores[i]['val_ap']))
    #                 # print('nodemf (' + mode1 + ') Test ROC score: ', str(nnm_edge_emb_scores[i]['test_roc']))
    #                 # print('nodemf (' + mode1 + ') Test AP score: ', str(nnm_edge_emb_scores[i]['test_ap']))
    #
    # return lp_scores, paras_str, df, val_roc_min, val_roc_max
    ### ---------- Z-NetMF-gt ---------- ###
    # try to find the best combination of P,Q,Z
    RANK = 256
    DIMENSIONS = 128
    WINDOW_SIZE = 10
    NEGATIVE = 1.0
    paras = [0.5, 1, 2]
    # z_paras = [0.4, 0.8, 1.2, 1.6, 1.7, 1.9, 2.2, 2.5, 2.8, 3.0, 3.5]
    z_paras = [0.4, 3.0, 3.5]
    paras_str += 'z_netmf_paras' + ','.join([str(i) for i in paras])
    df = pd.DataFrame([], columns=['mode', 'paras', 'val_roc'])
    val_roc_min = 1
    val_roc_max = 0
    for P in paras:
        for Q in paras:
            for Z in z_paras:
                modes, nnm_edge_emb_scores = nodemf_scores(g_train, train_test_split, RANK, DIMENSIONS,
                                                           WINDOW_SIZE, NEGATIVE, P, Q, modes, verbose, 'right', Z)
                for i in range(len(modes)):
                    mode = modes[i]
                    mode1 = mode + '_' + str(P) + '_' + str(Q) + '_' + str(Z)
                    lp_scores['nnm_' + mode1] = nnm_edge_emb_scores[i]

                    if nnm_edge_emb_scores[i]['val_roc'] < val_roc_min:
                        val_roc_min = nnm_edge_emb_scores[i]['val_roc']
                    if nnm_edge_emb_scores[i]['val_roc'] > val_roc_max:
                        val_roc_max = nnm_edge_emb_scores[i]['val_roc']
                    dff = pd.DataFrame([[mode, str(P) + ',' + str(Q) + ',' + str(Z), nnm_edge_emb_scores[i]['val_roc']]],
                                       columns=['mode', 'paras', 'val_roc'])
                    df = pd.concat([df, dff])

                    if verbose >= 1:
                        print('')
                        print('nodemf (' + mode1 + ') Validation ROC score: ', str(nnm_edge_emb_scores[i]['val_roc']))
                        # print('nodemf (' + mode1 + ') Validation AP score: ', str(nnm_edge_emb_scores[i]['val_ap']))
                        # print('nodemf (' + mode1 + ') Test ROC score: ', str(nnm_edge_emb_scores[i]['test_roc']))
                        # print('nodemf (' + mode1 + ') Test AP score: ', str(nnm_edge_emb_scores[i]['test_ap']))

    return lp_scores, paras_str, df, val_roc_min, val_roc_max