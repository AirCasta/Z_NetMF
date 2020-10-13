#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T
import networkx as nx
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'


def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]

def get_biased_matrix(A, p, q):
    logger.info("Reweighting...")
    g = nx.from_scipy_sparse_matrix(A)
    for edge in g.edges():
        s = edge[0]
        e = edge[1]
        if len(list(g.neighbors(s))) == 1 or len(list(g.neighbors(1))) == 1:
            g.add_edge(s, e, weight=1/p*g[s][e]['weight'])
        elif len(list(nx.common_neighbors(g,s,e)))>0:
            g.add_edge(s, e, weight=1*g[s][e]['weight'])
        else:
            g.add_edge(s, e, weight=1/q*g[s][e]['weight'])
        # print(g[s][e],p,q)
    A = nx.to_scipy_sparse_matrix(g)
    logger.info("Computed adjacency matrix.")
    # print(A)
    return A

def deepwalk_filter(evals, z, window):
    # print(np.max(evals), np.min(evals))
    evals = z * evals
    if z == 1:
        nor = window
    else:
        nor = z*(1 - z**window)/(1 - z)
    for i in range(len(evals)):
        x = evals[i]
        # evals[i] = 1 if x >= 1 else x*(1-x**window) / (1-x) / window
        # evals[i] = x*(1-x**window) / (1-x) / window
        evals[i] = 1 if x>=z else x * (1 - x ** window) / (1 - x) / nor
        # if np.abs(evals[i]-1) <= 0.000001:
        #     print(x)
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, z, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, z, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(Y)

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def znetmf(args):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", args.window)

    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
                              variable_name=args.matfile_variable_name)
    # print(A.sum())
    if args.p != 1 or args.q != 1:
        A = get_biased_matrix(A, args.p, args.q)
    vol = float(A.sum())
    # print(vol)
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=args.rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, args.z, D_rt_invU,
                                                  window=args.window,
                                                  vol=vol, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)

    logger.info("Save embedding to %s_emb_z_" + str('%.1f' % args.z) + "_p_" + str('%.1f' % args.p)
                + "_q_" + str('%.1f' % args.q) + '.npy', args.output)
    np.save(args.output + '_emb_z_' + str('%.1f' % args.z) + "_p_" + str('%.1f' % args.p)
                + "_q_" + str('%.1f' % args.q) + '.npy', deepwalk_embedding, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")
    parser.add_argument("--z", default=1.0, type=float,
            help="z")
    parser.add_argument("--p", default=1.0, type=float,
            help="single node bias")
    parser.add_argument("--q", default=1.0, type=float,
            help="far node bias")

    # parser.add_argument('--large', dest="large", action="store_true",
    #         help="using netmf for large window size")
    # parser.add_argument('--small', dest="large", action="store_false",
    #         help="using netmf for small window size")
    # parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp

    znetmf(args)

