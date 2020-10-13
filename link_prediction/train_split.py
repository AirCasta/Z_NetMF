#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 12:29
# @Author  : Yuchen Sun
# @FileName: train_split.py
# @Project: PyCharm
import numpy as np
import scipy.sparse as sp
import networkx as nx

# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    if verbose == True:
        print('preprocessing')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], 0), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)
    print("orig:",orig_num_cc)

    # upper triangle of adj
    adj_triu =sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0] # (coords, values, shape), edges only 1 way

    num_test = int(np.floor(edges.shape[0]*test_frac))
    num_val = int(np.floor(edges.shape[0]*val_frac))
    num_other = num_test+num_val

    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)
    test_edges = set()
    val_edges = set()
    other_edges = set()

    if verbose == True:
        print('generating test sets...')
    np.random.shuffle(edge_tuples)

    # random sample edges
    i = 0
    for edge in edge_tuples:
        if i < num_other:
            g.remove_edge(edge[0], edge[1])
            train_edges.remove(edge)
            other_edges.add(edge)
            i += 1
        else:
            break

    # union-find sets, every cp means a set, combine until orig_num_cc
    # assign nodes
    node_label = {}
    cps = {}
    i = 0
    for cp in nx.connected_components(g):
        cps[i] = []
        for node in g.subgraph(cp).nodes():
            cps[i].append(node)
            node_label[node] = i
        i += 1

    # record removed edge from other edges
    rm_edge = []
    j = 0
    for edge in other_edges:
        # if i==1:
        #     break
        u, v = edge
        if node_label[u] == node_label[v]:
            if j < num_test:
                test_edges.add(edge)
            else:
                val_edges.add(edge)
            j += 1
            continue
        g.add_edge(u, v)
        train_edges.add(edge)
        rm_edge.append((u, v))
        if len(cps[node_label[u]]) < len(cps[node_label[v]]):
            u, v = v, u
        for node in cps[node_label[v]]:
            node_label[node] = node_label[u]
            cps[node_label[u]].append(node)
        i -= 1

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('generating false test sets...')

    test_edges_false = set()

    while len(test_edges_false) < len(test_edges):
        s = np.random.randint(0, adj.shape[0])
        e = np.random.randint(0, adj.shape[0])
        if s == e:
            continue

        f_edge = (min(s,e), max(s,e))
        if f_edge in all_edge_tuples:
            continue
        if f_edge in test_edges_false:
            continue
        test_edges_false.add(f_edge)

    if verbose == True:
        print('generating false val sets...')

    val_edges_false = set()

    while len(val_edges_false) < len(val_edges):
        s = np.random.randint(0, adj.shape[0])
        e = np.random.randint(0, adj.shape[0])
        if s == e:
            continue

        f_edge = (min(s, e), max(s, e))
        if f_edge in all_edge_tuples:
            continue
        if f_edge in test_edges_false:
            continue
        if f_edge in val_edges_false:
            continue

        val_edges_false.add(f_edge)

    if verbose == True:
        print('generating false train sets...')

    train_edges_false = set()

    while len(train_edges_false) < len(train_edges):
        s = np.random.randint(0, adj.shape[0])
        e = np.random.randint(0, adj.shape[0])
        if s == e:
            continue

        f_edge = (min(s, e), max(s, e))
        if f_edge in all_edge_tuples:
            continue
        if f_edge in test_edges_false:
            continue
        if f_edge in val_edges_false:
            continue
        if f_edge in train_edges_false:
            continue

        train_edges_false.add(f_edge)

    if verbose:
        print('final checks for disjointness...')

        # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose:
        print('generating adj_train...')

    adj_train = nx.adjacency_matrix(g)
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
