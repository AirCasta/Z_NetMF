#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 10:53
# @Author  : Yuchen Sun
# @FileName: preprocess.py
# @Project: PyCharm

import networkx as nx
import os
import pickle

def load_network(edge_list_file, pkl_file, is_directed=False):
    network = nx.Graph()
    with open(edge_list_file, 'r') as ef:
        for line in ef:
            if line[0]=='#':
                continue
            s, e = line.split()
            # print(s,e)
            network.add_edge(s, e)
            # if not is_directed:
            #     network.add_edge(e, s)
    with open(pkl_file, 'wb') as pf:
        pickle.dump(network, pf)
    return network

def load_max_component(data_dir):
    graph = nx.Graph
    if os.path.splitext(data_dir)[-1] == '.txt':
        network = nx.read_edgelist(data_dir, create_using=graph, nodetype=int, data=(('weight', int),))
    elif os.path.splitext(data_dir)[-1] == '.csv':
        network = nx.read_edgelist(data_dir, delimiter=',', create_using=graph, nodetype=int, data=(('weight', int),))
    largest_component = max(nx.connected_components(network), key=len)
    largest_component = network.subgraph(largest_component)
    return largest_component
