#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 10:48
# @Author  : Yuchen Sun
# @FileName: main.py
# @Project: PyCharm

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import random
import json
import link_prediction as lp
from preprocess import *

NUM_REPEAT = 1
RANDOM_SEED = 0
FRAC_EDGE_HIDDEN = 0.5

# data_name = 'CA-AstroPh'
data_name = 'facebook'
# short_name = 'ca'
short_name = 'fb'
pre_dir = 'processed/'+data_name+'.pkl'
# pre_dir = 'processed/CA-AstroPh.pkl'
if not os.path.exists(pre_dir):
    data_dir = 'dataset/'+data_name+'.txt'
    if data_name == 'github':
        data_dir = 'dataset/'+data_name+'.csv'
    # data_dir = 'dataset/CA-AstroPh.txt'
    # network = load_network(data_dir, pre_dir)
    network = load_max_component(data_dir)
    with open(pre_dir, 'wb') as pf:
        pickle.dump(network, pf)
else:
    with open(pre_dir, 'rb') as f:
        network = pickle.load(f)

print('network order:', network.order())
print('network size:', network.size())

for i in range(NUM_REPEAT):

    past_results = os.listdir('./results/')
    txt_past_results = os.listdir('./results/txt/')
    experiment_num = 0
    experiment_file = short_name+'_{}_{}_results.pkl'.format(experiment_num, 'right')
    while (experiment_file in past_results):
        experiment_num += 1
        experiment_file = short_name+'_{}_{}_results.pkl'.format(experiment_num, 'right')

    txt_experiment_num = 0
    txt_experiment_file = short_name+'_{}_{}_results.txt'.format(experiment_num, 'right')
    while (txt_experiment_file in txt_past_results):
        txt_experiment_num += 1
        txt_experiment_file = short_name+'_{}_{}_results.txt'.format(experiment_num, 'right')


    # experiment_file = 'ca_results.pkl'
    RESULTS_DIR = 'results/'+experiment_file
    TXT_RESULTS_DIR = './results/txt/' + txt_experiment_file
    TRAIN_AND_TEST_PATH = 'train_and_test/'
    val_frac = 0.05
    test_frac = FRAC_EDGE_HIDDEN-val_frac

    adj = nx.to_scipy_sparse_matrix(network)

    experiment_name = short_name+'-{}-hidden'.format(FRAC_EDGE_HIDDEN)
    print('Current experiment:', experiment_name)
    train_test_file = TRAIN_AND_TEST_PATH + experiment_name + '.pkl'
    results, paras_str = lp.calculate_scores(adj, test_frac=test_frac, val_frac=val_frac, random_seed=RANDOM_SEED, verbose=2, train_test_file=train_test_file)

    with open(RESULTS_DIR, 'wb') as f:
        pickle.dump(results, f, protocol=2)

    with open(TXT_RESULTS_DIR, 'w') as f:
        json.dump(results, f, indent=4)

    with open('results/paras.txt', 'a') as f:
        f.write(str(experiment_num)+' '+paras_str+'\n')