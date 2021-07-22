#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 10:48
# @Author  : Yuchen Sun
# @FileName: main.py
# @Project: PyCharm
import math

import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import numpy as np
from link_prediction.prediction import calculate_scores
from link_prediction.preprocess import load_max_component

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

NUM_REPEAT = 1
RANDOM_SEED = 0
FRAC_EDGE_HIDDEN = 0.5

# data_name = 'github'
data_name = 'CA-AstroPh'
# short_name = 'github'
short_name = 'ca'
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
    results, paras_str, df, val_roc_min, val_roc_max = calculate_scores(adj, test_frac=test_frac, val_frac=val_frac, random_seed=RANDOM_SEED, verbose=2, train_test_file=train_test_file)

    sns.set_theme(style="darkgrid")
    plt.figure()

    # This is where the actual plot gets made
    ax = sns.barplot(data=df, x="paras", y="val_roc", hue="mode", palette=['#0485d1', 'red', '#3f9b0b'],
                     saturation=0.6)

    # Customise some display properties
    ax.set_title('Z-NetMF-gt')
    ax.grid(color='#cccccc')
    ax.set_ylabel('Validation ROC Score')
    ax.set_xlabel('P,Q,Z Value')
    ax.set_xticklabels(df["paras"].unique().astype(str), rotation=45)
    # Facebook
    # ax.set_ylim([0.9, 1])
    # ax.yaxis.set_ticks(np.arange(0.9, 1, 0.005))
    # CA
    min_y = round_decimals_down(val_roc_min)
    max_y = round_decimals_up(val_roc_max)
    ax.set_ylim([min_y, max_y])
    ax.yaxis.set_ticks(np.arange(min_y, max_y, 0.005))
    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # # Turn on the minor TICKS, which are required for the minor GRID
    # ax.minorticks_on()
    # # Customize the major grid
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # # Customize the minor grid
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    # Customize the grid
    ax.grid(linestyle=':', linewidth='0.5', color='black')

    # Ask Matplotlib to show it
    plt.show()
    ax.figure.savefig('./figure/z_netmf_gt/output_ca.png')

    with open(RESULTS_DIR, 'wb') as f:
        pickle.dump(results, f, protocol=2)

    with open(TXT_RESULTS_DIR, 'w') as f:
        json.dump(results, f, indent=4)

    with open('results/paras.txt', 'a') as f:
        f.write(str(experiment_num)+' '+paras_str+'\n')