#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 15:17
# @Author  : Yuchen Sun
# @FileName: test.py
# @Project: PyCharm

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
# a = (1,2)
# b, c = a[0], a[1]
# print(b,c)
def draw(l, size, cm, name):
    fig = plt.figure(figsize=size)
    nx.draw(BA, pos, with_labels=False, node_size=100, node_color=l + 0.5, cmap=cm)
    plt.show()
    fig.savefig(name, transparent=True)

BA = nx.random_graphs.barabasi_albert_graph(16, 1)
A = np.array(nx.adjacency_matrix(BA).todense())
a = np.sum(A, axis=0)
m = np.argmax(a)
l = np.zeros(a.shape)
l[m] = 1
D_inv = np.diag(1/a)
l1 = (l.dot(D_inv).dot(A) + l)/2
l2 = (l1.dot(D_inv).dot(A))/2
print(l)
print(l1)
print(l2)
pos = nx.spring_layout(BA)  # 图形的布局样式，这里是中心放射状
startcolor = '#cde6c7'
endcolor = '#1d953f'
cmap2 = colors.LinearSegmentedColormap.from_list('own2',[startcolor,endcolor])
# cm = plt.cm.YlOrRd
size = (6,1.5)
cm = cmap2

# fig = plt.figure(figsize=size)
# nx.draw(BA, pos, with_labels=False, node_size=100, node_color=l+0.5, cmap=cm)
# plt.show()
# plt.figure(figsize=size)
# nx.draw(BA, pos, with_labels=False, node_size=100, node_color=l1+0.5, cmap=cm)
# plt.show()
# plt.figure(figsize=size)
# nx.draw(BA, pos, with_labels=False, node_size=100, node_color=l2+0.5, cmap=cm)
# plt.show()

draw(l,size,cm,'l.png')
draw(l1,size,cm,'l1.png')
draw(l2,size,cm,'l2.png')