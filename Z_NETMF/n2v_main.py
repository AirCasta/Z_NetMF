import argparse
import os
import numpy as np
import scipy.io
import networkx as nx
import node2vec
from gensim.models import Word2Vec

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='mat/POS.mat',
	                    help='Input graph path')

	parser.add_argument('--matfile-variable-name', default='network',
						help='variable name of adjacency matrix inside a .mat file.')

	parser.add_argument('--output', nargs='?', default='embedding/n2v_pos.npy',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph_el():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def load_adjacency_matrix(file, variable_name="network"):
	data = scipy.io.loadmat(file)
	return data[variable_name]

def read_graph_mat():

	A = load_adjacency_matrix(args.input, args.matfile_variable_name)
	G = nx.from_scipy_sparse_matrix(A)
	# if args.weighted:
	# 	G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	# else:
	# 	G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
	# 	for edge in G.edges():
	# 		G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G


def learn_embeddings(walks):

    # Learn embeddings by optimizing the Skipgram objective using SGD.
    # model_w2v.wv.most_similar()
	# model_w2v.wv.get_vector()
	# model_w2v.wv.syn0  #  model_w2v.wv.vectors 
	# model_w2v.wv.vocab  # 
	# model_w2v.wv.index2word  # 
	out_path = os.path.splitext(args.output)[0]+'_p_'+str(args.p)+'_q_'+str(args.q)+os.path.splitext(args.output)[-1]
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	if os.path.splitext(args.output)[-1] == '.emb':
		model.wv.save_word2vec_format(args.output)
	elif os.path.splitext(args.output)[-1] == '.npy':
		emb = np.ndarray(np.shape(model.wv.syn0))
		for i in model.wv.vocab:
			idx = int(i)
			emb[idx] = model.wv.__getitem__(i)
		np.save(out_path, emb)
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if os.path.splitext(args.input)[-1] == '.edgelist':
		nx_G = read_graph_el()
	elif os.path.splitext(args.input)[-1] == '.mat':
		nx_G = read_graph_mat()
	else:
		return
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
