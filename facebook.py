
#TODO: Extract features of graph and random forest classifier.
#Can do more complicated techniques with the random forest


import math
import random
import numpy as np
import pandas as pd
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import warnings
warnings.simplefilter("ignore")


#read edge file and get edges/nodes
def readfile(filename):
	f =  open(filename, "r")
	lines = f.readlines()
	nodes = []
	edges = []
	for line in lines:
		edge = line.split()
		edge_int = [int(x) for x in edge]
		edges.append(edge_int)
		if edge_int[0] not in nodes:
			nodes.append(edge_int[0])
		if edge_int[1] not in nodes:
			nodes.append(edge_int[1])
	return (edges, nodes)


def populate_subset(nodes, number):
	subset = []
	newnodes = nodes
	#choosing nodes to be in the subset
	while(len(subset) < number):
		chosen_num = random.choice(nodes)
		newnodes.remove(chosen_num)
		subset.append(chosen_num)
	return (subset, newnodes)

def partition_nodes(nodes, overlap_number, san_number, aux_number):
	overlap = populate_subset(nodes, overlap_number)
	san_nodes = populate_subset(overlap[1], san_number)
	aux_nodes = populate_subset(san_nodes[1], aux_number)
	overlap_set = set(overlap[0])
	san_set = set(san_nodes[0])
	aux_set = set(aux_nodes[0])
	return (overlap_set, san_set, aux_set)


def induced_subraph(edges, node_set):
	#finds the induced subgraph of a set of vertices
	induced = []
	for edge in edges:
		if(edge[0] in node_set and edge[1] in node_set):
			induced.append(edge)
	return induced

def facebook():
	alpha = 0.25 #overlap proportion
	ego_node = 0 #ego node: connected to all others (ignored for now)
	parsed = readfile("facebook/1912edges.txt")
	edges = parsed[0]
	nodes = parsed[1]
	num_nodes = len(nodes)
	overlap_number = math.floor(alpha * num_nodes)
	#calculate number of nodes in sanitized and auxiliary graph
	san_number = math.floor((num_nodes-overlap_number)/2)
	aux_number = num_nodes - overlap_number - san_number
	partitioned = partition_nodes(nodes, overlap_number, san_number, aux_number)
	#making the overlapping sanitized and auxiliary graphs
	san = partitioned[1].union(partitioned[0])
	aux = partitioned[2].union(partitioned[0])
	san_graph = induced_subraph(edges, san)
	aux_graph = induced_subraph(edges, aux)
	random_forest(san_graph, aux_graph, san, aux)

def parse_feat(features_file, nodes, randomize_feat=0):
	features_file = open(features_file, "r")
	feat_strings = features_file.readlines()
	feat = []
	#make a data frame with the features given
	for line in feat_strings:
		node_feats_str = line.split()
		node_feats = [int(x) for x in node_feats_str]
		if(node_feats[0] in nodes):
			num_randomized = random.randint(min(randomize_feat, 1), randomize_feat)
			#choose some features to randomize
			for i in range(1, num_randomized):
				random_pos = random.randrange(1, len(node_feats))
				node_feats[i] = random.randint(0,1)
			feat.append(node_feats)
	cols = list(range(len(feat[0])))
	new_df = pd.DataFrame(columns=cols, data=feat)
	return new_df


def make_concat_df(aux_features, san_features):
	new_rows = []
	same_rows = []
	diff_rows = []
	aux_list = aux_features.values.tolist()
	san_list = san_features.values.tolist()
	same_count = 0
	#looping through auxiliary and sanitized and concatenating each feature vector
	#to every other 
	for row1 in aux_list:
		for row2 in san_list:
			same = 0
			new_row = list(row1[1:]) + (list(row2[1:]))
			#checking if the nodes are the same
			if(row1[0] == row2[0]):
				same_count+=1
				same = 1
				new_row.append(same)
				same_rows.append(new_row)
			else:
				new_row.append(same)
				diff_rows.append(new_row)
	new_rows = new_rows + same_rows
	#sampling some different pairs 
	diff_sample = random.sample(diff_rows, len(same_rows))
	new_rows = new_rows + diff_sample
	cols = list(range(len(new_rows[0])-1))
	cols.append('same')
	new_df = pd.DataFrame(columns=cols, data=new_rows)
	print(new_df.shape)
	print(same_count)
	return new_df


def random_forest(san_graph, aux_graph, san, aux):
	aux_features = parse_feat("facebook/feat/1912feat.txt", aux)
	san_features = parse_feat("facebook/feat/1912feat.txt", san)
	print(san_features.head())
	print(aux_features.head())
	combined = make_concat_df(aux_features, san_features)
	print(combined.head())
	y = np.array(combined['same'])
	X = combined.drop('same', axis = 1)
	feat_list = list(X.columns)
	X = np.array(X)
	#using scikit learn with random forests
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf=RandomForestClassifier(n_estimators=100)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	tree = clf.estimators_[5]
	#making output file of sample decision tree
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feat_list)# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
	graph.write_png('tree.png')

			



if __name__ == '__main__':
	facebook()