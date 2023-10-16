"""Plasmid clustering by using leidenalg with CPM.
  python-3.10.4
  leidenalg-0.9.1
  igraph-0.10.4
  pandas-2.0.1
  networkx-3.1
"""

import leidenalg as la
import igraph as ig
import pandas as pd
import networkx as nx

# import plasmid pairs with transformed edgeweights from file "fastANI_edgeweights". The edgeweights are deposited to DRYAD, please see README for the link.
df = pd.read_table('fastANI_edgeweights', header=None)

# converts to igraph object
df.columns = ['seq1', 'seq2', 'edge']
G_nx = nx.from_pandas_edgelist(df, 'seq1', 'seq2', edge_attr=["edge"])
G = ig.Graph.from_networkx(G_nx)
print("graph is created, now profiling...")

# resolution scanning with CPM
optimiser = la.Optimiser()
profile = optimiser.resolution_profile(G, la.CPMVertexPartition, resolution_range=(0, 1), weights="edge")
print("profiling has been done.")

# find the best partition (maximum modularity)
para = {} # store the resolution_parameter as key, modularity as value
for i in range(len(profile)):
  para[profile[i].resolution_parameter] = profile[i].modularity

# get the profile index
max_para = max(para, key=para.get)
max_index = list(para.keys()).index(max_para)

# saving results to tab delimited file "profile_parameter.txt"
print("saving results...")
para_output = open('profile_parameter.txt', 'w')

for gamma, Q in para.items():
  para_output.write(str(gamma) + "\t" + str(Q) + "\n")

para_output.close()

# add partitions into the graph
G.vs['cluster'] = profile[max_index].membership

# save the graph to gml file "plasmid_clusters.gml"
ig.save(G,'plasmid_clusters.gml')

# save the cluster information to "clusters_output.txt"
clus = {}
for i in range(len(G.vs)):
       clus[G.vs[i]['_nx_name']] = G.vs[i]['cluster']

cluster_output = open('clusters_output.txt', 'w')

for na, cl in clus.items():
  cluster_output.write(str(na) + "\t" + str(cl) + "\n")

cluster_output.close()
# get the final resolution and modularity
print("modularity= " + str(profile[max_index].modularity))
print("resolution= " + str(profile[max_index].resolution_parameter))
