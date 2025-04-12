from pygobnilp.gobnilp import Gobnilp
from pgmpy.readwrite import BIFReader
import networkx as nx


# 1. Simulate and save data from simulation.py
# 2. Create a GOBNILP object and run learning
g = Gobnilp()
g.learn('data/discrete.dat',score='DiscreteBIC')
print("Hello")
G = g.learned_bn #Calling the function from GOBNILP that extracts the learned BN as a networksx object
print(type(G))  # Should be <class 'networkx.DiGraph'>

#Load in our true network and create a networkx DiGraph from it
reader = BIFReader('//alarm.bif')
pgmpy_model = reader.get_model()
true_graph = nx.DiGraph()
true_graph.add_nodes_from(pgmpy_model.nodes())
true_graph.add_edges_from(pgmpy_model.edges())

def compute_shd(g1, g2):
    edges_g1 = set(g1.edges())
    edges_g2 = set(g2.edges())
    return len(edges_g1 - edges_g2) + len(edges_g2 - edges_g1)



shd = compute_shd(G, true_graph)
print(f"SHD = {shd}")


