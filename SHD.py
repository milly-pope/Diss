import pandas as pd
from pygobnilp.gobnilp import Gobnilp
from pgmpy.readwrite import BIFReader
import networkx as nx



def compute_shd(g1, g2):
    edges_g1 = set(g1.edges())
    edges_g2 = set(g2.edges())
    return len(edges_g1 - edges_g2) + len(edges_g2 - edges_g1)

# Load true network structure
reader = BIFReader("Evaluation/alarm.bif")
pgmpy_model = reader.get_model()
true_graph = nx.DiGraph()
true_graph.add_nodes_from(pgmpy_model.nodes())
true_graph.add_edges_from(pgmpy_model.edges())

sample_sizes = 50000
gammas = [0, 0.25, 0.5, 0.75]  # gamma=0 is BIC
results = []


for i in range(1, 6):
    datapath = f"Evaluation/alarm{i}_n50000"
    for gamma in gammas:
        g = Gobnilp()
        g.learn(data_source=datapath, score='DiscreteEBIC', gamma=gamma)
        learned_bn = g.learned_bn
        shd = compute_shd(learned_bn, true_graph)

        results.append({
            "Sample": i,
            "Size": 50000,
            "Gamma": gamma,
            "SHD": shd
        })


# Create a DataFrame for display and export
df = pd.DataFrame(results)

# Pretty print in terminal
print(df.pivot_table(index=["Size", "Sample"], columns="Gamma", values="SHD").to_markdown())

