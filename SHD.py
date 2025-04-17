import pandas as pd
from pygobnilp.gobnilp import Gobnilp
from pgmpy.readwrite import BIFReader
import networkx as nx
import causaldag as cd



# Load true network structure
reader = BIFReader("Evaluation/alarm.bif")
pgmpy_model = reader.get_model()
true_graph = nx.DiGraph()
true_graph.add_nodes_from(pgmpy_model.nodes())
true_graph.add_edges_from(pgmpy_model.edges())

#Convert tp causal
true = cd.DAG.from_nx(true_graph)
sample_sizes = 5000
gammas = [0, 0.25, 0.5, 0.75]  # gamma=0 is BIC
results = []
print(type(true))  # Should say causaldag.DAG


for i in range(1, 6):
    datapath = f"Evaluation/alarm{i}_n5000"
    for gamma in gammas:
        g = Gobnilp()
        g.learn(data_source=datapath, score='DiscreteEBIC', gamma=gamma)
        learned_bn = g.learned_bn
        learned = cd.DAG.from_nx(learned_bn)
        shd = true.shd(learned)

        results.append({
            "Sample": i,
            "Size": 5000,
            "Gamma": gamma,
            "SHD": shd
        })


# Create a DataFrame for display and export
df = pd.DataFrame(results)

# Pretty print in terminal
print(df.pivot_table(index=["Size", "Sample"], columns="Gamma", values="SHD").to_markdown())

