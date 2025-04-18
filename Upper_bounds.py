from pygobnilp.scoring import DiscreteEBIC, DiscreteBIC, DiscreteLL, DiscreteAIC,GaussianLL, GaussianBIC, GaussianEBIC,GaussianAIC
from pygobnilp.gobnilp import Gobnilp
from pygobnilp.scoring import DiscreteData, ContinuousData
import pandas as pd

data_disc = DiscreteData('data/discrete.dat')
data_cont = ContinuousData('data/gaussian.dat')
score_chosen = 'DiscreteAIC'
# POSSIBLE SCORES FOR ANALYSIS
scores = {
    'DiscreteEBIC': DiscreteEBIC(data_disc),
    'DiscreteAIC': DiscreteAIC(data_disc),
    'DiscreteBIC': DiscreteBIC(data_disc),
    'DiscreteLL' : DiscreteLL(data_disc),
    'GaussianEBIC' : GaussianEBIC(data_cont),
    'GaussianLL': GaussianLL(data_cont),
    'GaussianBIC': GaussianBIC(data_cont),
    'GaussianAIC': GaussianAIC(data_cont)
}

skore = scores[score_chosen]

results = []
pruned = []
#For bedugging if needed
invalid_ubs = []


# Load the local scores using GOBNILP
m = Gobnilp()
m.learn(data_source='data/discrete.dat', data_type='discrete', score=score_chosen , end='local scores', palim=6, pruning=False)
# Load the pruned and unpruned score dictionaries
all_scores = m.return_local_scores(skore.score, palim=6, pruning=False)
kept_scores = m.return_local_scores(skore.score, palim=6, pruning=True)
# Loop through all nodes
for node, score_dict in m.local_scores.items():

    for parent_set in score_dict:
        proper_supersets = [ps for ps in score_dict if parent_set < ps]
        # Get upper bound by recomputing score using your scoring class
        score, ub = skore.score(node, tuple(parent_set))

        for superset in proper_supersets:
            superset_score = score_dict[superset]
            valid = superset_score <= ub

            results.append({
                "Node": node,
                "Parent Set": str(set(parent_set)),
                "Upper Bound": round(ub, 2),
                "Superset": str(set(superset)),
                "Superset Score": round(superset_score, 2),
                "Valid Bound?": valid
            })
            #Debugging
            '''
            if not valid:
                invalid_ubs.append({
                    "Node": node,
                    "Parent Set": set(parent_set),
                    "Superset": set(superset),
                    "Upper Bound": ub,
                    "Superset Score": superset_score,
                    "Bound Gap": superset_score - ub,
                    "Parent Score": score,  # current set's own score
                    "Score Diff (superset-parent)": superset_score - score
                })
            '''
    # Identify pruned sets: all - kept
    pruned_sets = set(all_scores[node].keys()) - set(kept_scores[node].keys())

    for pruned_set in pruned_sets:
        pruned_tuple = tuple(pruned_set)
        score, ub = skore.score(node, pruned_tuple)
        pruned.append({
            "Node": node,
            "Pruned Set": str(set(pruned_set)),
            "Score": round(score, 2),
            "Upper Bound": round(ub, 2)
        })

df_results = pd.DataFrame(results)
df_pruned = pd.DataFrame(pruned)
df_invalid_ubs = pd.DataFrame(invalid_ubs)
#Evaluation

total_comparisons = len(results)
invalid_bounds = sum(not r["Valid Bound?"] for r in results)
total_prunes = len(pruned)

print(f"\nTotal comparisons made: {total_comparisons}")
print(f"Invalid upper bounds found: {invalid_bounds}")

#print(df_invalid_ubs.to_string(index=False))
#Reduction in search space

total_unpruned = sum(len(v) for v in kept_scores.values())  # From earlier
total_possible = total_prunes + total_unpruned

prune_percent = round((total_prunes / total_possible) * 100, 2)
print(f"\nPruning Summary: Number of Prunes {total_prunes}")
print(f"Total possible sets: {total_possible}")
print(f"Percentage pruned: {prune_percent}%")

#Tightness of the upper bound
gap_values = [r["Upper Bound"] - r["Superset Score"] for r in results]
avg_gap = sum(gap_values) / len(gap_values)
print(f"Average upper bound gap: {avg_gap:.2f}")

