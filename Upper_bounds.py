from pygobnilp.gobnilp import Gobnilp
try:
    from scoring import DiscreteEBIC
except ImportError as e:
    print("Could not import score generating code!")
    print(e)
m = Gobnilp()

m.learn('data/discrete.dat',score='DiscreteBIC', end='local scores', palim = 6, pruning = False)
for k, v in m.local_scores.items():
    print(k)
    print(v)
    print()

for node, score_dict in m.local_scores.items():
    print(f" Node: {node}")

    for parent_set, score in score_dict.items():
        # Proper supersets: all sets that are strict supersets of this parent set
        proper_supersets = [ps for ps in score_dict if parent_set < ps]

        if not proper_supersets:
            continue  # skip if there are no supersets (e.g. full set)

        # Find the best (max) score among supersets
        best_superset = max(proper_supersets, key=lambda ps: score_dict[ps])
        best_superset_score = score_dict[best_superset]

        print(f"Parent set: {set(parent_set)} | Score: {score:.2f}")
        print(f" Best superset: {set(best_superset)} | Superset score: {best_superset_score:.2f}")
        print()

test =  DiscreteEBIC('discrete.dat', 1).score
print(test)
