from pygobnilp.gobnilp import Gobnilp
try:
    from scoring import (
        DiscreteData, ContinuousData,
        BDeu, BGe,
        DiscreteLL, DiscreteBIC, DiscreteEBIC, DiscreteAIC,
        GaussianLL, GaussianBIC,GaussianEBIC, GaussianAIC, GaussianL0)
except ImportError as e:
    print("Could not import score generating code!")
    print(e)
m = Gobnilp()

m.learn('data/discrete.dat',score='DiscreteEBIC', end='local scores', palim = 6, pruning = False)
for k, v in m.local_scores.items():
    print(k)
    print(v)
    print()


test =  DiscreteBIC('discrete.dat', 1).score
print(test)