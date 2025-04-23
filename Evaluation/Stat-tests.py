import pandas as pd

from scipy.stats import wilcoxon

# swap out SHD results and re run for each sample size
data = {
    "BIC": [3, 4, 1, 3, 3],
    "EBIC_0.25": [3, 3, 3, 4, 2],
    "EBIC_0.5":[4, 2, 2, 3, 4],
    "EBIC_0.75": [1, 4, 4, 3, 3],
}

df = pd.DataFrame(data)


#Wilcoxon
comparisons = ["EBIC_0.25", "EBIC_0.5", "EBIC_0.75"]

for score in comparisons:
    stat, p = wilcoxon(df["BIC"], df[score])
    print(f"Wilcoxon Test - BIC vs {score}:")
    print(f"  Test statistic = {stat:.4f}, p-value = {p:.4f}\n")


#GOF
from scipy.stats import chisquare

observed_1000 = [8, 2, 5]      #
observed_5000 = [9, 4, 2]

observed_10000 = [5, 6, 4]
observed_50000 = [4, 5, 6]


expected = [5, 5, 5]

def run_gof_test(observed, expected, label):
    chi2, p = chisquare(f_obs=observed, f_exp=expected)
    print(f"\nChi-Squared Goodness-of-Fit Test ({label})")
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")
    print(f"Chi2 = {chi2:.3f}, p-value = {p:.10f}")
run_gof_test(observed_1000, expected, "n = 1000")
run_gof_test(observed_5000, expected, "n = 5000")
run_gof_test(observed_10000, expected, "n = 10000")
run_gof_test(observed_50000, expected, "n = 50000")

#Creating graphs from the data
import matplotlib.pyplot as plt
import pandas as pd

#SHD Results
data = {
    'Sample Size': [1000]*4 + [5000]*4 + [10000]*4,
    'Gamma': [0, 0.25, 0.5, 0.75]*3,
    'SHD': [
        13.6, 13.0, 13.2, 12.4,
        6.0, 6.8, 6.6, 6.8,
        6.2, 6.4, 6.2, 6.6
    ]
}

df = pd.DataFrame(data)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
sample_sizes = [1000, 5000, 10000]

for i, n in enumerate(sample_sizes):
    subset = df[df['Sample Size'] == n]
    axes[i].plot(subset['Gamma'], subset['SHD'], marker='o', linestyle='-')
    axes[i].set_title(f'Sample Size: {n}')
    axes[i].set_xlabel('Gamma')
    axes[i].set_ylim(0, 15)
    if i == 0:
        axes[i].set_ylabel('SHD')

plt.suptitle('SHD of EBIC for varying values of Gammas')
plt.show()

# Wlicoxon
pval_data = {
    'Sample Size': [1000, 5000, 10000],
    '0.25': [0.1936, 0.3573, 1.0],
    '0.5': [0.3125, 0.1441, 0.6547],
    '0.75': [0.1936, 0.1025, 0.7055]
}

pval_df = pd.DataFrame(pval_data)
plt.figure(figsize=(8, 5))
for gamma in ['0.25', '0.5', '0.75']:
    plt.plot(pval_df['Sample Size'], pval_df[gamma], marker='o', label=f'Gamma = {gamma}')

plt.title('Wilcoxon p-value vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('p-value')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()

