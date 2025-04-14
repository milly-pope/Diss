import pandas as pd
from scipy.stats import friedmanchisquare

# Structural Hamming Distance results for n = 1000
data = {
    "BIC": [22, 15, 15, 24, 23],
    "EBIC_0.25": [22, 13, 11, 20, 24],
    "EBIC_0.5": [19, 13, 12, 20, 24],
    "EBIC_0.75": [20, 16, 15, 21, 26],
}

df = pd.DataFrame(data)
print(df)
stat, p = friedmanchisquare(
    df["BIC"],
    df["EBIC_0.25"],
    df["EBIC_0.5"],
    df["EBIC_0.75"]
)

print("Friedman test statistic:", stat)
print("p-value:", p)
