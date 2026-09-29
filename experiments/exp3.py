# Author: Paula Sarrión Soriano

# ============================================
# EXPERIMENT 3 — PRIVACY–UTILITY TRADE-OFF
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from anjana.anonymity import k_anonymity, l_diversity, t_closeness

# Sklearn for utility measurement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===================================
# 1. Load and preprocess the dataset
# ===================================

cols = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native-country",
    "income"
]

df = pd.read_csv(
    "./adult/adult.data",
    names=cols,
    skipinitialspace=True
)

# Encode income as binary (0/1)
df["income"] = (df["income"] == ">50K").astype(int)

# Define Quasi-Identifiers and Sensitive attribute
QI = ["age", "sex", "education", "native-country", "marital-status"]
SENSITIVE = "occupation"

print("Loaded Dataset:", df.shape)


# ============================
# 2. Syntactic Anonymization 
# ============================

# Load hierarchies for Anjana
# These hierarchies define how each attribute can be progressively generalized during anonymization
hierarchies = {
    "age": dict(pd.read_csv("./adult/hierarchies/age.csv", header=None)),
    "education": dict(pd.read_csv("./adult/hierarchies/education.csv", header=None)),
    "marital-status": dict(pd.read_csv("./adult/hierarchies/marital.csv", header=None)),
    "occupation": dict(pd.read_csv("./adult/hierarchies/occupation.csv", header=None)),
    "sex": dict(pd.read_csv("./adult/hierarchies/sex.csv", header=None)),
    "native-country": dict(pd.read_csv("./adult/hierarchies/country.csv", header=None)),
}

SUPP_LEVEL = 50

# ================================================
# 2.1. Utility function (classification accuracy)
# ================================================

def utility_accuracy(data):
    """
    Defines the utility metric used for syntactic anonymization
    
    :param data: Database to evaluate
    :return: Test accuracy as a measure of data utility
    """
    X = data[QI]            # predictors
    y = data[SENSITIVE]     # target variable

    X_enc = OneHotEncoder(handle_unknown="ignore").fit_transform(X) # encodes categorical variables to make them usable by the classifier
    # Splits the data into training (70%) and test (30%) sets
    Xtr, Xte, ytr, yte = train_test_split(
        X_enc, y, test_size=0.3, random_state=42
    )

    # Trains a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))

# Computes accuracy on the original, non-anonymized dataset
baseline_accuracy = utility_accuracy(df)
print(f"Baseline accuracy (no privacy): {baseline_accuracy:.3f}")   # This serves as the maximum achievable utility


# ======================
# 2.2. Run experiments
# ======================

# k-anonymity 
k_values = [5, 7, 9, 10, 12,15, 17, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50]
k_acc = []

for k in k_values:
    df_k = k_anonymity(df, [], QI, k, SUPP_LEVEL, hierarchies)
    acc = utility_accuracy(df_k)    # Measures how anonymization affects predictive accuracy
    k_acc.append(acc)   # Stores the result
    print(f"k={k}: accuracy={acc:.3f}")

# l-diversity 
l_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
l_acc = []

for l in l_values:
    df_l = l_diversity(df, [], QI, SENSITIVE, 5, l, SUPP_LEVEL, hierarchies)
    acc = utility_accuracy(df_l)
    l_acc.append(acc)
    print(f"l={l}: accuracy={acc:.3f}")

# t-closeness
t_values = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
t_acc = []

for t in t_values:
    df_t = t_closeness(df, [], QI, SENSITIVE, 5, t, SUPP_LEVEL, hierarchies)
    acc = utility_accuracy(df_t)
    t_acc.append(acc)
    print(f"t={t}: accuracy={acc:.3f}")



# ==========================
# 3. Differential Privacy
# ==========================

def dp_query_error(data, mechanism, eps, delta=1e-5):
    """
    Defines a DP utility metric based on query error, not classification
    DP utility = mean absolute error on occupation counts per education

    :param data: Database to evaluate
    :param mechanism: "laplace" or "gaussian"
    :param eps: epsilon value
    :param delta: delta value for Gaussian mechanism
    :return: Mean absolute error between true and noisy counts, used as DP utility loss
    """
    # True counts of occupation per education level
    true_counts = (
        data.groupby("education")[SENSITIVE]
        .value_counts()
        .unstack(fill_value=0)
    )

    if mechanism == "laplace":  # Adds Laplace noise according to ε
        noisy = true_counts + np.random.laplace(0, 1/eps, size=true_counts.shape)
    else:   # Adds Gaussian noise according to (ε, δ)
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
        noisy = true_counts + np.random.normal(0, sigma, size=true_counts.shape)

    return np.mean(np.abs(true_counts - noisy))

eps_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005]
lap_errors = []
gau_errors = []

# Stores utility loss for each mechanism and epsilon value
for eps in eps_values:
    lap_errors.append(dp_query_error(df, "laplace", eps))
    gau_errors.append(dp_query_error(df, "gaussian", eps))
    print(f"ε={eps}: Laplace={lap_errors[-1]:.3f}, Gaussian={gau_errors[-1]:.3f}")


# ============================================
# 4. Unified normalized utility comparison
# ============================================

def rel_utility(acc):
    """
    Normalizes syntactic utility relative to the baseline accuracy

    :param acc: Accuracy value to normalize
    :return: Relative utility (normalized accuracy)
    """
    return acc / baseline_accuracy

# Converts DP error into a utility score in [0,1]
def dp_utility(err, max_err):
    return 1 - (err / max_err)

# Defines three comparable privacy levels
labels = ["Low", "Medium", "High"]


# Selects representative utilities
# k-anonymity
k_rep = [5, 15, 30]
k_idx = [k_values.index(k) for k in k_rep]
k_util = [rel_utility(k_acc[i]) for i in k_idx]

# l-diversity
l_rep = [2, 6, 10]
l_idx = [l_values.index(l) for l in l_rep]
l_util = [rel_utility(l_acc[i]) for i in l_idx]

# t-closeness (note: decreasing t = stronger privacy)
t_rep = [0.5, 0.2, 0.05]
t_idx = [list(t_values).index(t) for t in t_rep]
t_util = [rel_utility(t_acc[i]) for i in t_idx]

# DP (Laplace + Gaussian)
eps_rep = [1.0, 0.2, 0.05]
eps_idx = [list(eps_values).index(e) for e in eps_rep]

max_dp_err = max(max(lap_errors), max(gau_errors))

dp_lap_util = [
    dp_utility(lap_errors[i], max_dp_err) for i in eps_idx
]

dp_gau_util = [
    dp_utility(gau_errors[i], max_dp_err) for i in eps_idx
]



# ==========
# 5. Plots
# ==========

# Syntactic Anonymization
fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

axs[0].plot(k_values, k_acc, marker="o", color='C3', linewidth=3)
axs[0].axhline(baseline_accuracy, linestyle="--", linewidth=2.5, color='C7')
#axs[0].set_title(r"(a)", fontsize=18)
axs[0].set_xlabel(r"$k$", fontsize=18)
axs[0].set_ylabel("Accuracy", fontsize=18)
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].grid(alpha=0.3)

axs[1].plot(l_values, l_acc, marker="s", color='C4', linewidth=3)
axs[1].axhline(baseline_accuracy, linestyle="--", linewidth=2.5, color='C7')
#axs[1].set_title(r"(b)", fontsize=18)
axs[1].set_xlabel(r"$\ell$", fontsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=14)
axs[1].grid(alpha=0.3)

axs[2].plot(t_values, t_acc, marker="^", color='C2', linewidth=3)
axs[2].axhline(baseline_accuracy, linestyle="--", linewidth=2.5, color='C7')
axs[2].invert_xaxis()       # smaller t = stronger privacy
#axs[2].set_title(r"(c)", fontsize=18)
axs[2].set_xlabel(r"$t$", fontsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=14)
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Differential Privacy
plt.figure(figsize=(8, 5))
plt.plot(
    eps_values,
    lap_errors,
    linestyle="-",
    linewidth=3,
    label="Laplace mechanism",
    color='C0'
)
plt.plot(
    eps_values,
    gau_errors,
    linestyle="-",
    linewidth=3,
    label="Gaussian mechanism",
    color='C1'
)

plt.axhline(0, linestyle="--", linewidth=2.5, label="True answer (no noise)", color='C7')

plt.xlabel("ε", fontsize=18)
plt.ylabel("Mean Absolute Error (utility loss)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=16)
plt.show()



# Unified comparison
x = np.arange(len(labels))
width = 0.15

plt.figure(figsize=(11, 6))

plt.bar(x - 2*width, k_util, width, label="k-anonymity", color='C3')
plt.bar(x - width, l_util, width, label=r"$\ell$-diversity", color='C4')
plt.bar(x, t_util, width, label="t-closeness", color='C2')
plt.bar(x + width, dp_lap_util, width, label="DP (Laplace)", color='C0')
plt.bar(x + 2*width, dp_gau_util, width, label="DP (Gaussian)", color='C1')
plt.axhline(1.0, linestyle="--", linewidth=2.5, label="No-privacy baseline", color='C7')

plt.xticks(x, labels, fontsize=14)
plt.ylabel("Relative Utility (normalized)", fontsize=18)
plt.xlabel("Privacy Level", fontsize=18)
plt.tick_params(axis='y', which='major', labelsize=14)
plt.grid(axis="y", alpha=0.3)
plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



print("\nDone!")