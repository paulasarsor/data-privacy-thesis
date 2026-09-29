# Author: Paula Sarrión Soriano

# ====================================
# EXPERIMENT 2 — COMPOSITION ATTACK
# ====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from anjana.anonymity import k_anonymity, l_diversity, t_closeness

np.random.seed(47)

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

# Creates a unique identifier (uid) for each individual.
df = df.reset_index(drop=True)
df["uid"] = df.index
# Essential to track individuals across multiple releases
# and simulate overlapping populations.

print("Loaded Dataset:", df.shape)

# ===============================
# 2. Create overlapping releases
# ===============================

def create_overlapping_subsets(df, overlap_size=5000, subset_size=15000):
    """
    Creates two subsets from the original dataset 
    with partial overlap, simulating independent releases

    :param df: Original dataset
    :param overlap_size: Size of the overlapping portion
    :param subset_size: Size of each subset
    :return: The two subsets and the list of overlapping uids
    """
    # Extracts all unique individual identifiers
    uids = df["uid"].values

    # Randomly selects individuals that will appear in both releases
    overlap = np.random.choice(uids, overlap_size, replace=False)

    # Removes overlapping individuals from the pool
    remaining = np.setdiff1d(uids, overlap)
    # Selects the non-overlapping part of the first release
    s1_extra = np.random.choice(remaining, subset_size - overlap_size, replace=False)
    # Updates the remaining pool
    remaining = np.setdiff1d(remaining, s1_extra)
    # Selects the non-overlapping part of the second release
    s2_extra = np.random.choice(remaining, subset_size - overlap_size, replace=False)

    # Constructs the two subsets
    s1 = df[df["uid"].isin(np.concatenate([overlap, s1_extra]))].copy()
    s2 = df[df["uid"].isin(np.concatenate([overlap, s2_extra]))].copy()

    return s1.reset_index(drop=True), s2.reset_index(drop=True), overlap

# Generates the two releases
subset1, subset2, overlap_ids = create_overlapping_subsets(df)


# ===================================
# 3. Define neighboring datasets
# ===================================

# Randomly select one individual
idx = np.random.randint(len(df))

D = df.copy()
D_prime = df.drop(index=idx)

print(f"Neighboring datasets differ by individual at index {idx}")


# =============================
# 4. Syntactic Anonymization
# =============================

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

def anonymize(df, method, param, supp_level=50, k=10):
    """
    Wrapper function to anonymize a dataset
    using the specified method and parameters.
    
    :param df: Dataset to anonymize
    :param method: Anonymization method ("k", "l", or "t")
    :param param: Value of anonymization parameter (k, l, or t)
    :param supp_level: Suppression level
    :param k: Default k for l-diversity and t-closeness
    :return: Anonymized dataset
    """

    if method == "k":
        return k_anonymity(df, [], QI, param, supp_level, hierarchies )
    elif method == "l":
        return l_diversity(df, [], QI, SENSITIVE, k, param, supp_level, hierarchies)
    elif method == "t":
        return t_closeness(df, [], QI, SENSITIVE, k, param, supp_level, hierarchies)
    else:
        raise ValueError

# Apply anonymization to both releases
print("Running k-anonymity...")
Rk1 = anonymize(subset1, "k", 10)
Rk2 = anonymize(subset2, "k", 10)

print("Running l-diversity...")
Rl1 = anonymize(subset1, "l", 7)
Rl2 = anonymize(subset2, "l", 7)

print("Running t-closeness...")
Rt1 = anonymize(subset1, "t", 0.1)
Rt2 = anonymize(subset2, "t", 0.1)


# =========================================
# 4.1. Intersection of anonymized releases
# =========================================

def equivalence_classes(df, QI):
    """
    Builds equivalence classes based on QIs

    :param df: Dataset
    :param QI: Quasi-identifiers
    :return: Dictionary mapping QI value tuples to list of row indices
    """

    # Dictionary mapping QI tuples to row indices
    groups = defaultdict(list)

    # Groups rows sharing the same generalized QI values
    for idx, row in df.iterrows():
        key = tuple(row[q] for q in QI)
        groups[key].append(idx)

    return groups

def sensitive_sets(df, eqs, sensitive):
    """
    Maps each equivalence class to the set of sensitive values it contains

    :param df: Dataset
    :param eqs: Equivalence classes
    :param sensitive: Sensitive attribute
    :return: Dictionary mapping QI value tuples to sets of sensitive values
    """
    out = {}
    for k, idxs in eqs.items():
        out[k] = set(df.loc[idxs, sensitive])   # Collects unique sensitive values per class
    return out

def intersection_attack(R1, R2, overlap_uids, QI, sensitive):
    """
    Implements the intersection (composition) attack
    
    :param R1: Relase 1
    :param R2: Relase 2
    :param overlap_uids: List of overlapping individual IDs
    :param QI: Quasi-identifiers
    :param sensitive: Sensitive attribute
    :return: DataFrame with attack results per individual
    """
    # Computes equivalence classes in both releases
    eq1 = equivalence_classes(R1, QI)
    eq2 = equivalence_classes(R2, QI)

    # Gets sensitive value sets per equivalence class
    s1 = sensitive_sets(R1, eq1, sensitive)
    s2 = sensitive_sets(R2, eq2, sensitive)

    # Maps individuals to rows for fast lookup
    uid_to_row_1 = {uid: i for i, uid in enumerate(R1["uid"])}
    uid_to_row_2 = {uid: i for i, uid in enumerate(R2["uid"])}

    rows = []

    # Iterates over overlapping individuals
    for uid in overlap_uids:
        if uid not in uid_to_row_1 or uid not in uid_to_row_2:
            continue

        i1 = uid_to_row_1[uid]
        i2 = uid_to_row_2[uid]

        # Finds the equivalence class of the individual in each release
        k1 = tuple(R1.loc[i1, q] for q in QI)
        k2 = tuple(R2.loc[i2, q] for q in QI)

        if k1 not in s1 or k2 not in s2:
            continue

        # Computes prior and posterior anonymity (before and after intersecting releases)
        prior = min(len(s1[k1]), len(s2[k2]))
        post = len(s1[k1].intersection(s2[k2]))

        rows.append({
            "prior": prior,
            "posterior": post,
            "drop": prior - post,
            "confidence": 1 / post if post > 0 else 1.0     # Attacker confidence increases as the posterior set shrinks
        })

    return pd.DataFrame(rows)

k_attack_df = intersection_attack(Rk1, Rk2, overlap_ids, QI, SENSITIVE)
print("\nIntersection attack (k-anonymity) summary:")
print(k_attack_df.describe())

l_attack_df = intersection_attack(Rl1, Rl2, overlap_ids, QI, SENSITIVE)
print("\nIntersection attack (l-diversity) summary:")
print(l_attack_df.describe())

t_attack_df = intersection_attack(Rt1, Rt2, overlap_ids, QI, SENSITIVE)
print("\nIntersection attack (t-closeness) summary:")
print(t_attack_df.describe())


# =====================
# 4.2. Attack metrics
# =====================

def attack_metrics(df):
    """
    Computes summary metrics of the attack:
    - Vulnerable population: anonymity reduced
    - Perfect breach: posterior size = 1
    - Avg prior/posterior anonymity
    - Avg anonymity drop

    :param df: DataFrame with attack results
    :return: Dictionary with metrics
    """
    return {
        "Vulnerable population (%)": 100 * (df["drop"] > 0).mean(),
        "Perfect breach (%)": 100 * (df["posterior"] == 1).mean(),
        "Avg prior anonymity": df["prior"].mean(),
        "Avg posterior anonymity": df["posterior"].mean(),
        "Avg drop": df["drop"].mean()
    }

# Compute and print attack metrics
metrics = attack_metrics(k_attack_df)
print("\nMetrics (k-anonymity):", metrics)

metrics = attack_metrics(l_attack_df)
print("\nMetrics (l-diversity):", metrics)

metrics = attack_metrics(t_attack_df)
print("\nMetrics (t-closeness):", metrics)

# ==========
# 4.3 Plots
# ==========

plt.figure(figsize=(15, 4))

# Shows how anonymity collapses after intersection
ax1 = plt.subplot(1, 2, 1)
ax1.hist(k_attack_df["prior"], bins=20, alpha=0.6, label="Single release",color='C3')
ax1.hist(k_attack_df["posterior"], bins=20, alpha=0.6, label="After composition",color='C5')
ax1.set_xlabel("Number of possible sensitive values", fontsize=18)
ax1.set_ylabel("Number of individuals", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)

# SA - Visualizes how confident the attacker becomes
ax2 = plt.subplot(1, 2, 2)
ax2.hist(k_attack_df["confidence"], bins=20, alpha=0.6, color='C5')
ax2.set_xlabel("Adversarial confidence", fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Get legend from left subplot and display on right subplot
handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=13, loc='upper right')

plt.tight_layout()
plt.show()


# =========================
# 5. Differential Privacy 
# =========================
 
# Scalar query: count of sensitive value
def dp_query(true_val, eps, mechanism="laplace", delta=1e-5):
    """
    DP query: total count of a sensitive value
    """
    if mechanism == "laplace":
        noise = np.random.laplace(0, 1 / eps)
    else:
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
        noise = np.random.normal(0, sigma)
    return true_val + noise



# Repeated releases (sampling output distributions)
def sample_outputs(true_val, eps, k=1, mechanism="laplace", runs=1000):
    """
    Samples outputs from k composed DP releases
    """
    outs = []
    for _ in range(runs):
        total = 0
        for _ in range(k):
            total += dp_query(true_val, eps, mechanism)
        outs.append(total)
    return np.array(outs)


# ====================================
# 5.1. Privacy loss under composition
# ====================================

k_releases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
eps = 0.2

true_D = (D[SENSITIVE] == "Exec-managerial").sum()
true_Dp = (D_prime[SENSITIVE] == "Exec-managerial").sum()


# Plot composition behavior
plt.figure(figsize=(7, 5))
plt.plot(
    k_releases,
    [k * eps for k in k_releases],
    "-",
    label="Basic composition bound",
    color='C8',
    linewidth=3
)
plt.plot(
    k_releases,
    [np.sqrt(2 * k * np.log(1e5)) * eps + k * eps * ((np.exp(eps) - 1) / (np.exp(eps) + 1)) for k in k_releases],
    "-",
    label="Advanced composition bound",
    color='C2',
    linewidth=3
)

plt.xlabel("Number of DP releases", fontsize=18)
plt.ylabel("Privacy loss (ε̂)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.show()


# ==================================
# 5.2. Visual distribution overlap
# ==================================

eps = 0.2
k = 1
outs_D_lap = sample_outputs(true_D, eps, k=k, mechanism="laplace")
outs_Dp_lap = sample_outputs(true_Dp, eps, k=k, mechanism="laplace")
outs_D_gau = sample_outputs(true_D, eps, k=k, mechanism="gaussian")
outs_Dp_gau = sample_outputs(true_Dp, eps, k=k, mechanism="gaussian")

k = 5
outs_D_lap_k5 = sample_outputs(true_D, eps, k=k, mechanism="laplace")
outs_Dp_lap_k5 = sample_outputs(true_Dp, eps, k=k, mechanism="laplace")
outs_D_gau_k5 = sample_outputs(true_D, eps, k=k, mechanism="gaussian")
outs_Dp_gau_k5 = sample_outputs(true_Dp, eps, k=k, mechanism="gaussian")

k = 10
outs_D_lap_k10 = sample_outputs(true_D, eps, k=k, mechanism="laplace")
outs_Dp_lap_k10 = sample_outputs(true_Dp, eps, k=k, mechanism="laplace")
outs_D_gau_k10 = sample_outputs(true_D, eps, k=k, mechanism="gaussian")
outs_Dp_gau_k10 = sample_outputs(true_Dp, eps, k=k, mechanism="gaussian")

# Laplace mechanism: k=1, 5, 10 side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.hist(outs_D_lap, bins=50, alpha=0.6, label="Dataset D", color='C0')
ax1.hist(outs_Dp_lap, bins=50, alpha=0.6, label="Neighbor D'", color='C7')
ax1.set_title("k=1", fontsize=18)
ax1.set_ylabel("Frequency", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.locator_params(axis='x', nbins=4)

ax2.hist(outs_D_lap_k5, bins=50, alpha=0.6, label="Dataset D", color='C0')
ax2.hist(outs_Dp_lap_k5, bins=50, alpha=0.6, label="Neighbor D'", color='C7')
ax2.set_title("k=5", fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.locator_params(axis='x', nbins=4)

ax3.hist(outs_D_lap_k10, bins=50, alpha=0.6, label="Dataset D", color='C0')
ax3.hist(outs_Dp_lap_k10, bins=50, alpha=0.6, label="Neighbor D'", color='C7')
ax3.set_title("k=10", fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.locator_params(axis='x', nbins=4)

fig.text(0.5, 0.04, "Composed DP output", fontsize=18, ha='center')
ax3.legend(fontsize=13, loc='upper right')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# Gaussian mechanism: k=1, 5, 10 side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.hist(outs_D_gau, bins=50, alpha=0.6, label="Dataset D", color='C1')
ax1.hist(outs_Dp_gau, bins=50, alpha=0.6, label="Neighbor D'", color='C8')
ax1.set_title("k=1", fontsize=18)
ax1.set_ylabel("Frequency", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.locator_params(axis='x', nbins=4)

ax2.hist(outs_D_gau_k5, bins=50, alpha=0.6, label="Dataset D", color='C1')
ax2.hist(outs_Dp_gau_k5, bins=50, alpha=0.6, label="Neighbor D'", color='C8')
ax2.set_title("k=5", fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.locator_params(axis='x', nbins=4)

ax3.hist(outs_D_gau_k10, bins=50, alpha=0.6, label="Dataset D", color='C1')
ax3.hist(outs_Dp_gau_k10, bins=50, alpha=0.6, label="Neighbor D'", color='C8')
ax3.set_title("k=10", fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.locator_params(axis='x', nbins=4)

fig.text(0.5, 0.04, "Composed DP output", fontsize=18, ha='center')
ax3.legend(fontsize=11, loc='upper right')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


print("\nDone!")