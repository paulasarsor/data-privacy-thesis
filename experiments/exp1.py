# Author: Paula Sarrión Soriano

# ==================================
# EXPERIMENT 1 — INFERENCE ATTACK
# ==================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anjana
from anjana.anonymity import k_anonymity, l_diversity, t_closeness

#np.random.seed(67)
np.random.seed(389264)

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

#print(df.columns.tolist())

# Encode income as binary (0/1)
df["income"] = (df["income"] == ">50K").astype(int)

# Define Quasi-Identifiers and Sensitive attribute
QI = ["age", "sex", "education", "native-country", "marital-status"]
SENSITIVE = "income"


print("Dataset loaded:", df.shape)


# =========================
# 2. Differential Privacy
# =========================

def laplace_mechanism(true_value, epsilon, sensitivity=1.0): 
    """
    Laplace Mechanism for Differential Privacy for counting queries
    
    :param true_value: true query result
    :param epsilon: epsilon value
    :param sensitivity: l1-sensitivity of the query
    :return: noisy (private) result
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)     # Draws Laplace noise with mean 0 and the computed scale
    return true_value + noise               

def gaussian_mechanism(true_value, epsilon, delta=1e-5, sensitivity=1.0):
    """
    Gaussian Mechanism for Differential Privacy for counting queries

    :param true_value: true query result
    :param epsilon: epsilon value
    :param delta: delta value for Gaussian mechanism
    :param sensitivity: l2-sensitivity of the query
    :return: noisy (private) result
    """
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon   # Standard deviation for Gaussian noise
    noise = np.random.normal(0, sigma)  # Draws Gaussian noise with mean 0 and standard deviation sigma
    return true_value + noise           

# =========================
# 2.1. Inference accuracy
# =========================

def dp_inference_accuracy(df, mechanism, eps, delta=None):
    """
    Function that simulates an attribute inference attack 
    when only DP-protected aggregates are released
    
    :param df: dataset
    :param mechanism: mechanism to use ("laplace" or "gaussian")
    :param eps: epsilon value
    :param delta: delta value for Gaussian mechanism
    :return: inference accuracy
    """
    # Counters for correct predictions and total individuals
    correct = 0
    total = 0

    # Iterates over equivalence classes defined by the quasi-identifiers
    for _, group in df.groupby(QI):
        true_mean = group[SENSITIVE].mean()     # True mean of the sensitive attribute in the group
        true_label = int(true_mean >= 0.5)      # True label for the group based on majority

        # Apply the chosen DP mechanism to get a noisy estimate
        if mechanism == "laplace":
            noisy = laplace_mechanism(true_mean, eps)
        elif mechanism == "gaussian":
            noisy = gaussian_mechanism(true_mean, eps, delta)
        
        pred = int(noisy >= 0.5)    # Attacker predicts the sensitive value using the noisy statistic

        # Count correct predictions for all individuals in the group
        correct += (pred == true_label) * len(group)
        total += len(group)

    # Return the overall inference accuracy of the attacker
    return correct / total


# Evaluate inference accuracy (no microdata release) 
print("\nRunning DP inference...")

results_dp = {"laplace": {}, "gaussian": {}}
for eps in [1.0, 0.8, 0.5, 0.2, 0.1, 0.05]:
    acc_lap = dp_inference_accuracy(df, "laplace", eps)
    acc_gau = dp_inference_accuracy(df, "gaussian", eps, delta=1e-5)
    results_dp["laplace"][eps] = acc_lap
    results_dp["gaussian"][eps] = acc_gau
    print(f"ε={eps}: Laplace {acc_lap:.3f}, Gaussian {acc_gau:.3f}")


# ===========================
# 3. Syntactic Anonymization
# ===========================

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

# Allows up to 50% record suppression if anonymization constraints cannot be satisfied
supp_level = 50

# =======================
# 3.1. Attack definition
# =======================

def inference_accuracy(df_anonymized):
    """
    Background-knowledge attribute inference: attacker predicts
    sensitive value by majority vote within each equivalence class.

    :param df_anonymized: anonymized dataset
    :return: inference accuracy
    """
    preds = (
        df_anonymized
        .groupby(QI)[SENSITIVE]
        .transform(lambda g: int(g.mean() >= 0.5))
    )
    return (preds == df_anonymized[SENSITIVE]).mean()   # Computes the fraction of correctly inferred sensitive values


# =====================
# 3.2. Run experiments
# =====================

# Run syntactic anonymizations and evaluate inference accuracy
print("\nRunning syntactic anonymization with Anjana...")

# k-anonymity
results_k = {}
print("\nRunning k-anonymity...")
for k in [5, 10, 20, 30, 40, 50]:
    try:
        df_k = k_anonymity( df, [], QI, k, supp_level, hierarchies )    # Apply k-anonymity
        if df_k is None or df_k.empty:
            print(f"k={k}: cannot create anonymized dataset, skipping...")
            continue
        acc = inference_accuracy(df_k)      # Evaluate inference accuracy
        results_k[k] = acc
        print(f"k={k}: accuracy={acc:.3f}, rows={len(df_k)}")
    except Exception as e:
        print(f"k={k}: error -> {e}")


# l-diversity
results_l = {}
print("\nRunning l-diversity...")
for l in [1, 2, 3]:
    try:
        df_l = l_diversity(df, [], QI, SENSITIVE, 5, l, supp_level, hierarchies)    # Apply l-diversity
        if df_l is None or df_l.empty:
            print(f"l={l}: cannot create anonymized dataset, skipping...")
            continue
        acc = inference_accuracy(df_l)      # Evaluate inference accuracy
        results_l[l] = acc
        print(f"l={l}: accuracy={acc:.3f}, rows={len(df_l)}")
    except Exception as e:
        print(f"l={l}: error -> {e}")
    

# t-closeness
results_t = {}
print("\nRunning t-closeness...")
for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
    try:
        df_t = t_closeness(df, [], QI, SENSITIVE, 5, t, supp_level, hierarchies)    # Apply t-closeness
        if df_t is None or df_t.empty:
            print(f"t={t}: cannot create anonymized dataset, skipping...")
            continue
        acc = inference_accuracy(df_t)      # Evaluate inference accuracy
        results_t[t] = acc
        print(f"t={t}: accuracy={acc:.3f}, rows={len(df_t)}")
    except Exception as e:
        print(f"t={t}: error -> {e}")
    

# ===========
# 4. Plots
# ===========

# k-anonymity, l-diversity, and t-closeness side by side with shared y-axis
k_vals = list(results_k.keys())
k_accs = list(results_k.values())

l_vals = list(results_l.keys())
l_accs = list(results_l.values())

t_vals = list(results_t.keys())
t_accs = list(results_t.values())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

# k-anonymity
ax1.plot(k_vals, k_accs, marker='o', color='C3', linewidth=3)
ax1.axhline(0.5, linestyle="--", linewidth=3, color='C7')
ax1.set_xlabel("k", fontsize=18)
ax1.set_ylabel("Inference Accuracy", fontsize=18)
ax1.set_ylim(0.5, 0.9)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(True)

# l-diversity
ax2.plot(l_vals, l_accs, marker='o', color='C4', linewidth=3)
ax2.axhline(0.5, linestyle="--", linewidth=3, color='C7')
ax2.set_xlabel(r"$\ell$", fontsize=18)
ax2.set_ylim(0.5, 0.9)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True)

# t-closeness
ax3.plot(t_vals, t_accs, marker='o', color='C2', linewidth=3)
ax3.axhline(0.5, linestyle="--", linewidth=3, color='C7')
ax3.set_xlabel("t", fontsize=18)
ax3.invert_xaxis()  # x-axis is inverted because smaller t = stronger privacy
ax3.set_ylim(0.5, 0.9)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.grid(True)

plt.tight_layout()
plt.show()

# Differential Privacy
eps_vals = list(results_dp["laplace"].keys())
lap_accs = list(results_dp["laplace"].values())
gau_accs = list(results_dp["gaussian"].values())

plt.figure()
plt.plot(eps_vals, lap_accs, marker='o', label="Laplace", color='C0', linewidth=2.5)
plt.plot(eps_vals, gau_accs, marker='o', label="Gaussian", color='C1', linewidth=2.5)
plt.axhline(0.5, linestyle="--", linewidth=2.5, color='C7')
plt.xlabel(r"$\varepsilon$", fontsize=18)
plt.ylabel("Inference Accuracy", fontsize=18)
plt.ylim(0.4, 0.7)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()



print("\nDone!")