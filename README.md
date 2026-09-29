# Data Privacy: Syntactic Anonymization vs. Differential Privacy

Code for my Bachelor's Thesis (BSc in Mathematics, University of Barcelona, 2026):
**"Mathematical Framework for Data Privacy: A Comparative Study of Privacy-Preserving Methods"**
(supervisor: Dr. Nahuel Statuto).

The thesis compares classical syntactic anonymization (k-anonymity, ℓ-diversity, t-closeness)
with differential privacy (Laplace and Gaussian mechanisms). This repository contains the three
experiments of the empirical evaluation (Chapter 4), implemented in Python.


## Experiments

| Script | Thesis section | What it does |
|---|---|---|
| `experiments/exp1.py` | 4.3.1 | **Attribute inference attack.** An attacker who knows the quasi-identifiers infers the sensitive attribute (`income`) by majority vote inside each equivalence class (syntactic methods) or from noisy group means (DP). |
| `experiments/exp2.py` | 4.3.2 | **Composition attack.** Two overlapping releases (15,000 records each, 5,000 shared) are anonymized independently and intersected to measure the loss of anonymity. For DP, it plots the basic and advanced composition bounds and the output distributions of repeated Laplace/Gaussian releases. |
| `experiments/exp3.py` | 4.3.3 | **Privacy–utility trade-off.** Utility is the accuracy of a logistic regression (syntactic methods) or the mean absolute error of noisy contingency tables (DP), normalized to compare all methods at three privacy levels. |

**Setup common to all experiments:** [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
(32,561 records); quasi-identifiers `age, sex, education, native-country, marital-status`;
sensitive attribute `income` (exp1) or `occupation` (exp2, exp3); DP experiments use δ = 10⁻⁵.
Syntactic anonymization is done with the [`anjana`](https://github.com/IFCA-Advanced-Computing/anjana)
library, using the generalization hierarchies in `experiments/adult/hierarchies/`
(up to 50% record suppression allowed).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
└── experiments/
    ├── exp1.py              # Attribute inference attack
    ├── exp2.py              # Composition attack
    ├── exp3.py              # Privacy–utility trade-off
    └── adult/
        ├── adult.data       # UCI Adult dataset (train split)
        └── hierarchies/     # Generalization hierarchies used by anjana
```

## Installation

Tested with Python 3.12.

```bash
git clone <this-repo-url>
cd data-privacy-thesis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the experiments

The scripts use relative paths (`./adult/...`), so **run them from inside the `experiments/` folder**:

```bash
cd experiments
python exp1.py
python exp2.py
python exp3.py
```

Each script prints its results to the console and then opens the figures with `plt.show()`.
On a machine without a display, run with `MPLBACKEND=Agg python exp1.py` to skip the figures.
Each script takes a few minutes to run.

## Main results

Obtained by running these scripts (consistent with the thesis):

- **Inference attack (exp1).** With syntactic anonymization the attacker stays well above the
  0.5 random-guessing baseline: ~0.82 for k-anonymity (k = 5 to 50), 0.743 for ℓ = 2 and
  0.743 for t ≤ 0.3. With DP, accuracy falls toward random guessing as ε decreases
  (Laplace: 0.648 at ε = 1, ~0.51 at ε = 0.05).
- **Composition attack (exp2).** For k-anonymity (k = 10), the anonymity of about 48% of
  the overlapping individuals shrinks after intersecting the two releases. The ℓ-diversity (ℓ = 7)
  and t-closeness (t = 0.1) releases showed no loss, but at a high cost in utility (see exp3). For DP,
  the privacy loss follows the basic and advanced composition bounds and can be budgeted in advance.
- **Privacy–utility (exp3).** k-anonymity keeps utility but offers weak protection; ℓ-diversity and
  t-closeness collapse in utility as constraints tighten; DP degrades smoothly as ε decreases.

## Reproducibility notes

- `exp1.py` and `exp2.py` fix the NumPy seed, so the results are reproducible.
- In `exp3.py` the train/test split is seeded, but the DP noise is not, so the DP error values
  vary slightly between runs (the overall trends do not).
- `exp1.py` also tries ℓ = 3, which `anjana` cannot achieve for a binary sensitive attribute, so
  it prints a message and skips it. The thesis reports ℓ ∈ {1, 2}.
- Anonymized outputs may depend on the `anjana` version; the scripts were last verified with
  `anjana` 1.2.3 (see `requirements.txt`).

## Assumptions and limitations

- **DP experiments are simulations.** They add noise to aggregate statistics computed from the
  full dataset to compare against syntactic methods. They are not a production DP pipeline.
- **Sensitivity.** Counting queries use sensitivity 1, corresponding to the add/remove
  neighbouring model (e.g. `exp2.py` builds the neighbouring dataset by dropping one record).
  In `exp1.py`, group means are perturbed with sensitivity 1, which is a conservative choice
  (the true sensitivity of a group mean is 1/n for a group of size n) and means more noise than needed.
- **Gaussian mechanism.** The calibration σ = Δ₂·√(2 ln(1.25/δ))/ε is valid for ε ≤ 1, which
  is the range used here.
- **Utility metrics are not directly comparable** across families, so the unified comparison
  in exp3 uses a normalized score and three representative privacy levels.

## Data and references

- Dataset: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository
  (CC BY 4.0).
- Anonymization library: Sáinz-Pardo Díaz, J. & López García, Á. (2024). *An open source Python
  library for anonymizing sensitive data*. Scientific Data, 11(1), 1289.

## License

Released under the MIT License (see `LICENSE`).

## Author

Paula Sarrión Soriano · [www.linkedin.com/in/paulasarrionsoriano] · [paulasarrionsoriano@gmail.com]