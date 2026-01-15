"""
Application SoftImpute sur X_test - Version minimale
"""

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
import gc

print("Chargement X_test...")
X_test = pd.read_csv("data/X_test_XKVc4no.csv", index_col=0)

print(f"Shape X_test: {X_test.shape}")

# Les 1000 dernières colonnes sont trouées
holed_cols = X_test.columns[-1000:].tolist()
print(f"Colonnes trouées: {len(holed_cols)}")

# Normalisation
print("Normalisation...")
X_normalized = X_test.copy()
col_means = {}
col_stds = {}

for col in X_normalized.columns:
    valid_mask = ~X_normalized[col].isna()
    if valid_mask.sum() > 0:
        col_means[col] = X_normalized.loc[valid_mask, col].mean()
        col_stds[col] = X_normalized.loc[valid_mask, col].std()
        if col_stds[col] < 1e-8:
            col_stds[col] = 1.0
        X_normalized[col] = (X_normalized[col] - col_means[col]) / col_stds[col]

# SoftImpute
print("Application SoftImpute (peut prendre 8-10 min)...")
X_matrix = X_normalized.T.to_numpy().astype(np.float64)

model = SoftImpute(
    shrinkage_value=5,
    max_iters=100,
    convergence_threshold=1e-4,
    verbose=False
)

X_completed = model.fit_transform(X_matrix)

# Dénormalisation
print("Dénormalisation...")
X_completed_df = pd.DataFrame(X_completed.T, index=X_test.index, columns=X_test.columns)

for col in X_completed_df.columns:
    if col in col_means:
        X_completed_df[col] = (X_completed_df[col] * col_stds[col]) + col_means[col]

# Extraction des 1000 colonnes trouées
Y_test = X_completed_df[holed_cols]

# Vérifications
print(f"\nShape Y_test: {Y_test.shape}")
print(f"NaN restants: {Y_test.isna().sum().sum()}")

# Sauvegarde
Y_test.to_csv("Y_test_SUBMISSION.csv")
print("Fichier sauvegardé: Y_test_SUBMISSION.csv")
print("\nPrêt pour soumission sur ChallengeData!")

del X_matrix, X_completed, X_normalized
gc.collect()