import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

# random seed parameters
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# data upload
df = pd.read_csv("patient_data_up.csv", index_col="patient")

y = df["earlier_birth"]
X = df.drop(columns=["earlier_birth"])

categorical_vars = list(df.select_dtypes(include=["object"]).columns)
numerical_vars = list(X.select_dtypes(include=["int64", "float64"]).columns)

# Preprocessing - we can try with sparse = true or false but idk
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_vars),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_vars)
    ],
    remainder="drop"
)


# MODEL CHABGE HERE - choose your model and its parameters
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=7,
    random_state=RANDOM_SEED
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gb)
])

# # schemes validation
cv_schemes = [
    ("SKF5_rs42", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
    ("SKF5_rs2024", StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)),
    ("RSKF5x2", RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)),
]

# dictionnarys
imp_by_feature = defaultdict(list)   
metrics_log = []                     

def get_feature_names_after_fit(fitted_preprocessor) -> np.ndarray:
    """Получаем имена признаков после OneHot+Scaling из уже обученного препроцессора."""
    # preprocesses
    num_names = np.array(numerical_vars, dtype=object)

    # Categorial names after OHE
    ohe = fitted_preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_vars)

    return np.concatenate([num_names, cat_names])

# no leaks in test set!!!
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

for scheme_name, cv in cv_schemes:
    fold_idx = 0
    for train_idx, val_idx in cv.split(X_train_all, y_train_all):
        fold_idx += 1
        X_tr, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
        y_tr, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]

        # pipline train
        pipeline.fit(X_tr, y_tr)

        # ROC-AUC test
        p_val = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        metrics_log.append({
            "scheme": scheme_name,
            "fold": fold_idx,
            "roc_auc_val": auc
        })

        
        # (последний шаг пайплайна).
        fitted_prep = pipeline.named_steps["preprocessor"]
        est = pipeline.named_steps["model"]

        X_val_trans = fitted_prep.transform(X_val)  
        feat_names = get_feature_names_after_fit(fitted_prep)

        pi = permutation_importance(
            estimator=est,
            X=X_val_trans,
            y=y_val,
            scoring="roc_auc",
            n_repeats=15,   #change nuber here for number of shuffles!!
            random_state=RANDOM_SEED
        )

        #importances by feature
        for name, val in zip(feat_names, pi.importances_mean):
            imp_by_feature[name].append(val)

# choosing the right one
rows = []
for feat, vals in imp_by_feature.items():
    vals = np.array(vals, dtype=float)
    rows.append({
        "feature": feat,
        "importance_mean": vals.mean(),
        "importance_std": vals.std(),
        "n_evaluations": len(vals)
    })

imp_agg = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
imp_agg.to_csv("feature_importance.csv", index=False)

print("Aggregated permutation importances saved to 'feature_importance.csv'")
print("\nTop features by aggregated permutation importance:")
print(imp_agg.head(25).to_string(index=False))

#Final training on all train and test on hold-out
pipeline.fit(X_train_all, y_train_all)
p_test = pipeline.predict_proba(X_test_all)[:, 1]
auc_test = roc_auc_score(y_test_all, p_test)
print(f"\nFinal hold-out ROC-AUC on test: {auc_test:.4f}")

#auqlity of life improvements
metrics_df = pd.DataFrame(metrics_log).sort_values(["scheme", "fold"])
metrics_df.to_csv("cv_roc_auc_log.csv", index=False)
print("CV metrics saved to 'cv_roc_auc_log.csv'")

# Show top 20 features
topk = 20
top_imp = imp_agg.head(topk).iloc[::-1]  # change if inc or descrease,
plt.figure(figsize=(9, 7))
plt.barh(top_imp["feature"], top_imp["importance_mean"])
plt.xlabel("Permutation importance (mean across CV)")
plt.title(f"Top-{topk} features — GradientBoosting")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
print("Chart saved to 'feature_importance_top20.png'")


with open("final_results.txt", "w") as f:
    f.write("Aggregated permutation importances saved to 'feature_importance.csv'\n")
    f.write("\nAll features by aggregated permutation importance:\n")
    f.write(imp_agg.to_string(index=False))  # все строки, не только head
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write(f"\nFinal hold-out ROC-AUC on test: {auc_test:.4f}\n")