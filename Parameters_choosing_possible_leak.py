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
from sklearn.metrics import make_scorer, roc_auc_score, classification_report   # CHANGED: добавил classification_report (по желанию)
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

# random seed parameters
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# data upload
df = pd.read_csv("patient_data_two_cc.csv", index_col="patient")

y = df["earlier_birth"]
X = df.drop(columns=["earlier_birth"])

categorical_vars = list(df.select_dtypes(include=["object"]).columns)
numerical_vars = list(X.select_dtypes(include=["int64", "float64"]).columns)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_vars),
        # В новых версиях sklearn используем sparse_output=False
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_vars)
    ],
    remainder="drop"
)


#add multiclass roc-auc scorer
def multiclass_auc(estimator, X, y):
    """Скорер для permutation_importance: macro ROC-AUC (OvR) для многокласса."""
    proba = estimator.predict_proba(X)  # (n_samples, n_classes)
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

# MODEL
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

# CHANGED: многоклассовый AUC (One-vs-Rest), усреднение macro
multiclass_auc_scorer = make_scorer(
    roc_auc_score,
    needs_proba=True,          # ← здесь корректно
    multi_class="ovr",
    average="macro"
)

# schemes validation
cv_schemes = [
    ("SKF5_rs42", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
    ("SKF5_rs2024", StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)),
    ("RSKF5x2", RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)),
]

# dicts
imp_by_feature = defaultdict(list)
metrics_log = []

def get_feature_names_after_fit(fitted_preprocessor) -> np.ndarray:
    """Имена признаков после OneHot+Scaling из обученного препроцессора."""
    num_names = np.array(numerical_vars, dtype=object)
    ohe = fitted_preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_vars)
    return np.concatenate([num_names, cat_names])

# hold-out split (без утечки)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

for scheme_name, cv in cv_schemes:
    fold_idx = 0
    for train_idx, val_idx in cv.split(X_train_all, y_train_all):
        fold_idx += 1
        X_tr, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
        y_tr, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]

        # pipeline train
        pipeline.fit(X_tr, y_tr)

        # CHANGED: многоклассовый ROC-AUC — НЕ берем [:, 1], а подаем всю матрицу вероятностей
        p_val = pipeline.predict_proba(X_val)
        auc = roc_auc_score(y_val, p_val, multi_class="ovr", average="macro")
        metrics_log.append({
            "scheme": scheme_name,
            "fold": fold_idx,
            "roc_auc_val": auc
        })

        # подготовка для permutation importance
        fitted_prep = pipeline.named_steps["preprocessor"]
        est = pipeline.named_steps["model"]

        X_val_trans = fitted_prep.transform(X_val)  # dense np.array (из-за sparse_output=False)
        feat_names = get_feature_names_after_fit(fitted_prep)

        # CHANGED: передаем кастомный scorer для многоклассового AUC
        pi = permutation_importance(
            estimator=est,
            X=X_val_trans,
            y=y_val,
            scoring= multiclass_auc,    # CHANGED use my finction
            n_repeats=15,                     # можно менять для стабильности/времени
            random_state=RANDOM_SEED
        )

        # importances by feature
        for name, val in zip(feat_names, pi.importances_mean):
            imp_by_feature[name].append(val)

# aggregation
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

# CHANGED: единообразие имен файлов
imp_csv_path = "feature_importance_two_params.csv"
imp_agg.to_csv(imp_csv_path, index=False)

print(f"Aggregated permutation importances saved to '{imp_csv_path}'")
print("\nTop features by aggregated permutation importance with 2 control parameters:")
print(imp_agg.head(25).to_string(index=False))

# Final training on all train and test on hold-out
pipeline.fit(X_train_all, y_train_all)

# CHANGED: многоклассовый AUC на тесте
p_test = pipeline.predict_proba(X_test_all)
auc_test = roc_auc_score(y_test_all, p_test, multi_class="ovr", average="macro")
print(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}")

# QoL: save CV metrics
metrics_df = pd.DataFrame(metrics_log).sort_values(["scheme", "fold"])
cv_log_path = "cv_roc_auc_log_two_params.csv"   # CHANGED: согласовано с print
metrics_df.to_csv(cv_log_path, index=False)
print(f"CV metrics saved to '{cv_log_path}'")

# Plot top 20 features
topk = 20
top_imp = imp_agg.head(topk).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top_imp["feature"], top_imp["importance_mean"])
plt.xlabel("Permutation importance (mean across CV)")
plt.title(f"Top-{topk} features — GradientBoosting")
plt.tight_layout()
plot_path = "feature_importance_top20_two_params.png"  # CHANGED: согласовано с print
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Chart saved to '{plot_path}'")

# Save full text report
with open("final_results.txt", "w") as f:
    f.write(f"Aggregated permutation importances saved to '{imp_csv_path}'\n")
    f.write("\nAll features by aggregated permutation importance:\n")
    f.write(imp_agg.to_string(index=False))
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}\n")  # CHANGED: подпись метрики
