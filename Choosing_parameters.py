# =========================
# Parameters choosing + Aggregated Permutation Importance (Multiclass)
# =========================
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# -------------------------
# Reproducibility
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------
# Data
# -------------------------
# CHANGED: путь оставь свой
df = pd.read_csv("patient_data_two_cc.csv", index_col="patient")

# Целевой столбец
y = df["earlier_birth"]

# CHANGED: Автоматически убираем возможные источники утечки:
#  - любые колонки, начинающиеся на control_
#  - любые "копии" таргета (earlier_birth_copy и т.п.)
#  - любые колонки, которые текстом содержат "copy" или "control" рядом с earlier_birth
leak_patterns = [
    r"^control_.*", r".*control.*earlier.*", r".*earlier.*control.*",
    r".*copy.*earlier.*", r".*earlier.*copy.*", r"^earlier_birth_copy$",
    r"^control_parameter.*"
]
leakage_cols = []
for c in df.columns:
    if c == "earlier_birth":
        continue
    for p in leak_patterns:
        if re.match(p, c, flags=re.IGNORECASE):
            leakage_cols.append(c)
            break

# CHANGED: формируем X без таргета и без потенциальных «контрольных» колонок
drop_cols = ["earlier_birth"] + leakage_cols
X = df.drop(columns=drop_cols, errors="ignore")
print("Removed potential leakage columns:", leakage_cols)

# -------------------------
# Feature types
# -------------------------
categorical_vars = list(X.select_dtypes(include=["object"]).columns)
numerical_vars = list(X.select_dtypes(include=["int64", "float64"]).columns)

# -------------------------
# Preprocessing
# -------------------------
# CHANGED: Совместимость с версиями sklearn:
#   - в новых версиях: OneHotEncoder(..., sparse_output=False)
#   - в старых:        OneHotEncoder(..., sparse=False)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_vars),
        ("cat", ohe, categorical_vars),
    ],
    remainder="drop"
)

# -------------------------
# Model
# -------------------------
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

# -------------------------
# CV schemes
# -------------------------
cv_schemes = [
    ("SKF5_rs42",   StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
    ("SKF5_rs2024", StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)),
    ("RSKF5x2",     RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)),
]

# -------------------------
# Helper: names after preprocessing
# -------------------------
def get_feature_names_after_fit(fitted_preprocessor) -> np.ndarray:
    num_names = np.array(numerical_vars, dtype=object)
    ohe_step = fitted_preprocessor.named_transformers_.get("cat", None)
    if ohe_step is not None and hasattr(ohe_step, "get_feature_names_out"):
        cat_names = ohe_step.get_feature_names_out(categorical_vars)
    else:
        cat_names = np.array([], dtype=object)
    return np.concatenate([num_names, cat_names])

# -------------------------
# Hold-out split (no leakage)
# -------------------------
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# -------------------------
# Custom scorer for permutation_importance (multiclass ROC-AUC)
# -------------------------
def multiclass_auc(estimator, Xmat, y_true):
    """
    Переменная Xmat — уже трансформированная (dense) матрица признаков.
    Здесь мы вызываем predict_proba у final estimator (последней ступени пайплайна).
    """
    proba = estimator.predict_proba(Xmat)  # shape (n_samples, n_classes)
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")

# -------------------------
# Aggregation containers
# -------------------------
imp_by_feature = defaultdict(list)
metrics_log = []

# -------------------------
# Cross-validation loops
# -------------------------
for scheme_name, cv in cv_schemes:
    fold_idx = 0
    for tr_idx, va_idx in cv.split(X_train_all, y_train_all):
        fold_idx += 1
        X_tr, X_val = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
        y_tr, y_val = y_train_all.iloc[tr_idx], y_train_all.iloc[va_idx]

        # fit pipeline
        pipeline.fit(X_tr, y_tr)

        # многоклассовый ROC-AUC на валидации
        p_val = pipeline.predict_proba(X_val)  # (n, n_classes)
        auc_val = roc_auc_score(y_val, p_val, multi_class="ovr", average="macro")
        metrics_log.append({"scheme": scheme_name, "fold": fold_idx, "roc_auc_val": auc_val})

        # permutation importance на валидации (по transform'у препроцессора)
        fitted_prep = pipeline.named_steps["preprocessor"]
        est = pipeline.named_steps["model"]

        X_val_trans = fitted_prep.transform(X_val)  # dense np.array
        feat_names = get_feature_names_after_fit(fitted_prep)

        pi = permutation_importance(
            estimator=est,
            X=X_val_trans,
            y=y_val,
            scoring=multiclass_auc,   # CHANGED: своя функция-скорер без needs_proba
            n_repeats=15,             # можно увеличить для стабильности
            random_state=RANDOM_SEED
        )

        for name, val in zip(feat_names, pi.importances_mean):
            imp_by_feature[name].append(val)

# -------------------------
# Aggregate importances
# -------------------------
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

imp_csv_path = "feature_importance_two_params.csv"  # CHANGED: единообразие
imp_agg.to_csv(imp_csv_path, index=False)
print(f"Aggregated permutation importances saved to '{imp_csv_path}'")
print("\nTop features by aggregated permutation importance (preview):")
print(imp_agg.head(25).to_string(index=False))

# -------------------------
# Final fit on full train and report on hold-out test
# -------------------------
pipeline.fit(X_train_all, y_train_all)
p_test = pipeline.predict_proba(X_test_all)
auc_test = roc_auc_score(y_test_all, p_test, multi_class="ovr", average="macro")
print(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}")

# (опционально) отчёт по классам
# y_pred_test = pipeline.predict(X_test_all)
# print(classification_report(y_test_all, y_pred_test))

# -------------------------
# Save CV metrics
# -------------------------
metrics_df = pd.DataFrame(metrics_log).sort_values(["scheme", "fold"])
cv_log_path = "cv_roc_auc_log_two_params.csv"   # CHANGED
metrics_df.to_csv(cv_log_path, index=False)
print(f"CV metrics saved to '{cv_log_path}'")

# -------------------------
# Plot top-20 features
# -------------------------
topk = 20
top_imp = imp_agg.head(topk).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top_imp["feature"], top_imp["importance_mean"])
plt.xlabel("Permutation importance (mean across CV)")
plt.title(f"Top-{topk} features — GradientBoosting")
plt.tight_layout()
plot_path = "feature_importance_top20_two_params.png"  # CHANGED
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Chart saved to '{plot_path}'")

# -------------------------
# Save full text report
# -------------------------
with open("final_results.txt", "w") as f:
    f.write(f"Aggregated permutation importances saved to '{imp_csv_path}'\n")
    f.write("\nAll features by aggregated permutation importance:\n")
    f.write(imp_agg.to_string(index=False))
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}\n")
