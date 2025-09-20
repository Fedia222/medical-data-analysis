# =========================
# Fast & Safe pipeline: HistGradientBoosting + CV + Permutation Importance
# =========================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# -------------------------
# Reproducibility
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------
# Load data
# -------------------------
DATA_PATH = "patient_data_two_cc.csv"   # <-- при необходимости замените
df = pd.read_csv(DATA_PATH, index_col="patient")

# -------------------------
# Target / features
# -------------------------
y = df["earlier_birth"]

# Выявляем и запоминаем потенциальные колонки-утечки (контрольные копии таргета)
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

print("Контрольные (исключаем из обучения):", leakage_cols)

# X_clean — только «чистые» фичи (без таргета и без control_*)
X_clean = df.drop(columns=["earlier_birth"] + leakage_cols, errors="ignore")

# (Опционально) полный набор с control_* для последующего анализа:
X_with_controls = df.drop(columns=["earlier_birth"], errors="ignore")

# -------------------------
# DTypes
# -------------------------
categorical_vars = list(X_clean.select_dtypes(include=["object"]).columns)
numerical_vars   = list(X_clean.select_dtypes(include=["int64", "float64"]).columns)

# -------------------------
# Preprocessing
# OneHotEncoder: плотный выход; ограничим кардинальность (если доступно) для ускорения
# -------------------------
try:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,  # sklearn >= 1.2
        max_categories=20     # объединит редкие уровни в 'other' (если поддерживается)
    )
except TypeError:
    # старые версии sklearn
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
        # без max_categories, если недоступно
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_vars),
        ("cat", ohe, categorical_vars),
    ],
    remainder="drop"
)

# -------------------------
# Model: HistGradientBoosting — быстро и с early stopping
# -------------------------
gb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_SEED
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gb)
])

# -------------------------
# Hold-out split
# -------------------------
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_clean, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# -------------------------
# Helpers
# -------------------------
def get_feature_names_after_fit(fitted_preprocessor) -> np.ndarray:
    """Имена фич после ColumnTransformer (num + OHE cat)."""
    num_names = np.array(numerical_vars, dtype=object)

    cat_step = fitted_preprocessor.named_transformers_.get("cat", None)
    if cat_step is not None and hasattr(cat_step, "get_feature_names_out"):
        cat_names = cat_step.get_feature_names_out(categorical_vars)
    else:
        # на очень старых версиях
        try:
            cat_names = np.array(cat_step.get_feature_names(categorical_vars))
        except Exception:
            cat_names = np.array([], dtype=object)

    return np.concatenate([num_names, cat_names])

def multiclass_auc(estimator, Xmat, y_true):
    """Скорер для permutation_importance: macro ROC-AUC (OvR) многокласса."""
    proba = estimator.predict_proba(Xmat)  # (n_samples, n_classes)
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")

# -------------------------
# CV + Permutation Importance (ускорённая конфигурация)
# -------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
imp_by_feature = defaultdict(list)
metrics_log = []

for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train_all, y_train_all), start=1):
    X_tr, X_val = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
    y_tr, y_val = y_train_all.iloc[tr_idx], y_train_all.iloc[va_idx]

    pipeline.fit(X_tr, y_tr)

    # Валидационный AUC (многокласс)
    p_val = pipeline.predict_proba(X_val)
    auc_val = roc_auc_score(y_val, p_val, multi_class="ovr", average="macro")
    metrics_log.append({"scheme": "SKF5_rs42", "fold": fold_idx, "roc_auc_val": auc_val})

    # Permutation importance на валидации
    fitted_prep = pipeline.named_steps["preprocessor"]
    est = pipeline.named_steps["model"]
    X_val_trans = fitted_prep.transform(X_val)   # dense
    feat_names = get_feature_names_after_fit(fitted_prep)

    pi = permutation_importance(
        estimator=est,
        X=X_val_trans,
        y=y_val,
        scoring=multiclass_auc,
        n_repeats=5,      # быстрее
        n_jobs=-1,        # параллельно
        random_state=RANDOM_SEED
    )

    for name, val in zip(feat_names, pi.importances_mean):
        imp_by_feature[name].append(val)

# -------------------------
# Aggregate importances & save artifacts
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

imp_csv_path = "feature_importance_two_params.csv"
imp_agg.to_csv(imp_csv_path, index=False)
print(f"Permutation importances saved to '{imp_csv_path}'")

metrics_df = pd.DataFrame(metrics_log).sort_values(["scheme", "fold"])
cv_log_path = "cv_roc_auc_log_two_params.csv"
metrics_df.to_csv(cv_log_path, index=False)
print(f"CV metrics saved to '{cv_log_path}'")

# -------------------------
# Final model on full train + test AUC
# -------------------------
pipeline.fit(X_train_all, y_train_all)
p_test = pipeline.predict_proba(X_test_all)
auc_test = roc_auc_score(y_test_all, p_test, multi_class="ovr", average="macro")
print(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}")

# (опционально) можно посмотреть классификационный отчёт:
# y_pred_test = pipeline.predict(X_test_all)
# print(classification_report(y_test_all, y_pred_test))

# -------------------------
# Plot Top-20 features
# -------------------------
topk = 20
top_imp = imp_agg.head(topk).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top_imp["feature"], top_imp["importance_mean"])
plt.xlabel("Permutation importance (mean across CV)")
plt.title(f"Top-{topk} features — HistGradientBoosting")
plt.tight_layout()
plot_path = "feature_importance_top20_two_params.png"
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Chart saved to '{plot_path}'")

# -------------------------
# Save full text report
# -------------------------
with open("final_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Permutation importances saved to '{imp_csv_path}'\n")
    f.write("\nAll features by aggregated permutation importance:\n")
    f.write(imp_agg.to_string(index=False))
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write(f"\nFinal hold-out ROC-AUC (macro, ovr) on test: {auc_test:.4f}\n")

# -------------------------
# (Диагностика) Насколько control_* совпадают с таргетом
# -------------------------
if leakage_cols:
    print("\n[Diagnostics] control_* vs earlier_birth (share of exact matches):")
    for c in leakage_cols:
        same = (df[c] == y).mean()
        print(f"{c:30s}  same={same:.2%}")
