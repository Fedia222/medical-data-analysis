import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# load the dataset
df = pd.read_csv("patient_data_up.csv", index_col="patient")
print("Dataset loaded successfully.")

# target and features
y = df['earlier_birth']
X = df.drop(columns=['earlier_birth'])

# feature types
categorical_vars = list(df.select_dtypes(include=['object']).columns)
numerical_vars = X.select_dtypes(include=['int64', 'float64']).columns

# preprocess
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_vars),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_vars)
])

# models (добавил random_state где есть)
models = {
    'LogReg': LogisticRegression(max_iter=5000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),  # у KNN нет random_state
    'MLP': MLPClassifier(max_iter=4000, random_state=42)
}

# grids
params = {
    'LogReg': {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga'],  # ok
        # при желании можно добавить 'model__penalty': ['l2'] (а для 'saga' ещё 'l1','elasticnet')
    },
    'RandomForest': {
        'model__n_estimators': [40, 50, 100, 200, 400],
        'model__max_depth': [None, 10, 20, 30, 40],
        'model__min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'model__n_estimators': [50, 100, 200, 400],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    },
    'SVM': {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf']
    },
    'KNN': {
        'model__n_neighbors': [3, 5, 7],
        'model__weights': ['uniform', 'distance']
    },
    'MLP': {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001]
    }
}

best_models = {}
results = []

# s1tratifed k-fold with shuffle
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# train/test split (stratify и shuffle уже ок)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

for name, model in models.items():
    print(f"\nTraining and tuning {name}...")
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('model', model)])

    grid_search = GridSearchCV(
        pipeline,
        param_grid=params[name],
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
    print(f"ROC-AUC Score for {name}: {roc_auc_score(y_test, y_proba):.3f}\n")

    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,
        "Best ROC-AUC": roc_auc_score(y_test, y_proba)
    })

with open("best_models_results_with_shuffle.txt", "w") as f:
    for res in results:
        f.write(f"Model: {res['Model']}\n")
        f.write(f"Best Params: {res['Best Params']}\n")
        f.write(f"Best ROC-AUC: {res['Best ROC-AUC']:.3f}\n")
        f.write("-" * 50 + "\n")
