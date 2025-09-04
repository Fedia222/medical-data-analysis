#here we gonna choose an optimal model for predict pregnanciesw
#we will use different models and compare their results
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#load the dataset
df = pd.read_csv("patient_data_up.csv", index_col="patient")
print("Dataset loaded successfully.")


#tqarget variable
y = df['earlier_birth']
X = df.drop(columns=['earlier_birth'])


# Preprocess the data
#Convert categorical variables tonumerical using onehot encoding
#First lets get the list of categorical variables

s = df.select_dtypes(include=['object']).columns
categorical_vars = list(s)
numerical_vars = X.select_dtypes(include=['int64', 'float64']).columns
#print("Categorical variables:", categorical_vars)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_vars),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_vars)
    ])



#ours models for testiing

models = {
    'LogReg': LogisticRegression(max_iter=5000),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(max_iter=4000)
}

#pqrqmeters for gridseqrch

params = {
    'LogReg': {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga']
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

#loop for esting models

results = []  #list for results

for name, model in models.items():
    print(f"Training and tuning {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    grid_search = GridSearchCV(pipeline, param_grid=params[name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
    print(f"ROC-AUC Score for {name}: {roc_auc_score(y_test, y_proba)}\n")


    #make a list with our best models:
    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,
        "Best ROC-AUC": roc_auc_score(y_test, y_proba)
    })

#save results to a txt

with open("best_models_results_without_shuffle.txt", "w") as f:
    for res in results:
        f.write(f"Model: {res['Model']}\n")
        f.write(f"Best Params: {res['Best Params']}\n")
        f.write(f"Best ROC-AUC: {res['Best ROC-AUC']:.3f}\n")
        f.write("-" * 50 + "\n")


