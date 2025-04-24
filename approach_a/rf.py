import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load feature table from CSV
csv_path = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/chord_features.csv"
df = pd.read_csv(csv_path)

# 2) Split into X / y
X = df.drop('label', axis=1).values
y = df['label'].values

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4) Scale features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# 5) Train initial Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 6) Evaluate
y_pred = rf.predict(X_test)
print("=== Initial Random Forest ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7) Hyperparameter search
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=10, cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

print("\n=== Randomized Search Results ===")
print("Best params:", search.best_params_)
print("Best CV accuracy:", search.best_score_)
