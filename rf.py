# 1. import relevant libraties
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 2. Load data
data = pd.read_csv("train.csv")
X = data.drop("target", axis=1)
y = data["target"]

cat_cols = [col for col in X.columns if col.startswith("cat_")]
num_cols = [col for col in X.columns if col.startswith("num_") or col == "id"]

# 3. Split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Model construction
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
)
rf_model.fit(X_train, y_train)

# 5. Model test
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)