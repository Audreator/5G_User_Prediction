# 1. import relevant libraties
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 2. Load data
data = pd.read_csv('train.csv')
x = data.drop(['id', 'target'], axis=1)
y = data['target']

# 3. Split the data set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

# Characteristic standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Model construction
model = LogisticRegression(
    max_iter=1000,
)
model.fit(x_train, y_train)

# 5. Model test
y_pred = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(auc)