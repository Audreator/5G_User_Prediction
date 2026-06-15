# Logical regression
# import relevant libraties
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from visualization import roc
# Load data
data = pd.read_csv('train.csv')
x = data.drop(['id', 'target'], axis=1)
y = data['target']
# Split the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Characteristic standardization
scaler = StandardScaler()
x_train_lr = scaler.fit_transform(x_train)
x_test_lr = scaler.transform(x_test)
# Model construction
model = LogisticRegression(
    max_iter=1000,
)
model.fit(x_train_lr, y_train)
y_pred_lr = model.predict_proba(x_test_lr)[:,1]
# Model test
auc_lr = roc_auc_score(y_test, y_pred_lr)
print("logical regression AUC:" , auc_lr);
# Random forest
# Model construction
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    n_jobs= -1 # 多线程
)
model.fit(x_train, y_train)
# Model test
y_pred_rf = model.predict_proba(x_test)[:,1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
print("random forest AUC:" , auc_rf);
# lightGBM
# Model construction
model = lgb.LGBMClassifier(
    num_iterations=500,
    learning_rate=0.05,
    num_leaves=64,
)
model.fit(x_train, y_train)
# Model test
y_pred_lgb = model.predict_proba(x_test)[:,1]
auc_lgb = roc_auc_score(y_test, y_pred_lgb)
print("lightGBM AUC:" , auc_lgb);
# Weighted integration
# Logical regression、Random forest、LightGBM
total_auc_1 = auc_lr + auc_rf + auc_lgb
weight_lr = auc_lr / total_auc_1
weight_rf = auc_rf / total_auc_1
weight_lgb = auc_lgb / total_auc_1
y_pred_1 = weight_lr * y_pred_lr + weight_rf * y_pred_rf + weight_lgb * y_pred_lgb
auc_1 = roc_auc_score(y_test, y_pred_1)
print("Weighted integrationLogical regression、Random forest、LightGBM AUC:" , auc_1);
# Random forest、LightGBM
total_auc_2 = auc_rf + auc_lgb
weight_rf = auc_rf / total_auc_2
weight_lgb = auc_lgb / total_auc_2
y_pred_2 = weight_rf * y_pred_rf + weight_lgb * y_pred_lgb
auc_2 = roc_auc_score(y_test, y_pred_2)
print("Weighted integrationRandom forest、LightGBM AUC:" , auc_2);
# Visualization
roc(y_test, {'logical regression':y_pred_lr,'random forest':y_pred_rf, 'lightGBM':y_pred_lgb}, 'roc.png')