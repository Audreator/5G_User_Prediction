# 5G_User_Prediction
Use logistic regression, random forest, and LightGBM models to predict whether each sample in the test set is a 5G user.
# 5G User Prediction

Predict whether a customer is a 5G user based on telecommunication data (call, data usage, plan, region, etc.).

## Models Compared
- Logistic Regression (baseline)
- Random Forest (bagging)
- LightGBM (boosting)

## Key Results (AUC)

| Model | Best AUC |
|-------|----------|
| Logistic Regression | 0.8351 |
| Random Forest | 0.9108 |
| LightGBM | **0.9233** |
| RF+LGB Ensemble | 0.9231 |
| LR+RF+LGB Ensemble | 0.9173 |

- LightGBM alone outperformed all ensembles, showing that more models ≠ better performance.

## Tuning Highlights
- **LightGBM**: increased `num_iterations` to 500, lowered `learning_rate` to 0.05, set `num_leaves` = 64.
- **Random Forest**: increased `n_estimators` to 300, limited `max_depth` to 15.
- All runs averaged over 5 different random seeds to avoid chance.

## Visualizations
- ROC curve & feature importance (top 10) included.
