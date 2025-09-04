################################################################################################################
##########################################___SITOSHNAPRED_TERNARY___############################################
################################################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import pandas as pd
import numpy as np
import lightgbm as lgb

# Load dataset
data = pd.read_csv("Sitoshna_input_three_class.csv")

# Separate features and labels
X = data.drop('Labels', axis=1)
y = data['Labels'].astype(int)

# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True)

# To store metrics
accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

fold = 1
for train_index, test_index in skf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # LightGBM dataset
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    # Parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'num_class': 3,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    }

    # Train model
    gbm = lgb.train(
        params,
        dtrain,
        num_boost_round=10
    )

    # Predict probabilities
    y_pred_proba = gbm.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Classification report
    #print(classification_report(y_test, y_pred, digits=4))

    # Scores
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # One-hot encode true labels for AUC
    y_test_onehot = np.zeros((y_test.size, y_pred_proba.shape[1]))
    y_test_onehot[np.arange(y_test.size), y_test.values] = 1

    auc = roc_auc_score(y_test_onehot, y_pred_proba, multi_class='ovr')

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")

    # Save fold metrics
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    aucs.append(auc)

    fold += 1

# Average performance
print("\n=== Average Performance Across 5 Folds ===")
print(f"Avg Accuracy:  {np.mean(accuracies):.4f}")
print(f"Avg Precision: {np.mean(precisions):.4f}")
print(f"Avg Recall:    {np.mean(recalls):.4f}")
print(f"Avg F1-score:  {np.mean(f1_scores):.4f}")
print(f"Avg AUC:       {np.mean(aucs):.4f}")


import torch

# Save the LightGBM model
torch.save(gbm, 'lightgbm_ternary.pth')


################################################################################################################
#############################################___SHAP ANALYSIS___###############################################
################################################################################################################

import shap
explainer = shap.TreeExplainer(model =gbm)
shap_values= explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], features=X_test, max_display=15)


################################################################################################################
################################################################################################################


