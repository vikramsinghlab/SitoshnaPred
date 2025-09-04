################################################################################################################
##########################################___SITOSHNAPRED_BINARY___#############################################
################################################################################################################


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Sitoshna_input_binary.csv")

# Separate features and labels
X = data.drop('Labels', axis=1)
y = data['Labels']

# 5-Fold Stratified CV
kf = StratifiedKFold(n_splits=5, shuffle=True)

# Metrics storage
accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

# For ROC Curve plotting
tprs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 7))

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': 0,
}

fold = 1
for train_index, test_index in kf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
    dtest = lgb.Dataset(X_test, label=y_test, categorical_feature='auto')

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        num_boost_round=10,
    )

    y_pred_proba = model.predict(X_test)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    aucs.append(auc_score)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    plt.plot(fpr, tpr, lw=1.5, label=f'Fold {fold} AUC = {auc_score:.4f}')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC ROC:   {auc_score:.4f}")
    fold += 1

# Plot Mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='black', linestyle='-', lw=2,
         label=f'Mean ROC (AUC = {np.mean(aucs):.4f})')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('5-Fold Cross-Validated ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
# ✅ Save plot in HD (300 DPI)
plt.savefig("5fold_ROC_plot.png", dpi=1200)  # You can change the file name or format (e.g., .pdf)

# Show plot
plt.show()

# Print average metrics
print("\n=== Average Metrics Over 5 Folds ===")
print(f"Average Accuracy:  {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall:    {np.mean(recalls):.4f}")
print(f"Average F1-score:  {np.mean(f1_scores):.4f}")
print(f"Average AUC ROC:   {np.mean(aucs):.4f}")

import torch

# Save the LightGBM model
torch.save(gbm, 'lightgbm_binary.pth')


################################################################################################################
#############################################___SHAP ANALYSIS___###############################################
################################################################################################################

import shap
explainer = shap.TreeExplainer(model =model)
shap_values= explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], features=X_test, max_display=15)


################################################################################################################
################################################################################################################

