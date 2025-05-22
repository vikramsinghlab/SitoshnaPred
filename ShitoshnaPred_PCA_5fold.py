from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import early_stopping, log_evaluation
import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("input_sitoshnapred.csv")

# Separate features and labels
X = data.drop('Labels', axis=1)
y = data['Labels']

# Initialize cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
acc_scores, prec_scores, rec_scores, f1_scores, auc_scores = [], [], [], [], []

# For SHAP summary aggregation (optional)
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
    dtest = lgb.Dataset(X_test, label=y_test, categorical_feature='auto')

    params =     'objective': 'binary',  # Use 'multiclass' if you have more than two classes
    'metric': 'auc',  # Use 'multi_logloss' for multiclass classification
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'learning_rate': 0.05,  # Adjust for better performance
    'num_leaves': 31,  # Controls model complexity
    'feature_fraction': 0.8,  # Random feature selection
    'bagging_fraction': 0.8,  # Random row selection
    'bagging_freq': 5,  # Perform bagging every 5 iterations
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1,  # L2 regularization
    'verbose': 0,  # Suppress logs
}

    # Train with callbacks for early stopping and log suppression
    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        num_boost_round=10,
        callbacks=[
            early_stopping(stopping_rounds=10),
            log_evaluation(period=0)
        ]
    )

    y_pred = gbm.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    acc_scores.append(accuracy_score(y_test, y_pred_binary))
    prec_scores.append(precision_score(y_test, y_pred_binary))
    rec_scores.append(recall_score(y_test, y_pred_binary))
    f1_scores.append(f1_score(y_test, y_pred_binary))
    auc_scores.append(roc_auc_score(y_test, y_pred))

    print(f"Accuracy: {acc_scores[-1]:.4f}")
    print(f"Precision: {prec_scores[-1]:.4f}")
    print(f"Recall: {rec_scores[-1]:.4f}")
    print(f"F1-score: {f1_scores[-1]:.4f}")
    print(f"AUC ROC: {auc_scores[-1]:.4f}")

    # SHAP for last fold only
    if fold == 5:
        explainer = shap.TreeExplainer(gbm)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values[1], features=X_test, max_display=15)

# Print average metrics
print("\nAverage metrics over 5 folds:")
print(f"Accuracy: {np.mean(acc_scores):.4f}")
print(f"Precision: {np.mean(prec_scores):.4f}")
print(f"Recall: {np.mean(rec_scores):.4f}")
print(f"F1-score: {np.mean(f1_scores):.4f}")
print(f"AUC ROC: {np.mean(auc_scores):.4f}")
