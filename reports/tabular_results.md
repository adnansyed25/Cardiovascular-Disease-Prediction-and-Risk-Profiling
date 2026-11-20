# Clinical model results

Train: 54,848 patients | Test: 13,713 patients (stratified 80/20 split, 5-fold CV for tuning).

| model | cv_roc_auc | train_accuracy | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.802 | 0.744 | 0.733 | 0.752 | 0.686 | 0.718 | 0.804 |
| Hist Gradient Boosting | 0.801 | 0.741 | 0.733 | 0.750 | 0.689 | 0.718 | 0.803 |
| Random Forest | 0.800 | 0.748 | 0.731 | 0.753 | 0.679 | 0.714 | 0.802 |
| k-Nearest Neighbours | 0.796 | 0.735 | 0.729 | 0.754 | 0.671 | 0.710 | 0.797 |
| Logistic Regression | 0.791 | 0.728 | 0.725 | 0.751 | 0.666 | 0.706 | 0.791 |
| Naive Bayes | 0.784 | 0.721 | 0.718 | 0.769 | 0.615 | 0.684 | 0.783 |

Best model (by CV ROC-AUC): **XGBoost**. Top features: ap_hi, age_years, cholesterol, mean_arterial_pressure, bmi.
