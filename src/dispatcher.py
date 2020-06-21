from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import svm
import xgboost as xgb

"""
> This is a model dispatcher file, or basically a mapping of model names to corresponding model objects
> The idea is to keep the code clean, and free of dependencies of a particular model
> Using the technique, I was able to train a lot of models hassle free, and also organize the results
> We are primarily using all families of classification models, starting from 
> Logistic regression
> SVM
> Tree Based Models (CART)
> Ensemble Models (Random Forest, GradientBoostedTrees, XGB)
"""

MODELS = {
    "logistic_regression_c_1e-6" : linear_model.LogisticRegression(C=1e-6),
    "logistic_regression_c_1e-5" : linear_model.LogisticRegression(C=1e-5),
    "logistic_regression_c_1e-4" : linear_model.LogisticRegression(C=1e-4),
    "logistic_regression_c_1e-3" : linear_model.LogisticRegression(C=1e-3),
    "logistic_regression_c_1e-2" : linear_model.LogisticRegression(C=1e-2),
    "logistic_regression_c_1e-1" : linear_model.LogisticRegression(C=1e-1),
    "logistic_regression_c_1" : linear_model.LogisticRegression(C=1),
    "logistic_regression_c_1e1" : linear_model.LogisticRegression(C=1e1),
    "logistic_regression_c_1e2" : linear_model.LogisticRegression(C=1e2),
    "decision_tree" : tree.DecisionTreeClassifier(),
    "decision_tree_max_depth_3" : tree.DecisionTreeClassifier(max_depth=3),
    "decision_tree_max_depth_5" : tree.DecisionTreeClassifier(max_depth=5),
    "decision_tree_max_depth_7" : tree.DecisionTreeClassifier(max_depth=7),
    "decision_tree_max_depth_10" : tree.DecisionTreeClassifier(max_depth=10),
    "decision_tree_max_depth_12" : tree.DecisionTreeClassifier(max_depth=12),
    "svm_c_1e-4" : svm.SVC(C=1e-4),
    "svm_c_1e-3" : svm.SVC(C=1e-3),
    "svm_c_1e-2" : svm.SVC(C=1e-2),
    "svm_c_1e-1" : svm.SVC(C=1e-1),
    "svm_c_1" : svm.SVC(C=1.0),
    "svm_c_1e2" : svm.SVC(C=1e2),
    "svm_c_1e3" : svm.SVC(C=1e3),
    "svm_c_1e4" : svm.SVC(C=1e4),
    "svm_c_1e5" : svm.SVC(C=1e5),
    "svm_c_1e6" : svm.SVC(C=1e6),
    "random_forest_depth_5_estimator_10" : ensemble.RandomForestClassifier(n_estimators=10, max_depth=5),
    "random_forest_depth_5_estimator_100" : ensemble.RandomForestClassifier(n_estimators=100, max_depth=5),
    "random_forest_depth_10_estimator_10" : ensemble.RandomForestClassifier(n_estimators=10, max_depth=10),
    "random_forest_depth_10_estimator_100" : ensemble.RandomForestClassifier(n_estimators=100, max_depth=10),
    "gradient_boosting_depth_1_estimator_10" : ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=1),
    "gradient_boosting_depth_1_estimator_50" : ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=1),
    "gradient_boosting_depth_1_estimator_100" : ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=1),
    "gradient_boosting_depth_3_estimator_10" : ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=3),
    "gradient_boosting_depth_3_estimator_50" : ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=3),
    "gradient_boosting_depth_3_estimator_100" : ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=3),
    "gradient_boosting_depth_5_estimator_10" : ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=5),
    "gradient_boosting_depth_5_estimator_50" : ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=5),
    "gradient_boosting_depth_5_estimator_100" : ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=5),
    "gradient_boosting_depth_7_estimator_10" : ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=7),
    "gradient_boosting_depth_7_estimator_50" : ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=7),
    "gradient_boosting_depth_7_estimator_100" : ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7),
    "gradient_boosting_depth_10_estimator_10" : ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=10),
    "gradient_boosting_depth_10_estimator_50" : ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=10),
    "gradient_boosting_depth_10_estimator_100" : ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=10),
    "xgboost_estimator_depth_1_estimator_10" : xgb.XGBClassifier(n_estimators=10, max_depth=1),
    "xgboost_estimator_depth_1_estimator_50" : xgb.XGBClassifier(n_estimators=50, max_depth=1),
    "xgboost_estimator_depth_1_estimator_100" : xgb.XGBClassifier(n_estimators=100, max_depth=1),
    "xgboost_estimator_depth_3_estimator_10" : xgb.XGBClassifier(n_estimators=10, max_depth=3),
    "xgboost_estimator_depth_3_estimator_50" : xgb.XGBClassifier(n_estimators=50, max_depth=3),
    "xgboost_estimator_depth_3_estimator_100" : xgb.XGBClassifier(n_estimators=100, max_depth=3),
    "xgboost_estimator_depth_5_estimator_10" : xgb.XGBClassifier(n_estimators=10, max_depth=5),
    "xgboost_estimator_depth_5_estimator_50" : xgb.XGBClassifier(n_estimators=50, max_depth=5),
    "xgboost_estimator_depth_5_estimator_100" : xgb.XGBClassifier(n_estimators=100, max_depth=5),
    "xgboost_estimator_depth_7_estimator_10" : xgb.XGBClassifier(n_estimators=10, max_depth=7),
    "xgboost_estimator_depth_7_estimator_50" : xgb.XGBClassifier(n_estimators=50, max_depth=7),
    "xgboost_estimator_depth_7_estimator_100" : xgb.XGBClassifier(n_estimators=100, max_depth=7),
    "xgboost_estimator_depth_10_estimator_10" : xgb.XGBClassifier(n_estimators=10, max_depth=10),
    "xgboost_estimator_depth_10_estimator_50" : xgb.XGBClassifier(n_estimators=50, max_depth=10),
    "xgboost_estimator_depth_10_estimator_100" : xgb.XGBClassifier(n_estimators=100, max_depth=10),
}