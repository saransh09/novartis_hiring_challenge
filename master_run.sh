# This is the master_run.sh file which used the run.sh shell script
# To train the data on different models


# # Logistic Regression models with parameter changes
# bash run.sh logistic_regression_c_1e-6
# bash run.sh logistic_regression_c_1e-5
# bash run.sh logistic_regression_c_1e-4
# bash run.sh logistic_regression_c_1e-3
# bash run.sh logistic_regression_c_1e-2
# bash run.sh logistic_regression_c_1e-1
# bash run.sh logistic_regression_c_1
# bash run.sh logistic_regression_c_1e1
# bash run.sh logistic_regression_c_1e2


#### Using Tree based classifiers ####
# bash run.sh decision_tree
# bash run.sh decision_tree_max_depth_3
# bash run.sh decision_tree_max_depth_5
# bash run.sh decision_tree_max_depth_7
# bash run.sh decision_tree_max_depth_10
# bash run.sh decision_tree_max_depth_12


### SVM Based decision Classifiers ####
# bash run.sh svm_c_1e-4
# bash run.sh svm_c_1e-3
# bash run.sh svm_c_1e-2
# bash run.sh svm_c_1e-1
# bash run.sh svm_c_1
# bash run.sh svm_c_1e2
# bash run.sh svm_c_1e3
# bash run.sh svm_c_1e4

#
# # ###############################
# # ###TAKING WAY TOO MUCH TIME####
# # bash run.sh svm_c_1e5
# # bash run.sh svm_c_1e6
# # ###TAKING WAY TOO MUCH TIME####
# # ################################
#

############ Random forest models ##############
# bash run.sh random_forest_depth_5_estimator_10
# bash run.sh random_forest_depth_10_estimator_10
# bash run.sh random_forest_depth_5_estimator_100
# bash run.sh random_forest_depth_10_estimator_100

############ Gradient Boosted Trees ##############
# bash run.sh gradient_boosting_depth_1_estimator_10
# bash run.sh gradient_boosting_depth_1_estimator_50
# bash run.sh gradient_boosting_depth_1_estimator_100
# bash run.sh gradient_boosting_depth_3_estimator_10
# bash run.sh gradient_boosting_depth_3_estimator_50
# bash run.sh gradient_boosting_depth_3_estimator_100
# bash run.sh gradient_boosting_depth_5_estimator_10
# bash run.sh gradient_boosting_depth_5_estimator_50
# bash run.sh gradient_boosting_depth_5_estimator_100
# bash run.sh gradient_boosting_depth_7_estimator_10
# bash run.sh gradient_boosting_depth_7_estimator_50
# bash run.sh gradient_boosting_depth_7_estimator_100
# bash run.sh gradient_boosting_depth_10_estimator_10
# bash run.sh gradient_boosting_depth_10_estimator_50
# bash run.sh gradient_boosting_depth_10_estimator_100

###################### XGBoost #########################
# bash run.sh xgboost_estimator_depth_1_estimator_10
# bash run.sh xgboost_estimator_depth_1_estimator_50
# bash run.sh xgboost_estimator_depth_1_estimator_100
# bash run.sh xgboost_estimator_depth_3_estimator_10
# bash run.sh xgboost_estimator_depth_3_estimator_50
# bash run.sh xgboost_estimator_depth_3_estimator_100
# bash run.sh xgboost_estimator_depth_5_estimator_10
# bash run.sh xgboost_estimator_depth_5_estimator_50
# bash run.sh xgboost_estimator_depth_5_estimator_100
# bash run.sh xgboost_estimator_depth_7_estimator_10
# bash run.sh xgboost_estimator_depth_7_estimator_50
# bash run.sh xgboost_estimator_depth_7_estimator_100
# bash run.sh xgboost_estimator_depth_10_estimator_10
# bash run.sh xgboost_estimator_depth_10_estimator_50
# bash run.sh xgboost_estimator_depth_10_estimator_100

# python -m src.get_best_result

export BEST_K=10
python -m src.get_m_best_models

# export BEST_MODEL=xgboost_estimator_depth_7_estimator_100_label_encoder
# export TEST_DATA=input/test_label_encoding.csv
# export SAMPLE_SUBMISSION=input/sample_submission.csv
# python -m src.predict