import os
import pandas as pd
import joblib
import numpy as np
from scipy import stats

from . import dispatcher

"""
This is the prediction file, where we will use the trained models to perform prediction on the test
dataset
"""

# Use the environment variables
TEST_DATA = './input/test_label_encoding.csv'
MODEL_LIST = [
    'xgboost_estimator_depth_7_estimator_100_label_encoder',
    'xgboost_estimator_depth_7_estimator_50_label_encoder',
    'xgboost_estimator_depth_10_estimator_100_label_encoder',
    'xgboost_estimator_depth_5_estimator_100_label_encoder',
    'xgboost_estimator_depth_5_estimator_50_label_encoder',
    'xgboost_estimator_depth_3_estimator_100_label_encoder',
    'xgboost_estimator_depth_10_estimator_50_label_encoder',
    'gradient_boosting_depth_7_estimator_100_label_encoder',
    'gradient_boosting_depth_10_estimator_100_label_encoder',
    'gradient_boosting_depth_10_estimator_50_label_encoder',
]r', 0.9994734243822796) ('xgboost_estimator_depth_7_estimator_50_label_encoder', 0.9994734147533804) ('xgboost_estimator_depth_10_estimator_100_label_encoder', 0.9994295454880276) ('xgboost_estimator_depth_5_estimator_100_label_encode
def predict():
    # Defined predictions (Where the predictions from all the folds will be saved)
    final_predictions = None
    for model in MODEL_LIST:
        predictions = None
        for FOLD in range(5):
            # print(f"FOLD={FOLD}")
            df = pd.read_csv(TEST_DATA)
            df = df.drop(["INCIDENT_ID", "DATE"],axis=1)
            # Load the classifier which we have selected
            clf = joblib.load(os.path.join(f"models/{model}",f"fold_{FOLD}.pkl"))
            # Store the predictions from dataframe
            preds = clf.predict(df)
            if FOLD==0:
                predictions = preds
            else:
                # Store stack all the predictions together
                predictions = np.vstack((predictions,preds))
        predictions = stats.mode(predictions, axis=0).mode[0]
        if final_predictions is None:
            final_predictions = predictions
        else:
            final_predictions = np.vstack((final_predictions, predictions))
    # Finally from the 5 predictions per entry that we have
    # Select the entry which is most occuring (hence) most confident
    
    res = np.zeros(shape=(1,15903))
    for i in range(10):
        res[0,:] += (11-i)*final_predictions[i,:]

    print(pd.Series(res[0,:]).value_counts())

    res[res<61] = 0
    res[res>=61] = 1
    res = list(res[0,:].astype(np.int16))
    # print(res)
    
    # print(pd.Series(res[0,:]).value_counts())
    # print("res.shape = ", res.shape)    

    # final_predictions = stats.mode(final_predictions,axis=0).mode[0]
    # Restructure the test data to save the submission file
    df = pd.read_csv(TEST_DATA)
    # df['MULTIPLE_OFFENSE'] = final_predictions
    df['MULTIPLE_OFFENSE'] = res
    submission = df[['INCIDENT_ID','MULTIPLE_OFFENSE']]
    submission.to_csv(f'./submissions/ensemble_10.csv',index=False)

if __name__ == '__main__':
    predict()