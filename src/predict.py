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
TEST_DATA = os.environ.get("TEST_DATA")
BEST_MODEL = os.environ.get("BEST_MODEL")
SAMPLE_SUBMISSION = os.environ.get("SAMPLE_SUBMISSION")

def predict():
    # Defined predictions (Where the predictions from all the folds will be saved)
    predictions = None
    for FOLD in range(5):
        print(f"FOLD={FOLD}")
        df = pd.read_csv(TEST_DATA)
        df = df.drop(["INCIDENT_ID", "DATE"],axis=1)
        # Load the classifier which we have selected
        clf = joblib.load(os.path.join(f"models/{BEST_MODEL}",f"fold_{FOLD}.pkl"))
        # Store the predictions from dataframe
        preds = clf.predict(df)
        if FOLD==0:
            predictions = preds
        else:
            # Store stack all the predictions together
            predictions = np.vstack((predictions,preds))
    # Finally from the 5 predictions per entry that we have
    # Select the entry which is most occuring (hence) most confident
    final_predictions = stats.mode(predictions,axis=0).mode[0]
    # Restructure the test data to save the submission file
    df = pd.read_csv(TEST_DATA)
    df['MULTIPLE_OFFENSE'] = final_predictions
    submission = df[['INCIDENT_ID','MULTIPLE_OFFENSE']]
    submission.to_csv(f'./submissions/{BEST_MODEL}.csv',index=False)

if __name__ == '__main__':
    predict()