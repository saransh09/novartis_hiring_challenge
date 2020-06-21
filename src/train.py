import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from . import dispatcher
import joblib

"""
In this file we define the strategy to train the model

> We will be using environment variables to define the training data, testing data, fold to train,
model etc

> One peculiar thing that I noticed was the severe class imbalance in the dataset. While, I would
have liked generating synthetic data-samples using techniques like SMOTE Analysis, the kind of way
in which I have formulated the problem (as all nominal categorical variables) There is no SMOTE
technique that is available in the open source right now. Therefore, I decided to go forward with 
Random over sampling, however, this step proved out to be really beneficial for me
(When I was not oversampling, the validation accuracy reached as high as 1.0, but the predictions 
were biased towards class 1, which was corrected with the help of oversampling techniques)

> I have used different models from the model dispatcher file, trained them for specific folds and
saved them with proper naming convention in the /models/ folder (with .bin extension)
> Simulataneously, I have also created a .txt file which logs the recall_score per fold
> This helps me to keep track of the results, and compile it later
"""

# Call all the environment variables that we will be using
TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


# Defined the FOLD_MAPPINGS

# Initially I decided to use a 10-Fold cross validation, however, I realised that it was an overkill
# Therefore, I have settled for a 5-Fold validation for now

## FOLD_MAPPINGS FOR 10-FOLD CROSS VALIDATION
# FOLD_MAPPINGS = {
#     0: [1,2,3,4,5,6,7,8,9],
#     1: [0,2,3,4,5,6,7,8,9],
#     2: [0,1,3,4,5,6,7,8,9],
#     3: [0,1,2,4,5,6,7,8,9],
#     4: [0,1,2,3,5,6,7,8,9],
#     5: [0,1,2,3,4,6,7,8,9],
#     6: [0,1,2,3,4,5,7,8,9],
#     7: [0,1,2,3,4,5,6,8,9],
#     8: [0,1,2,3,4,5,6,7,9],
#     9: [0,1,2,3,4,5,6,7,8],
# }


## FOLD MAPPING FOR 5-FOLD CROSS VALIDATION
## I GUESS, THE 10 FOLD CROSS VALIDATION FOR THIS SMALL OF A DATA WAS AN OVERKILL
FOLD_MAPPINGS = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}


if __name__ == '__main__':

    # Load the training and the test dataframes
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    
    # Load all the data (which is not from the fold under consideration)
    train_df = df[df.kfold.isin(FOLD_MAPPINGS.get(FOLD))].reset_index(drop=True)

    # This was the turning point, which improved my accuracies a lot, when I did Random Oversampling
    # As the number of occurances of 0 class or when attack is not taking place were very less
    ###### TRYING IMBLEARN FOR OVER_SAMPLING (OWING TO THE EXCESSIVE IMBALANCE IN THE DATA) #####
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    train_df_x = train_df.drop(['INCIDENT_ID','DATE','MULTIPLE_OFFENSE','kfold'],axis=1)
    train_df_y = train_df['MULTIPLE_OFFENSE']
    train_df, y_res = ros.fit_resample(train_df_x, train_df_y)
    # train_df = pd.concat([train_df_x, train_df_y], axis=1)
    ##############################################################################################

    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ###### TRYING IMBLEARN FOR OVER_SAMPLING (OWING TO THE EXCESSIVE IMBALANCE IN THE DATA) #####
    y_train = y_res.values
    ##############################################################################################

    # y_train = train_df.MULTIPLE_OFFENSE.values
    y_valid = valid_df.MULTIPLE_OFFENSE.values

    # train_df = train_df.drop(["INCIDENT_ID", "DATE", "MULTIPLE_OFFENSE", "kfold"], axis=1)
    valid_df = valid_df.drop(["INCIDENT_ID", "DATE", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    # fetching the classifier to run from the dispatcher file
    clf = dispatcher.MODELS[MODEL]
    # Fit the data on the classifier
    clf.fit(train_df, y_train)
    # Predict the values
    preds = clf.predict(valid_df)
    # From the metrics call recall_score
    recall = metrics.recall_score(y_valid, preds)
    # Name the model
    model_name = f'{MODEL}_label_encoder'
    if not os.path.exists(f'models/{model_name}/'):
        os.mkdir(f'models/{model_name}/')
    # Use joblib to dump the model in the relvant folder
    joblib.dump(clf, f'models/{model_name}/fold_{FOLD}.pkl')
    s = f"{model_name}_fold_{FOLD} : {recall}\n"
    f = open(f"models/{model_name}/results.txt","a")
    f.write(s)
    f.close()