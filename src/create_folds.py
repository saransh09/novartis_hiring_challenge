import pandas as pd
from sklearn import model_selection

"""
Basic Idea:

Whenever we are tackling data science problems, having a strong validation strategy is the key
to actually being able to evaluate our models efficiently.
Therefore, in this script, I have used the perform Stratified K-Fold sampling for cross
validation. The reason why stratified sampling is important is that we want to ensure that the
ratios of the classes are preserved in all the splits.
"""

if __name__ == '__main__':
    # Reading the training data
    df = pd.read_csv("../input/Train.csv")
    # Creating a column for the k_folds that we will be using
    df['kfold'] = -1
    # Shuffle the dataframe once, before sampling
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize the StratifiedKFold() sampler
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    # Finally use the sampler to actually create difference folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.MULTIPLE_OFFENSE.values)):
        print(f"fold {fold}")
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    # Save the train_folds in the relevant input folder
    df.to_csv("../input/train_folds.csv", index=False)