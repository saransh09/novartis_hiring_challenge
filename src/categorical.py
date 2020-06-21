# This script is based on a video on applied machine learning by Abhishek Thakur
# Link to original : https://github.com/abhishekkrthakur/mlframework/blob/master/src/categorical.py

from sklearn import preprocessing

"""
From the dataset, I could make out, that most the features were categorical, and even if they were
not categorical, most of the times, it was inconclusive for me to decide whether they can be continuous
or even ordinal categorical variables (as no real world significance of the variables is given)

Therefore, it only makes sense that I use consider all the features (X_<num>) features as categorical

To handle categorical features, some of the standard techniques include one_hot_encoding, label_encoding,
label_binarizers. And some advanced techniques like categorical embeddings etc. (However we will jump)
to that only if we need to

So, this is a helper class / running script to handle the categorical features. 

Right now, I have used Label Encoding to handle the features, One thing is important to note here
that there are some entries which are present in the test set and not in the training set, therefore
I concatenated the train and the test set first, created label encodings and then split them back again
"""

class CategoricalFeatures:

    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """init function for the generic class to handle categorical features

        Args:
            df (pandas dataframe): The data to be encoded
            categorical_features (list(string)): All the features to be encoded
            encoding_type (string): The encoding that we need to perform
            handle_na (bool, optional): Whether to handle na values or not. Defaults to False.
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    
    def _label_binarizer(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:,j]
            self.binary_encoders[c] = lbl
        return self.output_df


    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
    

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarizer()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
    

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-9999999")

        if self.enc_type=="label":
            for c,lbl in self.label_encoders.items():
                dataframe.loc[:,c] = lbl.transform(dataframe[c].values)
            return dataframe
        
        elif self.enc_type=="binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:,j]
            return dataframe
        
        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        
        else:
            raise Exception("Encoding type not understood")
    

if __name__ == '__main__':
    import pandas as pd
    import joblib
    df = pd.read_csv("../input/train_folds.csv")
    original_train_cols = df.columns
    # As we have to merge the training and test set to create label encodings
    # Creating a column 'tr' which indicates whether the entry is from the training set or not
    df['tr'] = 1
    df_test = pd.read_csv("../input/Test.csv")
    original_test_cols = df_test.columns
    # Create pseudo labels to match the structure of the training dataframe 
    df_test['kfold'] = -1
    df_test['MULTIPLE_OFFENSE'] = -1
    df_test['tr'] = -1
    # Finally concatenate the dataframe
    df = pd.concat([df, df_test]).reset_index(drop=True)
    print(f"The total number of entries are : {len(df)}")
    print(f"The total number of entries with tr==-1 : {len(df[df['tr']==-1])}")
    cat_cols = [c for c in df.columns if c not in ["INCIDENT_ID", "DATE", "MULTIPLE_OFFENSE", "kfold", "tr"]]
    # Use these categorical column names to call the class and perform label encoding transform
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cat_cols,
                                    encoding_type="label",
                                    handle_na=True)
    df_transformed = cat_feats.fit_transform()
    X = df_transformed.loc[df_transformed.tr==1,:]
    X_test = df_transformed.loc[df_transformed.tr==-1,:]
    # Finally retrieve back the training and test dataframes
    df = X[original_train_cols]
    df_test = X_test[original_test_cols]
    df.to_csv("../input/train_label_endoing.csv", index=False)
    df_test.to_csv("../input/test_label_encoding.csv", index=False)
    # Dump the label_encoders in input folder itself, should a need arise to reverse transform
    # in the future
    joblib.dump(cat_feats.label_encoders, "../input/label_encoders.bin")