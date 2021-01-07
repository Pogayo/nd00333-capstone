
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import naive_bayes


import argparse
import os
import joblib
import pickle
from azureml.core import Workspace, Dataset
from azureml.core.run import Run 
from azureml.data.dataset_factory import TabularDatasetFactory

from xgboost import XGBClassifier


def clean_data(data):
   
    # Clean and encode data
    train_df = data.to_pandas_dataframe().dropna()

    train_df["label"]=train_df["label"].apply(int)


    train_df.describe()

    train_df['word_count'] =train_df["text"].apply(lambda x: len(str(x).split(" ")))
    train_df['char_count'] = train_df["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    train_df['sentence_count'] = train_df["text"].apply(lambda x: len(str(x).split(".")))
    train_df['avg_word_length'] = train_df['char_count'] / train_df['word_count']
    train_df['avg_sentence_lenght'] =train_df['word_count'] / train_df['sentence_count']
    train_df.head()

    train_df["label"]=train_df["label"].apply(lambda x:x+1)

    vectorizer=TfidfVectorizer(strip_accents='ascii', sublinear_tf=True, min_df=2, max_df=0.5)

    y=train_df.label

    train_df.drop("ID", axis=1, inplace=True)
    

    X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.20, random_state=42, stratify=y)

    vectorizer.fit(train_df.text)
    sparse_X_train=vectorizer.transform(X_train.text)
    sparse_X_val=vectorizer.transform(X_val.text)


    return  sparse_X_train, sparse_X_val, y_train, y_val




run = Run.get_context()
ws=run.experiment.workspace

   
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument('--n_estimators', type=int, default=100, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_depth', type=int, default=6, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    run.log("Number of estimators:", np.float(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))
    
    # TODO: Create TabularDataset
    
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    dataset.to_pandas_dataframe()
   

    X_train, X_test, y_train, y_test= clean_data(dataset)

    model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth).fit(X_train, y_train)
    
    #saving the model
    os.makedirs("outputs", exist_ok=True)
    filename = 'outputs/model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()