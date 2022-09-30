"""
Holdout test set comparison

Compare the final classifier to other network architectures.
"""
from typing import List, Tuple

import itertools
import argparse
import logging
import json
import pickle

import pandas as pd
import numpy as np

import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.svm
import sklearn.neighbors
import sklearn.linear_model

import tensorflow as tf
import tensorflow.keras as keras

# LABEL_MAP is the mapping from class names to class indices.
LABEL_MAP = {'Thermo': 0, 'Psychro': 1}

# RANDOM_SEED is the initial RNG state used to divide the dataset into training-test sets
# and cross-validation folds.
RANDOM_SEED = 1

# HOLDOUT_SET_SIZE is the ratio of points to set aside for the final holdout test set (unused in this experiment.)
# In our experiment, we set aside 10% of the points for the holdout set.
HOLDOUT_SET_SIZE = 0.1

def model_report(name, model, X_holdout_t, y_holdout):
    holdout_predictions = np.round(model.predict(X_holdout_t))
    report = sklearn.metrics.classification_report(y_holdout, holdout_predictions, output_dict=True)
    report['classifier'] = name
    return pd.json_normalize(report, sep='-')

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    dataset = pd.read_csv(args.datapath, names=(
        "Classification", "Label", "Sequence"))

    X, y = dataset['Sequence'], dataset['Label'].map(LABEL_MAP).to_numpy()
    X_cv, X_holdout, y_cv, y_holdout = sklearn.model_selection.train_test_split(
        X, y, shuffle=True, test_size=HOLDOUT_SET_SIZE, random_state=RANDOM_SEED)

    with open(args.pipeline, 'rb') as pipeline_file:
        pipeline = pickle.load(pipeline_file)

    X_cv_t = pipeline.transform(X_cv).toarray()
    X_holdout_t = pipeline.transform(X_holdout).toarray()
    
    our_model = keras.models.load_model(args.model)
    our_report = model_report('ours', our_model, X_holdout_t, y_holdout)

    rf_model = sklearn.ensemble.RandomForestClassifier(max_depth=3)
    rf_model.fit(X_cv_t, y_cv)
    rf_report = model_report('rf', rf_model, X_holdout_t, y_holdout)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X_cv_t, y_cv)
    lr_report = model_report('lr', lr_model, X_holdout_t, y_holdout)    

    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(X_cv_t, y_cv)
    knn_report = model_report('knn', knn_model, X_holdout_t, y_holdout)

    svm_model = sklearn.svm.SVC()
    svm_model.fit(X_cv_t, y_cv)
    svm_report = model_report('svm', svm_model, X_holdout_t, y_holdout)

    df = pd.concat([
        our_report, rf_report, lr_report, svm_report, knn_report
    ])

    logging.info("Saving results to " + args.results)

    df.to_csv(args.results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="dataset/database-pdb.csv",
                        help="The path to the .CSV with the experiment data.")
    parser.add_argument("--results", default="final-results.csv",
                        help="Where to save the results of the classifier comparison.")
    parser.add_argument("--pipeline", default="psychornot-pipeline",
                        help="Where to find the preprocessing pipeline for the model.")
    parser.add_argument("--model", default="psychornot-ensemble",
                        help="Where to find the ensemble of the best performing models.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log results to the console.")
    args = parser.parse_args()
    main(args)