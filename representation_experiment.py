"""
PsychOrNot - Experiment 1

"""

from typing import List, Tuple

import itertools
import argparse
import logging
import json

import pandas as pd
import numpy as np

import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics

# HYPERPARAMETER_OPTIONS contain the hyperparameter options to do a grid search over:
HYPERPARAMETER_OPTIONS = {
    "ngram_range": [(1, 1), (2, 2), (3, 3), (4, 4), (1, 3)], # Which set of n-grams to include
    "normalization": ["none", "tf", "tfidf"] # The normalization to apply 
}

# AMINO_ACIDS is the vocabulary used to construct the features.
# These correspond to the FASTA format symbols, except for
# pyrrolysine and selenocysteine (which do not appear in the dataset) 
# and special characters, which are ignored.
AMINO_ACIDS = "AGSTNQVILMFYWHPKREDC"

# LABEL_MAP is the mapping from class names to class indices.
LABEL_MAP = {'Thermo': 0, 'Psychro': 1}

# RANDOM_SEED is the initial RNG state used to divide the dataset into training-test sets
# and cross-validation folds.
RANDOM_SEED = 1

# HOLDOUT_SET_SIZE is the ratio of points to set aside for the final holdout test set (unused in this experiment.)
# In our experiment, we set aside 10% of the points for the holdout set.
HOLDOUT_SET_SIZE = 0.1

# CROSSVALIDATION_FOLDS is the number of folds to use.
CROSSVALIDATION_FOLDS = 10

# To ensure features appear in the same order,
# we provide a helper to compute the vocabulary for the n-gram representation:
def feature_names_for_ngram_range(vocabulary: str, ngram_range: Tuple[int, int]) -> List[str]:
    """
    feature_names_for_ngram_range returns the set of possible n-grams
    for the given vocabulary and n in (ngram_range)

    >>> feature_names_for_ngram_range("ab", 1, 2)
    ['a', 'b', 'aa', 'ab', 'ba', 'bb']
    """
    features = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        features += [''.join(c)
                     for c in itertools.product(vocabulary, repeat=n)]
    return features


def make_classifier():
    """
    make_classifier returns the classifier used to compare the representations.
    """
    return sklearn.ensemble.RandomForestClassifier(max_depth=3)


def hyperparameter_grid(choices: dict) -> dict:
    """
    hyperparameter_grid returns the list of possible choices of the hyperparameters
    from the dictionary in (choices).
    """
    return [dict(zip(choices, x)) for x in itertools.product(*choices.values())]


def run_sweep(sweep: dict, X_train: pd.Series, y_train: pd.Series, X_validation: pd.Series, y_validation: pd.Series) -> dict:
    """
    run_sweep runs one fold of the experiment for a given hyperparameter configuration.
    """
    results = {}
    logging.debug(f"Evaluating with hyperparameters: {sweep}")
    transform_steps = [('vectorize', sklearn.feature_extraction.text.CountVectorizer(
        analyzer='char',
        lowercase=False,
        ngram_range=sweep["ngram_range"],
        vocabulary=feature_names_for_ngram_range(
            AMINO_ACIDS, sweep["ngram_range"])
    ))]
    if sweep['normalization'] != 'none':
        transform_steps.append(('normalize', sklearn.feature_extraction.text.TfidfTransformer(
            use_idf=(sweep['normalization'] == 'tfidf')
        )))
    pipeline = sklearn.pipeline.Pipeline(transform_steps)
    X_train_t = pipeline.fit_transform(X_train)

    clf = make_classifier()
    clf.fit(X_train_t, y_train)
    y_pred = clf.predict(X_train_t)
    results['train'] = sklearn.metrics.classification_report(
        y_train, y_pred, output_dict=True)

    X_validation_t = pipeline.transform(X_validation)
    y_pred = clf.predict(X_validation_t)
    results['validation'] = sklearn.metrics.classification_report(
        y_validation, y_pred, output_dict=True)
    return results


def main(args):
    dataset = pd.read_csv(args.datapath, names=(
        "Classification", "Label", "Sequence"))

    X, y = dataset['Sequence'], dataset['Label'].map(LABEL_MAP)
    X_cv, X_holdout, y_cv, y_holdout = sklearn.model_selection.train_test_split(
        X, y, shuffle=True, test_size=HOLDOUT_SET_SIZE, random_state=RANDOM_SEED)
    kfold = sklearn.model_selection.StratifiedKFold(
        n_splits=CROSSVALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    experiment_results = []
    for sweep in hyperparameter_grid(HYPERPARAMETER_OPTIONS):
        sweep_results = []
        for fold_idx, (train_indices, validation_indices) in enumerate(kfold.split(X_cv, y_cv)):
            logging.debug(f"Running fold {fold_idx}")
            X_train, y_train = X_cv.iloc[train_indices], y_cv.iloc[train_indices]
            X_validation, y_validation = X_cv.iloc[validation_indices], y_cv.iloc[validation_indices]

            fold_results = run_sweep(
                sweep, X_train, y_train, X_validation, y_validation)
            sweep_results.append(fold_results)
        experiment_results.append({'params': sweep, 'results': sweep_results})

    with open(args.results, 'w') as result_file:
        json.dump(experiment_results, result_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare representations for the PsychOrNot classifier.")
    parser.add_argument("--datapath", default="dataset/database-pdb.csv",
                        help="The path to the .CSV with the experiment data.")
    parser.add_argument("--results", default="representation-results.json",
                        help="Where to save the results of the representation comparison.")
    args = parser.parse_args()
    main(args)
