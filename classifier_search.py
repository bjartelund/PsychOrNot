"""
PsychOrNot - Experiment 2

"""
from typing import List, Tuple

import itertools
import argparse
import logging
import random
import pickle

import pandas as pd
import numpy as np

import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.svm

import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner
import keras_tuner.tuners
import keras_tuner_cv.inner_cv
import keras_tuner_cv.utils


def build_input_pipeline():
    transform_steps = [('vectorize', sklearn.feature_extraction.text.CountVectorizer(
        analyzer='char',
        lowercase=False,
        ngram_range=(2, 2),
        vocabulary=feature_names_for_ngram_range(
            AMINO_ACIDS, (2, 2))
    ))]
    transform_steps.append(
        ('normalize', sklearn.feature_extraction.text.TfidfTransformer(use_idf=True)))
    return sklearn.pipeline.Pipeline(transform_steps)


def build_model(hp):
    # Since we ensemble these models later in construct_ensemble
    # and Keras requires unique model names,
    # assign a random identifier to avoid renaming them later
    model = keras.Sequential([
        keras.Input(shape=(400,))], name=f"sequential_{random.randint(0, 2**31)}")

    regularization = hp.Choice('regularization', ['none', 'l1', 'l2'])
    if regularization == 'none':
        regularization = None

    for i in range(hp.Int("layers", 1, 3)):
        model.add(keras.layers.Dense(units=hp.Int(
            f"units_{i}", min_value=32, max_value=512, step=32), 
            activation=hp.Choice("activation", ["relu", "sigmoid"]),
            kernel_regularizer=regularization,
            bias_regularizer=regularization))

    model.add(keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.2, step=0.05)))

    # Add the prediction head:
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', ['sgd', 'adam', 'rmsprop'])
    lr = hp.Float('learning-rate', max_value=1.0, min_value=1e-5, sampling='log')

    if optimizer_choice == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)


    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


def construct_ensemble(models):
    input_layer = keras.Input(shape=(400,))
    linked_models = [model(input_layer) for model in models]
    out = keras.layers.Average()(linked_models)
    model = keras.Model(inputs=input_layer, outputs=out)
    return model


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

# MAX_EPOCHS is the number of epochs to train each model for at most
MAX_EPOCHS = 100

# MAX_TRIALS is the number of hyperparameter configurations to try out.
MAX_TRIALS = 1000

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


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    dataset = pd.read_csv(args.datapath, names=(
        "Classification", "Label", "Sequence"))

    X, y = dataset['Sequence'], dataset['Label'].map(LABEL_MAP).to_numpy()
    X_cv, X_holdout, y_cv, y_holdout = sklearn.model_selection.train_test_split(
        X, y, shuffle=True, test_size=HOLDOUT_SET_SIZE, random_state=RANDOM_SEED)
    kfold = sklearn.model_selection.StratifiedKFold(
        n_splits=CROSSVALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    pipeline = build_input_pipeline()
    X_cv_t = pipeline.fit_transform(X_cv).toarray()
    
    if args.pipeline:
        logging.info("Writing pipeline to " + args.pipeline)
        with open(args.pipeline, 'wb') as pipeline_file: 
            pickle.dump(pipeline, pipeline_file)

    parameter_tuner = keras_tuner_cv.inner_cv.inner_cv(keras_tuner.tuners.RandomSearch)(
        build_model,
        kfold,
        max_trials=MAX_TRIALS,
        objective="val_accuracy",
        overwrite=True,
        directory="classifier_search",
        project_name="psychornot",
        save_history=True,
        save_output=True
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    board_logging = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    parameter_tuner.search(X_cv_t, y_cv, epochs=MAX_EPOCHS, callbacks=[stop_early, board_logging])

    df = keras_tuner_cv.utils.pd_inner_cv_get_result(parameter_tuner)
    print(df.head())
    df.to_csv(args.results)

    best_models = parameter_tuner.get_best_models(num_models=1)
    best_ensemble = construct_ensemble(best_models[0])
    best_ensemble.build(input_shape=(400,))
    best_ensemble.compile(optimizer=keras.optimizers.RMSprop(),
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy'])
    best_ensemble.save(args.model)

    X_holdout_t = pipeline.transform(X_holdout).toarray()
    print("Final holdout test set accuracy:")
    best_ensemble.evaluate(X_holdout_t, y_holdout)
    y_holdout_predictions = np.round(best_ensemble.predict(X_holdout_t))
    classification_report = sklearn.metrics.classification_report(
        y_holdout, y_holdout_predictions, digits=3)
    print(classification_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare classifier architectures for the PsychOrNot classifier.")
    parser.add_argument("--datapath", default="dataset/database-pdb.csv",
                        help="The path to the .CSV with the experiment data.")
    parser.add_argument("--results", default="classification-results.csv",
                        help="Where to save the results of the classifier search.")
    parser.add_argument("--pipeline", default="psychornot-pipeline",
                        help="Where to save the preprocessing pipeline for the model.")
    parser.add_argument("--model", default="psychornot-ensemble",
                        help="Where to save the ensemble of the best performing models.")
    parser.add_argument("--logdir", default="logs",
                        help="Where to save the logs.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log results to the console.")
    args = parser.parse_args()
    main(args)
