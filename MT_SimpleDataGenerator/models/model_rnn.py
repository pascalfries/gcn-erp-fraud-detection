from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, GRU, Dense, Masking, Dropout, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from timeseries.TimeseriesExtractor import TimeseriesExtractor
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds, plot_history, plot_confusion_matrix, aggregate_sets_multi, apply_model_to_sap_data_timeseries

import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import config as cfg
import database_config
import os
import pickle
import functools

# https://www.tensorflow.org/guide/keras/rnn


# CONFIG ===============================================================================================================
MAX_EPOCHS = 1#_000
TRAIN_SIZE_RELATIVE = 0.75
VALIDATION_SIZE_RELATIVE_TEST = 0.60

TIMESERIES_GEN_WINDOW_DURATION = 2

FORCE_TIMESERIES_REGENERATION = False
STORAGE_PATH_TIMESERIES = r'C:\Users\Pasi\Documents (offline)\timeseries'

ITEM_FEATURES = ['price', 'old_value', 'new_value', 'timestamp', 'record_id']
ITEM_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE CODE BELOW THIS COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# SET SEED =============================================================================================================
set_all_seeds(cfg.RANDOM_SEED_MODEL)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# GENERATE TIMESERIES - ALL TIMES ======================================================================================
item_feature_count = len(ITEM_FEATURES) + len(ITEM_TYPES)

if os.path.isfile(STORAGE_PATH_TIMESERIES + rf'\rnn_timeseries_{TIMESERIES_GEN_WINDOW_DURATION}.npy') and not FORCE_TIMESERIES_REGENERATION:
    print('LOADING TIMESERIES DATA FROM PATH ' + STORAGE_PATH_TIMESERIES)
    timeseries = np.load(STORAGE_PATH_TIMESERIES + rf'\rnn_timeseries_{TIMESERIES_GEN_WINDOW_DURATION}.npy')
    labels = np.load(STORAGE_PATH_TIMESERIES + rf'\rnn_labels_{TIMESERIES_GEN_WINDOW_DURATION}.npy')

    with open(STORAGE_PATH_TIMESERIES + rf'\fraud_ids_{TIMESERIES_GEN_WINDOW_DURATION}.pickle', 'rb') as fraud_ids_file:
        fraud_ids = pickle.load(fraud_ids_file)
else:
    database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
    timeseries_extractor = TimeseriesExtractor(db=database_config.db, max_simulation_time=cfg.SIMULATION_END_TIME)
    timeseries, labels, fraud_ids, names = timeseries_extractor.generate_timeseries(TIMESERIES_GEN_WINDOW_DURATION, ITEM_FEATURES, ITEM_TYPES)

    np.save(STORAGE_PATH_TIMESERIES + rf'\rnn_timeseries_{TIMESERIES_GEN_WINDOW_DURATION}.npy', timeseries)
    np.save(STORAGE_PATH_TIMESERIES + rf'\rnn_labels_{TIMESERIES_GEN_WINDOW_DURATION}.npy', labels)

    with open(STORAGE_PATH_TIMESERIES + rf'\fraud_ids_{TIMESERIES_GEN_WINDOW_DURATION}.pickle', 'wb') as fraud_ids_file:
        pickle.dump(fraud_ids, fraud_ids_file)

timeseries[:, :, len(ITEM_TYPES):item_feature_count] = ((timeseries - np.nanmean(timeseries, axis=(0, 1))) / np.nanstd(timeseries, axis=(0, 1)))[:, :, len(ITEM_TYPES):item_feature_count]
np.nan_to_num(timeseries, copy=False, nan=-1)

# TRAIN/TEST SPLIT =====================================================================================================
labels = np.array([True if x > 0 else False for x in labels])
all_idx = list(range(0, len(labels) - 1))

train_idx, test_idx = model_selection.train_test_split(
    all_idx, train_size=int(len(all_idx) * TRAIN_SIZE_RELATIVE), test_size=None
)

val_idx, test_idx = model_selection.train_test_split(
    test_idx, train_size=int(len(test_idx) * VALIDATION_SIZE_RELATIVE_TEST), test_size=None
)

train_subjects = timeseries[train_idx]
train_labels = labels[train_idx]

val_subjects = timeseries[val_idx]
val_labels = labels[val_idx]

test_subjects = timeseries[test_idx]
test_labels = labels[test_idx]
test_fraud_ids = list(fraud_ids[i] for i in test_idx)


# DATA OVERVIEW ========================================================================================================
print('Data Shape:', timeseries.shape)

print('------------------------\nALL:\n', pd.Series(labels).value_counts())
print('------------------------\nTRAIN:\n', pd.Series(train_labels).value_counts())
print('------------------------\nVALIDATION:\n', pd.Series(val_labels).value_counts())
print('------------------------\nTEST:\n', pd.Series(test_labels).value_counts())

# MAIN CODE ============================================================================================================
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
auc = tf.keras.metrics.AUC()

with tf.device('/CPU:0'):
    model = Sequential([
        Masking(mask_value=-1),
        GRU(timeseries.shape[2] * 2, input_shape=timeseries.shape[1:]), # shape = (max_time, feat_count) check
        Dense(10, activation=tf.keras.activations.relu),
        Dense(2),
        Softmax()
    ])

    model.compile(
        loss=tf.losses.categorical_crossentropy,
        optimizer=optimizers.Adam(lr=0.001, amsgrad=True),
        metrics=['acc', auc]
    )

    history = model.fit(
        x=train_subjects,
        y=train_labels,
        epochs=MAX_EPOCHS,
        batch_size=1,
        validation_data=(val_subjects, val_labels),
        shuffle=False,
        callbacks=[es_callback],
        verbose=2
    )

    model.summary()
    # apply_model_to_sap_data_timeseries(model, ITEM_FEATURES, ITEM_TYPES)

    # model.save(cfg.STORAGE_BASE_PATH_MODELS + rf'\rnn_{TIMESERIES_GEN_WINDOW_DURATION}')
    # plot_history(history, es_callback, f'GRU\nWindow Duration {TIMESERIES_GEN_WINDOW_DURATION}',
    #              cfg.STORAGE_BASE_THESIS_IMG + rf'\rnn_{TIMESERIES_GEN_WINDOW_DURATION}.pdf')


# TEST MODEL ===========================================================================================================
#     all_predictions = K.argmax(model.predict(timeseries)).numpy().tolist()
#     test_predictions = K.argmax(model.predict(test_subjects)).numpy().tolist()
#
#     all_predictions_df = pd.DataFrame({"Predicted": all_predictions,
#                                        "True": labels,
#                                        "Fraud IDs": fraud_ids})
#
#     all_predictions_df['is_correct'] = all_predictions_df['Predicted'] == all_predictions_df['True']
#     all_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_all_rnn_{TIMESERIES_GEN_WINDOW_DURATION}.csv', sep=';')
#     all_identified_cases = all_predictions_df[all_predictions_df['is_correct'] == True]['Fraud IDs']
#
#     test_predictions_df = pd.DataFrame({"Predicted": test_predictions,
#                                         "True": labels[test_idx],
#                                         "Fraud IDs": test_fraud_ids})
#     test_predictions_df['is_correct'] = test_predictions_df['Predicted'] == test_predictions_df['True']
#     test_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_test_rnn_{TIMESERIES_GEN_WINDOW_DURATION}.csv', sep=';')
#     all_identified_cases_test = test_predictions_df[test_predictions_df['is_correct'] == True]['Fraud IDs']
#
#     plot_confusion_matrix('Confusion Matrix - All Data', all_predictions, labels,
#                           cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_all_rnn_{TIMESERIES_GEN_WINDOW_DURATION}.pdf')
#     plot_confusion_matrix('Confusion Matrix - Test Data', test_predictions, labels[test_idx],
#                           cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_test_rnn_{TIMESERIES_GEN_WINDOW_DURATION}.pdf')
#
#     print('IDENTIFIED IN ALL: ', functools.reduce(aggregate_sets_multi, all_identified_cases.dropna(), set()))
#     print('IDENTIFIED IN TEST: ', functools.reduce(aggregate_sets_multi, all_identified_cases_test.dropna(), set()))

