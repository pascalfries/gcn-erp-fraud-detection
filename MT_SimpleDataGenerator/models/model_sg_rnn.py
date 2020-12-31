from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, GRU, Dense, Masking, Dropout, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from timeseries.TimeseriesExtractor import TimeseriesExtractor
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds
from sklearn.metrics import confusion_matrix

import tensorflow.keras.activations as activation
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
import stellargraph as sg
import config as cfg
import database_config
import os

# todo https://www.tensorflow.org/guide/keras/rnn


# CONFIG ===============================================================================================================
RANDOM_SEED = 123

MAX_EPOCHS = 1_000
TRAIN_SIZE_RELATIVE = 0.70
VALIDATION_SIZE_RELATIVE_TEST = 0.60

TIMESERIES_GEN_WINDOW_DURATION = 3

FORCE_TIMESERIES_REGENERATION = False
STORAGE_PATH_TIMESERIES = r'C:\Users\Pasi\Desktop'

ITEM_FEATURES = ['price', 'old_value', 'new_value', 'timestamp', 'record_id']
ITEM_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE CODE BELOW THIS COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# SET SEED =============================================================================================================
# set_all_seeds(RANDOM_SEED)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# GENERATE TIMESERIES - ALL TIMES ======================================================================================
if os.path.isfile(STORAGE_PATH_TIMESERIES + r'\rnn_timeseries.npy') and not FORCE_TIMESERIES_REGENERATION:
    print('LOADING TIMESERIES DATA FROM PATH ' + STORAGE_PATH_TIMESERIES)
    timeseries = np.load(STORAGE_PATH_TIMESERIES + r'\rnn_timeseries.npy')
    labels = np.load(STORAGE_PATH_TIMESERIES + r'\rnn_labels.npy')
else:
    database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
    timeseries_extractor = TimeseriesExtractor(db=database_config.db, max_simulation_time=cfg.SIMULATION_END_TIME)
    (timeseries, labels) = timeseries_extractor.generate_timeseries(TIMESERIES_GEN_WINDOW_DURATION, ITEM_FEATURES, ITEM_TYPES)

    np.save(STORAGE_PATH_TIMESERIES + r'\rnn_timeseries.npy', timeseries)
    np.save(STORAGE_PATH_TIMESERIES + r'\rnn_labels.npy', labels)


# TRAIN/TEST SPLIT =====================================================================================================
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


# DATA OVERVIEW ========================================================================================================
print('Data Shape:', timeseries.shape)

print('------------------------\nALL:\n', pd.Series(labels).value_counts())
print('------------------------\nTRAIN:\n', pd.Series(train_labels).value_counts())
print('------------------------\nVALIDATION:\n', pd.Series(val_labels).value_counts())
print('------------------------\nTEST:\n', pd.Series(test_labels).value_counts())
exit()

# MAIN CODE ============================================================================================================
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
auc = tf.keras.metrics.AUC()

with tf.device('/CPU:0'):
    model = Sequential([
        # Masking(mask_value=-1),
        # tf.keras.layers.Embedding(input_dim=10, output_dim=10),
        # GRU(timeseries.shape[2], input_shape=timeseries.shape[1:]), # shape = (max_time, feat_count)
        GRU(timeseries.shape[2] * 2, input_shape=timeseries.shape[1:]), # shape = (max_time, feat_count)
        # Dense(10, activation=activation.relu),
        # Dropout(0.5),
        Dense(2),
        Softmax()
    ])

    model.compile(
        loss=tf.losses.categorical_crossentropy,
        optimizer=optimizers.Adam(lr=0.001, amsgrad=False),
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
        verbose=2,
        # class_weight={0: 0.7, 1: 1.0}
    )

    model.summary()
    sg.utils.plot_history(history)
    plt.show()


# PREDICT ALL ==========================================================================================================
    all_predictions = K.argmax(model.predict(timeseries)).numpy().tolist()
    test_predictions = K.argmax(model.predict(test_subjects)).numpy().tolist()

    all_predictions_df = pd.DataFrame({"Predicted": all_predictions, "True": labels})
    all_predictions_df['is_correct'] = all_predictions_df['Predicted'] == all_predictions_df['True']
    all_predictions_df_err = all_predictions_df[all_predictions_df['is_correct'] == False]
    all_predictions_df_err.to_csv(r'C:\Users\Pasi\Desktop\results_all.csv', sep=';')

    test_predictions_df = pd.DataFrame({"Predicted": test_predictions, "True": labels[test_idx]})
    test_predictions_df['is_correct'] = test_predictions_df['Predicted'] == test_predictions_df['True']
    test_predictions_df_err = test_predictions_df[test_predictions_df['is_correct'] == False]
    test_predictions_df_err.to_csv(r'C:\Users\Pasi\Desktop\results_test.csv', sep=';')

    print('------------------------\nALL:\n', confusion_matrix(all_predictions_df['True'], all_predictions_df['Predicted']))
    print('------------------------\nTEST:\n', confusion_matrix(test_predictions_df['True'], test_predictions_df['Predicted']))
