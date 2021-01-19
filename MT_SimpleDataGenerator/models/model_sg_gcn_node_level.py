from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from graph.GraphGenerator import GraphGenerator
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds, plot_history, plot_confusion_matrix, aggregate_sets

import pandas as pd
import tensorflow as tf
import config as cfg
import time
import database_config
import functools


# CONFIG ===============================================================================================================
MAX_EPOCHS = 1_000
TRAIN_SIZE_RELATIVE = 0.75
VALIDATION_SIZE_RELATIVE_TEST = 0.60

NODE_FEATURES = ['price', 'old_value', 'new_value', 'timestamp']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# SET SEED =============================================================================================================
set_all_seeds(321)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE CODE BELOW THIS COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# GENERATE GRAPH - ALL TIMES ===========================================================================================
database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)

time_start = time.perf_counter()
graph_gen = GraphGenerator()
graph = graph_gen.generate_graph(db=database_config.db)
time_end = time.perf_counter()
print(f"Generation took {time_end - time_start:0.4f} seconds")

graph._name = 'all'

time_start = time.perf_counter()
graph.prune(min_cluster_size=cfg.GRAPH_PRUNING_MIN_CLUSTER_SIZE)
time_end = time.perf_counter()
print(f"Pruning took {time_end - time_start:0.4f} seconds")

graph.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\all.txt')

time_start = time.perf_counter()
node_feature_count = len(NODE_FEATURES) + len(NODE_TYPES)
graph_stellar, graph_labels, graph_name, fraud_ids = graph.serialize_stellargraph_node_level(NODE_FEATURES, NODE_TYPES)
time_end = time.perf_counter()
print(f"Serialize StellarGraph took {time_end - time_start:0.4f} seconds")

labels_and_fraud_ids = list(zip(graph_labels, fraud_ids, graph_labels.index))

# TRAIN/TEST SPLIT =====================================================================================================
a, b = model_selection.train_test_split(
    graph_labels, train_size=int(len(graph_labels) * TRAIN_SIZE_RELATIVE), test_size=None
)

train_subjects_and_ids, test_subjects_and_ids = model_selection.train_test_split(
    labels_and_fraud_ids, train_size=int(len(graph_labels) * TRAIN_SIZE_RELATIVE), test_size=None
)

val_subjects_and_ids, test_subjects_and_ids = model_selection.train_test_split(
    test_subjects_and_ids, train_size=int(len(test_subjects_and_ids) * VALIDATION_SIZE_RELATIVE_TEST), test_size=None
)

train_subjects, _, train_idx = list(zip(*train_subjects_and_ids))
test_subjects, test_fraud_ids, test_idx = list(zip(*val_subjects_and_ids))
val_subjects, _, val_idx = list(zip(*test_subjects_and_ids))

train_subjects = pd.Series(train_subjects, index=train_idx)
test_subjects = pd.Series(test_subjects, index=test_idx)
val_subjects = pd.Series(val_subjects, index=val_idx)

# DATA OVERVIEW ========================================================================================================
print(graph_stellar.info())

print('------------------------\nALL:', graph_labels.value_counts().to_frame())
print('------------------------\nTRAIN:', train_subjects.value_counts().to_frame())
print('------------------------\nTEST:', test_subjects.value_counts().to_frame())
print('------------------------\nVALIDATION:', val_subjects.value_counts().to_frame())


# MAIN CODE ============================================================================================================
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)
all_targets = target_encoding.transform(graph_labels)

generator = FullBatchNodeGenerator(graph_stellar, method="gcn")

train_gen = generator.flow(train_subjects.index, train_targets)
val_gen = generator.flow(val_subjects.index, val_targets)
test_gen = generator.flow(test_subjects.index, test_targets)
all_gen = generator.flow(graph_labels.index, all_targets)

es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
auc = tf.keras.metrics.AUC()

with tf.device('/CPU:0'):
    gcn = GCN(
        layer_sizes=[2*node_feature_count, 2*node_feature_count],
        activations=['relu', 'relu'],
        generator=generator
    )

    x_inp, x_out = gcn.in_out_tensors()

    # predictions = Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    predictions = Dense(units=10)(x_out)
    predictions = tf.keras.activations.relu(predictions)
    predictions = Dense(units=train_targets.shape[1], activation="softmax")(predictions)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(
        optimizer=optimizers.Adam(lr=0.05, amsgrad=True),
        loss=tf.losses.categorical_crossentropy,
        metrics=['acc', auc]
    )

    history = model.fit(
        train_gen,
        epochs=MAX_EPOCHS,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback]
    )

    model.summary()
    plot_history(history, es_callback, f'Node Level GCN', cfg.STORAGE_BASE_THESIS_IMG + rf'\gcn_node.pdf')


# TEST MODEL ===========================================================================================================
    all_predictions = model.predict(all_gen)
    test_predictions = model.predict(test_gen)

    all_node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
    test_node_predictions = target_encoding.inverse_transform(test_predictions.squeeze())

    all_predictions_df = pd.DataFrame({"Predicted": all_node_predictions,
                                       "True": graph_labels,
                                       "Fraud IDs": fraud_ids}, index=graph_labels.index)
    all_predictions_df['is_correct'] = all_predictions_df['Predicted'] == all_predictions_df['True']
    all_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_all_gcn_node.csv', sep=';')

    all_identified_cases = all_predictions_df[all_predictions_df['is_correct'] == True]['Fraud IDs']
    print('IDENTIFIED IN ALL: ', functools.reduce(aggregate_sets, all_identified_cases.dropna(), set()))

    test_predictions_df = pd.DataFrame({"Predicted": test_node_predictions,
                                        "True": target_encoding.inverse_transform(test_targets.squeeze()),
                                        "Fraud IDs": test_fraud_ids}, index=test_subjects.index)
    test_predictions_df['is_correct'] = test_predictions_df['Predicted'] == test_predictions_df['True']
    test_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_test_gcn_node.csv', sep=';')

    test_identified_cases = test_predictions_df[test_predictions_df['is_correct'] == True]['Fraud IDs']
    print('IDENTIFIED IN ALL: ', functools.reduce(aggregate_sets, test_identified_cases.dropna(), set()))

    plot_confusion_matrix('Confusion Matrix - All Data', all_node_predictions, graph_labels.tolist(),
                          cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_all_gcn_node.pdf')
    plot_confusion_matrix('Confusion Matrix - Test Data', test_node_predictions, target_encoding.inverse_transform(test_targets.squeeze()),
                          cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_test_gcn_node.pdf')
