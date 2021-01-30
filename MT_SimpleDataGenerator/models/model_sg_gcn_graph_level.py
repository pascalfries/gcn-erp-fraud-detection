from graph.GraphCollection import GraphCollection
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn import model_selection
from helpers import set_all_seeds, with_probability, plot_history, plot_confusion_matrix, aggregate_sets_multi, apply_model_to_sap_data
from tensorflow.python.keras.utils.np_utils import to_categorical

import tensorflow as tf
import keras.backend as K
import config as cfg
import pandas as pd
import functools


# CONFIG ===============================================================================================================
MAX_EPOCHS = 1_000
TRAIN_SIZE_RELATIVE = 0.75
VALIDATION_SIZE_RELATIVE_TEST = 0.60

TIMESERIES_GEN_WINDOW_DURATION = 3

NODE_FEATURES = ['price', 'old_value', 'new_value']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# SET SEED =============================================================================================================
set_all_seeds(cfg.RANDOM_SEED_MODEL)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE CODE BELOW THIS COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# LOAD DATA ============================================================================================================
graphs = GraphCollection(directory_name=rf'{cfg.STORAGE_BASE_PATH_PY_GRAPHS}\window_duration_{TIMESERIES_GEN_WINDOW_DURATION}')
graphs_gt_stellar = graphs.serialize_stellargraph(NODE_FEATURES, NODE_TYPES)

node_feature_count = len(NODE_FEATURES) + len(NODE_TYPES)

# TEST/TRAIN SPLIT =====================================================================================================
train_subjects, test_subjects = model_selection.train_test_split(
    graphs_gt_stellar, train_size=int(len(graphs_gt_stellar) * TRAIN_SIZE_RELATIVE), test_size=None
)

val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=int(len(test_subjects) * VALIDATION_SIZE_RELATIVE_TEST), test_size=None
)


graphs_stellar_train = [item[0] for item in train_subjects]
graph_labels_train = [item[1] for item in train_subjects]

graphs_stellar_val = [item[0] for item in val_subjects]
graph_labels_val = [item[1] for item in val_subjects]

graphs_stellar_all = [item[0] for item in graphs_gt_stellar]
graph_labels_all = [item[1] for item in graphs_gt_stellar]
graph_fraud_id_all = [item[3] for item in graphs_gt_stellar]

graphs_stellar_test = [item[0] for item in test_subjects]
graph_labels_test = [item[1] for item in test_subjects]
graph_fraud_id_test = [item[3] for item in test_subjects]


# DATA OVERVIEW ========================================================================================================
print('ALL SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_all],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_all).value_counts().to_frame())

print('TRAIN SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_train],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_train).value_counts().to_frame())

print('\n-----------------------\nVAL SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_val],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_test).value_counts().to_frame())

print('\n-----------------------\nTEST SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_test],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_test).value_counts().to_frame())


# MAIN CODE ============================================================================================================
train_gt = to_categorical(graph_labels_train)
val_gt = to_categorical(graph_labels_val)
all_gt = to_categorical(graph_labels_all)
test_gt = to_categorical(graph_labels_test)

train_generator = PaddedGraphGenerator(graphs=graphs_stellar_train)
val_generator = PaddedGraphGenerator(graphs=graphs_stellar_val)
all_generator = PaddedGraphGenerator(graphs=graphs_stellar_all)
test_generator = PaddedGraphGenerator(graphs=graphs_stellar_test)

train_sequence = train_generator.flow(range(len(train_gt)), targets=train_gt, batch_size=1)
val_sequence = val_generator.flow(range(len(val_gt)), targets=val_gt, batch_size=1)
all_sequence = all_generator.flow(range(len(all_gt)), targets=all_gt, batch_size=1)
test_sequence = test_generator.flow(range(len(test_gt)), targets=test_gt, batch_size=1)

auc = tf.keras.metrics.AUC()
es_callback = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.00001, restore_best_weights=True)


with tf.device('/CPU:0'):
    gcn = GCNSupervisedGraphClassification(
        layer_sizes=[2*node_feature_count, 2*node_feature_count],
        activations=['relu', 'relu'],
        generator=train_generator,
        dropout=0.0,
    )

    x_inp, x_out = gcn.in_out_tensors()

    predictions = Dense(units=10)(x_out)
    predictions = tf.keras.activations.relu(predictions)
    predictions = Dense(units=2, activation="softmax")(predictions)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(
        optimizer=Adam(0.005, amsgrad=True),
        loss=categorical_crossentropy,
        metrics=[auc, 'acc']
    )

    history = model.fit(
        train_sequence,
        epochs=MAX_EPOCHS,
        validation_data=val_sequence,
        verbose=2,
        callbacks=[es_callback]
    )

    model.save(cfg.STORAGE_BASE_PATH_MODELS + rf'\gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}')

    # plot_history(history, es_callback, f'Graph Level GCN\n Window Duration {TIMESERIES_GEN_WINDOW_DURATION}',
    #              cfg.STORAGE_BASE_THESIS_IMG + rf'\gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}.pdf', sma_size=10)

# TEST MODEL ===========================================================================================================
    # all
    # all_predictions = K.argmax(model.predict(all_sequence)).numpy().tolist()
    # graph_names = [g[2] for g in graphs_gt_stellar]
    #
    # all_predictions_df = pd.DataFrame({"Slice": graph_names,
    #                    "Predicted is Fraud": all_predictions,
    #                    "True is Fraud": graph_labels_all,
    #                    "Fraud IDs": graph_fraud_id_all})
    # all_predictions_df['is_correct'] = all_predictions_df['Predicted is Fraud'] == all_predictions_df['True is Fraud']
    # all_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_all_gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}.csv', sep=';')
    #
    # all_identified_cases = all_predictions_df[all_predictions_df['is_correct'] == True]['Fraud IDs']
    #
    # plot_confusion_matrix('Confusion Matrix - All Data', all_predictions, graph_labels_all,
    #                       cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_all_gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}.pdf')
    #
    # # test
    # test_predictions = K.argmax(model.predict(test_sequence)).numpy().tolist()
    # graph_names = [g[2] for g in test_subjects]
    #
    # test_predictions_df = pd.DataFrame({"Slice": graph_names,
    #                    "Predicted is Fraud": test_predictions,
    #                    "True is Fraud": graph_labels_test,
    #                    "Fraud IDs": graph_fraud_id_test})
    # test_predictions_df['is_correct'] = test_predictions_df['Predicted is Fraud'] == test_predictions_df['True is Fraud']
    # test_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_test_gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}.csv', sep=';')
    #
    # all_identified_cases_test = test_predictions_df[test_predictions_df['is_correct'] == True]['Fraud IDs']
    #
    # plot_confusion_matrix('Confusion Matrix - Test Data', test_predictions, graph_labels_test,
    #                       cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_test_gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}.pdf')
    #
    # print('IDENTIFIED IN ALL: ', functools.reduce(aggregate_sets_multi, all_identified_cases.dropna(), set()))
    # print('IDENTIFIED IN TEST: ', functools.reduce(aggregate_sets_multi, all_identified_cases_test.dropna(), set()))
