from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from graph.GraphGenerator import GraphGenerator
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds, plot_history
from sklearn.metrics import confusion_matrix

import pandas as pd
import tensorflow as tf
import config as cfg
import time
import database_config

# todo extract ONE graph will all times (graph transformer must generate timestamped nodes for price changes)
# todo masking for non-existing

# todo make predictions better, maybe works automatically after bugfix (see top of file)
# todo fix graph node extractor


# CONFIG ===============================================================================================================
MAX_EPOCHS = 1_000
TRAIN_SIZE_RELATIVE = 0.60
VALIDATION_SIZE_RELATIVE_TEST = 0.50

NODE_FEATURES = ['price', 'old_value', 'new_value', 'timestamp', 'record_id']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# SET SEED =============================================================================================================
set_all_seeds(cfg.RANDOM_SEED_MODEL)
set_all_seeds(999)

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
graph_stellar, graph_labels, graph_name = graph.serialize_stellargraph_node_level(NODE_FEATURES, NODE_TYPES)
time_end = time.perf_counter()
print(f"Serialize StellarGraph took {time_end - time_start:0.4f} seconds")


# TRAIN/TEST SPLIT =====================================================================================================
train_subjects, test_subjects = model_selection.train_test_split(
    graph_labels, train_size=int(len(graph_labels) * TRAIN_SIZE_RELATIVE), test_size=None
)

val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=int(len(test_subjects) * VALIDATION_SIZE_RELATIVE_TEST), test_size=None
)


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

es_callback = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.00001, restore_best_weights=True)
auc = tf.keras.metrics.AUC()

with tf.device('/CPU:0'):
    gcn = GCN(
        layer_sizes=[node_feature_count, node_feature_count], activations=['relu', 'relu'], generator=generator,
    )

    x_inp, x_out = gcn.in_out_tensors()

    predictions = Dense(units=train_targets.shape[1], activation="softmax")(x_out)

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
    plot_history(history, es_callback, f'Node Level GCN (GCN [{node_feature_count}, {node_feature_count}], Dense {train_targets.shape[1]}, Softmax)',
                 cfg.STORAGE_BASE_THESIS_IMG + rf'\gcn_node.pdf', 10)


# TEST MODEL ===========================================================================================================
    all_predictions = model.predict(all_gen)
    test_predictions = model.predict(test_gen)

    all_node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
    test_node_predictions = target_encoding.inverse_transform(test_predictions.squeeze())

    all_predictions_df = pd.DataFrame({"Predicted": all_node_predictions, "True": graph_labels})
    all_predictions_df['is_correct'] = all_predictions_df['Predicted'] == all_predictions_df['True']
    all_predictions_df_err = all_predictions_df[all_predictions_df['is_correct'] == False]
    all_predictions_df_err.to_csv(r'C:\Users\Pasi\Desktop\results_all.csv', sep=';')

    test_predictions_df = pd.DataFrame({"Predicted": test_node_predictions, "True": target_encoding.inverse_transform(test_targets.squeeze())})
    test_predictions_df['is_correct'] = test_predictions_df['Predicted'] == test_predictions_df['True']
    test_predictions_df_err = test_predictions_df[test_predictions_df['is_correct'] == False]
    test_predictions_df_err.to_csv(r'C:\Users\Pasi\Desktop\results_test.csv', sep=';')

    print('------------------------\nALL:\n',
          confusion_matrix(all_predictions_df['True'], all_predictions_df['Predicted']))
    print('------------------------\nTEST:\n',
          confusion_matrix(test_predictions_df['True'], test_predictions_df['Predicted']))

