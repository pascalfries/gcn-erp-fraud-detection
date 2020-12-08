from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from graph.GraphGenerator import GraphGenerator
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import stellargraph as sg
import config as cfg
import time
import database_config
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds
from sklearn.metrics import confusion_matrix
from gcn_testing.graph_gen_node_level import generate_graph

# todo extract ONE graph will all times (graph transformer must generate timestamped nodes for price changes)
# todo https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html


def leaky_relu(value):
    return tf.keras.activations.relu(value, alpha=0.01)


def extract_gt(labels):
    return [[0, 1] if item == 'fraud' else [1, 0] for item in labels]


# CONFIG ===============================================================================================================
EPOCHS = 1_000
TRAIN_SIZE_RELATIVE = 0.5
VALIDATION_SIZE_RELATIVE_TRAIN = 0.50


# SET SEED =============================================================================================================
# set_all_seeds(13)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


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
print(f"Prunning took {time_end - time_start:0.4f} seconds")

graph.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\all.txt')

time_start = time.perf_counter()
NODE_FEATURES = ['price', 'old_value', 'new_value', 'timestamp']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS', 'MST_ADDRESSES']
node_feature_count = len(NODE_FEATURES) + len(NODE_TYPES)
graph_stellar, graph_labels, graph_name = graph.serialize_stellargraph_node_level(NODE_FEATURES, NODE_TYPES)
time_end = time.perf_counter()
print(f"Serialize StellarGraph took {time_end - time_start:0.4f} seconds")

# dataset = sg.datasets.Cora()
# graph_stellar, graph_labels = dataset.load()

# G, gt = generate_graph()
# NODE_FEATURES = []
# NODE_TYPES = ['type-1', 'type-2', 'type-3']
# graph_stellar, graph_labels = G.serialize_stellargraph_node_level(NODE_FEATURES, NODE_TYPES)[0], gt


# DATA OVERVIEW ========================================================================================================
print(graph_stellar.info())


# TRAIN/TEST SPLIT =====================================================================================================
train_subjects, test_subjects = model_selection.train_test_split(
    graph_labels, train_size=int(len(graph_labels) * TRAIN_SIZE_RELATIVE), test_size=None, stratify=graph_labels
)

val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=int(len(test_subjects) * VALIDATION_SIZE_RELATIVE_TRAIN), test_size=None
)

print('------------------------\nALL:', graph_labels.value_counts().to_frame())
print('------------------------\nTRAIN:', train_subjects.value_counts().to_frame())
print('------------------------\nTEST:', test_subjects.value_counts().to_frame())
print('------------------------\nVALIDATION:', val_subjects.value_counts().to_frame())


# MAIN CODE ============================================================================================================
with tf.device('/CPU:0'):
    es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    auc = tf.keras.metrics.AUC()

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    generator = FullBatchNodeGenerator(graph_stellar, method="gcn")

    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    all_gen = generator.flow(graph_labels.index, graph_labels)

    gcn = GCN(
        layer_sizes=[30, 30], activations=['sigmoid', 'sigmoid'], generator=generator,
        # layer_sizes=[node_feature_count, node_feature_count], activations=["relu", "relu"], generator=generator
    )

    x_inp, x_out = gcn.in_out_tensors()
    predictions = Dense(units=16)(x_out)
    predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    # predictions = Dense(units=16)(predictions)
    # predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    # predictions = Dense(units=train_targets.shape[1], activation="softmax")(predictions)
    predictions = Dense(units=train_targets.shape[1])(predictions)
    predictions = tf.keras.activations.softmax(predictions)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.05, amsgrad=True),
        loss=losses.categorical_crossentropy,
        metrics=['acc', auc]
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback],
        # class_weight={0: 0.90, 1: 0.05}#, 2: 0.10}
    )

    sg.utils.plot_history(history)
    plt.show()

    all_predictions = model.predict(all_gen)
    node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
    df = pd.DataFrame({"Predicted": node_predictions, "True": graph_labels, "RAW": [str(x) for x in all_predictions.squeeze()]})
    df['is_correct'] = df['Predicted'] == df['True']
    df_err = df[df['is_correct'] == False]
    df_err.to_csv(r'C:\Users\Pasi\Desktop\results.csv', sep=';')
    print(df_err)
    print(df['is_correct'].value_counts())

    print(confusion_matrix(df['True'], df['Predicted']))

    # todo fix constant predictions
    # todo fix graph node extractor
