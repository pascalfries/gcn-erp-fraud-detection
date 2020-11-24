from graph.GraphCollection import GraphCollection
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import stellargraph.random as sgrand
import tensorflow as tf
import config as cfg
import pandas as pd
import numpy as np
import time
import random
import os


# SEED
# SEED = 13
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# sgrand.set_seed(SEED)
#
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


# CONFIG
EPOCHS = 10_000

# Load graphs
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)
graphs.prune(4)

NODE_FEATURES = ['price', 'old_value', 'new_value']
# NODE_FEATURES = ['price', 'old_value', 'new_value', 'amount', 'name']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS', 'MST_ADDRESSES']
NODE_FEATURE_COUNT = len(NODE_FEATURES) + len(NODE_TYPES)
graphs_gt_stellar = graphs.serialize_stellargraph(NODE_FEATURES, NODE_TYPES)

# graphs_gt_stellar_train = graphs_gt_stellar[:400]
# graphs_gt_stellar_test = graphs_gt_stellar[401:]

# graphs_gt_stellar = [item for item in graphs_gt_stellar if item[1] or with_probability(0.5)]

graphs_stellar = [item[0] for item in graphs_gt_stellar]
graph_labels = [item[1] for item in graphs_gt_stellar]

# graphs_stellar_train = [item[0] for item in graphs_gt_stellar_train]
# graph_labels_train = [item[1] for item in graphs_gt_stellar_train]
# graphs_stellar_test = [item[0] for item in graphs_gt_stellar_test]
# graph_labels_test = [item[1] for item in graphs_gt_stellar_test]

# DATA OVERVIEW ===================================================================================================================
# summary = pd.DataFrame(
#     [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar],
#     columns=["nodes", "edges"],
# )
# print(summary.describe().round(1))
# print(pd.DataFrame(graph_labels).value_counts().to_frame())

# MAIN CODE ======================================================================================================================
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs_stellar)

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)

auc = tf.keras.metrics.AUC()

def train(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )

    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]
    test_auc = test_metrics[model.metrics_names.index("auc")]

    return history, test_acc, test_auc


def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


def leaky_relu(value):
    return tf.keras.activations.relu(value, alpha=0.01)


with tf.device('/CPU:0'):
    time_start = time.perf_counter()
    graph_count = len(graphs_gt_stellar)
    train_gen, test_gen = get_generators(
        range(0, int(graph_count * 0.70)), range(0, graph_count), graph_labels, batch_size=1
    )

    # generate model
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[12, 12],
        # layer_sizes=[NODE_FEATURE_COUNT, NODE_FEATURE_COUNT],
        activations=[leaky_relu, leaky_relu],
        generator=generator,
        dropout=0.0,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32)(x_out)
    predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    predictions = Dense(units=16)(predictions)
    predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.001, amsgrad=True), loss=binary_crossentropy, metrics=['acc', auc])

    history, acc_val, auc_val = train(model, train_gen, test_gen, es, EPOCHS)

    time_end = time.perf_counter()

print(f"Accuracy: {acc_val*100:.3}%\nAUC: {auc_val:.3}")
print(f"Training took {time_end - time_start:0.4f} seconds")

# predict all
all_gen = generator.flow(graphs_stellar)
all_predictions = [x[0] > 0.5 for x in model.predict(all_gen).tolist()]
graph_names = [g.get_name() for g in graphs.get_raw_list()]

df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar], "RAW val.": model.predict(all_gen).tolist()})
df.to_csv(r'C:\Users\Pasi\Desktop\result.csv', sep=';')

# predict in-sample
# all_predictions = [x[0] > 0.5 for x in model.predict(train_gen).tolist()]
# graph_names = [g.get_name() for g in graphs.get_raw_list()]

# df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar], "RAW val.": model.predict(train_gen).tolist()})
# df.to_csv(r'C:\Users\Pasi\Desktop\result.csv', sep=';')


# predict out-of-sample
# all_gen = generator.flow(graphs_stellar)
# all_predictions = [x[0] > 0.5 for x in model.predict(test_gen).tolist()]
# graph_names = [g.get_name() for g in graphs.get_raw_list()]

# df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar], "RAW val.": model.predict(test_gen).tolist()})
# df.to_csv(r'C:\Users\Pasi\Desktop\result.csv', sep=';')
