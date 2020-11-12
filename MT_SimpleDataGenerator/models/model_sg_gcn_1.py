import pandas as pd
import numpy as np
import stellargraph.random as sgrand
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
import random
import os

import config as cfg
from graph.GraphCollection import GraphCollection

# SEED
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
sgrand.set_seed(SEED)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# CONFIG
EPOCHS = 1_000

# Load graphs
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)
graphs.prune(4)

NODE_FEATURES = ['price', 'old_value', 'new_value']
# NODE_FEATURES = ['price', 'old_value', 'new_value', 'amount', 'name']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS', 'MST_ADDRESSES']
NODE_FEATURE_COUNT = len(NODE_FEATURES) + len(NODE_TYPES)
graphs_gt_stellar = graphs.serialize_stellargraph(NODE_FEATURES, NODE_TYPES)

# graphs_gt_stellar = [item for item in graphs_gt_stellar if item[1] or with_probability(0.5)]

graphs_stellar = [item[0] for item in graphs_gt_stellar]
graph_labels = [item[1] for item in graphs_gt_stellar]

# DATA OVERVIEW ===================================================================================================================
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels).value_counts().to_frame())

# MAIN CODE ======================================================================================================================
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs_stellar)


# MODEL
es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)

auc = tf.keras.metrics.AUC()


def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[12, 12],
        # layer_sizes=[NODE_FEATURE_COUNT, NODE_FEATURE_COUNT],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.0,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=['acc', auc])

    return model


def train(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
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


# with tf.device('/CPU:0'):
time_start = time.perf_counter()
train_gen, test_gen = get_generators(
    range(len(graphs_gt_stellar)), range(len(graphs_gt_stellar)), graph_labels, batch_size=10
)

model = create_graph_classification_model(generator)

history, acc_val, auc_val = train(model, train_gen, test_gen, es, EPOCHS)

print(
    f"Accuracy: {acc_val*100:.3}%\n"
    f"AUC: {auc_val:.3}"
)
time_end = time.perf_counter()
print(f"Training took {time_end - time_start:0.4f} seconds")

# plt.figure(figsize=(8, 6))
# plt.hist([test_accs, test_aucs], label=['ACC', 'AUC'])
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.legend(loc='upper left')
# plt.show()

# predict all
all_gen = generator.flow(graphs_stellar)
all_predictions = [x[0] > 0.5 for x in model.predict(all_gen).tolist()]
graph_names = [g.get_name() for g in graphs.get_raw_list()]

# df = pd.DataFrame({"RAW val.": model.predict(all_gen).tolist(), "Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar]})
df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar], "RAW val.": model.predict(all_gen).tolist()})
df.to_csv(r'C:\Users\Pasi\Desktop\result.csv', sep=';')
