from graph.GraphCollection import GraphCollection
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn import model_selection
import stellargraph.random as sgrand
import matplotlib.pyplot as plt
import tensorflow as tf
import config as cfg
import pandas as pd
import numpy as np
import time
import random
import os


import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection


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
TRAINING_ITERATIONS = 1
EPOCHS = 3 #10_000
TRAIN_SIZE_RELATIVE = 0.8
TRAIN_SIZE_RELATIVE_TRAIN = 0.50

# Load graphs
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)

NODE_FEATURES = ['price', 'old_value', 'new_value']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS', 'MST_ADDRESSES']
node_feature_count = len(NODE_FEATURES) + len(NODE_TYPES)
graphs_gt_stellar = graphs.serialize_stellargraph_node_level(NODE_FEATURES, NODE_TYPES)

graphs_stellar = [item[0] for item in graphs_gt_stellar]
graph_labels = [item[1] for item in graphs_gt_stellar]

# todo: HAS TO BE DONE PER GRAPH, manually! (first: complete for one graph)
# todo https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html

# DATA OVERVIEW ========================================================================================================
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels).value_counts().to_frame())

# TRAIN/TEST SPLIT =====================================================================================================
train_graphs, test_graphs = model_selection.train_test_split(
    graphs_gt_stellar, train_size=int(len(graphs_gt_stellar) * TRAIN_SIZE_RELATIVE), test_size=None
)

validation_graphs, test_graphs = model_selection.train_test_split(
    test_graphs, train_size=int(len(test_graphs) * TRAIN_SIZE_RELATIVE_TRAIN), test_size=None
)


# MAIN CODE ============================================================================================================
es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

auc = tf.keras.metrics.AUC()
history = []
for iteration in range(TRAINING_ITERATIONS):
    for G, node_subjects, name in train_graphs:
        train_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=int(len(node_subjects) * TRAIN_SIZE_RELATIVE), test_size=None, stratify=node_subjects
        )

        val_subjects, test_subjects = model_selection.train_test_split(
            test_subjects, train_size=int(len(test_subjects) * TRAIN_SIZE_RELATIVE_TRAIN), test_size=None, stratify=test_subjects
        )

        target_encoding = preprocessing.LabelBinarizer()

        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)

        print(f'train_targets: {len(train_targets)}\nval_targets: {len(val_targets)}\ntest_targets: {len(test_targets)}')

        generator = FullBatchNodeGenerator(G, method="gcn")

        train_gen = generator.flow(train_subjects.index, train_targets)
        test_gen = generator.flow(test_subjects.index, test_targets)
        val_gen = generator.flow(val_subjects.index, val_targets)

        gcn = GCN(
            layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator
        )

        x_inp, x_out = gcn.in_out_tensors()

        predictions = layers.Dense(units=1, activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss=losses.binary_crossentropy, # old val.: categorical_crossentropy
            metrics=["acc", auc],
        )

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            verbose=0,
            shuffle=False,
            callbacks=[es_callback],
        )

sg.utils.plot_history(history)
plt.title('Training over Time')
plt.show()
