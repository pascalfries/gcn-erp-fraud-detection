import pandas as pd
import numpy as np
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import time

import config as cfg
from graph.GraphCollection import GraphCollection
# Load graphs
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)
graphs_gt_stellar = graphs.serialize_stellargraph(['price', 'old_value', 'new_value'])

graphs_stellar = [item[0] for item in graphs_gt_stellar]
graph_labels = [item[1] for item in graphs_gt_stellar]

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(graphs_stellar[0][0].info())

# summary = pd.DataFrame(
#     [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar],
#     columns=["nodes", "edges"],
# )
# print(summary.describe().round(1))
# print(pd.DataFrame(graph_labels).value_counts().to_frame())

# MAIN CODE ======================================================================================================================
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs_stellar)

# MODEL
def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[12, 12],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.3,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.01), loss=binary_crossentropy, metrics=["acc"])

    return model


epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc


def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


test_accs = []

stratified_folds = model_selection.StratifiedKFold(
    n_splits=folds
).split(graph_labels, graph_labels)

# with tf.device('/CPU:0'):
time_start = time.perf_counter()
for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds}...")
    train_gen, test_gen = get_generators(
        train_index, test_index, graph_labels, batch_size=10
    )

    model = create_graph_classification_model(generator)

    history, acc = train_fold(model, train_gen, test_gen, es, epochs)

    test_accs.append(acc)

print(
    f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
)
time_end = time.perf_counter()
print(f"Training took {time_end - time_start:0.4f} seconds")

# plt.figure(figsize=(8, 6))
# plt.hist(test_accs)
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.show()













#
# train_subjects, test_subjects = model_selection.train_test_split(
#     node_subjects, train_size=140, test_size=None, stratify=node_subjects
# )
#
# val_subjects, test_subjects = model_selection.train_test_split(
#     test_subjects, train_size=500, test_size=None, stratify=test_subjects
# )
#
# target_encoding = preprocessing.LabelBinarizer()
#
# train_targets = target_encoding.fit_transform(train_subjects)
# val_targets = target_encoding.transform(val_subjects)
# test_targets = target_encoding.transform(test_subjects)
#
# generator = FullBatchNodeGenerator(G, method="gcn")
# train_gen = generator.flow(train_subjects.index, train_targets)
# val_gen = generator.flow(val_subjects.index, val_targets)
# test_gen = generator.flow(test_subjects.index, test_targets)
#
# gcn = GCN(
#     layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.3
# )
#
# x_inp, x_out = gcn.in_out_tensors()
# predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
#
# # CREATE MODEL
# model = Model(inputs=x_inp, outputs=predictions)
# model.compile(
#     optimizer=optimizers.Adam(lr=0.01),
#     loss=losses.categorical_crossentropy,
#     metrics=["acc"],
# )
#
# # es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
# history = model.fit(
#     train_gen,
#     epochs=40,
#     validation_data=val_gen,
#     verbose=2,
#     shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
#     # callbacks=[es_callback],
# )
#
# sg.utils.plot_history(history)
#
# plt.show()
#
# test_metrics = model.evaluate(test_gen)
# print("\nTest Set Metrics:")
# for name, val in zip(model.metrics_names, test_metrics):
#     print("\t{}: {:0.4f}".format(name, val))
#
# # PREDICT
# all_nodes = node_subjects.index
# all_gen = generator.flow(all_nodes)
# all_predictions = model.predict(all_gen)
#
# node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
#
# df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
# print(df.head(20))
#
