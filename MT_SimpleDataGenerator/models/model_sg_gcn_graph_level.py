from graph.GraphCollection import GraphCollection
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn import model_selection
from helpers import set_all_seeds, with_probability
import tensorflow as tf
import matplotlib.pyplot as plt
import config as cfg
import pandas as pd
import time


# SEED
# set_all_seeds(13)

# CONFIG
EPOCHS = 10_000
TRAIN_SIZE_RELATIVE = 0.80

NODE_FEATURES = ['price', 'old_value', 'new_value', 'is_fraud']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS', 'MST_ADDRESSES']


# LOAD DATA ============================================================================================================
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)
graphs_gt_stellar = graphs.serialize_stellargraph(NODE_FEATURES, NODE_TYPES)

# graphs_gt_stellar = [g for g in graphs_gt_stellar if g[1] or with_probability(0.2)]

node_feature_count = len(NODE_FEATURES) + len(NODE_TYPES)

# TEST/TRAIN SPLIT =====================================================================================================
train_subjects, test_subjects = model_selection.train_test_split(
    graphs_gt_stellar, train_size=int(len(graphs_gt_stellar) * TRAIN_SIZE_RELATIVE), test_size=None, stratify=None
)

# train_subjects_fraud = [g for g in train_subjects if g[1]]
# train_subjects = train_subjects + train_subjects_fraud + train_subjects_fraud
# random.shuffle(train_subjects)

graphs_stellar_train = [item[0] for item in train_subjects]
graph_labels_train = [item[1] for item in train_subjects]

graphs_stellar_test = [item[0] for item in test_subjects]
graph_labels_test = [item[1] for item in test_subjects]


# DATA OVERVIEW ========================================================================================================
print('TRAIN SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_train],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_train).value_counts().to_frame())

print('\n-----------------------\nTEST SET:')
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_stellar_test],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
print(pd.DataFrame(graph_labels_test).value_counts().to_frame())


# MAIN CODE ============================================================================================================
train_gt = pd.get_dummies(graph_labels_train, drop_first=True)
test_gt = pd.get_dummies(graph_labels_test, drop_first=True)

train_generator = PaddedGraphGenerator(graphs=graphs_stellar_train)
test_generator = PaddedGraphGenerator(graphs=graphs_stellar_test)

auc = tf.keras.metrics.AUC()
es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=15, restore_best_weights=True
)


def train(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        verbose=1,
        callbacks=[es],
        # class_weight={0: 1.0, 1: 2.0} # 0: no_fraud, 1: fraud
    )

    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]
    test_auc = test_metrics[model.metrics_names.index("auc")]

    return history, test_acc, test_auc


def get_sequences(train_labels, test_labels, batch_size):
    train_sequence = train_generator.flow(
        range(len(train_labels)), targets=train_labels.values, batch_size=batch_size
    )

    test_sequence = test_generator.flow(
        range(len(test_labels)), targets=test_labels.values, batch_size=batch_size
    )

    return train_sequence, test_sequence


def leaky_relu(value):
    return tf.keras.activations.relu(value, alpha=0.01)


with tf.device('/CPU:0'):
    time_start = time.perf_counter()
    train_sequence, test_sequence = get_sequences(train_gt, test_gt, batch_size=20)

    # generate model
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[12, 12],
        # layer_sizes=[node_feature_count, node_feature_count],
        activations=[leaky_relu, leaky_relu],
        generator=train_generator,
        dropout=0.0,
    )

    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32)(x_out)
    predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    predictions = Dense(units=16)(predictions)
    predictions = tf.keras.activations.relu(predictions, alpha=0.01)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.001, amsgrad=True), loss=binary_crossentropy, metrics=[auc, 'acc'])
    # model.compile(optimizer='rmsprop', loss=binary_crossentropy, metrics=[auc])

    history, acc_val, auc_val = train(model, train_sequence, train_sequence, es, EPOCHS)

    time_end = time.perf_counter()

    print(f"Accuracy: {acc_val*100:.3}%\nAUC: {auc_val:.3}")
    print(f"Training took {time_end - time_start:0.4f} seconds")

    # predict in-sample
    print('IN SAMPLE:')
    train_sequence, test_sequence = get_sequences(train_gt, test_gt, batch_size=1)
    all_predictions = [x[0] > 0.5 for x in model.predict(train_sequence).tolist()]
    graph_names = [g[2] for g in train_subjects]
    gt = [item[1] for item in train_subjects]

    df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": gt, "RAW val.": model.predict(train_sequence).tolist()})
    df.to_csv(cfg.STORAGE_ROOT_PATH + r'\result_train.csv', sep=';')

    print(tf.math.confusion_matrix(labels=gt, predictions=all_predictions).numpy())

    # predict out-of-sample
    print('OUT OF SAMPLE:')
    all_predictions = [x[0] > 0.5 for x in model.predict(test_sequence).tolist()]
    graph_names = [g[2] for g in test_subjects]
    gt = [item[1] for item in test_subjects]

    df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": gt, "RAW val.": model.predict(test_sequence).tolist()})
    df.to_csv(cfg.STORAGE_ROOT_PATH + r'\result_test.csv', sep=';')

    print(tf.math.confusion_matrix(labels=gt, predictions=all_predictions).numpy())
