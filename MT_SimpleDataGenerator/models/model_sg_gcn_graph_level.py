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
import stellargraph as sg
import pandas as pd
import time


# CONFIG ===============================================================================================================
RANDOM_SEED = 123
MAX_EPOCHS = 10_000
TRAIN_SIZE_RELATIVE = 0.70
VALIDATION_SIZE_RELATIVE_TEST = 0.60

NODE_FEATURES = ['price', 'old_value', 'new_value', 'timestamp', 'record_id']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


# SET SEED =============================================================================================================
# set_all_seeds(RANDOM_SEED)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE CODE BELOW THIS COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# LOAD DATA ============================================================================================================
graphs = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)
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
train_gt = pd.get_dummies(graph_labels_train, drop_first=True)
val_gt = pd.get_dummies(graph_labels_val, drop_first=True)
test_gt = pd.get_dummies(graph_labels_test, drop_first=True)

train_generator = PaddedGraphGenerator(graphs=graphs_stellar_train)
val_generator = PaddedGraphGenerator(graphs=graphs_stellar_val)
test_generator = PaddedGraphGenerator(graphs=graphs_stellar_test)

auc = tf.keras.metrics.AUC()
es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


with tf.device('/CPU:0'):
    time_start = time.perf_counter()
    batch_size = 10

    train_sequence = train_generator.flow(range(len(train_gt)), targets=train_gt.values, batch_size=batch_size)
    val_sequence = val_generator.flow(range(len(val_gt)), targets=val_gt.values, batch_size=batch_size)
    test_sequence = test_generator.flow(range(len(test_gt)), targets=test_gt.values, batch_size=1)

    # generate model
    gc_model = GCNSupervisedGraphClassification(
        # layer_sizes= #todo this worked well
        layer_sizes=[node_feature_count, node_feature_count],
        activations=['relu', 'relu'],
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

    history = model.fit(
        train_sequence,
        epochs=MAX_EPOCHS,
        validation_data=val_sequence,
        verbose=1,
        callbacks=[es_callback],
        # class_weight={0: 1.0, 1: 2.0} # 0: no_fraud, 1: fraud
    )

    test_metrics = model.evaluate(test_sequence, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]
    test_auc = test_metrics[model.metrics_names.index("auc")]

    time_end = time.perf_counter()

    print(f"Accuracy: {test_acc*100:.3}%\nAUC: {test_auc:.3}")
    print(f"Training took {time_end - time_start:0.4f} seconds")

    sg.utils.plot_history(history)
    plt.show()

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
