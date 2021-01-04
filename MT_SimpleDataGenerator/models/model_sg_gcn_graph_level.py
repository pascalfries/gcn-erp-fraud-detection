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


# CONFIG ===============================================================================================================
RANDOM_SEED = 123
MAX_EPOCHS = 1_000
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

graphs_stellar_all = [item[0] for item in graphs_gt_stellar]
graph_labels_all = [item[1] for item in graphs_gt_stellar]

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
all_gt = pd.get_dummies(graph_labels_all, drop_first=True)
test_gt = pd.get_dummies(graph_labels_test, drop_first=True)

train_generator = PaddedGraphGenerator(graphs=graphs_stellar_train)
val_generator = PaddedGraphGenerator(graphs=graphs_stellar_val)
all_generator = PaddedGraphGenerator(graphs=graphs_stellar_all)
test_generator = PaddedGraphGenerator(graphs=graphs_stellar_test)

train_sequence = train_generator.flow(range(len(train_gt)), targets=train_gt.values, batch_size=10)
val_sequence = val_generator.flow(range(len(val_gt)), targets=val_gt.values, batch_size=10)
all_sequence = all_generator.flow(range(len(all_gt)), targets=all_gt.values, batch_size=1)
test_sequence = test_generator.flow(range(len(test_gt)), targets=test_gt.values, batch_size=1)

auc = tf.keras.metrics.AUC()
es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


with tf.device('/CPU:0'):
    gc_model = GCNSupervisedGraphClassification(
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
    model.compile(
        optimizer=Adam(0.001, amsgrad=True),
        loss=binary_crossentropy,
        metrics=[auc, 'acc']
    )

    history = model.fit(
        train_sequence,
        epochs=MAX_EPOCHS,
        validation_data=val_sequence,
        verbose=2,
        callbacks=[es_callback]
    )

    sg.utils.plot_history(history)
    plt.show()

# TEST MODEL ===========================================================================================================
    print('ALL:')
    all_predictions = [x[0] > 0.5 for x in model.predict(all_sequence).tolist()]
    graph_names = [g[2] for g in graphs_gt_stellar]

    df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": all_predictions, "True is Fraud": graph_labels_all})
    df.to_csv(cfg.STORAGE_ROOT_PATH + r'\result_train.csv', sep=';')

    print(tf.math.confusion_matrix(labels=graph_labels_all, predictions=all_predictions).numpy())

    print('TRAIN:')
    test_predictions = [x[0] > 0.5 for x in model.predict(test_sequence).tolist()]
    graph_names = [g[2] for g in test_subjects]

    df = pd.DataFrame({"Slice": graph_names, "Predicted is Fraud": test_predictions, "True is Fraud": graphs_stellar_test})
    df.to_csv(cfg.STORAGE_ROOT_PATH + r'\result_test.csv', sep=';')

    print(tf.math.confusion_matrix(labels=graph_labels_test, predictions=test_predictions).numpy())
