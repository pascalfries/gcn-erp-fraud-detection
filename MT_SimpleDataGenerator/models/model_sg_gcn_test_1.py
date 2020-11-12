import pandas as pd
import numpy as np
from stellargraph.mapper import FullBatchNodeGenerator, PaddedGraphGenerator
from stellargraph.layer import GCN, GCNSupervisedGraphClassification
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import tensorflow as tf
import config as cfg
from graph.GraphCollection import GraphCollection


# Load
graphs = GraphCollection(directory_name=cfg.STORAGE_ROOT_PATH + r'\gcn_testing')
graphs_gt_stellar = graphs.serialize_stellargraph(['old_price', 'new_price'], ['PRODUCT', 'CHANGE'])

graphs_stellar = [item[0] for item in graphs_gt_stellar]
graph_labels = [item[1] for item in graphs_gt_stellar]

# DATA OVERVIEW ===================================================================================================================
print(graphs_stellar[0].info())

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
auc = tf.keras.metrics.AUC()
def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[12, 12],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.0,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="sigmoid")(x_out)
    predictions = Dense(units=16, activation="sigmoid")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=['acc', auc])

    return model


def train_fold(model, train_gen, test_gen, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0,
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


epochs = 20  # maximum number of training epochs
folds = 4  # the number of folds for k-fold cross validation

test_accs = []
test_aucs = []

stratified_folds = model_selection.StratifiedKFold(
    n_splits=folds
).split(graph_labels, graph_labels)

time_start = time.perf_counter()
model = None
# with tf.device('/CPU:0'):
for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds}...")
    train_gen, test_gen = get_generators(
        train_index, test_index, graph_labels, batch_size=10
    )

    model = create_graph_classification_model(generator)

    history, acc, auc_val = train_fold(model, train_gen, test_gen, epochs)

    test_accs.append(acc)
    test_aucs.append(auc_val)

print(
    f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%\n"
    f"AUC over all folds mean: {np.mean(test_aucs):.3}"
)
time_end = time.perf_counter()
print(f"Training took {time_end - time_start:0.4f} seconds")

plt.figure(figsize=(8, 6))
plt.hist([test_accs, test_aucs], label=['ACC', 'AUC'])
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.legend(loc='upper left')
plt.show()

# predict all
all_gen = generator.flow(graphs_stellar)
all_predictions = [x[0] > 0.5 for x in model.predict(all_gen).tolist()]

df = pd.DataFrame({"Predicted is Fraud": all_predictions, "True is Fraud": [item[1] for item in graphs_gt_stellar]})
df.to_csv(r'C:\Users\Pasi\Desktop\result.csv', sep=';')
