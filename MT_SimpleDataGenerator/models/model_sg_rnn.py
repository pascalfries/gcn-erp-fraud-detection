from keras.utils.test_utils import get_test_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

from graph.GraphGenerator import GraphGenerator
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import stellargraph as sg
import config as cfg
import time
import database_config
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from helpers import set_all_seeds
from sklearn.metrics import confusion_matrix

# todo extract all timestamped nodes in order
# todo https://www.tensorflow.org/guide/keras/rnn

# todo masking for non-existing attributes


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
(x, labels), _ = get_test_data(num_train=600, input_shape=(5, 10), classification=True, num_classes=2)
print('X_train:', labels.shape)
print('X_test:', labels.shape)

labels = pd.DataFrame(labels)

# DATA OVERVIEW ========================================================================================================

# TRAIN/TEST SPLIT =====================================================================================================
train_subjects, test_subjects = model_selection.train_test_split(
    labels, train_size=int(len(labels) * TRAIN_SIZE_RELATIVE), test_size=None, stratify=labels
)

val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=int(len(test_subjects) * VALIDATION_SIZE_RELATIVE_TRAIN), test_size=None
)

print('------------------------\nALL:\n', labels.value_counts())
print('------------------------\nTRAIN:\n', train_subjects.value_counts())
print('------------------------\nTEST:\n', test_subjects.value_counts())
print('------------------------\nVALIDATION:\n', val_subjects.value_counts())
#todo finish/fix train/test/val split

# MAIN CODE ============================================================================================================
train_subjects = to_categorical(train_subjects)
test = to_categorical(test_subjects)
val_subjects = to_categorical(val_subjects)

es_callback = EarlyStopping(monitor="val_acc", patience=5, restore_best_weights=True)
auc = tf.keras.metrics.AUC()

model = Sequential([
    GRU(2),
    # tf.keras.layers.Dropout(0.5),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.05, amsgrad=True), metrics=['acc', auc])
history = model.fit(
    x=train_subjects,
    y=test_subjects,
    epochs=EPOCHS,
    batch_size=16,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[es_callback],
    verbose=2,
    class_weight={0: 0.90, 1: 0.05}
)

sg.utils.plot_history(history)
plt.show()


exit()


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



# MAIN CODE ============================================================================================================
if False:
    es_callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    auc = tf.keras.metrics.AUC()

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    generator = FullBatchNodeGenerator(graph_stellar, method="gcn")

    train_gen = generator.flow(train_subjects.index, train_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    all_gen = generator.flow(graph_labels.index, graph_labels)

    all_predictions = model.predict(all_gen)
    node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
    df = pd.DataFrame({"Predicted": node_predictions, "True": graph_labels, "RAW": [str(x) for x in all_predictions.squeeze()]})
    df['is_correct'] = df['Predicted'] == df['True']
    df_err = df[df['is_correct'] == False]
    df_err.to_csv(r'C:\Users\Pasi\Desktop\results.csv', sep=';')
    print(df_err)
    print(df['is_correct'].value_counts())

    print(confusion_matrix(df['True'], df['Predicted']))