from sklearn.metrics import confusion_matrix
from sap.SapExtractor import SapExtractor
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from timeseries.TimeseriesExtractor import TimeseriesExtractor

import config as cfg
import stellargraph as sg
import keras.backend as K
import tensorflow as tf
import numpy as np
import stellargraph.random as sgrand
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import os


def with_probability(probability: float) -> bool:
    if probability >= 1:
        return True
    elif probability <= 0:
        return False

    return random.randint(1, 100000) <= 100000 * probability


def rand_float(min: float = 0, max: float = 0) -> float:
    return random.randint(int(100000 * min), int(100000 * max)) / 100000.0


def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sgrand.set_seed(seed)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def plot_history(history, es_callback, title=None, save_path=None, sma_size=None):
    fig = sg.utils.plot_history(history, return_figure=True)

    fig.axes[0].axvline(es_callback.stopped_epoch - es_callback.patience, color='#00c700', linestyle='dashed')
    fig.axes[1].axvline(es_callback.stopped_epoch - es_callback.patience, color='#00c700', linestyle='dashed')
    fig.axes[2].axvline(es_callback.stopped_epoch - es_callback.patience, color='#00c700', linestyle='dashed')

    if sma_size is not None:
        fig.axes[0].plot(history.epoch[:-sma_size + 1], moving_average(history.history['acc'], sma_size).tolist(),
                         color='#4eb5fc',
                         linestyle='dashed',
                         label=f'SMA {sma_size} train')
        fig.axes[0].plot(history.epoch[:-sma_size + 1], moving_average(history.history['val_acc'], sma_size).tolist(),
                         color='#ffb06b',
                         linestyle='dashed',
                         label=f'SMA {sma_size} validation')

        fig.axes[1].plot(history.epoch[:-sma_size + 1], moving_average(history.history['auc'], sma_size).tolist(),
                         color='#4eb5fc',
                         linestyle='dashed',
                         label=f'SMA {sma_size} train')
        fig.axes[1].plot(history.epoch[:-sma_size + 1], moving_average(history.history['val_auc'], sma_size).tolist(),
                         color='#ffb06b',
                         linestyle='dashed',
                         label=f'SMA {sma_size} validation')

        fig.axes[2].plot(history.epoch[:-sma_size + 1], moving_average(history.history['loss'], sma_size).tolist(),
                         color='#4eb5fc',
                         linestyle='dashed',
                         label=f'SMA {sma_size} train')
        fig.axes[2].plot(history.epoch[:-sma_size + 1], moving_average(history.history['val_loss'], sma_size).tolist(),
                         color='#ffb06b',
                         linestyle='dashed',
                         label=f'SMA {sma_size} validation')

    if title is not None:
        fig.axes[0].set_title(title, fontsize=18)

    plt.tight_layout(pad=1.06, h_pad=0)

    if save_path is not None:
        plt.draw()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    else:
        plt.show(bbox_inches='tight')


def plot_confusion_matrix(title, predictions, ground_truth, save_path=None):
    plt.rcParams.update({'font.size': 20})

    classes = set(ground_truth)

    conf_matrix = confusion_matrix(ground_truth, predictions, labels=list(classes))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap='RdYlGn')
    plt.tight_layout()
    plt.title(title)

    if len(classes) == 2:
        plt.ylabel('True is_fraud')
        plt.xlabel('Predicted is_fraud')
    else:
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')

    if save_path is not None:
        plt.draw()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    else:
        plt.show(bbox_inches='tight')


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def aggregate_sets(aggr, elem):
    aggr.add(elem)
    return aggr


def aggregate_sets_multi(aggr, set):
    aggr.update(set)
    return aggr


def apply_model_to_sap_data(model, node_features, node_types):
    sap_extractor = SapExtractor(cfg.STORAGE_BASE_SAP_DATA)
    graphs = sap_extractor.extract_slices(cfg.SAP_DATA_WINDOW_DURATION, cfg.SAP_DATA_WINDOW_STRIDE)
    graphs_stellar_with_gt = graphs.serialize_stellargraph(node_features, node_types)
    graphs_stellar_with_gt = [g for g in graphs_stellar_with_gt if len(g[0]._nodes) > 0]

    graphs_stellar = [item[0] for item in graphs_stellar_with_gt]
    graph_labels = [item[1] for item in graphs_stellar_with_gt]
    graph_names = [item[2] for item in graphs_stellar_with_gt]
    all_gt = to_categorical(graph_labels)

    sap_generator = PaddedGraphGenerator(graphs=graphs_stellar)
    sap_sequence = sap_generator.flow(range(len(all_gt)), targets=all_gt, batch_size=1)

    raw_predictions = model.predict(sap_sequence)
    all_predictions = K.argmax(raw_predictions).numpy().tolist()
    print(all_predictions)

    all_predictions_df = pd.DataFrame({"Slice": graph_names,
                       "Predicted is Fraud": all_predictions,
                       "RAW": raw_predictions.tolist(),
                                       "Labels": graph_labels})
    all_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_all_sap_gcn_graph.csv', sep=';')
    plot_confusion_matrix('Confusion Matrix - SAP Data', all_predictions, graph_labels,
                          cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_all_gcn_graph_sap.pdf')


def apply_model_to_sap_data_timeseries(model, node_features, node_types):
    item_feature_count = len(node_features) + len(node_types)
    sap_extractor = SapExtractor(cfg.STORAGE_BASE_SAP_DATA)
    db, min_time, max_time = sap_extractor.extract()

    timeseries_extractor = TimeseriesExtractor(db=db, max_simulation_time=max_time, min_simulation_time=min_time)

    timeseries, labels, _, names = timeseries_extractor.generate_timeseries(cfg.SAP_DATA_WINDOW_DURATION, node_features, node_types, window_stride=cfg.SAP_DATA_WINDOW_STRIDE)

    timeseries[:, :, len(node_types):item_feature_count] = ((timeseries - np.nanmean(timeseries,
                                                                                     axis=(0, 1))) / np.nanstd(
    timeseries, axis=(0, 1)))[:, :, len(node_types):item_feature_count]
    np.nan_to_num(timeseries, copy=False, nan=-1)

    all_gt = to_categorical(labels)

    raw_predictions = model.predict(timeseries)
    all_predictions = K.argmax(raw_predictions).numpy().tolist()
    print(all_predictions)

    all_predictions_df = pd.DataFrame({"Slice": names,
                       "Predicted is Fraud": all_predictions,
                       "RAW": raw_predictions.tolist(),
                                       "Labels": labels})
    all_predictions_df.to_csv(cfg.STORAGE_ROOT_PATH + rf'\results_all_sap_rnn.csv', sep=';')

    plot_confusion_matrix('Confusion Matrix - SAP Data', all_predictions, labels,
                              cfg.STORAGE_BASE_THESIS_IMG + rf'\conf_matrix_all_rnn_sap.pdf')
