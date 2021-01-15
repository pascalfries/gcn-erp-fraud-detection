from sklearn.metrics import confusion_matrix

import stellargraph as sg
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
