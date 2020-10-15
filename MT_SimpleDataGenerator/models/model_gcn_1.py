from __future__ import division
from __future__ import print_function
import pandas as pd
import scipy.sparse as sp
from bunch import Bunch
from sklearn.metrics import accuracy_score

import database_config
import config as cfg
from data.DatabaseSlicer import DatabaseSlicer
from graph.GraphGenerator import GraphGenerator
from graph.GraphCollection import GraphCollection

import random
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, encode_onehot
from pygcn.models import GCN


# helpers
def readout_output(output, mode: str = 'node'):
    if mode == 'node':
        return output
    elif mode == 'graph':
        contains_fraud = any([node[1] >= node[0] for node in output])
        return torch.tensor([[0.001, 0.999] if contains_fraud else [0.999, 0.001]])


def readout_labels(labels, mode: str = 'node'):
    if mode == 'node':
        return labels
    elif mode == 'graph':
        return torch.tensor([int(sum(labels).item() > 0)], dtype=torch.long)


def is_fraud_output(output, mode: str = 'node'):
    if mode == 'node':
        return output
    elif mode == 'graph':
        return any([node[1] >= node[0] for node in output])


def if_fraud_target(labels, mode: str = 'node'):
    if mode == 'node':
        return labels
    elif mode == 'graph':
        return sum(labels).item() > 0

# Get Data
database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)

# Generate Slices & Graphs
GENERATE_GRAPHS_FROM_DB = False

if GENERATE_GRAPHS_FROM_DB:
    db_slicer = DatabaseSlicer(db=database_config.db, max_simulation_time=cfg.SIMULATION_END_TIME)
    slices = db_slicer.generate_slices_sliding_window(window_duration=2)

    graph_gen = GraphGenerator()
    graphs = graph_gen.generate_graphs(databases=slices)
    graphs.save(cfg.STORAGE_BASE_PATH_PY_GRAPHS)
else:
    graphs = GraphCollection()
    graphs.load(cfg.STORAGE_BASE_PATH_PY_GRAPHS)

with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\generate_graphs.bat', 'w') as graphviz_script:
    for index, history_item in enumerate(graphs.get_raw_list()):
        history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\{history_item.get_name()}.txt')
        print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg', file=graphviz_script)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=cfg.RANDOM_SEED_MODEL, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,  help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--classification_mode', type=str, default='node', help='Node- vs Graph-level classification (values: node, graph))')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
dataset = []
graphs.save_numpy(r'C:\Users\Pasi\OneDrive\Documents\Uni\MSem. 4 - SS 20\MT - Master Thesis\Simulator and Models\MT_SimpleDataGenerator\pygcn\data\sdg_fraud\\', ['price', 'new_value', 'old_value'])

for graph in graphs.get_raw_list():
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../pygcn/data/sdg_fraud/", dataset=graph.get_name(), train_size=len(graph) - 1, validation_size=0)

    dataset.append(Bunch(
        name=graph.get_name(),
        adj=adj,
        features=features,
        # labels_raw=labels,
        labels=readout_labels(labels, args.classification_mode),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test
    ))

random.shuffle(dataset)
TRAIN_SET = dataset[:550]
VALIDATION_SET = dataset[550:600]
TEST_SET = dataset[550:-1]

# Model and optimizer
model = GCN(nfeat=dataset[0].features.shape[1],
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()

    avg_loss_train = 0
    avg_acc_train = 0
    avg_loss_validation = 0
    avg_acc_validation = 0

    for dataitem in TRAIN_SET:
        # print(f'Traing {dataitem.name}')
        optimizer.zero_grad()

        output = model(dataitem.features, dataitem.adj)
        output_transformed = readout_output(output, args.classification_mode)

        loss_train = F.nll_loss(output_transformed, dataitem.labels)
        avg_acc_train += accuracy(output_transformed, dataitem.labels).item() / len(TRAIN_SET)

        avg_loss_train += loss_train.item()

        loss_train.backward()
        optimizer.step()

    if not args.fastmode:
        for dataitem in VALIDATION_SET:
            model.eval()
            output = model(dataitem.features, dataitem.adj)

            output_transformed = readout_output(output, args.classification_mode)

            avg_loss_validation += F.nll_loss(output_transformed, dataitem.labels) / len(VALIDATION_SET)
            avg_acc_validation += accuracy(output_transformed, dataitem.labels) / len(VALIDATION_SET)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(avg_loss_train),
          'acc_train: {:.4f}'.format(avg_acc_train),
          'loss_val: {:.4f}'.format(avg_loss_validation),
          'acc_val: {:.4f}'.format(avg_acc_validation),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    avg_loss_train = 0
    avg_accuracy_train = 0
    correct_graph_classification_count = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for dataitem in TEST_SET:
        output = model(dataitem.features, dataitem.adj)

        avg_loss_train += F.nll_loss(output, dataitem.labels).item() / len(TEST_SET)
        avg_accuracy_train += accuracy(output, dataitem.labels).item() / len(TEST_SET)

        pred = is_fraud_output(output, "graph")
        gt = if_fraud_target(dataitem.labels, "graph")

        if gt and pred:
            tp += 1
        elif not gt and not pred:
            tn += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1

        if is_fraud_output(output, "graph") == if_fraud_target(dataitem.labels, "graph"):
            correct_graph_classification_count += 1
            print(f'GRAPH: {dataitem.name}\nis_fraud_output: {is_fraud_output(output, "graph")}\nis_fraud_target: {if_fraud_target(dataitem.labels, "graph")}\n')

    print("Test set results:",
          "\nloss= {:.4f}".format(avg_loss_train),
          "\naccuracy= {:.4f}".format(avg_accuracy_train),
          f'\ncorrect graph class.: {correct_graph_classification_count} / {len(TEST_SET)}')

    print("\nConfusion Matrix (GT\\Pred.):",
          f"\n{tn} | {fp}",
          f"\n{fn} | {tp}")


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
