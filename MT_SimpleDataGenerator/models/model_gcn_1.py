from __future__ import division
from __future__ import print_function
import pandas as pd
from bunch import Bunch
import database_config
import config as cfg
from data.DatabaseSlicer import DatabaseSlicer
from graph.GraphGenerator import GraphGenerator
from graph.GraphCollection import GraphCollection

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

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

# with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\generate_graphs.bat', 'w') as graphviz_script:
#     for index, history_item in enumerate(graphs):
#         history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\{history_item.get_name()}.txt')
#         print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg', file=graphviz_script)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=cfg.RANDOM_SEED_MODEL, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,  help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../pygcn/data/cora/")

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # print('\nRESULTS:\n---------------------')
    # print('Accuracy:', accuracy_score(target_test, predictions))
    # print('Confusion Matrix:\n', confusion_matrix(target_test, predictions))
    # plot_confusion_matrix(decision_tree, data_test, target_test, cmap='RdYlGn', normalize='true')
    # plt.show()

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


# Configuration
TEST_SET_SIZE = 1500

# Get Data
changes_df = database_config.db.get_table('MTA_CHANGES').get_data()
update_changes_df = pd.DataFrame(changes_df[changes_df['change_type'] == 'update'])

update_changes_df['price_delta'] = update_changes_df.old_value.astype(float) - update_changes_df.new_value.astype(float)

dataset = Bunch(
    data=update_changes_df[['price_delta']].values,
    # data=update_changes_df[['old_value', 'new_value']].values,
    # data=update_changes_df[['old_value', 'new_value', 'salesperson_id']].values,
    target=update_changes_df['is_fraud'].values
)

data_train = dataset.data[:-TEST_SET_SIZE]
target_train = dataset.target[:-TEST_SET_SIZE]

data_test = dataset.data[-TEST_SET_SIZE:]
target_test = dataset.target[-TEST_SET_SIZE:]

