import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
import graphviz
from bunch import Bunch
import database_config
import config as cfg


# Configuration
TEST_SET_SIZE = 1500
VIEW_GRAPH = False

# Model Setup
decision_tree = tree.DecisionTreeClassifier(max_depth=4)

# Get Data
database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
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

print('Size Train:', len(data_train))
print('Size Test:', len(data_test))

# Fit Model
decision_tree.fit(data_train, target_train)


# Evaluate Results
predictions = decision_tree.predict(data_test)

print('\nRESULTS:\n---------------------')
print('Accuracy:', accuracy_score(target_test, predictions))
print('Confusion Matrix:\n', confusion_matrix(target_test, predictions))
plot_confusion_matrix(decision_tree, data_test, target_test, cmap='RdYlGn', normalize='true')
plt.show()

# Export Tree to GraphViz
# dot_data = tree.export_graphviz(decision_tree, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render('decision_tree', view=VIEW_GRAPH)
