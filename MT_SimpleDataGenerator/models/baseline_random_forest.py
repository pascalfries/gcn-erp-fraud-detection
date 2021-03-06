import matplotlib.pyplot as plt
from sklearn import ensemble
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from bunch import Bunch
import database_config
import config as cfg


# Configuration
TRAIN_SIZE_RELATIVE = 0.75
VIEW_GRAPH = False

# Model Setup
random_forest = ensemble.RandomForestClassifier(n_estimators=10, max_depth=5)

# Get Data
database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
changes_df = database_config.db.get_table('MTA_CHANGES').get_data()
update_changes_df = pd.DataFrame(changes_df[changes_df['change_type'] == 'update'])

update_changes_df['price_delta'] = update_changes_df.old_value.astype(float) - update_changes_df.new_value.astype(float)

dataset = Bunch(
    # data=update_changes_df[['price_delta']].values,
    # data=update_changes_df[['old_value', 'new_value']].values,
    data=update_changes_df[['old_value', 'new_value', 'price_delta']].values,
    # data=update_changes_df[['old_value', 'new_value', 'salesperson_id']].values,
    target=update_changes_df['is_fraud'].values
)

test_set_size = int(len(dataset.data) * TRAIN_SIZE_RELATIVE)
data_train = dataset.data[:-test_set_size]
target_train = dataset.target[:-test_set_size]

data_test = dataset.data[-test_set_size:]
target_test = dataset.target[-test_set_size:]

print('Size Train:', len(data_train))
print('Size Test:', len(data_test))

# Fit Model
random_forest.fit(data_train, target_train)


# Evaluate Results
predictions = random_forest.predict(data_test)

print('\nRESULTS:\n---------------------')
print('Accuracy:', accuracy_score(target_test, predictions))
print('Confusion Matrix:\n', confusion_matrix(target_test, predictions))
plot_confusion_matrix(random_forest, data_test, target_test, cmap='RdYlGn', normalize='true')
plt.show()
