import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, plot_confusion_matrix
from bunch import Bunch
import database_config
import config as cfg


# Configuration
TEST_SET_SIZE = 1500

# Model Setup
logistic_regressor = linear_model.LogisticRegression()

# Get Data
database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
changes_df = database_config.db.get_table('MTA_CHANGES').get_data()
update_changes_df = pd.DataFrame(changes_df[changes_df['change_type'] == 'update'])

update_changes_df['price_delta'] = update_changes_df.old_value.astype(float) - update_changes_df.new_value.astype(float)

dataset = Bunch(
    data=[abs(delta) for delta in update_changes_df[['price_delta']].values],
    target=update_changes_df['is_fraud'].values
)

data_train = dataset.data[:-TEST_SET_SIZE]
target_train = dataset.target[:-TEST_SET_SIZE]

data_test = dataset.data[-TEST_SET_SIZE:]
target_test = dataset.target[-TEST_SET_SIZE:]

print('Size Train:', len(data_train))
print('Size Test:', len(data_test))

# Fit Model
logistic_regressor.fit(data_train, target_train)


# Evaluate Results
predictions = logistic_regressor.predict(data_test)

print('\nRESULTS:\n---------------------')
print('Accuracy:', accuracy_score(target_test, predictions))
print('ROC AUC:', roc_auc_score(target_test, predictions))
print('Confusion Matrix:\n', confusion_matrix(target_test, predictions))
plot_confusion_matrix(logistic_regressor, data_test, target_test, cmap='RdYlGn', normalize='true')
plt.show()
