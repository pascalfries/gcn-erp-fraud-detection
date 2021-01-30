from helpers import apply_model_to_sap_data, apply_model_to_sap_data_timeseries

import tensorflow as tf
import config as cfg


TIMESERIES_GEN_WINDOW_DURATION = 2

NODE_FEATURES = ['price', 'old_value', 'new_value']
NODE_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']

ITEM_FEATURES = ['price', 'old_value', 'new_value', 'timestamp', 'record_id']
ITEM_TYPES = ['MST_PRODUCTS', 'MST_CUSTOMERS', 'MST_SALESPERSONS', 'TRC_SALES', 'MTA_CHANGES', 'TRM_SALE_PRODUCTS',
              'MST_ADDRESSES']


with tf.device('/CPU:0'):
    # APPLY GRAPH GCN TO SAP DATA ======================================================================================
    # model = tf.keras.models.load_model(cfg.STORAGE_BASE_PATH_MODELS + rf'\gcn_graph_{TIMESERIES_GEN_WINDOW_DURATION}')
    # apply_model_to_sap_data(model, NODE_FEATURES, NODE_TYPES)


    # APPLY RNN TO SAP DATA ============================================================================================
    model = tf.keras.models.load_model(cfg.STORAGE_BASE_PATH_MODELS + rf'\rnn_{TIMESERIES_GEN_WINDOW_DURATION}')
    apply_model_to_sap_data_timeseries(model, ITEM_FEATURES, ITEM_TYPES)
