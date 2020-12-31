from data.DatabaseSlicer import DatabaseSlicer
from data.Database import Database
from typing import List

import math
import numpy as np


class TimeseriesExtractor:
    def __init__(self, db: Database, max_simulation_time: int):
        self._db = db
        self._max_simulation_time = max_simulation_time

    # labels: 0: no_fraud, 1: fraud
    def generate_timeseries(self, window_duration, attributes: List[str], types: List[str]) -> (np.ndarray, np.ndarray):
        db_slicer = DatabaseSlicer(db=self._db, max_simulation_time=self._max_simulation_time)
        slices = db_slicer.generate_slices_sliding_window(window_duration=window_duration)[1:] # test ignore 1st timestamp
        slices_lengths = [len(db.get_table('MTA_CHANGES').get_data()) for db in slices]
        max_slice_length = max(slices_lengths)

        feature_count = len(attributes) + len(types)

        # [batch, timesteps, feature]
        timeseries = np.ndarray(shape=(len(slices), 2 * max_slice_length, feature_count))

        # [batch]
        labels = np.ndarray(shape=(len(slices),))

        for index, db in enumerate(slices):
            series, label = self.serialize_timeseries(db, 2 * max_slice_length, attributes, types)

            labels[index] = label
            timeseries[index::] = series

        return timeseries, labels

    def serialize_timeseries(self, db: Database, max_slice_length, attributes: List[str], types: List[str]) -> (np.ndarray, str):
        feature_count = len(attributes) + len(types)
        tbl_meta_changes = db.get_table('MTA_CHANGES').get_data()

        series = np.ndarray(shape=(max_slice_length, feature_count))
        series.fill(-1)
        label = 0 # no fraud

        # generate nodes
        row_index = 0
        for _, row in tbl_meta_changes.iterrows():
            dst_table_name, dst_column_name = row['table_column_ref'].split('.', 2)

            # insert change record
            for type_index, type in enumerate(types):
                series[row_index, type_index] = 1 if type == 'MTA_CHANGES' else 0

            for attribute_index, attribute in enumerate(attributes):
                series[row_index, len(types) + attribute_index] = self.get_attribute_normalized(row, attribute)

            if 'is_fraud' in row and row['is_fraud'] == True:
                label = 1  # fraud

            row_index += 1

            # insert actual record
            for type_index, type in enumerate(types):
                series[row_index, type_index] = 1 if type == dst_table_name else 0

            dst_table = db.get_table(dst_table_name)
            dst_record = dst_table.get_record(row['record_id'])
            for attribute_index, attribute in enumerate(attributes):
                series[row_index, len(types) + attribute_index] = self.get_attribute_normalized(dst_record, attribute)

            if 'is_fraud' in dst_record and dst_record['is_fraud'] == True:
                label = 1 # fraud

            row_index += 1

        return series, label

    def get_attribute_normalized(self, record, attribute_name):
        if attribute_name not in record:
            return -1

        if isinstance(record[attribute_name], bool):
            return 1 if record[attribute_name] else 0

        if isinstance(record[attribute_name], int) or isinstance(record[attribute_name], float):
            if np.isnan(record[attribute_name]):
                return -1
            else:
                if record[attribute_name] == 0:
                    return record[attribute_name]
                else:
                    return math.log(record[attribute_name])

        if isinstance(record[attribute_name], str):
            num = float(record[attribute_name])

            if np.isnan(num):
                return -1
            else:
                if num == 0:
                    return num
                else:
                    return math.log(num)

        return -1
