import sys
import json
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from data.DataColumn import DataColumn
from data.ForeignKey import ForeignKey


class DataTable:
    def __init__(self, name: str,
                 trace_changes: bool = False,
                 is_mapping_table: bool = False,
                 data: Optional[pd.DataFrame] = None):
        self._name: str = name
        self._trace_changes: bool = trace_changes
        self._enable_tracing: bool = True
        self._is_mapping_table: bool = is_mapping_table
        self._data: pd.Dataframe = data if data is not None else pd.DataFrame()
        self._parent_db = None
        self._foreign_keys: List[ForeignKey] = []

        if len(self._data) > 0:
            self._columns: Dict[str, DataColumn] = {column_name: DataColumn(column_name) for column_name in
                                                    self._data.columns.values}
        else:
            self._columns: Dict[str, DataColumn] = {}

    def get_random_item(self):
        sample = self._data.sample()
        return sample.iloc[0], sample.index[0]

    def get_random_item_id(self) -> int:
        return self._data.sample().index[0]

    def enable_tracing(self) -> None:
        self._enable_tracing = True

    def disable_tracing(self) -> None:
        self._enable_tracing = False

    def get_name(self) -> str:
        return self._name

    def get_data(self) -> pd.DataFrame:
        return self._data

    def set_data(self, new_data: pd.DataFrame) -> None:
        self._data = new_data

    def get_data_is_time_dependent(self) -> bool:
        for column in self._columns.values():
            if column.get_is_timestamp():
                return True

        return False

    def get_is_mapping_table(self) -> bool:
        return self._is_mapping_table

    def get_columns(self) -> Dict[str, DataColumn]:
        return self._columns

    def get_all_foreign_keys(self) -> List[ForeignKey]:
        return [fk for column in self._columns.values() for fk in column.get_foreign_keys()]

    def add_columns(self, columns: List[DataColumn], reindex=True) -> None:
        for column in columns:
            self._columns[column.get_name()] = column

            if reindex:
                self._data[column.get_name()] = None

        if reindex and not self._is_mapping_table:
            self._data.set_index([c.get_name() for c in columns if c.get_is_primary()], inplace=True)

    def link_database(self, db):
        self._parent_db = db

    def insert_record(self, record, is_fraud: bool = False, fraud_id: str = '', changed_by=0) -> int:
        return self.insert_record_with_id(len(self._data), record, is_fraud, fraud_id, changed_by)

    def insert_record_with_id(self, new_id: int, record, is_fraud: bool = False, fraud_id: str = '', changed_by=0) -> int:
        self._data.loc[new_id] = record

        if self._trace_changes and self._enable_tracing:
            change_record = [f'{self._name}.*', new_id, 'create', None, self._data.loc[new_id].to_dict(), changed_by, self._parent_db.get_simulation_time(), is_fraud, fraud_id]
            self._parent_db.get_changelog().insert_record(change_record)

        return new_id

    def insert_records(self, records: list, is_fraud: bool = False, fraud_id: str = '', changed_by=0) -> None:
        for record in records:
            self.insert_record(record, is_fraud, fraud_id, changed_by)

    def truncate(self) -> None:
        self._data = self._data[0:0]

    def remove_record(self, num: int, is_fraud: bool = False, fraud_id: str = '', changed_by=0):
        old_record = self.get_record(num)

        if old_record is not None:
            self._data.drop(num, inplace=True)

            if self._trace_changes and self._enable_tracing:
                change_record = [f'{self._name}.*', num, 'delete', old_record.to_dict(), None, changed_by, self._parent_db.get_simulation_time(), is_fraud, fraud_id]
                self._parent_db.get_changelog().insert_record(change_record)

    def get_record(self, num: int):
        if num in self._data.index:
            return self._data.loc[num]
        else:
            return None

    def find_record(self, column_name: str, search_value):
        return self.get_record(self.find_record_index(column_name, search_value))

    def find_record_index(self, column_name: str, search_value):
        if self._columns[column_name].get_is_primary():
            return search_value

        return self._data.index[self._data[column_name] == search_value].tolist()[0]

    def update_record(self, num: int, col_name: str, new_value, is_fraud: bool = False, fraud_id: str = '', changed_by=0):
        if col_name in self._columns.keys():
            old_record = self.get_record(num)

            if old_record is not None:
                old_value = old_record[col_name]
                self._data.loc[num, col_name] = new_value

                if self._trace_changes and self._enable_tracing:
                    change_record = [f'{self._name}.{col_name}', num, 'update', old_value, new_value, changed_by, self._parent_db.get_simulation_time(), is_fraud, fraud_id]

                    self._parent_db.get_changelog().insert_record(change_record)
        else:
            print(f'ERROR: Column {col_name} does not exist in table {self._name}.', file=sys.stderr)

    def save(self, file_name_path: str) -> None:
        self._data.to_csv(file_name_path, sep=';')

    def load(self, file_name_path: str) -> None:
        self._data = pd.read_csv(file_name_path, sep=';')

        if not self._is_mapping_table:
            self._data.set_index([c.get_name() for c in self._columns.values() if c.get_is_primary()], inplace=True)
        else:
            self._data.drop('Unnamed: 0', axis=1, inplace=True)

        if self.get_name() == 'MTA_CHANGES':
            self._data['old_value'] = self._data['old_value'].map(
                lambda i: None if pd.isna(i) else json.loads(str(i).replace('\'', '"')) if str(i).startswith('{') else i, na_action=None
            )

            self._data['new_value'] = self._data['new_value'].map(
                lambda i: None if pd.isna(i) is None else json.loads(str(i).replace('\'', '"')) if str(i).startswith('{') else i, na_action=None
            )

    def normalize_z_score(self):
        for column_name in self._data.columns:
            column_data = self._data[column_name]

            if not np.issubdtype(column_data, np.number):
                continue

            self._data [column_name] = ((column_data - np.nanmean(column_data)) / np.nanstd(column_data))

    def copy(self):
        table = DataTable(self._name, self._trace_changes, self._is_mapping_table, self._data.copy(deep=True))
        table.add_columns(list(self._columns.values()), False)

        return table

    def __len__(self) -> int:
        return len(self._data)
