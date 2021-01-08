from typing import Dict, List
import os
from data.DataTable import DataTable
from data.ForeignKey import ForeignKey
from data.DataColumn import DataColumn


class Database:
    def __init__(self, name: str):
        self._simulation = None
        self._name: str = name
        self._tables: Dict[str, DataTable] = {}

        self._meta_table_changes = DataTable('MTA_CHANGES')
        self._meta_table_changes.add_columns(
            [DataColumn('id', is_primary=True), DataColumn('table_column_ref', is_hidden=True), DataColumn('record_id', is_hidden=True), DataColumn('change_type'),
             DataColumn('old_value'), DataColumn('new_value'), DataColumn('salesperson_id', is_hidden=True),
             DataColumn('timestamp', is_timestamp=True, is_hidden=False), DataColumn('is_fraud'), DataColumn('fraud_id')])
        self._meta_table_changes.link_database(self)
        self.add_table(self._meta_table_changes)

    def get_name(self) -> str:
        return self._name

    def get_changelog(self) -> DataTable:
        return self._meta_table_changes

    def set_name(self, name: str) -> None:
        self._name = name

    def bind_simulation(self, simulation) -> None:
        self._simulation = simulation

    def get_simulation_time(self) -> int:
        return self._simulation.get_current_time() if self._simulation is not None else 0

    def add_table(self, table: DataTable) -> None:
        self._tables[table.get_name()] = table
        table.link_database(self)

    def truncate_tables(self) -> None:
        for table in self._tables.values():
            table.truncate()

    def get_table(self, table_name: str) -> DataTable:
        if table_name in self._tables.keys():
            return self._tables[table_name]
        else:
            print(f'ERROR: Table {table_name} does not exits in DB.')

    def get_tables(self) -> List[DataTable]:
        return list(self._tables.values())

    def add_foreign_key(self, foreign_key: ForeignKey) -> None:
        if foreign_key.get_src_table() in self._tables.keys() and foreign_key.get_dst_table() in self._tables.keys():
            src_table = self._tables[foreign_key.get_src_table()]
            dst_table = self._tables[foreign_key.get_dst_table()]

            if foreign_key.get_src_column() in src_table.get_columns().keys() and foreign_key.get_dst_column() in dst_table.get_columns().keys():
                src_table.get_columns()[foreign_key.get_src_column()].add_foreign_key(foreign_key)

                # print(f'Adding FK {foreign_key.get_key_name()} to {foreign_key.get_src_table()}.{foreign_key.get_src_column()} -> {foreign_key.get_dst_table()}.{foreign_key.get_dst_column()}.')
            else:
                print(
                    f'ERROR: Column {foreign_key.get_src_table()}.{foreign_key.get_src_column()} or column {foreign_key.get_dst_table()}.{foreign_key.get_dst_column()} doesn\'t exist.')
        else:
            print(
                f'ERROR: Table {foreign_key.get_src_table()} or table {foreign_key.get_dst_table()} does not exits in DB.')

    def save(self, directory_name: str) -> None:
        for table in self._tables.values():
            table.save(f'{os.path.join(directory_name, table.get_name())}.csv')

    def load(self, directory_name: str) -> None:
        for file_name in os.listdir(directory_name):
            if file_name.endswith('.csv'):
                table_name = file_name[:-4]

                if table_name in self._tables.keys():
                    self._tables[table_name].load(os.path.join(directory_name, file_name))

    def copy(self):
        new_db = Database(self._name)

        for table in self._tables.values():
            new_db.add_table(table.copy())

        return new_db

    def normalize_z_score(self):
        pass # todo implement

    def enable_tracing(self) -> None:
        for table in self._tables.values():
            table.enable_tracing()

    def disable_tracing(self) -> None:
        for table in self._tables.values():
            table.disable_tracing()
