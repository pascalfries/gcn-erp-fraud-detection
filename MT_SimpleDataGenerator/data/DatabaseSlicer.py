import sys
from typing import List
from data.Database import Database


class DatabaseSlicer:
    def __init__(self, db: Database, max_simulation_time: int):
        self._db = db
        self._max_simulation_time = max_simulation_time

    def generate_slices_sliding_window(self, window_duration: int, window_stride: int = 1, normalize_z_score: bool = False) -> List[Database]:
        databases_history = []

        changes_table = self._db.get_table('MTA_CHANGES').get_data()

        for start_time in range(1, self._max_simulation_time - window_duration + 2, window_stride):
            end_time = start_time + window_duration - 1

            new_db = self._db.copy()
            new_db.disable_tracing()
            new_db.set_name(f'{new_db.get_name()}_{start_time}_{end_time}')
            print(f'---> GENERATING DB WINDOW {new_db.get_name()} FOR INTERVAL [{start_time}; {end_time}]')

            changes_to_undo = changes_table[changes_table['timestamp'] >= start_time]

            changed_keys = []
            for index, change_record in changes_to_undo[::-1].iterrows():
                # print(f'REVERTING change {index}')
                dst_table_name, dst_column_name = change_record['table_column_ref'].split('.', 2)

                if change_record['timestamp'] <= end_time:
                    item_key = f'{dst_table_name}[{change_record["record_id"]}]'
                    changed_keys.append(item_key)

                    if item_key in changed_keys:
                        continue

                if change_record['change_type'] == 'create':
                    new_db.get_table(dst_table_name).remove_record(change_record['record_id'])
                elif change_record['change_type'] == 'update':
                    new_db.get_table(dst_table_name).update_record(change_record['record_id'], dst_column_name,
                                                                   change_record['old_value'])
                elif change_record['change_type'] == 'delete':
                    new_db.get_table(dst_table_name).insert_record_with_id(change_record['record_id'],
                                                                           list(change_record['old_value'].values()))
                else:
                    print(f'Invalid change type: {change_record["change_type"]}', file=sys.stderr)

            for table in new_db.get_tables():
                for column in table.get_columns().values():
                    if column.get_is_timestamp():
                        table_data = table.get_data()
                        # invert filter and replace entire data-backing df
                        records_to_keep = table_data[(table_data[column.get_name()] >= start_time) & (table_data[column.get_name()] <= end_time)]
                        table.set_data(records_to_keep)
                        break

            if normalize_z_score:
                new_db.normalize_z_score()

            databases_history.append(new_db)
        
        return databases_history

    def generate_slices_per_timestamp(self) -> List[Database]: # obsolete
        databases_history = []

        for time in self._db.get_table('MTA_CHANGES').get_data()['timestamp'].unique():
            new_db = self._db.copy()
            new_db.disable_tracing()
            new_db.set_name(f'{new_db.get_name()}_{time}')
            print(f'---> GENERATING DB SNAPSHOT {new_db.get_name()} FOR TIME {time}')

            changes = self._db.get_table('MTA_CHANGES').get_data()[self._db.get_table('MTA_CHANGES').get_data()['timestamp'] > time]

            for index, change_record in changes[::-1].iterrows():
                print(f'REVERTING change {index}')
                dst_table_name, dst_column_name = change_record['table_column_ref'].split('.', 2)

                if change_record['change_type'] == 'create':
                    new_db.get_table(dst_table_name).remove_record(change_record['record_id'])
                elif change_record['change_type'] == 'update':
                    new_db.get_table(dst_table_name).update_record(change_record['record_id'], dst_column_name, change_record['old_value'])
                elif change_record['change_type'] == 'delete':
                    new_db.get_table(dst_table_name).insert_record_with_id(change_record['record_id'], list(change_record['old_value'].values()))
                else:
                    print(f'Invalid change type: {change_record["change_type"]}', file=sys.stderr)

            for table in new_db.get_tables():
                for column in table.get_columns().values():
                    if column.get_is_timestamp():
                        for index, record in table.get_data().iterrows():
                            if record[column.get_name()] > time:
                                table.remove_record(index)

            databases_history.append(new_db)

        return databases_history
