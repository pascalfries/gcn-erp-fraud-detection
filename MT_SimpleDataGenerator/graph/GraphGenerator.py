import sys
from typing import List, Dict, Union, Any
from graph.Graph import Graph
from data.Database import Database


class GraphGenerator:
    def __init__(self, db: Database, max_simulation_time: int):
        self._db = db
        self._max_simulation_time = max_simulation_time

    def generate_graphs_sliding_window(self, window_duration: int, window_stride: int = 1) -> List[Dict[str, Union[Graph, Any]]]:
        databases_history = []

        for start_time in range(1, self._max_simulation_time - window_duration + 2, window_stride):
            end_time = start_time + window_duration - 1

            new_db = self._db.copy()
            new_db.disable_tracing()
            new_db.set_name(f'{new_db.get_name()}_{start_time}_{end_time}')
            print(f'---> GENERATING DB WINDOW {new_db.get_name()} FOR INTERVAL [{start_time}; {end_time}]')

            changes_table = self._db.get_table('MTA_CHANGES').get_data()
            changes_to_undo = changes_table[(changes_table['timestamp'] > end_time)]

            for index, change_record in changes_to_undo[::1].iterrows():
                # print(f'REVERTING change {index}')
                dst_table_name, dst_column_name = change_record['table_column_ref'].split('.', 2)

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
                        for index, record in table.get_data().iterrows():
                            if record[column.get_name()] < start_time or record[column.get_name()] > end_time:
                                table.remove_record(index)

            databases_history.append(new_db)

        return list(map(lambda db: {'db': db, 'graph': self.generate_graph(db)}, databases_history))

    def generate_graphs_per_timestamp(self) -> List[Dict[str, Union[Graph, Any]]]:
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

        return list(map(lambda db: { 'db': db, 'graph': self.generate_graph(db)}, databases_history))

    def generate_graph(self, db: Database) -> Graph:
        # print(f'generating graph for {db.get_name()}')
        graph = Graph()

        # generate nodes
        for table in db.get_tables():
            # print(f'adding nodes for table {table.get_name()}')
            table_data = table.get_data()

            for index, row in table_data.iterrows():
                properties = {column.get_name(): row[column.get_name()] for column in table.get_columns().values() if not column.get_is_primary() and not column.get_is_hidden()}

                if table.get_name() == 'MTA_CHANGES':
                    graph.add_node(properties, f'{table.get_name()}[{index}]', 'crimson' if row['is_fraud'] else 'black')
                else:
                    graph.add_node(properties, f'{table.get_name()}[{index}]')

        # generate DELETE nodes from MTA_CHANGES
        for _, src_record in db.get_table('MTA_CHANGES').get_data().iterrows():
            if(src_record['change_type']) == 'delete':
                dst_table_name, _ = src_record['table_column_ref'].split('.', 2)
                graph.add_node(src_record['old_value'], f'{dst_table_name}[{src_record["record_id"]}]', 'crimson' if src_record['is_fraud'] else 'black')

        # generate links
        for table in db.get_tables():
            # print(f'TABLE KEYS: {table.get_name()}')
            foreign_keys = table.get_all_foreign_keys()
            table_data = table.get_data()

            for index, src_record in table_data.iterrows():
                src_node = graph.get_node_by_key(f'{table.get_name()}[{index}]')

                for foreign_key in foreign_keys:
                    if foreign_key.get_src_table() != table.get_name():
                        continue

                    dst_table = db.get_table(foreign_key.get_dst_table())
                    dst_data_entry_index = dst_table.find_record_index(foreign_key.get_dst_column(), src_record[foreign_key.get_src_column()])
                    dst_node = graph.get_node_by_key(f'{foreign_key.get_dst_table()}[{dst_data_entry_index}]')

                    if dst_node is not None:
                        if foreign_key.get_reverse_relation():
                            dst_node.add_neighbor(src_node, foreign_key.get_color())
                        else:
                            src_node.add_neighbor(dst_node, foreign_key.get_color())

        # generate meta links
        COLORS = {'create': 'green3', 'update': 'darkorange', 'delete': 'crimson'}

        for index, src_record in db.get_table('MTA_CHANGES').get_data().iterrows():
            src_node = graph.get_node_by_key(f'MTA_CHANGES[{index}]')
            dst_table_name, dst_column_name = src_record['table_column_ref'].split('.', 2)
            dst_node = graph.get_node_by_key(f'{dst_table_name}[{src_record["record_id"]}]')

            src_node.add_neighbor(dst_node, COLORS[src_record['change_type']])

        return graph
