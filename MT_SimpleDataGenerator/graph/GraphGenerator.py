from typing import List, Dict
from graph.Graph import Graph
from data.Database import Database
from graph.GraphCollection import GraphCollection
import time


class GraphGenerator:
    def __init__(self, edge_colors: Dict[str, str] = None, default_node_color: str = 'black', fraud_node_color: str = 'crimson'):
        self._edge_colors = {'create': 'green3', 'update': 'darkorange', 'delete': 'crimson'}

        if edge_colors is not None:
            self._edge_colors = edge_colors

        self._default_node_color = default_node_color
        self._fraud_node_color = fraud_node_color

    def generate_graphs(self, databases: List[Database]) -> GraphCollection:
        return GraphCollection(list(map(lambda db: self.generate_graph(db), databases)))

    def generate_graph(self, db: Database) -> Graph:
        print(f'generating graph for {db.get_name()}')
        graph = Graph(name=db.get_name())

        # generate nodes
        time_start = time.perf_counter()
        for table in db.get_tables():
            # print(f'adding nodes for table {table.get_name()}')
            table_data = table.get_data()

            for index, row in table_data.iterrows():
                properties = {column.get_name(): row[column.get_name()] for column in table.get_columns().values() if not column.get_is_primary() and not column.get_is_hidden()}

                if table.get_name() == 'MTA_CHANGES':
                    graph.add_node(properties, f'{table.get_name()}[{index}]', self._fraud_node_color if row['is_fraud'] else self._default_node_color, node_type=table.get_name())
                else:
                    graph.add_node(properties, f'{table.get_name()}[{index}]', node_type=table.get_name())

        time_end = time.perf_counter()
        # print(f"generate nodes took {time_end - time_start:0.4f} seconds")

        # generate DELETE nodes from MTA_CHANGES
        time_start = time.perf_counter()
        for _, src_record in db.get_table('MTA_CHANGES').get_data().iterrows():
            if(src_record['change_type']) == 'delete':
                dst_table_name, _ = src_record['table_column_ref'].split('.', 2)
                graph.add_node(src_record['old_value'], f'{dst_table_name}[{src_record["record_id"]}]', self._fraud_node_color if src_record['is_fraud'] else self._default_node_color, node_type='MTA_CHANGES')

        time_end = time.perf_counter()
        # print(f"delete nodes took {time_end - time_start:0.4f} seconds")

        # generate links
        time_start = time.perf_counter()
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

        time_end = time.perf_counter()
        # print(f"generate links took {time_end - time_start:0.4f} seconds")

        # generate meta links
        time_start = time.perf_counter()
        for index, src_record in db.get_table('MTA_CHANGES').get_data().iterrows():
            src_node = graph.get_node_by_key(f'MTA_CHANGES[{index}]')
            dst_table_name, dst_column_name = src_record['table_column_ref'].split('.', 2)
            dst_node = graph.get_node_by_key(f'{dst_table_name}[{src_record["record_id"]}]')

            src_node.add_neighbor(dst_node, self._edge_colors[src_record['change_type']])

        time_end = time.perf_counter()
        # print(f"generate meta links took {time_end - time_start:0.4f} seconds")

        return graph
