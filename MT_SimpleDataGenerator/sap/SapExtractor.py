from data.Database import Database
from typing import Tuple
from graph.GraphCollection import GraphCollection
from graph.GraphGenerator import GraphGenerator
from data.DatabaseSlicer import DatabaseSlicer

import config as cfg
import pandas as pd
import database_config


class SapExtractor:
    def __init__(self, storage_path):
        self._storage_path = storage_path

        self._sap_kna1 = pd.read_excel(self._storage_path + rf'\KNA1.xlsx', engine='openpyxl')
        self._sap_vbak = pd.read_excel(self._storage_path + rf'\VBAK.xlsx', engine='openpyxl')
        self._sap_vbap = pd.read_excel(self._storage_path + rf'\VBAP.xlsx', engine='openpyxl')
        self._sap_mbew = pd.read_excel(self._storage_path + rf'\MBEW.xlsx', engine='openpyxl')
        self._sap_a306 = pd.read_excel(self._storage_path + rf'\A306.xlsx', engine='openpyxl')
        self._sap_cdpos = pd.read_excel(self._storage_path + rf'\CDPOS.xlsx', engine='openpyxl')
        self._sap_cdhdr = pd.read_excel(self._storage_path + rf'\CDHDR.xlsx', engine='openpyxl')

        self._sap_cdpos = self._sap_cdpos[self._sap_cdpos['Tabellenname'] == 'KONPAE']
        self._sap_cdpos['Objektwert'] = pd.to_numeric(self._sap_cdpos['Objektwert'])

        self._min_time = 999_999_999_999
        self._max_time = 0

        self._product_ids = {}
        self._next_product_id = 0

    def extract_slices(self, window_duration, window_stride) -> GraphCollection:
        db, min_time, max_time = self.extract()

        # slice data
        print(f'MAX TIME: {self._max_time}')
        db.save(cfg.STORAGE_ROOT_PATH + r'\sap_db')

        data_slicer = DatabaseSlicer(db, max_simulation_time=max_time, min_time=min_time)
        db_slices = data_slicer.generate_slices_sliding_window(window_duration, window_stride)

        # generate graphs
        graph_gen = GraphGenerator()
        graphs = graph_gen.generate_graphs(db_slices)

        graphs.prune(min_cluster_size=cfg.GRAPH_PRUNING_MIN_CLUSTER_SIZE)
        graphs.save(cfg.STORAGE_BASE_PATH_PY_GRAPHS + rf'\sap')

        with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\sap\generate_graphs.bat', 'w') as graphviz_script:
            for index, history_item in enumerate(graphs.get_raw_list()):
                history_item.export_graphviz(
                    rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\sap\{history_item.get_name()}.txt')
                print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
                      file=graphviz_script)

        return graphs

    def extract(self) -> Tuple[Database, int, int]:
        db = database_config.db
        db.disable_tracing()
        db.set_name('SAP_DATA')

        self.extract_trc_sales(db)
        self.extract_mst_addresses(db)
        self.extract_mst_customers(db)
        self.extract_trm_sale_products(db)
        self.extract_mst_products(db)
        self.extract_mta_changes(db)

        return db, int(self._min_time), int(self._max_time)

    def get_product_id_from_name(self, product_name: str) -> str:
        if product_name not in self._product_ids:
            self._product_ids[product_name] = self._next_product_id
            self._next_product_id += 1

        return str(self._product_ids[product_name])

    def extract_mst_products(self, db):
        tbl_products = db.get_table('MST_PRODUCTS')

        for index, sap_product in self._sap_mbew.iterrows():
            tbl_products.insert_record_with_id(self.get_product_id_from_name(sap_product['Material']), [sap_product['Material'], sap_product['Standardpreis']])

    def extract_mta_changes(self, db):
        tbl_changes = db.get_table('MTA_CHANGES')

        sap_cdpos_a306 = self._sap_cdpos.merge(self._sap_a306, left_on='Objektwert', right_on='KNUMH')\
            .merge(self._sap_cdhdr, on='Belegnummer')

        for index, sap_change in sap_cdpos_a306.iterrows():
            old_value = float(sap_change['alter Wert'])
            new_value = float(sap_change['neuer Wert'])
            is_fraud = True if old_value > 50 or old_value < 0.5 or new_value > 50 or new_value < 0.5 else False

            tbl_changes.insert_record_with_id(index, ['MST_PRODUCTS.price', self.get_product_id_from_name(sap_change['MATNR']), 'update', sap_change['alter Wert'], sap_change['neuer Wert'], sap_change['Benutzer'], self.get_precise_datestamp(sap_change['Datum'], sap_change['Uhrzeit']), is_fraud, ''])

    def extract_trm_sale_products(self, db: Database):
        tbl_sale_products = db.get_table('TRM_SALE_PRODUCTS')

        for index, sap_purchase in self._sap_vbap.iterrows():
            tbl_sale_products.insert_record([self.get_product_id_from_name(sap_purchase['Material']), sap_purchase['Verkaufsbeleg'], sap_purchase['Auftragsmenge'], self.get_precise_datestamp(sap_purchase['Angelegt am'], sap_purchase['Uhrzeit'])])

    def extract_trc_sales(self, db: Database):
        tbl_sales = db.get_table('TRC_SALES')
        tbl_salespersons = db.get_table('MST_SALESPERSONS')

        for index, sap_sale in self._sap_vbak.iterrows():
            salesperson_id = tbl_salespersons.find_record_index('name', sap_sale['Angelegt von'])

            if salesperson_id is None:
                salesperson_id = tbl_salespersons.insert_record_with_id(sap_sale['Angelegt von'], [sap_sale['Angelegt von'], sap_sale['Angelegt von']])

            tbl_sales.insert_record_with_id(sap_sale['Verkaufsbeleg'], ['', sap_sale['Auftraggeber'], salesperson_id, False, '', self.get_precise_datestamp(sap_sale['Bestelldatum'], sap_sale['Uhrzeit'])])

    def extract_mst_customers(self, db: Database):
        tbl_customers = db.get_table('MST_CUSTOMERS')

        for index, sap_customer in self._sap_kna1.iterrows():
            tbl_customers.insert_record_with_id(sap_customer['Debitor'], [sap_customer['Name 1'], sap_customer['Debitor']])

    def extract_mst_addresses(self, db: Database):
        tbl_addresses = db.get_table('MST_ADDRESSES')

        for index, sap_address in self._sap_kna1.iterrows():
            tbl_addresses.insert_record_with_id(sap_address['Debitor'], [sap_address['Straße'], 0, sap_address['Ort'], sap_address['Postleitzahl']])

    def get_precise_datestamp(self, date, time=None):
        time_ms = 0
        if time is not None:
            time_ms = sum(factor * int(t) for factor, t in zip([3600, 60, 1], str(time).split(":")))

        time_total = date.timestamp() + time_ms

        if time_total < self._min_time:
            self._min_time = time_total

        if time_total > self._max_time:
            self._max_time = time_total

        return time_total
