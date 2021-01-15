from data.Database import Database
import pandas as pd
import database_config


class SapExtractor:
    def __init__(self, storage_path):
        self._storage_path = storage_path

        self._sap_kna1 = pd.read_excel(self._storage_path + rf'\kna1.xlsx', engine='openpyxl')
        self._sap_vbak = pd.read_excel(self._storage_path + rf'\vbak.xlsx', engine='openpyxl')
        self._sap_vbap = pd.read_excel(self._storage_path + rf'\vbap.xlsx', engine='openpyxl')

    def extract(self) -> Database:
        db = database_config.db
        db.disable_tracing()
        db.set_name('SAP DATA')

        self.extract_trc_sales(db)
        self.extract_mst_addresses(db)
        self.extract_mst_customers(db)
        self.extract_tmr_sale_products(db)

        return db

    def extract_mst_products(self):
        pass

    def extract_mta_changes(self):
        pass

    def extract_tmr_sale_products(self, db: Database):
        tbl_sale_products = db.get_table('TRM_SALE_PRODUCTS')

        for index, sap_purchase in self._sap_vbap.iterrows():
            tbl_sale_products.insert_record([sap_purchase['Material'], sap_purchase['Verkaufsbeleg'], sap_purchase['Auftragsmenge'], sap_purchase['Angelegt am'].timestamp()])

    def extract_trc_sales(self, db: Database):
        tbl_sales = db.get_table('TRC_SALES')
        tbl_salespersons = db.get_table('MST_SALESPERSONS')

        for index, sap_sale in self._sap_vbak.iterrows():
            salesperson_id = tbl_salespersons.find_record_index('name', sap_sale['Angelegt von'])

            if salesperson_id is None:
                salesperson_id = tbl_salespersons.insert_record_with_id(sap_sale['Angelegt von'], [sap_sale['Angelegt von'], sap_sale['Angelegt von']])

            tbl_sales.insert_record_with_id(sap_sale['Verkaufsbeleg'], ['', sap_sale['Auftraggeber'], salesperson_id, False, '', sap_sale['Bestelldatum'].timestamp()])

    def extract_mst_customers(self, db: Database):
        tbl_customers = db.get_table('MST_CUSTOMERS')

        for index, sap_customer in self._sap_kna1.iterrows():
            tbl_customers.insert_record_with_id(sap_customer['Debitor'], [sap_customer['Name 1'], sap_customer['Debitor']])

    def extract_mst_addresses(self, db: Database):
        tbl_addresses = db.get_table('MST_ADDRESSES')

        for index, sap_address in self._sap_kna1.iterrows():
            tbl_addresses.insert_record_with_id(sap_address['Debitor'], [sap_address['Straße'], 0, sap_address['Ort'], sap_address['Postleitzahl']])
