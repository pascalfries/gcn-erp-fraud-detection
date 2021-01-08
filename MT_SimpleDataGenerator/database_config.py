from data.Database import Database
from data.DataTable import DataTable
from data.DataColumn import DataColumn
from data.ForeignKey import ForeignKey

# ===============================================================
# ======================== TABLE SETUP ==========================
# ===============================================================
db = Database('SDG_DATA')

# TABLE: PRODUCTS
table_products = DataTable('MST_PRODUCTS', True)
table_products.add_columns([DataColumn('id', True), DataColumn('name'), DataColumn('price')])
db.add_table(table_products)

# TABLE: SALES
table_sales = DataTable('TRC_SALES')
table_sales.add_columns([DataColumn('id', True), DataColumn('description'), DataColumn('customer_id', is_hidden=True), DataColumn('salesperson_id', is_hidden=True), DataColumn('is_fraud'), DataColumn('fraud_id'), DataColumn('timestamp', is_timestamp=True)])
db.add_table(table_sales)

# TABLE: SALE_PRODUCTS
table_sale_products = DataTable('TRM_SALE_PRODUCTS', is_mapping_table=True)
table_sale_products.add_columns([DataColumn('product_id', True), DataColumn('sale_id', True), DataColumn('amount'), DataColumn('timestamp', is_timestamp=True)])
db.add_table(table_sale_products)

# TABLE: CUSTOMERS
table_customers = DataTable('MST_CUSTOMERS', True)
table_customers.add_columns([DataColumn('id', True), DataColumn('name'), DataColumn('address_id')])
db.add_table(table_customers)

# TABLE: SALESPERSONS
table_salespersons = DataTable('MST_SALESPERSONS', True)
table_salespersons.add_columns([DataColumn('id', True), DataColumn('name'), DataColumn('address_id')])
db.add_table(table_salespersons)

# TABLE: ADDRESSES
table_addresses = DataTable('MST_ADDRESSES', True)
table_addresses.add_columns([DataColumn('id', True), DataColumn('street'), DataColumn('house_number'), DataColumn('city'), DataColumn('zip_code')])
db.add_table(table_addresses)

# ===============================================================
# ========================= TABLE KEYS ==========================
# ===============================================================
db.add_foreign_key(ForeignKey('key_saleproducts_sale', 'TRM_SALE_PRODUCTS', 'sale_id', 'TRC_SALES', 'id', reverse_relation=True, color='dodgerblue3'))
db.add_foreign_key(ForeignKey('key_saleproducts_product', 'TRM_SALE_PRODUCTS', 'product_id', 'MST_PRODUCTS', 'id', color='dodgerblue3'))
db.add_foreign_key(ForeignKey('key_sales_salesperson', 'TRC_SALES', 'salesperson_id', 'MST_SALESPERSONS', 'id', reverse_relation=True))
db.add_foreign_key(ForeignKey('key_sales_customer', 'TRC_SALES', 'customer_id', 'MST_CUSTOMERS', 'id', reverse_relation=True, color='darkorchid3'))
db.add_foreign_key(ForeignKey('key_changes_salesperson', 'MTA_CHANGES', 'salesperson_id', 'MST_SALESPERSONS', 'id', reverse_relation=True))
db.add_foreign_key(ForeignKey('key_salesperson_address', 'MST_SALESPERSONS', 'address_id', 'MST_ADDRESSES', 'id'))
db.add_foreign_key(ForeignKey('key_customer_address', 'MST_CUSTOMERS', 'address_id', 'MST_ADDRESSES', 'id'))
