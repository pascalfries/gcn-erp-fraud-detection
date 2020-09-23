from typing import List
from generators.TableGeneratorInterface import TableGeneratorInterface
from data.Database import Database
import config as cfg
from generators.helper import get_random_name, get_random_address


class CustomersGenerator(TableGeneratorInterface):
    def generate(self, db: Database) -> List:
        salespersons = db.get_table('MST_SALESPERSONS')
        customer_salesperson_mapping = {6: 6, 85: 1, 28: 3, 14: 8} # x:y -> customer x is salesperson y
        customers = []

        for id in range(cfg.INIT_GEN_CUSTOMER_COUNT):
            if id in customer_salesperson_mapping.keys():
                salesperson_record = salespersons.get_record(customer_salesperson_mapping[id])

                address_id = salesperson_record['address_id']
                name = salesperson_record['name']
            else:
                address = get_random_address()
                address_id = db.get_table('MST_ADDRESSES').insert_record([address[0], address[1], address[2], address[3]])
                name = get_random_name()

            customers.append([name, address_id])

        return customers
