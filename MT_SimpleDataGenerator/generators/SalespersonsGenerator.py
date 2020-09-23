from typing import List
from generators.TableGeneratorInterface import TableGeneratorInterface
from data.Database import Database
import config as cfg
from generators.helper import get_random_address, get_random_name


class SalespersonsGenerator(TableGeneratorInterface):
    def generate(self, db: Database) -> List:
        salespersons = []

        for id in range(cfg.INIT_GEN_SALESPERSON_COUNT):
            address = get_random_address()
            address_id = db.get_table('MST_ADDRESSES').insert_record([address[0], address[1], address[2], address[3]])

            salespersons.append([get_random_name(), address_id])

        return salespersons
