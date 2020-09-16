from typing import List
from generators.TableGeneratorInterface import TableGeneratorInterface
from data.Database import Database
import config as cfg


class CustomersGenerator(TableGeneratorInterface):
    def generate(self, db: Database) -> List:
        return [[f'Kunde {id}', f'Musterstrasse {id}'] for id in range(cfg.INIT_GEN_CUSTOMER_COUNT)]