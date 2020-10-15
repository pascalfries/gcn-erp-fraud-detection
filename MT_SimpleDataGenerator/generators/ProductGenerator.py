from typing import List
from generators.TableGeneratorInterface import TableGeneratorInterface
import random
from data.Database import Database
import config as cfg


class ProductGenerator(TableGeneratorInterface):
    def generate(self, db: Database) -> List:
        return [[f'Product {id}', random.randrange(1, 401)] for id in range(cfg.INIT_GEN_PRODUCT_COUNT)]