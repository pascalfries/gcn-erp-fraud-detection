from typing import List
from generators.TableGeneratorInterface import TableGeneratorInterface
from data.Database import Database
import config as cfg


class SalespersonsGenerator(TableGeneratorInterface):
    def generate(self, db: Database) -> List:
        return [[f'Verkäufer {id}'] for id in range(cfg.INIT_GEN_SALESPERSON_COUNT)]