from typing import List
from data.Database import Database


class TableGeneratorInterface:
    def generate(self, db: Database) -> List:
        pass
