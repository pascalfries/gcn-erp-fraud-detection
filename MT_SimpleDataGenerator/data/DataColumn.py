from typing import Optional, List
from data.ForeignKey import ForeignKey


class DataColumn:
    def __init__(self, name: str, is_primary: bool = False, is_timestamp: bool = False, foreign_keys: Optional[List[ForeignKey]] = [], is_hidden: bool = False):
        self._name: str = name
        self._is_primary: bool = is_primary
        self._is_timestamp: bool = is_timestamp
        self._foreign_keys: List[ForeignKey] = foreign_keys
        self._is_hidden: bool = is_hidden

    def get_name(self) -> str:
        return self._name

    def get_is_primary(self) -> bool:
        return self._is_primary

    def get_is_timestamp(self) -> bool:
        return self._is_timestamp

    def get_is_hidden(self) -> bool:
        return self._is_hidden

    def get_foreign_keys(self) -> List[ForeignKey]:
        return self._foreign_keys

    def add_foreign_key(self, foreign_key: ForeignKey) -> None:
        self._foreign_keys.append(foreign_key)