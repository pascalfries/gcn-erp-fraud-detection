class ForeignKey:
    def __init__(self, key_name: str, src_table: str, src_column: str, dst_table: str, dst_column: str, reverse_relation: bool = False, color: str = None):
        self._key_name = key_name
        self._src_table = src_table
        self._src_column = src_column
        self._dst_table = dst_table
        self._dst_column = dst_column
        self._reverse_relation = reverse_relation
        self._color = color

    def get_key_name(self) -> str:
        return self._key_name

    def get_src_table(self) -> str:
        return self._src_table

    def get_src_column(self) -> str:
        return self._src_column

    def get_dst_table(self) -> str:
        return self._dst_table

    def get_dst_column(self) -> str:
        return self._dst_column

    def get_reverse_relation(self) -> bool:
        return self._reverse_relation

    def get_color(self) -> str:
        return self._color
