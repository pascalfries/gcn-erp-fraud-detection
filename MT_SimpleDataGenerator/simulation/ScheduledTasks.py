from data.Database import Database
from typing import Callable, Dict


class ScheduledTask:
    def __init__(self, exec_at: int, table_name: str):
        self._exec_at = exec_at
        self._table_name = table_name

    def get_exec_time(self) -> int:
        return self._exec_at

    def get_table_name(self) -> str:
        return self._table_name

    def exec(self, db: Database) -> bool:
        pass


class LambdaTask(ScheduledTask):
    def __init__(self, exec_at: int, table_name: str, task: Callable[[Database, Dict], bool], args: Dict):
        super().__init__(exec_at, table_name)

        self._task = task
        self._args = args

    def exec(self, db: Database) -> bool:
        return self._task(db, self._args)


class InsertTask(ScheduledTask):
    def __init__(self, exec_at: int, table_name: str, new_record, changed_by=0):
        super().__init__(exec_at, table_name)

        self._new_record = new_record
        self._changed_by = changed_by

    def exec(self, db: Database) -> bool:
        table = db.get_table(self._table_name)

        if table is not None:
            table.insert_record(self._new_record, changed_by=self._changed_by)
            return True
        else:
            return False


class UpdateTask(ScheduledTask):
    def __init__(self, exec_at: int, table_name: str, item_id: int, field_name: str, new_value, is_fraud: bool = False, changed_by=0):
        super().__init__(exec_at, table_name)

        self._item_id = item_id
        self._field_name = field_name
        self._new_value = new_value
        self._is_fraud = is_fraud
        self._changed_by = changed_by

    def exec(self, db: Database) -> bool:
        table = db.get_table(self._table_name)

        if table is not None:
            table.update_record(num=self._item_id,
                                col_name=self._field_name,
                                new_value=self._new_value,
                                is_fraud=self._is_fraud,
                                changed_by=self._changed_by)
            return True
        else:
            return False


class DeleteTask(ScheduledTask):
    def __init__(self, exec_at: int, table_name: str, item_id: int, is_fraud: bool = False, changed_by=0):
        super().__init__(exec_at, table_name)

        self._item_id = item_id
        self._is_fraud = is_fraud
        self._changed_by = changed_by

    def exec(self, db: Database) -> bool:
        table = db.get_table(self._table_name)

        if table is not None:
            table.remove_record(self._item_id, is_fraud=self._is_fraud, changed_by=self._changed_by)
            return True
        else:
            return False
