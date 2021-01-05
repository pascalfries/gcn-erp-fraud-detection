from typing import List
from simulation.ScheduledTasks import ScheduledTask
from simulation.agents.AgentInterface import AgentInterface
from data.Database import Database

import sys
import random
import config as cfg


class Simulation:
    def __init__(self, db: Database, end_time: int):
        self._db: Database = db
        self._db.bind_simulation(simulation=self)

        self._current_time: int = 0
        self._end_time: int = end_time
        self._scheduled_tasks: List[ScheduledTask] = []
        self._agents: List[AgentInterface] = []

    def run(self):
        random.seed(cfg.RANDOM_SEED_SIMULATION)
        print('******************************\n***** RUNNING SIMULATION *****\n******************************')

        for tick in range(self._end_time):
            print(f'\n\nTICK {self._current_time + 1} / {self._end_time}\n******************************\n'
                  f'sched. tasks: {len(self._scheduled_tasks)}\n'
                  f'reg. agents: {len(self._agents)}')
            self.increment_time()

        print('******************************\n**** SIMULATION COMPLETED ****\n******************************')

    def get_db(self) -> Database:
        return self._db

    def get_current_time(self) -> int:
        return self._current_time

    def increment_time(self, increment_by: int = 1) -> None:
        for time_offset in range(increment_by):
            self._current_time += 1

            self.exec_scheduled_tasks_for_time(self._current_time)

            if cfg.VERBOSE_OUTPUT:
                print('\nINVOKE AGENTS\n------------------------------')

            for agent in self._agents:
                if cfg.VERBOSE_OUTPUT:
                    print(f' > {agent.get_name()}')

                agent.tick(simulation=self)

    def add_scheduled_task(self, task: ScheduledTask) -> None:
        self._scheduled_tasks.append(task)

    def add_agent(self, agent: AgentInterface) -> None:
        self._agents.append(agent)

    def exec_scheduled_tasks_for_time(self, time: int) -> None:
        if cfg.VERBOSE_OUTPUT:
            print('\nEXECUTING TASKS\n------------------------------')

        executed_task_count = 0

        for task in self._scheduled_tasks:
            if task.get_exec_time() == time:
                is_result_ok = task.exec(self._db)
                executed_task_count += 1

                if cfg.VERBOSE_OUTPUT:
                    if is_result_ok:
                        print(f' > {type(task).__name__} on {task.get_table_name()}: DONE')
                    else:
                        print(f' > {type(task).__name__} on {task.get_table_name()}: FAILED', file=sys.stderr)

        if executed_task_count == 0 and cfg.VERBOSE_OUTPUT:
            print(f' > no tasks to execute')

        self._scheduled_tasks[:] = [task for task in self._scheduled_tasks if task.get_exec_time() > time]

