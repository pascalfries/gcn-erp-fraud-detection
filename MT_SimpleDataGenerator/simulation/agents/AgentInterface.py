from simulation import Simulation


class AgentInterface:
    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def tick(self, simulation: Simulation) -> None:
        pass
