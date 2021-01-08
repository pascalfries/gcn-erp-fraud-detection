from simulation.agents.AgentInterface import AgentInterface
from fraud.FraudPurchaseDefinition import FraudPurchaseDefinition
from simulation.Simulation import Simulation
from typing import List


class Fraudster(AgentInterface):
    def __init__(self,
                 name: str,
                 buyer_customer_id: int,
                 fraudster_salesperson_id: int,
                 products_to_buy: List[FraudPurchaseDefinition]):
        super().__init__(name)

        self._buyer_customer_id = buyer_customer_id
        self._fraudster_salesperson_id = fraudster_salesperson_id
        self._products_to_buy = products_to_buy

    def tick(self, simulation: Simulation) -> None:
        if simulation.get_current_time() == 1:
            for index, purchase in enumerate(self._products_to_buy):
                purchase.schedule_events(simulation=simulation,
                                         buyer_customer_id=self._buyer_customer_id,
                                         fraudster_salesperson_id=self._fraudster_salesperson_id,
                                         fraud_id=f'{self._name}.{index}')
