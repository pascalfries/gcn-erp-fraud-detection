from simulation.agents.AgentInterface import AgentInterface
from simulation.Simulation import Simulation
from helpers import with_probability, rand_float


class Salesperson(AgentInterface):
    def __init__(self, name: str,
                 salesperson_id: int,
                 tick_action_probability: float = 0.2,
                 min_product_count: int = 1,
                 max_product_count: int = 5,
                 price_decrease_probability: float = 0.5,
                 min_price_decrease_percentage: float = 0.005,
                 max_price_decrease_percentage: float = 0.03,
                 min_price_increase_percentage: float = 0.005,
                 max_price_increase_percentage: float = 0.03):
        super(Salesperson, self).__init__(name)

        self._salesperson_id = salesperson_id
        self._tick_action_probability = tick_action_probability
        self._min_product_count = min_product_count
        self._max_product_count = max_product_count
        self._price_decrease_probability = price_decrease_probability

        self._min_price_decrease_percentage = min_price_decrease_percentage
        self._max_price_decrease_percentage = max_price_decrease_percentage
        self._min_price_increase_percentage = min_price_increase_percentage
        self._max_price_increase_percentage = max_price_increase_percentage

    def tick(self, simulation: Simulation) -> None:
        # Dec 24 is day 359

        tick_action_probability = self._tick_action_probability
        price_decrease_probability = self._price_decrease_probability
        min_price_decrease_percentage = self._min_price_decrease_percentage
        max_price_decrease_percentage = self._max_price_decrease_percentage
        min_price_increase_percentage = self._min_price_increase_percentage
        max_price_increase_percentage = self._max_price_increase_percentage
        min_product_count = self._min_product_count
        max_product_count = self._max_product_count

        if simulation.get_current_time() in [355, 720]:
            tick_action_probability = 1
            price_decrease_probability = 0.85
            min_price_decrease_percentage = 3 * self._min_price_decrease_percentage
            max_price_decrease_percentage = 2 * self._max_price_decrease_percentage
            min_product_count = 2 * self._min_product_count
            max_product_count = 3 * self._max_product_count
        elif simulation.get_current_time() in [361, 362, 363, 726, 727, 728]:
            tick_action_probability = 0.75
            price_decrease_probability = 0.25
            min_price_increase_percentage = 2 * self._min_price_increase_percentage
            max_price_increase_percentage = 2 * self._max_price_increase_percentage
            max_product_count = 2 * self._max_product_count

        if with_probability(tick_action_probability):
            products_table = simulation.get_db().get_table('MST_PRODUCTS')
            changed_product_ids = []

            for count in range(min_product_count, max_product_count + 1):
                product_change, product_change_id = products_table.get_random_item()

                if product_change_id not in changed_product_ids:
                    changed_product_ids.append(product_change_id)

                    if with_probability(price_decrease_probability): # decrease price
                        discount_factor = 1.0 - rand_float(min_price_decrease_percentage, max_price_decrease_percentage)
                        new_price = product_change['price'] * discount_factor
                    else:  # increase price
                        markup_factor = 1.0 + rand_float(min_price_increase_percentage, max_price_increase_percentage)
                        new_price = product_change['price'] * markup_factor

                    products_table.update_record(num=product_change_id,
                                                 col_name='price',
                                                 new_value=int(round(new_price)),
                                                 changed_by=self._salesperson_id)
