import random
from simulation.agents.AgentInterface import AgentInterface
from simulation.Simulation import Simulation
from helpers import with_probability
from typing import Optional


class Customer(AgentInterface):
    def __init__(self, name: str,
                 customer_id: int,
                 tick_buy_probability: float = 0.5,
                 min_product_count: int = 1,
                 max_product_count: int = 10,
                 min_amount_per_product: int = 1,
                 max_amount_per_product: int = 5,
                 use_salesperson_id: Optional[int] = None):
        super(Customer, self).__init__(name)

        self._customer_id = customer_id
        self._tick_buy_probability = tick_buy_probability
        self._min_product_count = min_product_count
        self._max_product_count = max_product_count
        self._min_amount_per_product = min_amount_per_product
        self._max_amount_per_product = max_amount_per_product
        self._use_salesperson_id = use_salesperson_id

    def tick(self, simulation: Simulation) -> None:
        if with_probability(self._tick_buy_probability):
            db = simulation.get_db()

            products_table = simulation.get_db().get_table('MST_PRODUCTS')
            sales_table = db.get_table('TRC_SALES')
            sales_products_table = db.get_table('TRM_SALE_PRODUCTS')
            salespersons_table = db.get_table('MST_SALESPERSONS')

            salesperson_id = self._use_salesperson_id if self._use_salesperson_id is not None else salespersons_table.get_random_item_id()
            sales_table.insert_record([f'Purchase of {self._name}', self._customer_id, salesperson_id, simulation.get_current_time()])

            purchased_products = []
            for product_count in range(random.randint(self._min_product_count, self._max_product_count)):
                product_buy_id = products_table.get_random_item_id()

                if product_buy_id not in purchased_products:
                    purchased_products.append(product_buy_id)

                    product_buy_amount = random.randint(self._min_amount_per_product, self._max_amount_per_product)
                    sales_products_table.insert_record([product_buy_id, len(sales_table) - 1, product_buy_amount, simulation.get_current_time()]) # todo get actual Id instead of len
