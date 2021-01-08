from typing import Callable, Dict
from simulation.Simulation import Simulation
from data.Database import Database
import simulation.ScheduledTasks as task


def _default_price_decrement(time: int, steps: int, initial_price: int, target_price: int) -> int:
    diff_per_step = (initial_price - target_price) / steps
    return int(diff_per_step)


def _default_price_increment(time: int, steps: int, starting_price: int, target_price: int) -> int:
    diff_per_step = (target_price - starting_price) / steps
    return int(diff_per_step)


class FraudPurchaseDefinition:
    def __init__(self,
                 product_id: int,
                 purchase_amount: int,
                 purchase_time: int,
                 new_product_price: float = None,
                 new_product_price_percent: float = None,
                 price_decrease_increments: int = 1,
                 price_decrease_to_buy_delay: int = 0,
                 price_increase_increments: int = 1,
                 price_buy_to_increment_delay: int = 0,
                 price_decrement_function: Callable[[int, int, int, int], int] = _default_price_decrement,
                 price_increment_function: Callable[[int, int, int, int], int] = _default_price_increment):
        self._product_id = product_id
        self._purchase_amount = purchase_amount
        self._purchase_time = purchase_time
        self._new_product_price = new_product_price
        self._new_product_price_percent = new_product_price_percent

        self._price_decrease_to_buy_delay = price_decrease_to_buy_delay
        self._price_decrease_increments = price_decrease_increments

        self._price_buy_to_increment_delay = price_buy_to_increment_delay
        self._price_increase_increments = price_increase_increments
        self._price_decrement_function = price_decrement_function
        self._price_increment_function = price_increment_function

    def get_product_id(self) -> int:
        return self._product_id

    def schedule_events(self, simulation: Simulation, buyer_customer_id: int, fraudster_salesperson_id: int, fraud_id: str) -> None:
        # reduce price
        simulation.add_scheduled_task(task.LambdaTask(
            exec_at=self._purchase_time - self._price_decrease_to_buy_delay - self._price_decrease_increments - 1,
            table_name='MST_PRODUCTS',
            task=self._schedule_fraud_price_changes,
            args={
                'simulation': simulation,
                'product_id': self._product_id,
                'buy_time': self._purchase_time,
                'fraudster_salesperson_id': fraudster_salesperson_id,
                'fraud_id': fraud_id
            }
        ))

        # make purchase
        simulation.add_scheduled_task(task.LambdaTask(
                exec_at=self._purchase_time,
                table_name='MST_PRODUCTS',
                task=self._schedule_fraud_purchase,
                args={
                    'buyer_customer_id': buyer_customer_id,
                    'fraudster_salesperson_id': fraudster_salesperson_id,
                    'buy_time': self._purchase_time,
                    'fraud_id': fraud_id
                }
        ))

    def _schedule_fraud_price_changes(self, db: Database, args: Dict) -> bool:
        initial_product_price = db.get_table('MST_PRODUCTS').get_record(args['product_id'])['price']
        current_price = initial_product_price
        new_price = int(initial_product_price * self._new_product_price_percent) if self._new_product_price_percent is not None else self._new_product_price if self._new_product_price is not None else initial_product_price

        for increment_time_offset in range(self._price_decrease_increments):
            current_time = args['buy_time'] - self._price_decrease_to_buy_delay - self._price_decrease_increments + increment_time_offset

            if increment_time_offset == self._price_decrease_increments - 1:
                current_price = new_price
            else:
                current_price -= self._price_decrement_function(current_time, self._price_decrease_increments,
                                                                initial_product_price, new_price)

            args['simulation'].add_scheduled_task(task.UpdateTask(
                exec_at=current_time,
                table_name='MST_PRODUCTS',
                item_id=self._product_id,
                field_name='price',
                new_value=current_price,
                is_fraud=True,
                fraud_id=args['fraud_id'],
                changed_by=args['fraudster_salesperson_id']
            ))

        # increase price
        for increment_time_offset in range(self._price_increase_increments):
            current_time = args['buy_time'] + self._price_buy_to_increment_delay + increment_time_offset + 1

            if increment_time_offset == self._price_increase_increments - 1:
                current_price = initial_product_price
            else:
                current_price += self._price_increment_function(current_time, self._price_increase_increments,
                                                                new_price, initial_product_price)

            args['simulation'].add_scheduled_task(task.UpdateTask(
                exec_at=current_time,
                table_name='MST_PRODUCTS',
                item_id=self._product_id,
                field_name='price',
                new_value=current_price,
                is_fraud=True,
                fraud_id=args['fraud_id'],
                changed_by=args['fraudster_salesperson_id']
            ))

        return True

    def _schedule_fraud_purchase(self, db: Database, args: Dict) -> bool:
        db.get_table('TRC_SALES').insert_record([f'FPurchase of Customer {args["buyer_customer_id"]}', args["buyer_customer_id"], args['fraudster_salesperson_id'], True, args['fraud_id'], args['buy_time']])
        db.get_table('TRM_SALE_PRODUCTS').insert_record([self._product_id, len(db.get_table('TRC_SALES')) - 1, self._purchase_amount, args['buy_time']], changed_by=args['fraudster_salesperson_id'])

        return True
