from generators.ProductGenerator import ProductGenerator
from generators.CustomersGenerator import CustomersGenerator
from generators.SalespersonsGenerator import SalespersonsGenerator
from simulation.Simulation import Simulation
from simulation.agents.Customer import Customer
from simulation.agents.Fraudster import Fraudster
from simulation.agents.SalesPerson import Salesperson
from fraud.FraudPurchaseDefinition import FraudPurchaseDefinition
from data.DatabaseSlicer import DatabaseSlicer
from graph.GraphGenerator import GraphGenerator

import config as cfg
import random
import database_config
import os
import sys


sys.setrecursionlimit(10_000)

# ===============================================================
# ========================= RNG SETUP ===========================
# ===============================================================
random.seed(cfg.RANDOM_SEED_GENERATOR)


# ===============================================================
# ==================== BASE DATA GENERATION =====================
# ===============================================================
if cfg.CONF_GENERATE_MASTER_DATA:
    database_config.table_products.insert_records(ProductGenerator().generate(database_config.db))
    database_config.table_salespersons.insert_records(SalespersonsGenerator().generate(database_config.db))
    database_config.table_customers.insert_records(CustomersGenerator().generate(database_config.db))

    database_config.db.save(cfg.STORAGE_BASE_PATH_MASTER_DATA)
else:
    database_config.db.load(cfg.STORAGE_BASE_PATH_MASTER_DATA)


# ===============================================================
# ========================= SIMULATION ==========================
# ===============================================================
if cfg.CONF_RUN_SIMULATION:
    simulation = Simulation(db=database_config.db, end_time=cfg.SIMULATION_END_TIME)

    # generate salespersons
    simulation.add_agent(Salesperson(name='Salesperson 1', salesperson_id=1, tick_action_probability=0.08, min_price_increase_percentage=0.02, max_price_increase_percentage=0.06))
    simulation.add_agent(Salesperson(name='Salesperson 2', salesperson_id=2, tick_action_probability=0.11, max_price_decrease_percentage=0.035, max_product_count=8))
    simulation.add_agent(Salesperson(name='Salesperson 4', salesperson_id=4, tick_action_probability=0.16, min_price_increase_percentage=0.02, max_price_increase_percentage=0.05, min_price_decrease_percentage=0.015, max_price_decrease_percentage=0.05))
    simulation.add_agent(Salesperson(name='Salesperson 7', salesperson_id=7, tick_action_probability=0.21, price_decrease_probability=0.7))
    simulation.add_agent(Salesperson(name='Salesperson 9', salesperson_id=9, tick_action_probability=0.12, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))
    simulation.add_agent(Salesperson(name='Salesperson 10', salesperson_id=10, tick_action_probability=0.07, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))
    simulation.add_agent(Salesperson(name='Salesperson 11', salesperson_id=11, tick_action_probability=0.11, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))
    simulation.add_agent(Salesperson(name='Salesperson 12', salesperson_id=12, tick_action_probability=0.08, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))
    simulation.add_agent(Salesperson(name='Salesperson 13', salesperson_id=13, tick_action_probability=0.11, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))
    simulation.add_agent(Salesperson(name='Salesperson 14', salesperson_id=14, tick_action_probability=0.03, price_decrease_probability=0.2, min_price_decrease_percentage=0.05, max_price_decrease_percentage=0.10))

    # generate legitimate small customers
    for i in range(cfg.INIT_GEN_CUSTOMER_COUNT):
        if i in cfg.SIMULATION_CUSTOMER_IDS_NO_BUY or i in cfg.SIMULATION_CUSTOMER_IDS_LARGE_BUY:
            continue

        simulation.add_agent(Customer(name=f'Small Customer {i}', customer_id=i, tick_buy_probability=random.randint(0, 60) / 1000, max_amount_per_product=1, max_product_count=random.randint(1, 5)))

    # generate legitimate large customers
    for i in cfg.SIMULATION_CUSTOMER_IDS_LARGE_BUY:
        min_amount_per_product = random.randint(1, 11)

        simulation.add_agent(Customer(name=f'Large Customer {i}',
                                      tick_buy_probability=random.randint(50, 450) / 1000,
                                      max_product_count=random.randint(1, 4),
                                      min_amount_per_product=min_amount_per_product,
                                      max_amount_per_product=max(min_amount_per_product, random.choice([5, 10, 15, 20, 25, 50, 100])),
                                      customer_id=i,
                                      use_salesperson_id=random.choice([None, None, 0, 3, 5, 6, 8, 14])))

    # generate fraud customers
    # Selfish Fraudsters
    simulation.add_agent(Fraudster(name='SF1', fraudster_salesperson_id=6, buyer_customer_id=6, products_to_buy=[
        FraudPurchaseDefinition(product_id=19, new_product_price_percent=0.79, purchase_amount=3, purchase_time=300),
        FraudPurchaseDefinition(product_id=107, new_product_price_percent=0.78, purchase_amount=1, purchase_time=426),
        FraudPurchaseDefinition(product_id=98, new_product_price_percent=0.83, purchase_amount=1, purchase_time=638),
        FraudPurchaseDefinition(product_id=185, new_product_price_percent=0.70, purchase_amount=3, purchase_time=796),
        FraudPurchaseDefinition(product_id=4, new_product_price_percent=0.64, purchase_amount=5, purchase_time=1_067),
        FraudPurchaseDefinition(product_id=73, new_product_price_percent=0.60, purchase_amount=2, purchase_time=1_234),
        FraudPurchaseDefinition(product_id=191, new_product_price_percent=0.61, purchase_amount=3, purchase_time=1_630),
    ]))

    simulation.add_agent(Fraudster(name='SF2', fraudster_salesperson_id=13, buyer_customer_id=143, products_to_buy=[
        FraudPurchaseDefinition(product_id=7, new_product_price_percent=0.80, purchase_amount=2, purchase_time=816),
        FraudPurchaseDefinition(product_id=158, new_product_price_percent=0.80, purchase_amount=4, purchase_time=978),
        FraudPurchaseDefinition(product_id=97, new_product_price_percent=0.80, purchase_amount=2, purchase_time=1_567),
        FraudPurchaseDefinition(product_id=199, new_product_price_percent=0.67, purchase_amount=2, purchase_time=1_676),
        FraudPurchaseDefinition(product_id=109, new_product_price_percent=0.80, purchase_amount=2, purchase_time=1_801),
        FraudPurchaseDefinition(product_id=153, new_product_price_percent=0.65, purchase_amount=2, purchase_time=1_799),
    ]))

    # Fraudsters with Accomplices
    simulation.add_agent(Fraudster(name='FA1', fraudster_salesperson_id=6, buyer_customer_id=89, products_to_buy=[
        FraudPurchaseDefinition(product_id=146, new_product_price_percent=0.86, purchase_amount=1, purchase_time=123),
        FraudPurchaseDefinition(product_id=10, new_product_price_percent=0.75, purchase_amount=1, purchase_time=407),
        FraudPurchaseDefinition(product_id=10, new_product_price_percent=0.75, purchase_amount=2, purchase_time=613),
        FraudPurchaseDefinition(product_id=91, new_product_price_percent=0.70, purchase_amount=2, purchase_time=722),
        FraudPurchaseDefinition(product_id=83, new_product_price_percent=0.80, purchase_amount=1, purchase_time=994),
    ]))

    simulation.add_agent(Fraudster(name='FA2', fraudster_salesperson_id=6, buyer_customer_id=64, products_to_buy=[
        FraudPurchaseDefinition(product_id=42, new_product_price_percent=0.80, purchase_amount=1, purchase_time=162),
        FraudPurchaseDefinition(product_id=134, new_product_price_percent=0.84, purchase_amount=1, purchase_time=466),
        FraudPurchaseDefinition(product_id=64, new_product_price_percent=0.80, purchase_amount=2, purchase_time=609),
        FraudPurchaseDefinition(product_id=118, new_product_price_percent=0.90, purchase_amount=1, purchase_time=1042),
    ]))

    simulation.add_agent(Fraudster(name='FA3', fraudster_salesperson_id=13, buyer_customer_id=90, products_to_buy=[
        FraudPurchaseDefinition(product_id=33, new_product_price_percent=0.61, purchase_amount=10, purchase_time=552),
        FraudPurchaseDefinition(product_id=43, new_product_price_percent=0.78, purchase_amount=5, purchase_time=956),
        FraudPurchaseDefinition(product_id=24, new_product_price_percent=0.72, purchase_amount=8, purchase_time=1_034)
    ]))

    simulation.add_agent(Fraudster(name='FA4', fraudster_salesperson_id=11, buyer_customer_id=104, products_to_buy=[
        FraudPurchaseDefinition(product_id=41, new_product_price_percent=0.85, purchase_amount=2, purchase_time=1_455),
        FraudPurchaseDefinition(product_id=184, new_product_price_percent=0.76, purchase_amount=2, purchase_time=1_813)
    ]))

    # Patient Fraudsters
    simulation.add_agent(Fraudster(name='PF1', fraudster_salesperson_id=4, buyer_customer_id=55, products_to_buy=[
        FraudPurchaseDefinition(product_id=80, new_product_price_percent=0.80, purchase_amount=2, price_buy_to_increment_delay=2, price_decrease_to_buy_delay=1, purchase_time=10),
        FraudPurchaseDefinition(product_id=18, new_product_price_percent=0.83, purchase_amount=1, price_buy_to_increment_delay=4, price_decrease_to_buy_delay=1, purchase_time=917),
    ]))

    simulation.add_agent(Fraudster(name='PF2', fraudster_salesperson_id=4, buyer_customer_id=44, products_to_buy=[
        FraudPurchaseDefinition(product_id=7, new_product_price_percent=0.67, purchase_amount=1, price_buy_to_increment_delay=2, price_decrease_to_buy_delay=1, purchase_time=237),
        FraudPurchaseDefinition(product_id=132, new_product_price_percent=0.74, purchase_amount=1, price_buy_to_increment_delay=6, price_decrease_to_buy_delay=3, purchase_time=321),
        FraudPurchaseDefinition(product_id=86, new_product_price_percent=0.90, purchase_amount=3, price_buy_to_increment_delay=4, price_decrease_to_buy_delay=2, purchase_time=413),
        FraudPurchaseDefinition(product_id=91, new_product_price_percent=0.78, purchase_amount=2, price_buy_to_increment_delay=3, price_decrease_to_buy_delay=3, purchase_time=699),
        FraudPurchaseDefinition(product_id=135, new_product_price_percent=0.72, purchase_amount=4, price_buy_to_increment_delay=2, price_decrease_to_buy_delay=2, purchase_time=1_397),
    ]))

    simulation.add_agent(Fraudster(name='PF3', fraudster_salesperson_id=13, buyer_customer_id=44, products_to_buy=[
        FraudPurchaseDefinition(product_id=32, new_product_price_percent=0.75, purchase_amount=1, price_buy_to_increment_delay=6, price_decrease_to_buy_delay=3, purchase_time=321),
        FraudPurchaseDefinition(product_id=171, new_product_price_percent=0.71, purchase_amount=1, price_buy_to_increment_delay=6, price_decrease_to_buy_delay=2, purchase_time=1_047),
        FraudPurchaseDefinition(product_id=168, new_product_price_percent=0.88, purchase_amount=3, price_buy_to_increment_delay=7, price_decrease_to_buy_delay=3, purchase_time=1_479),
        FraudPurchaseDefinition(product_id=34, new_product_price_percent=0.45, purchase_amount=3, price_buy_to_increment_delay=8, price_decrease_to_buy_delay=5, purchase_time=1_560),
        FraudPurchaseDefinition(product_id=34, new_product_price_percent=0.40, purchase_amount=4, price_buy_to_increment_delay=6, price_decrease_to_buy_delay=2, purchase_time=1_574),
    ]))

    # Stealthy Fraudsters
    simulation.add_agent(Fraudster(name='YF1', fraudster_salesperson_id=4, buyer_customer_id=26, products_to_buy=[
        FraudPurchaseDefinition(product_id=67, new_product_price_percent=0.80, purchase_amount=1, price_decrease_increments=8, price_increase_increments=5, purchase_time=100),
        FraudPurchaseDefinition(product_id=21, new_product_price_percent=0.53, purchase_amount=2, price_decrease_increments=5, price_increase_increments=5, purchase_time=356),
        FraudPurchaseDefinition(product_id=87, new_product_price_percent=0.72, purchase_amount=1, price_decrease_increments=20, price_decrease_to_buy_delay=30, price_buy_to_increment_delay=10, price_increase_increments=5, purchase_time=103),
    ]))

    simulation.add_agent(Fraudster(name='YF2', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=94, new_product_price_percent=0.90, purchase_amount=40, price_decrease_increments=3, price_increase_increments=3, purchase_time=216),
    ]))

    simulation.add_agent(Fraudster(name='YF3', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=49, new_product_price_percent=0.73, purchase_amount=20, price_decrease_increments=5, price_increase_increments=5, purchase_time=739),
    ]))

    simulation.add_agent(Fraudster(name='YF4', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=55, new_product_price_percent=0.80, purchase_amount=25, price_decrease_increments=6, price_increase_increments=6, purchase_time=894),
    ]))

    simulation.add_agent(Fraudster(name='YF5', fraudster_salesperson_id=13, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=55, new_product_price_percent=0.76, purchase_amount=10, price_decrease_increments=5, price_increase_increments=4, purchase_time=1_396),
        FraudPurchaseDefinition(product_id=55, new_product_price_percent=0.72, purchase_amount=25, price_decrease_increments=5, price_increase_increments=5, purchase_time=1_416),
    ]))

    simulation.add_agent(Fraudster(name='YF6', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=123, new_product_price_percent=0.61, purchase_amount=20, price_decrease_increments=5, price_increase_increments=6, purchase_time=1_637),
    ]))

    simulation.run()

    database_config.db.save(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
else:
    database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)

if cfg.CONF_GENERATE_SLICES_AND_GRAPHS:
    graph_gen = GraphGenerator()
    db_slicer = DatabaseSlicer(db=database_config.db, max_simulation_time=cfg.SIMULATION_END_TIME)

    for window_duration in cfg.GRAPH_SLICER_WINDOW_DURATIONS:
        print(f'GENERATING DB FOR WINDOW DURATION = {window_duration}')

        if not os.path.exists(rf'{cfg.STORAGE_BASE_PATH_PY_GRAPHS}\window_duration_{window_duration}'):
            os.mkdir(rf'{cfg.STORAGE_BASE_PATH_PY_GRAPHS}\window_duration_{window_duration}')

        if not os.path.exists(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\window_duration_{window_duration}'):
            os.mkdir(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\window_duration_{window_duration}')

        slices = db_slicer.generate_slices_sliding_window(window_duration=window_duration)

        graphs = graph_gen.generate_graphs(databases=slices)
        graphs.prune(min_cluster_size=cfg.GRAPH_PRUNING_MIN_CLUSTER_SIZE)
        graphs.save(cfg.STORAGE_BASE_PATH_PY_GRAPHS + rf'\window_duration_{window_duration}')

        with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\window_duration_{window_duration}\generate_graphs.bat', 'w') as graphviz_script:
            for index, history_item in enumerate(graphs.get_raw_list()):
                history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\window_duration_{window_duration}\{history_item.get_name()}.txt')
                print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
                      file=graphviz_script)
