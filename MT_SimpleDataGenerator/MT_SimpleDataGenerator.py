import random
import database_config
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

    # generate legitimate small customers
    for i in range(0, 90):
        if i in [6, 55]:
            continue

        simulation.add_agent(Customer(name=f'Small Customer {i}', customer_id=i, tick_buy_probability=random.randint(0, 60) / 1000, max_amount_per_product=1, max_product_count=random.randint(1, 5)))

    # generate legitimate large customers
    simulation.add_agent(Customer(name=f'Large Customer 90', tick_buy_probability=0.31, max_product_count=3, min_amount_per_product=10, max_amount_per_product=20, customer_id=90, use_salesperson_id=8))
    simulation.add_agent(Customer(name=f'Large Customer 91', tick_buy_probability=0.13, max_product_count=2, max_amount_per_product=25, customer_id=91))
    simulation.add_agent(Customer(name=f'Large Customer 92', tick_buy_probability=0.24, customer_id=92, use_salesperson_id=3))
    simulation.add_agent(Customer(name=f'Large Customer 93', tick_buy_probability=0.11, max_amount_per_product=100, customer_id=93))
    simulation.add_agent(Customer(name=f'Large Customer 94', tick_buy_probability=0.53, customer_id=94, use_salesperson_id=6))
    simulation.add_agent(Customer(name=f'Large Customer 95', tick_buy_probability=0.45, customer_id=95, use_salesperson_id=5))
    simulation.add_agent(Customer(name=f'Large Customer 96', tick_buy_probability=0.05, max_product_count=1, min_amount_per_product=10, max_amount_per_product=100, customer_id=96))
    simulation.add_agent(Customer(name=f'Large Customer 97', tick_buy_probability=0.48, customer_id=97, use_salesperson_id=0))
    simulation.add_agent(Customer(name=f'Large Customer 98', tick_buy_probability=0.34, customer_id=98))
    simulation.add_agent(Customer(name=f'Large Customer 99', tick_buy_probability=0.09, max_amount_per_product=20, customer_id=99))

    # generate fraud customers
    # Selfish Fraudsters
    simulation.add_agent(Fraudster(name='Selfish Fraudster 1', fraudster_salesperson_id=6, buyer_customer_id=6, products_to_buy=[
        FraudPurchaseDefinition(product_id=7, new_product_price_percent=0.80, purchase_amount=2, purchase_time=16),
        FraudPurchaseDefinition(product_id=19, new_product_price_percent=0.80, purchase_amount=3, purchase_time=300),
        FraudPurchaseDefinition(product_id=7, new_product_price_percent=0.70, purchase_amount=1, purchase_time=426),
        FraudPurchaseDefinition(product_id=98, new_product_price_percent=0.84, purchase_amount=1, purchase_time=638),
        FraudPurchaseDefinition(product_id=85, new_product_price_percent=0.70, purchase_amount=3, purchase_time=796),
        FraudPurchaseDefinition(product_id=4, new_product_price_percent=0.65, purchase_amount=5, purchase_time=1_067)
    ]))

    simulation.add_agent(Fraudster(name='Selfish Fraudster 2', fraudster_salesperson_id=6, buyer_customer_id=90, products_to_buy=[
        FraudPurchaseDefinition(product_id=33, new_product_price_percent=0.62, purchase_amount=10, purchase_time=552),
        FraudPurchaseDefinition(product_id=43, new_product_price_percent=0.81, purchase_amount=5, purchase_time=956),
        FraudPurchaseDefinition(product_id=24, new_product_price_percent=0.73, purchase_amount=8, purchase_time=1_034)
    ]))

    # Fraudsters with Accomplices
    simulation.add_agent(Fraudster(name='Fraudster w/ Accomplice 1', fraudster_salesperson_id=6, buyer_customer_id=89, products_to_buy=[
        FraudPurchaseDefinition(product_id=46, new_product_price_percent=0.90, purchase_amount=1, purchase_time=123),
        FraudPurchaseDefinition(product_id=10, new_product_price_percent=0.75, purchase_amount=1, purchase_time=407),
        FraudPurchaseDefinition(product_id=10, new_product_price_percent=0.75, purchase_amount=2, purchase_time=613),
        FraudPurchaseDefinition(product_id=91, new_product_price_percent=0.70, purchase_amount=2, purchase_time=722),
        FraudPurchaseDefinition(product_id=83, new_product_price_percent=0.80, purchase_amount=1, purchase_time=994),
    ]))

    simulation.add_agent(Fraudster(name='Fraudster w/ Accomplice 2', fraudster_salesperson_id=6, buyer_customer_id=64, products_to_buy=[
        FraudPurchaseDefinition(product_id=42, new_product_price_percent=0.80, purchase_amount=1, purchase_time=162),
        FraudPurchaseDefinition(product_id=34, new_product_price_percent=0.85, purchase_amount=1, purchase_time=466),
        FraudPurchaseDefinition(product_id=64, new_product_price_percent=0.80, purchase_amount=2, purchase_time=609),
        FraudPurchaseDefinition(product_id=18, new_product_price_percent=0.90, purchase_amount=1, purchase_time=1042),
    ]))

    # Patient Fraudsters
    simulation.add_agent(Fraudster(name='Patient Fraudster 1', fraudster_salesperson_id=4, buyer_customer_id=55, products_to_buy=[
        FraudPurchaseDefinition(product_id=80, new_product_price_percent=0.90, purchase_amount=2, price_buy_to_increment_delay=2, price_decrease_to_buy_delay=1, purchase_time=10),
        FraudPurchaseDefinition(product_id=18, new_product_price_percent=0.83, purchase_amount=1, price_buy_to_increment_delay=4, price_decrease_to_buy_delay=1, purchase_time=917),
    ]))

    simulation.add_agent(Fraudster(name='Patient Fraudster 2', fraudster_salesperson_id=4, buyer_customer_id=44, products_to_buy=[
        FraudPurchaseDefinition(product_id=7, new_product_price_percent=0.90, purchase_amount=1, price_buy_to_increment_delay=2, price_decrease_to_buy_delay=1, purchase_time=237),
        FraudPurchaseDefinition(product_id=32, new_product_price_percent=0.75, purchase_amount=1, price_buy_to_increment_delay=6, price_decrease_to_buy_delay=3, purchase_time=321),
        FraudPurchaseDefinition(product_id=86, new_product_price_percent=0.91, purchase_amount=3, price_buy_to_increment_delay=4, price_decrease_to_buy_delay=2, purchase_time=413),
        FraudPurchaseDefinition(product_id=91, new_product_price_percent=0.80, purchase_amount=2, price_buy_to_increment_delay=3, price_decrease_to_buy_delay=3, purchase_time=699),
    ]))

    # Stealthy Fraudsters
    simulation.add_agent(Fraudster(name='Stealthy Fraudster 1', fraudster_salesperson_id=4, buyer_customer_id=26, products_to_buy=[
        FraudPurchaseDefinition(product_id=67, new_product_price_percent=0.80, purchase_amount=1, price_decrease_increments=8, price_increase_increments=5, purchase_time=100),
        FraudPurchaseDefinition(product_id=21, new_product_price_percent=0.55, purchase_amount=2, price_decrease_increments=5, price_increase_increments=5, purchase_time=356),
        FraudPurchaseDefinition(product_id=87, new_product_price_percent=0.75, purchase_amount=1, price_decrease_increments=20, price_decrease_to_buy_delay=30, price_buy_to_increment_delay=10, price_increase_increments=5, purchase_time=103),
    ]))

    simulation.add_agent(Fraudster(name='Stealthy Fraudster 2', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=94, new_product_price_percent=0.90, purchase_amount=40, price_decrease_increments=3, price_increase_increments=3, purchase_time=216),
    ]))

    simulation.add_agent(Fraudster(name='Stealthy Fraudster 3', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=39, new_product_price_percent=0.78, purchase_amount=20, price_decrease_increments=5, price_increase_increments=5, purchase_time=739),
    ]))

    simulation.add_agent(Fraudster(name='Stealthy Fraudster 4', fraudster_salesperson_id=6, buyer_customer_id=94, products_to_buy=[
        FraudPurchaseDefinition(product_id=55, new_product_price_percent=0.80, purchase_amount=25, price_decrease_increments=6, price_increase_increments=6, purchase_time=894),
    ]))

    simulation.run()

    database_config.db.save(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
else:
    database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)

if cfg.CONF_GENERATE_SLICES_AND_GRAPHS:
    db_slicer = DatabaseSlicer(db=database_config.db, max_simulation_time=cfg.SIMULATION_END_TIME)
    slices = db_slicer.generate_slices_sliding_window(window_duration=cfg.GRAPH_SLICER_WINDOW_DURATION)

    graph_gen = GraphGenerator()
    graphs = graph_gen.generate_graphs(databases=slices)
    graphs.prune(min_cluster_size=cfg.GRAPH_PRUNING_MIN_CLUSTER_SIZE)
    graphs.save(cfg.STORAGE_BASE_PATH_PY_GRAPHS)

    with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\generate_graphs.bat', 'w') as graphviz_script:
        for index, history_item in enumerate(graphs.get_raw_list()):
            history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\{history_item.get_name()}.txt')
            print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
                  file=graphviz_script)