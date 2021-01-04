# randomization setup
RANDOM_SEED_GENERATOR = 123
RANDOM_SEED_SIMULATION = 456
RANDOM_SEED_MODEL = 789

# storage paths
STORAGE_ROOT_PATH = r'C:\Users\Pasi\Documents (offline)'

STORAGE_BASE_PATH_MASTER_DATA = STORAGE_ROOT_PATH + r'\Data Thesis'
STORAGE_BASE_PATH_SIMULATED_DATA = STORAGE_ROOT_PATH + r'\Data Thesis Simulated'
STORAGE_BASE_PATH_PY_GRAPHS = STORAGE_ROOT_PATH + r'\Data Thesis Simulated\pygraphs'
STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS = STORAGE_ROOT_PATH + r'\Data Thesis Simulated\graphs'
STORAGE_BASE_PATH_PLOTS = STORAGE_ROOT_PATH + r'\Data Thesis Simulated\plots'
STORAGE_BASE_THESIS_IMG = r'C:\Users\Pasi\OneDrive\Documents\Uni\MSem. 4 - SS 20\MT - Master Thesis\Schriftlich\Thesis\img\auto'

# runtime setup
CONF_GENERATE_MASTER_DATA = True
CONF_RUN_SIMULATION = True
CONF_GENERATE_SLICES_AND_GRAPHS = True

# initial master data generator setup
INIT_GEN_CUSTOMER_COUNT = 150
INIT_GEN_PRODUCT_COUNT = 200
INIT_GEN_SALESPERSON_COUNT = 15

# simulation setup
SIMULATION_END_TIME = 365 * 5 # = 1_825
SIMULATION_SPECIAL_EVENT_TIMES = [355, 720, 1_085, 1_450, 1_815]

# database slice generation & pruning
GRAPH_SLICER_WINDOW_DURATIONS = [2, 3, 5, 10]
GRAPH_PRUNING_MIN_CLUSTER_SIZE = 4

# evaluation script
EVAL_SIMULATION_END_TIME = 365 * 2

# output
VERBOSE_OUTPUT = False

