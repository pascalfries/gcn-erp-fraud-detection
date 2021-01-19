import config as cfg
from sap.SapExtractor import SapExtractor
from graph.GraphGenerator import GraphGenerator
from data.DatabaseSlicer import DatabaseSlicer

# config
WINDOW_DURATION = 1_800 # 30min
WINDOW_STRIDE = 900 # 15min

# extract data
sap_extractor = SapExtractor(r'C:\Users\Pasi\OneDrive\Documents\Uni\MSem. 4 - SS 20\MT - Master Thesis\Data\ERPSimPlanspielAlleDaten\V2_mitFraud PJS1920')
db, min_time, max_time = sap_extractor.extract()

# slice data
print(f'MAX TIME: {max_time}')
data_slicer = DatabaseSlicer(db, max_simulation_time=max_time, min_time=min_time)
db_slices = data_slicer.generate_slices_sliding_window(WINDOW_DURATION, WINDOW_STRIDE)

# generate graphs
graph_gen = GraphGenerator()
graphs = graph_gen.generate_graphs(db_slices)

graphs.prune(min_cluster_size=cfg.GRAPH_PRUNING_MIN_CLUSTER_SIZE)
graphs.save(cfg.STORAGE_BASE_PATH_PY_GRAPHS + rf'\sap')

with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\sap\generate_graphs.bat', 'w') as graphviz_script:
    for index, history_item in enumerate(graphs.get_raw_list()):
        history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\sap\{history_item.get_name()}.txt')
        print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
              file=graphviz_script)

print('done')
