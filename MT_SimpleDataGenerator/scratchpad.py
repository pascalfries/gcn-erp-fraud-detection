import config as cfg
from graph.GraphCollection import GraphCollection
from sap.SapExtractor import SapExtractor
from graph.GraphGenerator import GraphGenerator

sap_extractor = SapExtractor(r'C:\Users\Pasi\OneDrive\Documents\Uni\MSem. 4 - SS 20\MT - Master Thesis\Data\ERPSimPlanspielAlleDaten\V2_mitFraud PJS1920')

db = sap_extractor.extract()

graph_gen = GraphGenerator()
graph = graph_gen.generate_graph(db)

graph.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\tmp\sap.txt') # dot -Tsvg sap.txt -o sap.svg

print('done')













exit()

gc = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)

with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\generate_graphs.bat', 'w') as graphviz_script:
    for index, history_item in enumerate(gc.get_raw_list()):
        history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\{history_item.get_name()}.txt')
        print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
              file=graphviz_script)
