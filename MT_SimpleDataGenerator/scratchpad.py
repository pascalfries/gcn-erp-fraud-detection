import config as cfg
from graph.GraphCollection import GraphCollection

gc = GraphCollection(directory_name=cfg.STORAGE_BASE_PATH_PY_GRAPHS)

with open(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\generate_graphs.bat', 'w') as graphviz_script:
    for index, history_item in enumerate(gc.get_raw_list()):
        history_item.export_graphviz(rf'{cfg.STORAGE_BASE_PATH_GRAPHVIZ_GRAPHS}\{history_item.get_name()}.txt')
        print(f'dot -Tsvg {history_item.get_name()}.txt -o graph_{history_item.get_name()}.svg',
              file=graphviz_script)
