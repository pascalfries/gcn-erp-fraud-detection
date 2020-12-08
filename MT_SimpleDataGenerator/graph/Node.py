from typing import List, Dict


class Node:
    __nextId: int = 1

    def __init__(self, properties: Dict, key: str = None, color: str = None, node_color: str = 'black', node_type: str = 'unknown'):
        self._id: int = Node.__nextId
        Node.__nextId += 1

        self._neighbors: List[Node] = []
        self._references: List[Node] = []
        self._neighbor_colors: List[str] = []
        self._properties: Dict = properties
        self._key: str = key
        self._color: str = color
        self._type: str = node_type
        self._node_color: str = node_color
        self._is_found = False
        self._is_part_of_large_cluster = False

    def get_is_found(self) -> bool:
        return self._is_found

    def get_is_part_of_large_cluster(self) -> bool:
        return self._is_part_of_large_cluster

    def set_is_found(self, is_found: bool):
        self._is_found = is_found

    def set_is_part_of_large_cluster(self, is_part_of_large_cluster: bool):
        self._is_part_of_large_cluster = is_part_of_large_cluster

    def get_id(self) -> int:
        return self._id

    def get_type(self) -> str:
        return self._type

    def get_property_value(self, property_name: str):
        return self._properties[property_name]

    def add_neighbor(self, neighbor, color: str = None) -> None:
        if neighbor is not None and neighbor not in self._neighbors:
            self._neighbors.append(neighbor)
            self._neighbor_colors.append(color)

            neighbor.add_backreference(self)

    def add_backreference(self, neighbor_of):
        if neighbor_of is not None and neighbor_of not in self._references:
            self._references.append(neighbor_of)

    def remove_from_neighbors(self,) -> None:
        for neighbor in (self._neighbors + self._references):
            if self in neighbor._neighbors:
                neighbor._neighbors.remove(self)

            if self in neighbor._references:
                neighbor._references.remove(self)

    #     if neighbor is not None and neighbor in self._neighbors:
    #         index = self._neighbors.index(neighbor)
    #
    #         neighbor.remove_backreference(self)
    #         del self._neighbors[index]
    #         del self._neighbor_colors[index]
    #
    # def remove_backreference(self, neighbor_of):
    #     if neighbor_of is not None and neighbor_of in self._references:
    #         self._references.append(neighbor_of)

    def get_neighbors(self) -> List:
        return self._neighbors

    def get_references(self) -> List:
        return self._references

    def get_neighbor_colors(self) -> List[str]:
        return self._neighbor_colors

    def get_properties(self) -> Dict:
        return self._properties

    def get_key(self) -> str:
        return self._key

    def generate_graphviz(self):
        property_string = '\n\n' + '\n'.join([f'{property}: {value}' for property, value in self._properties.items()])
        link_string = '\n'.join([f'node_{self._id} -> node_{neighbor.get_id()};\n' if color is None else f'node_{self._id} -> node_{neighbor.get_id()} [color={color}];' for neighbor, color in zip(self._neighbors, self._neighbor_colors)])

        if self._key is not None:
            return f'node_{self._id} [label="{self._key}{property_string}",color={self._node_color}];\n{link_string}'
        else:
            return f'node_{self._id} [label="Node {str(self._id)}{property_string},color={self._node_color}];\n{link_string}'
