from genetic_algorithm.constants import OUTPUT_NODE_NAME
from typing import Tuple
import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import count
import numpy as np

class Genome:
    global_innovation_number = count()
    
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.global_innovation_number = count(start=num_features)

    def newNet(self, random_weights: bool = True) -> nx.DiGraph:
        weight_gen = np.random.random if random_weights else lambda : 1

        # Create the basic network
        net = nx.DiGraph()
        out_node_id = self.num_features
        w_edges = [(i, out_node_id, {"weight": weight_gen(), "gin": i, "feature": i}) for i in range(self.num_features)]
        net.add_weighted_edges_from(w_edges)
        nx.set_node_attributes(net, {out_node_id: True}, name=OUTPUT_NODE_NAME) 
        print(net)
        return net

    def clone(self):
        return deepcopy(self)

    @classmethod
    def newNode(cls, net: nx.DiGraph) -> nx.DiGraph:
        edges = list(net.edges(data=True))

        # Get random edge to split
        i = np.random.randint(len(edges))
        src_node, snk_node, data = edges[i]
        new_node = np.max(list(net.nodes())) + 1
 
        # Add the new edge
        net.add_edges_from([
            (src_node, new_node, {"weight": 1, "gin": next(cls.global_innovation_number)}),
            (new_node, snk_node, data)
        ])

        # Disable the original edge
        nx.set_edge_attributes(net, {(src_node, snk_node): {"disabled": True}})

        # Ensure the net is still a DAG
        assert nx.is_directed_acyclic_graph(net)
        return net

    @classmethod
    def newConnection(cls, net: nx.DiGraph) -> nx.DiGraph:
        nodes = net.nodes()
        edges = net.edges()

        # There is

        return net

    @classmethod
    def evaluate(cls, net: nx.DiGraph, dataset: pd.DataFrame) -> np.array:
        out_node = [node for node in net.nodes(data=True) if OUTPUT_NODE_NAME in node][0]
        type(out_node)
        return cls.__evaluate_helper(net, out_node, dataset) 
    
    @classmethod
    def __evaluate_helper(cls, net: nx.DiGraph, out_node: Tuple[int, dict], dataset: pd.DataFrame) -> np.array:
        # If the node is an input, just get the corresponding feature
        node_id, data_dict = out_node
        if "feature" in data_dict:
            return dataset[data_dict["feature"]].to_numpy()

        # else, compute all input nodes
        incoming_edges = net.in_edges(node_id, data=True)
        result = np.array([])
        for edge in incoming_edges:
            assert edge[1] == node_id

            inp = cls.__evaluate_helper(net, (edge[0], net.nodes[edge[0]]), dataset)
            weight: float = edge[2]["weight"]
            if np.prod(result.shape) == 0:
                result = np.dot(inp, weight)
            else:
                result += np.dot(inp, weight)

        return result 

        
    @classmethod
    def crossover(cls, first: nx.DiGraph, second: nx.DiGraph) -> nx.DiGraph:
        return first
    
