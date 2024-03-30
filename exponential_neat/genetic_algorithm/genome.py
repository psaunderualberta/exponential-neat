from genetic_algorithm.constants import OUTPUT_NODE_NAME
from typing import Tuple, Any, List, Callable
import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import count
import numpy as np
from random import choice


class Genome:
    global_innovation_number = count()
    activ = lambda x: np.maximum(0, x)

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.global_innovation_number = count(start=num_features)

    def newNet(self, random_weights: bool = True) -> nx.DiGraph:
        weight_gen = np.random.random if random_weights else lambda: 1

        # Create the basic network
        net = nx.DiGraph()
        out_node_id = self.num_features
        w_edges = [
            (i, out_node_id, {"weight": weight_gen(), "gin": i})
            for i in range(self.num_features)
        ]
        net.add_edges_from(w_edges)
        nx.set_node_attributes(net, {out_node_id: True}, name=OUTPUT_NODE_NAME)
        for i in range(self.num_features):
            nx.set_node_attributes(
                net, {i: i for i in range(self.num_features)}, name="feature"
            )
        return net

    @classmethod
    def clone(cls, net: nx.DiGraph) -> nx.DiGraph:
        return deepcopy(net)

    @classmethod
    def newNode(
        cls, net: nx.DiGraph, edge2split: Tuple[int, int] | None = None
    ) -> nx.DiGraph:
        net = deepcopy(net)
        edges: List[Tuple[int, int, dict[str, Any]]] = list(net.edges(data=True))

        if edge2split is None:
            edge = choice(edges)
        else:
            edge = (*edge2split, net.edges[edge2split[0], edge2split[1]])
        src_node, snk_node, data = edge
        new_node = np.max(list(net.nodes())) + 1

        # Add the new edge
        net.add_edges_from(
            [
                (
                    src_node,
                    new_node,
                    {"weight": 1, "gin": next(cls.global_innovation_number)},
                ),
                (new_node, snk_node, data),
            ]
        )

        # Disable the original edge
        nx.set_edge_attributes(net, {(src_node, snk_node): {"disabled": True}})

        # Ensure the net is still a DAG
        assert nx.is_directed_acyclic_graph(net)
        return net

    @classmethod
    def newConnection(cls, net: nx.DiGraph) -> nx.DiGraph:
        nodes = list(net.nodes())
        edges = list(net.edges())
        out_node = [
            node for node in net.nodes(data=True) if OUTPUT_NODE_NAME in node[1]
        ][0]
        input_node_ids = set(range(out_node[0]))

        # Edge case where every connection already exists
        for _ in range(1000):
            n1 = choice(nodes)
            n2 = choice(nodes)

            if n1 == n2 or (n1, n2) in edges or n2 in input_node_ids:
                continue

            nc = deepcopy(net)
            nc.add_edge(n1, n2, weight=np.random.random())

            if nx.is_directed_acyclic_graph(nc):
                nx.set_edge_attributes(
                    nc, {(n1, n2): {"gin": next(cls.global_innovation_number)}}
                )
                return nc

        raise Exception("Cannot add to net and keep DAG nature")

    @classmethod
    def evaluate(cls, net: nx.DiGraph, dataset: np.array) -> np.array:
        print(net.nodes(data=True))
        print(net.edges(data=True))
        out_node = [
            node for node in net.nodes(data=True) if OUTPUT_NODE_NAME in node[1]
        ][0]
        return cls.__evaluateHelper(net, out_node, dataset)

    @classmethod
    def __evaluateHelper(
        cls, net: nx.DiGraph, out_node: Tuple[int, dict], dataset: np.array
    ) -> np.array:
        # If the node is an input, just get the corresponding feature
        node_id, data_dict = out_node
        if "feature" in data_dict:
            return dataset[:, data_dict["feature"]]

        # else, compute all input nodes
        incoming_edges = net.in_edges(node_id, data=True)
        result = np.array([])
        for edge in incoming_edges:
            assert edge[1] == node_id

            # Skip disabled edges
            if edge[2].get("disabled", False):
                continue

            inp = cls.__evaluateHelper(net, (edge[0], net.nodes[edge[0]]), dataset)
            weight = edge[2]["weight"]
            if np.prod(result.shape) == 0:
                result = np.dot(inp, weight)
            else:
                result += np.dot(inp, weight)

        print(result)
        return cls.activ(result)

    @classmethod
    def crossover(cls, first: nx.DiGraph, second: nx.DiGraph) -> nx.DiGraph:
        return first
