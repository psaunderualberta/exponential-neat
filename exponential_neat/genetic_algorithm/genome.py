from util.constants import OUTPUT_NODE_NAME
from typing import Tuple, Any, List, Callable
import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import count
import numpy as np
from random import choice
from util.util import getEdgeTypes


class Genome:
    global_innovation_number = count()

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.global_innovation_number = count(start=num_features + 1)
        self.activ = lambda x: 1 / (1 + np.exp(-5 * x))

    def newNet(self, random_weights: bool = True) -> nx.DiGraph:
        weight_gen = np.random.normal if random_weights else lambda: 1

        # Create the basic network
        net = nx.DiGraph()
        out_node_id = self.num_features + 1
        w_edges = [
            (i, out_node_id, {"weight": weight_gen(), "gin": i})
            for i in range(self.num_features)
        ] + [
            (
                self.num_features,
                out_node_id,
                {"weight": weight_gen(), "gin": self.num_features},
            )
        ]
        net.add_edges_from(w_edges)

        # Add a single hidden node
        hidden_node_id = self.num_features + 2
        net.add_edges_from(
            [
                (i, hidden_node_id, {"weight": weight_gen(), "gin": self.num_features + i + 1})
                for i in range(self.num_features + 1)
            ] + [(hidden_node_id, out_node_id, {"weight": weight_gen(), "gin": 100})]
        )

        return self.annotateNodes(net)

    def annotateNodes(self, net: nx.DiGraph) -> nx.DiGraph:
        # Add the nodes if they are not already present
        for i in range(self.num_features + 2):
            net.add_node(i)

        nx.set_node_attributes(
            net, {self.num_features + 1: True}, name=OUTPUT_NODE_NAME
        )
        nx.set_node_attributes(
            net, {i: i for i in range(self.num_features)}, name="feature"
        )
        nx.set_node_attributes(net, {self.num_features: True}, name="bias")
        return net

    def clone(self, net: nx.DiGraph) -> nx.DiGraph:
        return deepcopy(net)

    def newNode(
        self, net: nx.DiGraph, edge2split: Tuple[int, int] | None = None
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
                    {"weight": 1, "gin": next(self.global_innovation_number)},
                ),
                (
                    new_node,
                    snk_node,
                    {
                        "weight": data["weight"],
                        "gin": next(self.global_innovation_number),
                    },
                ),
            ]
        )

        # Disable the original edge
        nx.set_edge_attributes(net, {(src_node, snk_node): {"disabled": True}})

        # Ensure the net is still a DAG
        assert nx.is_directed_acyclic_graph(net)
        return net

    def newConnection(self, net: nx.DiGraph) -> nx.DiGraph:
        nodes = list(net.nodes())
        edges = set(net.edges())
        out_node = [
            node for node in net.nodes(data=True) if OUTPUT_NODE_NAME in node[1]
        ][0]
        input_node_ids = set(range(out_node[0]))
        output_node_id = out_node[0]

        for _ in range(1000):
            n1 = choice(nodes)
            n2 = choice(nodes)

            if (
                n1 == n2
                or n2 in input_node_ids
                or n1 == output_node_id
                or (n1, n2) in edges
            ):
                continue

            nc = deepcopy(net)
            nc.add_edge(n1, n2, weight=np.random.random())

            if nx.is_directed_acyclic_graph(nc):
                nx.set_edge_attributes(
                    nc, {(n1, n2): {"gin": next(self.global_innovation_number)}}
                )
                return nc

        # Edge case where every connection (probably) already exists
        return net

    def predict(self, net: nx.DiGraph, dataset: np.array) -> np.array:
        out_node = [
            node for node in net.nodes(data=True) if OUTPUT_NODE_NAME in node[1]
        ][0]
        return self.__predictHelper(net, out_node, dataset)

    def __predictHelper(
        self, net: nx.DiGraph, out_node: Tuple[int, dict], dataset: np.array
    ) -> np.array:
        # If the node is an input, just get the corresponding feature
        node_id, data_dict = out_node
        if "feature" in data_dict:
            return dataset[:, data_dict["feature"]].reshape(-1, 1)
        elif "bias" in data_dict:
            return np.ones((dataset.shape[0], 1))

        # else, compute all input nodes
        incoming_edges = net.in_edges(node_id, data=True)
        result = np.zeros((dataset.shape[0], 1))
        for edge in incoming_edges:
            assert edge[1] == node_id

            # Skip disabled edges
            if edge[2].get("disabled", False):
                continue

            inp = self.__predictHelper(net, (edge[0], net.nodes[edge[0]]), dataset)
            weight = edge[2]["weight"]
            result += np.dot(inp, weight)

        return self.activ(result)

    def crossover(self, fitter: nx.DiGraph, weaker: nx.DiGraph) -> nx.DiGraph:
        # Create a new network with the same nodes as the first parent
        if fitter.graph["fitness"] < weaker.graph["fitness"]:
            fitter, weaker = weaker, fitter

        child = self.annotateNodes(nx.DiGraph())

        # Get the edge types
        matching, disjoint, excess = getEdgeTypes(fitter, weaker)

        # Randomly select edges to inherit
        for fitter_edge, weaker_edge in matching:
            if np.random.random() < 0.5:
                child.add_edge(*fitter_edge[:2], **fitter_edge[2])
            else:
                child.add_edge(*weaker_edge[:2], **weaker_edge[2])
            
            # Each disabled edge has a 75% chance of being disabled if either parent has it disabled
            if fitter_edge[2].get("disabled", False) or weaker_edge[2].get("disabled", False):
                child.edges[fitter_edge[:2]]["disabled"] = np.random.random() < 0.75

        # Inherit disjoint and excess genes from fitter parent
        assert len(disjoint) + len(excess) == 0
        for edge in filter(lambda e: e[0] == 1, disjoint + excess):
            child.add_edge(*edge[1][:2], **edge[1][2])

        assert nx.is_directed_acyclic_graph(child)
        return child
