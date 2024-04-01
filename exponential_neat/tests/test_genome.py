from genetic_algorithm.genome import Genome
import numpy as np
import pytest
import pandas as pd
import networkx as nx
from util.constants import OUTPUT_NODE_NAME


class TestGenome:
    G = Genome(3)

    def test_create(self):
        net = self.G.newNet(random_weights=False)
        assert list(net.edges()) == [(0, 4), (1, 4), (2, 4), (3, 4)]

    def test_evaluate(self):
        pass

    def test_crossover(self):
        pass

    def test_new_node_set_edge(self):
        G = Genome(3)

        net = G.newNet()
        assert list(net.edges()) == [(0, 4), (1, 4), (2, 4), (3, 4)]

        edge2split = (0, 4)
        newnet = G.newNode(net, edge2split)

        # Ensure the correct edge & node was added
        assert set(newnet.nodes()) - set(net.nodes()) == set([5])
        assert set(newnet.edges()) - set(net.edges()) == set([(0, 5), (5, 4)])
        assert newnet.edges[0, 4].get("disabled", False) == True

        old_weight = net.edges[0, 4]["weight"]
        new_weight_first = newnet.edges[0, 5]["weight"]
        new_weight_second = newnet.edges[5, 4]["weight"]

        assert new_weight_first == 1.0
        assert new_weight_second == old_weight

    def test_new_node_random(self):
        G = Genome(3)

        net = G.newNet()
        assert list(net.edges()) == [(0, 4), (1, 4), (2, 4), (3, 4)]

        newnet = G.newNode(net)

        # Ensure the correct edge & node was added
        assert set(newnet.nodes()) - set(net.nodes()) == set([5])

        # Get the newly added node
        new_edge_data = sorted(list(set(newnet.edges()) - set(net.edges())))
        nsrc = new_edge_data[0][0]
        nsnk = new_edge_data[1][1]

        old_weight = net.edges[nsrc, nsnk]["weight"]
        new_weight_first = newnet.edges[nsrc, 5]["weight"]
        new_weight_second = newnet.edges[5, nsnk]["weight"]

        assert new_weight_first == 1.0
        assert new_weight_second == old_weight
        assert newnet.edges[nsrc, nsnk].get("disabled", False) == True

    def test_new_connection(self):
        G = Genome(3)

        net = G.newNet()
        assert list(net.edges()) == [(0, 4), (1, 4), (2, 4), (3, 4)]

        net.remove_edge(2, 4)
        newnet = G.newConnection(net)

        assert len(list(newnet.nodes())) == len(list(net.nodes()))
        assert len(newnet.edges()) == 1 + len(net.edges())

    def test_unable_to_add_connection(self):
        G = Genome(2)

        net = G.newNet()
        net.add_edge(0, 1)

        assert G.newConnection(net) == net

    # def test_evaluate_simple(self):
    #     G = Genome(3)

    #     data = np.array([[1, 0, 1], [-1, -1, -1], [5, 3, 1]]).astype(np.float32)

    #     net = G.newNet(False)

    #     result = G.predict(net, data)
    #     assert np.all(result == np.array([2, 0, 9]))

    def test_evaluate_complex(self):
        num_features = 2
        G = Genome(num_features)
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]]).astype(np.float32)
        y = np.array([1, 1, 0, 0]).astype(np.float32)

        net = nx.DiGraph()
        net.add_edges_from(
            [
                (0, 4, {"weight": -1.62}),
                (4, 3, {"weight": -4.40}),
                (1, 3, {"weight": -7.22}),
                (2, 3, {"weight": -5.08}),
            ]
        )

        nx.set_node_attributes(net, {num_features + 1: True}, name=OUTPUT_NODE_NAME)
        nx.set_node_attributes(net, {i: i for i in range(num_features)}, name="feature")
        nx.set_node_attributes(net, {num_features: True}, name="bias")

        result = G.predict(net, X)
        print(result)
        print(y)
        assert np.all(result == y)
