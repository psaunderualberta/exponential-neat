import pytest
import networkx as nx
from util.util import getEdgeTypes
from util.constants import OUTPUT_NODE_NAME


class TestUtil:
    def test_get_edge_types(self):
        """ This example is taken from the NEAT paper by Stanley and Miikkulainen (2002)"""

        g1 = nx.DiGraph()
        g1.add_edges_from(
            [
                (1, 4, {"weight": 1, "gin": 1}),
                (2, 4, {"weight": 1, "gin": 2}),
                (3, 4, {"weight": 1, "gin": 3}),
                (2, 5, {"weight": 1, "gin": 4}),
                (5, 4, {"weight": 1, "gin": 5}),
                (1, 5, {"weight": 1, "gin": 8}),
            ]
        )

        g2 = nx.DiGraph()
        g2.add_edges_from(
            [
                (1, 4, {"weight": 1, "gin": 1}),
                (2, 4, {"weight": 1, "gin": 2}),
                (3, 4, {"weight": 1, "gin": 3}),
                (2, 5, {"weight": 1, "gin": 4}),
                (5, 4, {"weight": 1, "gin": 5}),
                (5, 6, {"weight": 1, "gin": 6}),
                (6, 4, {"weight": 1, "gin": 7}),
                (3, 5, {"weight": 1, "gin": 9}),
                (1, 6, {"weight": 1, "gin": 10}),
            ]
        )

        nx.set_node_attributes(
            g1,
            {
                1: {"feature": 0},
                2: {"feature": 1},
                3: {"feature": 2},
                4: {"feature": None, OUTPUT_NODE_NAME: True},
                5: {"feature": None},
            },
        )

        nx.set_node_attributes(
            g2,
            {
                1: {"feature": 0},
                2: {"feature": 1},
                3: {"feature": 2},
                4: {"feature": None, OUTPUT_NODE_NAME: True},
                5: {"feature": None},
                6: {"feature": None},
            },
        )

        matching, disjoint, excess = getEdgeTypes(g1, g2)
        matching_gins = set([e[2]["gin"] for e in matching])
        disjoint_gins = set([e[2]["gin"] for e in disjoint])
        excess_gins = set([e[2]["gin"] for e in excess])

        assert matching_gins == {1, 2, 3, 4, 5}
        assert disjoint_gins == {6, 7, 8}
        assert excess_gins == {9, 10}
