import numpy as np
import networkx as nx
import numpy as np
from typing import Tuple, List


def getMatchingGenes(
    g1: nx.DiGraph, g2: nx.DiGraph
) -> List[Tuple[nx.DiGraph, nx.DiGraph]]:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    g2gins = {e[2]["gin"]: e for e in g2edges}
    g1gins = {e[2]["gin"]: e for e in g1edges}

    matching_gins = set(g1gins.keys()).intersection(set(g2gins.keys()))
    matching = [(g1gins[gin], g2gins[gin]) for gin in matching_gins]

    assert [(u, v) for (u, v, _), _ in matching] == [(u, v) for _, (u, v, _) in matching]
    return matching

def getDisjointGenes(g1: nx.DiGraph, g2: nx.DiGraph) -> List[Tuple[int, nx.DiGraph]]:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    g1max = max([e[2]["gin"] for e in g1edges])
    g2max = max([e[2]["gin"] for e in g2edges])
    g1gins = set([e[2]["gin"] for e in g1edges])
    g2gins = set([e[2]["gin"] for e in g2edges])

    disjoint = []

    for edge in g1edges:
        if edge[2]["gin"] not in g2gins and edge[2]["gin"] < g2max:
            disjoint.append((1, edge))

    for edge in g2edges:
        if edge[2]["gin"] not in g1gins and edge[2]["gin"] < g1max:
            disjoint.append((2, edge))

    return disjoint


def getExcessGenes(g1: nx.DiGraph, g2: nx.DiGraph) -> List[Tuple[int, nx.DiGraph]]:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    g1gins = set([e[2]["gin"] for e in g1edges])
    g2gins = set([e[2]["gin"] for e in g2edges])
    g1max = max([e[2]["gin"] for e in g1edges])
    g2max = max([e[2]["gin"] for e in g2edges])

    excess = []

    for edge in g1edges:
        if edge[2]["gin"] not in g2gins and edge[2]["gin"] > g2max:
            excess.append((1, edge))

    for edge in g2edges:
        if edge[2]["gin"] not in g1gins and edge[2]["gin"] > g1max:
            excess.append((2, edge))

    return excess


def getEdgeTypes(g1: nx.DiGraph, g2: nx.DiGraph) -> Tuple[list, list, list]:
    return (getMatchingGenes(g1, g2), getDisjointGenes(g1, g2), getExcessGenes(g1, g2))


def delta(g1: nx.DiGraph, g2: nx.DiGraph, config: dict) -> float:
    matching, disjoint, excess = getEdgeTypes(g1, g2)

    N = max(len(g1.edges()), len(g2.edges()))

    E = len(excess)
    D = len(disjoint)
    W_bar = sum(
        [np.abs(e1[2]["weight"] - e2[2]["weight"]) for e1, e2 in matching]
    ) / len(matching)

    return config["C1"] * D / N + config["C2"] * E / N + config["C3"] * W_bar


def MSE(y_true: np.array, y_pred: np.array) -> float:
    return -np.sum(y_true - y_pred) ** 2


def unif(start=0, end=1):
    return np.random.uniform(start, end)
