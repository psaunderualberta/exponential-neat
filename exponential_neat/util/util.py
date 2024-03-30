import numpy as np
import networkx as nx
import numpy as np
from typing import Tuple

def getMatchingGenes(g1: nx.DiGraph, g2: nx.DiGraph) -> tuple:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    print(g2edges)
    g2gins = set([e[2]["gin"] for e in g2edges])

    matching = []
    for edge in g1edges:
        if edge[2]["gin"] in g2gins:
            matching.append(edge)
    
    return matching

def getDisjointGenes(g1: nx.DiGraph, g2: nx.DiGraph) -> tuple:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    g1gins = set([e[2]["gin"] for e in g1edges])
    g2gins = set([e[2]["gin"] for e in g2edges])
    g1max = max([e[2]["gin"] for e in g1edges])
    g2max = max([e[2]["gin"] for e in g2edges])

    disjoint = []

    for edge in g1edges:
        if edge[2]["gin"] not in g2gins and edge[2]["gin"] < g2max:
            disjoint.append(edge)

    for edge in g2edges:
        if edge[2]["gin"] not in g1gins and edge[2]["gin"] < g1max:
            disjoint.append(edge)
    
    return disjoint

def getExcessGenes(g1: nx.DiGraph, g2: nx.DiGraph) -> tuple:
    g1edges = g1.edges(data=True)
    g2edges = g2.edges(data=True)

    g1gins = set([e[2]["gin"] for e in g1edges])
    g2gins = set([e[2]["gin"] for e in g2edges])
    g1max = max([e[2]["gin"] for e in g1edges])
    g2max = max([e[2]["gin"] for e in g2edges])

    excess = []

    for edge in g1edges:
        if edge[2]["gin"] not in g2gins and edge[2]["gin"] > g2max:
            excess.append(edge)

    for edge in g2edges:
        if edge[2]["gin"] not in g1gins and edge[2]["gin"] > g1max:
            excess.append(edge)
    
    return excess

def getEdgeTypes(g1: nx.DiGraph, g2: nx.DiGraph) -> Tuple[list, list, list]:
    return (
        getMatchingGenes(g1, g2),
        getDisjointGenes(g1, g2),
        getExcessGenes(g1, g2)
    )

def delta(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
    matching, disjoint, excess = getEdgeTypes(g1, g2)

    # TODO: Set this as a parameter
    w = 0.5
    N = max(len(g1.edges()), len(g2.edges()))

    return w * len(disjoint) / N + w * len(excess) / N + (1 - w) * sum(
        [np.abs(e1[2]["weight"] - e2[2]["weight"]) for e1, e2 in matching]
    ) / len(matching)

def MSE(y_true: np.array, y_pred: np.array) -> float:
    return np.power(y_true - y_pred, 2).mean()

def unif(start=0, end=1):
    return np.random.uniform(start, end)
