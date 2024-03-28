import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import count
import numpy as np

class Genome:
    global_innovation_number = count()
    
    def __init__(self, num_features: int, random_weights: bool = False):
        self.num_features = num_features
        self.global_innovation_number = count(start=num_features)

    def new_net(self):
        self.net = nx.DiGraph()

        # Create the basic network
        out_node_id = self.num_features
        w_edges = [(i, out_node_id, {"weight": np.random.random(), "gin": i}) for i in range(self.num_features)]
        self.net.add_weighted_edges_from(w_edges)
        print(self.net)

    def clone(self):
        return deepcopy(self)

    def new_node(self):
        edges = self.net.g 

    def new_connection(self):
        pass

    def evaluate(self, dataset: pd.DataFrame):
        pass
        
    def crossover(self, other: Genome):
        pass
    
