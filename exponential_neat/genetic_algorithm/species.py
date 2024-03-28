import networkx as nx

class Species:
    def __init__(self):
        self.genomes = []
        self.stalled_gens = 0
        self.representative_genome = []
        
