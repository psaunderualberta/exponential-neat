import networkx as nx
import numpy as np
from random import choice


class Species:
    def __init__(self, representative: nx.DiGraph):
        self.representative_genome = representative
        self.genomes = [representative]
        self.stalled_gens = 0

    def getRepresentativeGenome(self) -> nx.DiGraph:
        return self.representative_genome

    def getRandomGenome(self) -> nx.DiGraph:
        return choice(self.genomes)

    def updateRepresentativeGenome(self) -> nx.DiGraph:
        self.representative_genome = choice(self.genomes)
        return self.representative_genome

    def pruneWorstGenomes(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.genomes)

    def isStagnated(self) -> bool:
        return False

    def addGenome(self, genome: nx.DiGraph) -> nx.DiGraph:
        self.genomes.append(genome)
        return genome
