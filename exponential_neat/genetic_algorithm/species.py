import networkx as nx
import numpy as np
from random import choice


class Species:
    def __init__(self, representative: nx.DiGraph, config: dict = {}):
        self.representative_genome = representative
        self.genomes = [representative]
        self.stalled_gens = 0
        self.best_fitness = -np.inf
        self.config = config

    def getRepresentativeGenome(self) -> nx.DiGraph:
        return self.representative_genome

    def getChampion(self) -> nx.DiGraph:
        return max(self.genomes, key=lambda x: x.graph["fitness"])

    def getRandomGenome(self) -> nx.DiGraph:
        return choice(self.genomes)

    def updateRepresentativeGenome(self) -> nx.DiGraph:
        self.representative_genome = choice(self.genomes)
        return self.representative_genome

    def pruneWorstGenomes(self) -> None:
        sorted_genomes = sorted(self.genomes, key=lambda x: x.graph["fitness"])
        num_to_keep = max(self.config["min-species-size"], int(len(self.genomes) * self.config["survival-threshold"]))
        self.genomes = sorted_genomes[:num_to_keep]

    def __len__(self) -> int:
        return len(self.genomes)

    def isStagnated(self) -> bool:
        return self.stalled_gens >= 20  # TODO: Make this a config parameter

    def addGenome(self, genome: nx.DiGraph) -> nx.DiGraph:
        self.genomes.append(genome)
        return genome
    
    def logGeneration(self) -> None:
        best_new_fitness = max([g.graph["fitness"] for g in self.genomes])
        if self.best_fitness < best_new_fitness:
            self.stalled_gens = 0
            self.best_fitness = best_new_fitness
        else:
            self.stalled_gens += 1
