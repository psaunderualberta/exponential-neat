import networkx as nx
from typing import List
from util.constants import (
    DELTA_T,
    P_CROSSOVER,
    P_NEW_CONNECTION,
    P_NEW_NODE,
    P_WEIGHT,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_MU,
    WEIGHT_SIGMA,
    POPULATION_SIZE,
)
from genetic_algorithm.genome import Genome
from genetic_algorithm.species import Species
from util.util import delta, unif
import numpy as np
from collections import Counter


class Population:
    def __init__(self, num_features: int, config: dict = {}):
        self.species: List[Species] = []
        self.new_genomes: List[nx.DiGraph] = []
        self.config = config
        self.num_features = num_features
        self.genome = Genome(num_features)

    def updateRepresentativeGenomes(self):
        for species in self.species:
            species.updateRepresentativeGenome()

    def removeStagnatedSpecies(self) -> None:
        non_stagnated = []
        for species in self.species:
            if not species.isStagnated():
                non_stagnated.append(species)

        self.species = non_stagnated

    def getRepresentativeGenomes(self) -> List[nx.DiGraph]:
        return [s.getRepresentativeGenome() for s in self.species]

    def updateSpecies(self) -> List[Species]:
        for genome in self.new_genomes:
            inserted = False

            for species in self.species:
                rep = species.getRepresentativeGenome()
                if delta(rep, genome, self.config) <= self.config[DELTA_T]:
                    species.addGenome(genome)
                    inserted = True
                    break

            if not inserted:
                self.species.append(Species(genome, self.config))

        for species in self.species:
            species.logGeneration()

        return self.species

    def normalizeFitnesses(self) -> List[float]:
        fitnesses = []
        for species in self.species:
            fitnesses.append([g.graph["fitness"] / len(species) for g in species.genomes])
        
        return fitnesses

    def computeSpawnAmounts(self, norm_fitnesses: List[List[float]]) -> List[int]:
        ids = np.arange(len(norm_fitnesses))
        summed_norm_fitnesses = np.array([np.sum(fs) for fs in norm_fitnesses])
        spawns = np.random.choice(ids, size=self.config[POPULATION_SIZE], p=summed_norm_fitnesses / summed_norm_fitnesses.sum())
        c = Counter(spawns)
        return [c[i] for i in ids]

    def getNextPopulation(self) -> List[nx.DiGraph]:
        self.removeStagnatedSpecies()

        if len(self.species) == 0:
            self.initializePopulation()
            return self.new_genomes

        norms = self.normalizeFitnesses()
        assert len(norms) == len(self.species), f"{len(norms)} != {len(self.species)}"
        assert [len(n) == len(s) for n, s in zip(norms, self.species)]

        spawn_amounts = self.computeSpawnAmounts(norms)
        assert len(spawn_amounts) == len(self.species)

        for species, sa in zip(self.species, spawn_amounts):
            species.pruneWorstGenomes()

            # Copy over champion
            if sa > 0:
                self.new_genomes.append(species.getRepresentativeGenome())

            for _ in range(sa - 1):
                g = species.getRandomGenome()

                # Crossover w/ some probability
                if unif() <= self.config[P_CROSSOVER]:
                    g = Genome.crossover(g, species.getRandomGenome())

                edges = list(g.edges(data=True))

                # Perturb each weight w/ some probability
                for edge in edges:
                    if unif() <= self.config[P_WEIGHT]:
                        # Perturb the weight by the specified noise
                        edge[2]["weight"] += np.random.normal(
                            self.config[WEIGHT_MU], self.config[WEIGHT_SIGMA]
                        )

                        # Clip the weight to ensure its not too big or too small
                        edge[2]["weight"] = max(
                            min(edge[2]["weight"], self.config[WEIGHT_MAX]),
                            self.config[WEIGHT_MIN],
                        )
                    

                # Add a new node w/ some probability
                if unif() <= self.config[P_NEW_NODE]:
                    g = Genome.newNode(g)

                # Add a new connection w/ some probability
                if unif() <= self.config[P_NEW_CONNECTION]:
                    g = Genome.newConnection(g)

                # Append the genome to the list of new genomes
                self.new_genomes.append(g)

            # Update the representative genomes, as the old may have been removed
            species.updateRepresentativeGenome()

            n = species.getRepresentativeGenome()
            # print(n)
            # print(', '.join([f"{u} -> {v}" for u, v in n.edges()]))

        return self.new_genomes

    def initializePopulation(self) -> None:
        self.new_genomes = []
        for _ in range(self.config[POPULATION_SIZE]):
            self.new_genomes.append(self.genome.newNet())

        return