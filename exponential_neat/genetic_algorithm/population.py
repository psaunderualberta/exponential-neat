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
    WEIGHT_LOWER,
    WEIGHT_UPPER,
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
        for i, species in enumerate(self.species):
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
        fitnesses = [[g.graph["fitness"] for g in s.genomes] for s in self.species]
        min_fitness = np.min([min(fs) for fs in fitnesses])
        max_fitness = np.max([max(fs) for fs in fitnesses])

        fitness_range = max(1.0, max_fitness - min_fitness)

        norm_fitnesses = []
        for fs in fitnesses:
            norm_fitnesses.append(
                [(max_fitness - f) / (len(fs) * fitness_range) for f in fs]
            )

        return fitnesses

    def computeSpawnAmounts(self, norm_fitnesses: List[List[float]]) -> List[int]:
        ids = np.arange(len(norm_fitnesses))
        mean_norm_fitnesses = np.array([np.mean(fs) for fs in norm_fitnesses])

        spawns = np.random.choice(
            ids,
            size=self.config[POPULATION_SIZE],
            p=mean_norm_fitnesses / mean_norm_fitnesses.sum(),
        )
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

        # Update the representative genomes

        for species, sa in zip(self.species, spawn_amounts):
            species.updateRepresentativeGenome()
            species.pruneWorstGenomes()

            # Copy over champion
            if sa > 0:
                self.new_genomes.append(species.getChampion())
                sa -= 1

            for _ in range(sa):
                g = species.getRandomGenome()

                # Crossover w/ some probability
                if unif() <= self.config[P_CROSSOVER]:
                    g = self.genome.crossover(g, species.getRandomGenome())

                edges = list(g.edges(data=True))

                # Perturb each weight w/ some probability
                for edge in edges:
                    if unif() <= self.config[P_WEIGHT]:
                        # Perturb the weight by the specified noise
                        edge[2]["weight"] += unif(
                            self.config[WEIGHT_LOWER], self.config[WEIGHT_UPPER]
                        )

                        # Clip the weight to ensure its not too big or too small
                        edge[2]["weight"] = max(
                            min(edge[2]["weight"], self.config[WEIGHT_MAX]),
                            self.config[WEIGHT_MIN],
                        )

                # Add a new node w/ some probability
                if unif() <= self.config[P_NEW_NODE]:
                    g = self.genome.newNode(g)

                # Add a new connection w/ some probability
                if unif() <= self.config[P_NEW_CONNECTION]:
                    g = self.genome.newConnection(g)

                # Append the genome to the list of new genomes
                self.new_genomes.append(g)

        return self.new_genomes

    def initializePopulation(self) -> None:
        self.new_genomes = []
        for _ in range(self.config[POPULATION_SIZE]):
            self.new_genomes.append(self.genome.newNet())

        return
