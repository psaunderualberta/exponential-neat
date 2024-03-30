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
)
from genetic_algorithm.genome import Genome
from genetic_algorithm.species import Species
from util.comparisons import delta
from util.util import unif


class Population:
    def __init__(self, new_genomes: List[nx.DiGraph] = [], config: dict = {}):
        self.species: List[Species] = []
        self.new_genomes: List[nx.DiGraph] = new_genomes
        self.config = config

    def updateRepresentativeGenomes(self):
        for species in self.species:
            species.updateRepresentativeGenome()

    def removeStagnatedSpecies(self) -> None:
        non_stagnated = []
        for species in self.species:
            if not species.isStagnated:
                non_stagnated.append(species)

        self.species = non_stagnated

    def getRepresentativeGenomes(self) -> List[nx.DiGraph]:
        return [s.getRepresentativeGenome() for s in self.species]

    def updateSpecies(self) -> List[Species]:
        for genome in self.new_genomes:
            inserted = False

            for species in self.species:
                rep = species.getRepresentativeGenome()
                if delta(rep, genome) <= self.config[DELTA_T]:
                    species.addGenome(genome)
                    inserted = True
                    break

            if not inserted:
                self.species.append(Species(genome))

        # Reset the new genomes
        return self.species

    def normalizeFitnesses(self) -> List[float]:
        # TODO
        return [0.0]

    def computeSpawnAmounts(self) -> List[int]:
        # TODO
        return [0]

    def getNextPopulation(self) -> List[nx.DiGraph]:
        self.removeStagnatedSpecies()
        self.normalizeFitnesses()
        spawn_amounts = self.computeSpawnAmounts()

        assert len(spawn_amounts) == len(self.species)

        for species, sa in zip(self.species, spawn_amounts):
            species.pruneWorstGenomes()
            for _ in range(sa):
                g = species.getRandomGenome()

                # Crossover w/ some probability
                if unif() <= self.config[P_CROSSOVER]:
                    g = Genome.crossover(g, species.getRandomGenome())

                edges = list(g.edges())

                # Perturb each weight w/ some probability
                for edge in edges:
                    if unif() <= self.config[P_WEIGHT]:
                        # Perturb the weight by the specified noise
                        edge["weight"] += np.random.normal(
                            self.config[WEIGHT_MU], self.config[WEIGHT_SIGMA]
                        )

                        # Clip the weight to ensure its not too big or too small
                        edge["weight"] = max(
                            min(edge["weight"], self.config[WEIGHT_MAX]),
                            self.config[WEIGHT_MIN],
                        )

                # Add a new node w/ some probability
                if unif() <= self.config[P_NEW_NODE]:
                    Genome.newNode(g)

                # Add a new connection w/ some probability
                if unif() <= self.config[P_NEW_CONNECTION]:
                    g = Genome.newConnection(g)

                # Append the genome to the list of new genomes
                self.new_genomes.append(g)

            # Update the representative genomes, as the old may have been removed
            species.updateRepresentativeGenome()

        return self.new_genomes
