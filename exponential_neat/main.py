import pandas as pd
import networkx as nx
import numpy as np
import os
from genetic_algorithm.genome import Genome
from genetic_algorithm.population import Population
import json
from util.constants import FITNESS_FUNCS


def main():
    # Read in config
    configfile = os.path.join(os.path.dirname(__file__), "config", "config.json")
    with open(configfile, "r") as f:
        config = json.load(f)

    # Read in data
    datafile = os.path.join(os.path.dirname(__file__), "data", "xor.csv")
    df = pd.read_csv(datafile, header=None)
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    # Initialize population
    num_features = X.shape[1]
    G = Genome(num_features)
    first_population = [G.newNet() for _ in range(config["population_size"])]
    population = Population(first_population, config)
    
    evaluator = FITNESS_FUNCS[config["fitness_func"]]

    # For each iteration
    for i in range(config["generations"]):
        fitnesses = []

        # Evaluate the population
        for net in population.new_genomes:
            preds = G.predict(net, X)
            fitnesses.append(evaluator(y, preds))
            net.graph["fitness"] = fitnesses[-1]

        # Update the species
        population.updateSpecies()

        # Get Next Population
        population.getNextPopulation()


if __name__ == "__main__":
    main()
