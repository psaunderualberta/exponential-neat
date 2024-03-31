import pandas as pd
import networkx as nx
import numpy as np
import os
from genetic_algorithm.genome import Genome
from genetic_algorithm.population import Population
import json
from util.constants import FITNESS_FUNCS
import cProfile
from pstats import SortKey
from tqdm import tqdm

def main():
    # Read in config
    configfile = os.path.join(os.path.dirname(__file__), "config", "config.json")
    with open(configfile, "r") as f:
        config = json.load(f)

    # Read in data
    datafile = os.path.join(os.path.dirname(__file__), "data", "xor.csv")
    df = pd.read_csv(datafile, header=None)
    data = df.to_numpy().astype(np.float32)
    X = data[:, :-1]
    y = data[:, -1]

    # Initialize population
    num_features = X.shape[1]
    population = Population(num_features, config)
    population.initializePopulation()
    
    evaluator = FITNESS_FUNCS[config["fitness_func"]]

    # For each iteration
    # with cProfile.Profile() as pr:
    for i in tqdm(range(config["generations"])):
        fitnesses = []

        # Evaluate the population
        for net in population.new_genomes:
            preds = population.genome.predict(net, X)
            fitnesses.append(evaluator(y, preds))
            net.graph["fitness"] = fitnesses[-1]

            if abs(fitnesses[-1]) <= 1e-5:
                print("Solution found")
                print(f"Gen: {i}, Fitness: {fitnesses[-1]}")
                print(net)
                print(net.nodes(data=True))
                print(net.edges(data=True))
                return

        # Update the species
        population.updateSpecies()

        # Get Next Population
        population.getNextPopulation()
    
    print("No solution found")
        # pr.print_stats(SortKey.CUMULATIVE)


if __name__ == "__main__":
    main()
