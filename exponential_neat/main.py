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
import matplotlib.pyplot as plt

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

            if abs(fitnesses[-1]) < 1e-6:
                print("Solution found")
                print(preds)
                print(net.edges(data=True))
                print(f"Gen: {i}, Fitness: {fitnesses[-1]}")
                draw_net("xor", net, fitnesses[-1])
                return

        # Update the species
        population.updateSpecies()

        # Get Next Population
        population.getNextPopulation()
    
    print("No solution found")
        # pr.print_stats(SortKey.CUMULATIVE)

# From https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
def draw_net(problem: str, net: nx.DiGraph, fitness: float):
    for layer, nodes in enumerate(nx.topological_generations(net)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            net.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(net, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(net, pos=pos, ax=ax)

    for attr in ["weight", "disabled"]:
        edge_labels = nx.get_edge_attributes(net, attr)

        if attr == "weight":
            for label in edge_labels:
                edge_labels[label] = f"{edge_labels[label]:.2f}"

        nx.draw_networkx_edge_labels(net, pos, edge_labels)

    ax.set_title(f"Problem: {problem}, Fitness: {fitness}")
    fig.tight_layout()
    
    # Save file
    outfile = os.path.join(os.path.dirname(__file__), "out", "xor.pdf")
    plt.savefig(outfile, bbox_inches='tight')

if __name__ == "__main__":
    main()
