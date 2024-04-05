import neat
import os
from reporters.reporting import DifferentialPrivacyDemoReporter
from problems.xor.xor import eval_genomes, xor_inputs, xor_outputs, XOR_SENSIIVITY
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import argparse

# Load configuration.
def run(problempath):
    configfile = os.path.join(problempath, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         configfile)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    reporter = DifferentialPrivacyDemoReporter(False)
    p.add_reporter(reporter)

    # Run until a solution is found.
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    return reporter.get_fitnesses()

def evaluate_dp(values, epsilon, sensitivity, num_samples = 100):
    fitnesses = np.array(values)
    weights = np.exp((epsilon * fitnesses) / (2 * sensitivity))
    return np.random.choice(fitnesses, size=(1, num_samples), p=weights / np.sum(weights))

def hist(values):
    return np.histogram(values, range=(0, 4), bins=100, density=True)

def main():
    dirname = os.path.dirname(__file__)
    problems = os.listdir(os.path.join(dirname, "problems"))
    problems = list(filter(lambda a: a not in [".", ".."], problems))

    # Setup & Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate DP performance on NEAT results.")
    parser.add_argument("--problem", required=True, choices=problems, help="The specified problem to solve.")
    args = parser.parse_args()

    problempath = os.path.join(dirname, "problems", args.problem)
    outputpath = os.path.join(problempath, "outputs")
    os.makedirs(outputpath, exist_ok=True)

    epsilons = np.arange(1, 50)
    num_synthesis_runs = 100 

    with mp.Pool(os.cpu_count()) as p:
        fitnesses = p.map(run, [problempath] * num_synthesis_runs) 

    print(sum(map(lambda arr: arr.size, fitnesses)))

    hists = []
    bin_edges = []
    epsilons_records = []
    for eps in epsilons:
        hs = []
        es = []
        for f in fitnesses:
            private_f = evaluate_dp(f, eps, XOR_SENSIIVITY)
            h, e = hist(private_f)
            hs.append(h)
            es.append(e)
        assert all(np.all(es[0] == e) for e in es) == 1, "Bin edges are not the same across iterations"

        # Average the histograms
        h = np.mean(hs, axis=0)
        e = es[0]

        # Append to the lists
        hists.append(h)
        bin_edges.append(e[:-1])
        epsilons_records.append([eps for _ in range(h.shape[0])])
    
    # Convert to numpy arrays
    hists = np.array(hists)
    bin_edges = np.array(bin_edges)
    epsilons_records = np.array(epsilons_records)

    # Plot the surface
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(epsilons_records, bin_edges, hists, vmin=hists.min() * 2)

    ax.set(
        xlabel=r"$\varepsilon$",
        ylabel="Network Performance Density",
        zlabel="Density",
        title="Density of repeated private sampling at different epsilon settings"
    )

    outfile = os.path.join(outputpath, "xor-density.pdf")
    plt.tight_layout()
    plt.savefig(outfile)


    # Plot the actual distribution of fitnesses
    hs = []
    es = []
    for f in fitnesses:
        h, e = hist(f)
        hs.append(h)
        es.append(e)
    assert all(np.all(es[0] == e) for e in es) == 1, "Bin edges are not the same across iterations"
    
    # Average the histograms
    h = np.mean(hs, axis=0)
    e = es[0]

    fig = plt.figure()
    plt.hist(e[:-1], e, weights=h)
    plt.title("True Fitness Density")
    plt.xlabel("Network Fitness")
    plt.ylabel("Density")

    plt.tight_layout()
    outfile = os.path.join(outputpath, "xor-true-density.pdf")
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
