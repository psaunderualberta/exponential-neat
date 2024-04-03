import neat
import os
from reporters.reporting import DifferentialPrivacyDemoReporter
from problems.xor.xor import eval_genomes, xor_inputs, xor_outputs, XOR_SENSIIVITY
import matplotlib.pyplot as plt
import numpy as np

# Load configuration.
def run(eps):
    configfile = os.path.join('.', 'problems', 'xor', 'config-feedforward')
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


    print('\nDifferential Privacy Results:')
    dp_results = reporter.evaluate_dp(eps, XOR_SENSIIVITY, 10_000)
    return np.histogram(dp_results[-1, :], range=(0, 4), bins=50, density=True)

def main():
    epsilons = np.arange(1, 50)
    hists = []
    bin_edges = []
    epsilons_records = []
    for eps in epsilons:
        h, e = run(eps)
        hists.append(h)
        bin_edges.append(e[:-1])
        epsilons_records.append([eps for _ in range(h.shape[0])])
    

    hists = np.array(hists)
    bin_edges = np.array(bin_edges)
    epsilons_records = np.array(epsilons_records)


    # Plot the surface
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(epsilons_records, bin_edges, hists, vmin=hists.min() * 2)

    ax.set(
        xlabel="Epsilons",
        ylabel="Sampled Network Performance",
        zlabel="Density",
        title="Density of repeated private sampling at different epsilon settings"
    )

    outfile = os.path.join(".", "outputs", "xor-density.png")
    plt.tight_layout()
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
