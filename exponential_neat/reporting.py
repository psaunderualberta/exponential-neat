from neat import StdOutReporter
import numpy as np

class DifferentialPrivacyDemoReporter(StdOutReporter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitnesses = []
        self.genomes = []

    def end_generation(self, config, population, species_set):
        super().end_generation(config, population, species_set)

        self.fitnesses.append([[gen_id, g.fitness] for gen_id, g in population.items()])
        self.genomes.append([[gen_id, g.fitness] for gen_id, g in population.items()])

    def evaluate_dp(self, epsilon, sensitivity, num_samples = 1000):
        weights = np.array([])
        fitnesses = []
        genomes = []
        samples = []
        for i, (f, g) in zip(self.fitnesses, self.genomes):
            new_weights = np.exp((epsilon * f[:, 1]) / (2 * sensitivity))
            weights = np.append(weights, new_weights)
            genomes.extend(g)
        
            cs = np.cumsum(weights)
            samples.append(np.random.choice(fitnesses, size=num_samples, p=cs / cs[-1]))
            
        return samples
