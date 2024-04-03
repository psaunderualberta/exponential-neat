from neat import StdOutReporter
import numpy as np

class DifferentialPrivacyDemoReporter(StdOutReporter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitnesses = []
        self.champions = []

    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)

        self.fitnesses.append([[gen_id, g.fitness] for gen_id, g in population.items()])
        self.champions.append(best_genome)

    def evaluate_dp(self, epsilon, sensitivity, num_samples = 1000):
        weights = np.array([])
        fitnesses = np.array([])
        samples = np.array([]) 
        for f in map(np.array, self.fitnesses):
            f = f[:, 1]  # Just get the fitness, don't currently care about the genome ID 
            fitnesses = np.append(fitnesses, f)
            new_weights = np.exp((epsilon * f) / (2 * sensitivity))
            weights = np.append(weights, new_weights)
        
            new_samples = np.random.choice(fitnesses, size=(1, num_samples), p=weights / np.sum(weights))
            if samples.shape[0] == 0:
                samples = new_samples
            else:
                samples = np.append(samples, new_samples, axis=0)
            
        return samples

    def get_champions(self):
        return self.champions
