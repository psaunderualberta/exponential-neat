import neat
import numpy as np

CLASSIFICATION = "classification"
REGRESSION = "regression"

class Evaluator:
    def __init__(self, inputs, outputs, task):
        self.inputs = np.array(inputs).astype(np.float32)
        self.outputs = np.array(outputs)

        # Perform validation on input format
        assert self.inputs.shape[0] == self.outputs.shape[0], "Number of inputs != Number of outputs"

        # Perform task validation
        if task == CLASSIFICATION:
            categories = np.unique(self.outputs).reshape(1, -1)
            self.outputs = (categories == self.outputs).astype(np.int32)
            
            # Check if we can reduce to logistic regression
            if self.outputs.shape[1] == 2:
                self.outputs = np.argmax(self.outputs, axis=1).reshape(-1, 1)

            self.min_output = 0
            self.max_output = self.outputs.shape[1] 
        elif task == REGRESSION:
            ValueError("Regression is not yet supported")
        else:
            ValueError(f"Task of type '{task}' is not yet supported." \
                        + f"Please choose from '{[CLASSIFICATION, REGRESSION]}'")
    
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.inputs.shape[0] * self.outputs.shape[1]
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            t = []
            for xi, xo in zip(self.inputs, self.outputs):
                output = net.activate(xi)
                output = [o / max(sum(output), 1e-10) for o in output]

                f = np.sum(np.abs(output - xo))
                f = min(max(f, self.min_output), self.max_output)
                t.append(f)
                genome.fitness -= f
                
    def predict_genome(self, genome, config, inputs = None):
        if inputs is None:
            inputs = self.inputs

        outputs = []
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(self.inputs, self.outputs):
            output = net.activate(xi)

            if self.outputs.shape[1] > 2:
                output = [o / sum(output) for o in output]

            outputs.append(output)

        return outputs

    def get_sensitivity(self):
        return (self.max_output - self.min_output) ** 2 
