"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import neat
import os
import numpy as np

# 2-input XOR inputs and expected outputs.
__dirname = os.path.dirname(__file__)
__data = np.genfromtxt(os.path.join(__dirname, 'iris.csv'), delimiter=',', dtype=str)
__targets = __data[:, -1].reshape(-1, 1)
__categories = np.unique(__targets).reshape(1, -1)
 
__outputs = (__categories == __targets).astype(np.int32)
__inputs = __data[1:, :-1].astype(np.float64)

__MAX_OUTPUT = 3.0
__MIN_OUTPUT = 0.0

IRIS_SENSIIVITY = (__MAX_OUTPUT - __MIN_OUTPUT) ** 2


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = __inputs.shape[0]
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(__inputs, __outputs):
            output = net.activate(xi)
            
            f = np.sum((output[0] - xo[0]) ** 2)
            f = min(max(f, __MIN_OUTPUT), __MAX_OUTPUT)
            genome.fitness -= f

