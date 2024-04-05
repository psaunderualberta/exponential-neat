"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import neat

# 2-input XOR inputs and expected outputs.

__MAX_OUTPUT = 1.0
__MIN_OUTPUT = 0.0

XOR_SENSIIVITY = (__MAX_OUTPUT - __MIN_OUTPUT) ** 2

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            
            f = (output[0] - xo[0]) ** 2
            f = min(max(f, __MIN_OUTPUT), __MAX_OUTPUT)
            genome.fitness -= f

