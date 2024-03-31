from util.util import MSE

OUTPUT_NODE_NAME = "output"

POPULATION_SIZE = "population-size"

# Parameter names
DELTA_T = "delta_t"
P_CROSSOVER = "p_crossover"
P_WEIGHT = "p_weight"
P_NEW_NODE = "p_new-node"
P_NEW_CONNECTION = "p_new-connection"
WEIGHT_MU = "weight-mu"
WEIGHT_SIGMA = "weight-sigma"
WEIGHT_MIN = "weight-min"
WEIGHT_MAX = "weight-max"

FITNESS_FUNCS = {
    "MSE": MSE,
}

MIN_SPECIES_SIZE = "min-species-size"
SURVIVAL_THRESHOLD = "survival-threshold"
ELITISM = "elitism"
