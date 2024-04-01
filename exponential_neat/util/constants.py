from util.util import MSE

OUTPUT_NODE_NAME = "output"

POPULATION_SIZE = "population-size"

# Parameter names
DELTA_T = "delta_t"
P_CROSSOVER = "p_crossover"
P_WEIGHT = "p_weight"
P_NEW_NODE = "p_new-node"
P_NEW_CONNECTION = "p_new-connection"
WEIGHT_LOWER = "weight-lower"
WEIGHT_UPPER = "weight-upper"
WEIGHT_MIN = "weight-min"
WEIGHT_MAX = "weight-max"
NET_OUTPUT_MIN = "net-output-min"
NET_OUTPUT_MAX = "net-output-max"

FITNESS_FUNCS = {
    "MSE": MSE,
}

MIN_SPECIES_SIZE = "min-species-size"
SURVIVAL_THRESHOLD = "survival-threshold"
ELITISM = "elitism"
