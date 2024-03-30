from util.util import MSE

OUTPUT_NODE_NAME = "output"

# Parameter names
DELTA_T = "delta_t"
P_CROSSOVER = "p_crossover"
P_WEIGHT = "p_weight"
P_NEW_NODE = "p_new-node"
P_NEW_CONNECTION = "p_new-connection"
WEIGHT_MU = "weight-mu"
WEIGHT_SIGMA = "weight-sigma"
WEIGHT_MIN = "weight-min"
WEIGHT_MAX = "weight_max"

FITNESS_FUNCS = {
    "MSE": MSE,
}