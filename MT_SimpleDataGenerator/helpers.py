import tensorflow as tf
import numpy as np
import stellargraph.random as sgrand
import random
import os


def with_probability(probability: float) -> bool:
    if probability >= 1:
        return True
    elif probability <= 0:
        return False

    return random.randint(1, 100000) <= 100000 * probability


def rand_float(min: float = 0, max: float = 0) -> float:
    return random.randint(int(100000 * min), int(100000 * max)) / 100000.0


def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sgrand.set_seed(seed)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
