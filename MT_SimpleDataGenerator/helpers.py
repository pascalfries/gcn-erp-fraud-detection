import random


def with_probability(probability: float) -> bool:
    if probability >= 1:
        return True
    elif probability <= 0:
        return False

    return random.randint(1, 100000) <= 100000 * probability


def rand_float(min: float = 0, max: float = 0) -> float:
    return random.randint(int(100000 * min), int(100000 * max)) / 100000.0
