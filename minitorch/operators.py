"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, List

#
# Implementation of a prelude of elementary functions.


# - mul
def mul(a: float, b: float) -> float:
    return a * b


# - id
def id(x: float) -> float:
    return x


# - add
def add(a: float, b: float) -> float:
    return a + b


# - neg
def neg(a: float) -> float:
    return -1.0 * a


# - lt
def lt(a: float, b: float) -> bool:
    return float(a < b)


# - eq
def eq(a: float, b: float) -> bool:
    return float(a == b)


# - max
def max(a: float, b: float) -> float:
    return a if a > b else b


# - is_close
def is_close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    return max(0.0, x)


# - log
def log(x: float) -> float:
    return math.log(x)


# - exp
def exp(x: float) -> float:
    return math.exp(x)


# - log_back
def log_back(x: float, d: float) -> float:
    return d / x


# - inv
def inv(x: float) -> float:
    return 1.0 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


# - relu_back
def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(lst: List[float], fn: Callable[[float], float]) -> List[float]:
    return [fn(x) for x in lst]


# - zipWith
def zipWith(lst1: List[float], lst2: List[float], fn: Callable[[float, float], float]) -> List[float]:
    return [fn(a, b) for a, b in zip(lst1, lst2)]


# - reduce
def reduce(lst: List[float], fn: Callable[[float, float], float], init: float) -> float:
    result = init
    for x in lst:
        result = fn(result, x)
    return result


#
# Use these to implement
# - negList: negate a list
def negList(lst: List[float]) -> List[float]:
    return map(lst, lambda x: -x)


# - addLists: add two lists together
def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    return zipWith(lst1, lst2, lambda x, y: x + y)


# - sum: sum lists
def sum(lst: List[float]) -> float:
    return reduce(lst, lambda x, y: x + y, 0.0)


# - prod: take the product of lists
def prod(lst: List[float]) -> float:
    return reduce(lst, lambda x, y: x * y, 1.0)
