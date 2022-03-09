import random
import copy
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from cec2017.functions import f4


def evaluation_function(q: Callable, p: npt.ArrayLike) -> npt.ArrayLike:
    """
    Calculate funcion for each element in population.

    Args:
        q (Callable) : Funcion.
        p (array) : Population.

    Returns:
        (array) : Evaluation numpy array.
    """
    return np.array([q(each_p) for each_p in p])

    
def find_best(p: npt.ArrayLike, e: npt.ArrayLike) -> Tuple:
    """
    Find best object in population.

    Args:
        p (array) : Population array.
        e (array) : Evaluation arr funcionay.

    Returns:
        (tuple) : Best object with evaluation.
    """
    min_e = np.amin(e)
    index = list(e).index(min_e)

    return p[index], e[index]


def tournament_selection(p: npt.ArrayLike, e: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies tournament selection on population.

    Args:
        p (array) : Population array.
        e (array) : Evaluation array.
    
    Returns: 
        (array) : Population array after reproduction.
    """
    new_p = copy.deepcopy(p)
    zipped = list(zip(new_p, e))

    for i, each in enumerate(zipped):
        rival = random.choice(zipped)

        if each[1] <= rival[1]: new_p[i] = copy.deepcopy(each[0])
        else: new_p[i] = copy.deepcopy(rival[0])

    return new_p


def mutation(p: npt.ArrayLike, ms: float) -> npt.ArrayLike:
    """
    Applies mutation on populaion.

    Args:
        p (array) : Population array.
        ms (float) : Mutation strength.

    Returns 
        (array) : Population array after mutation
    """
    return np.array([p_ + ms * np.random.normal(0, 1, p.shape[1]) for p_ in p]) 


def elite_succession(p: npt.ArrayLike, new_p: npt.ArrayLike, k: int,
                     e: npt.ArrayLike, new_e: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies elite succession with k = 1.

    Args:
        p (array) : Old population.
        new_p (array) : New population.
        k (int) : Elite.
        e (array) : Evaluation of population.
        new_e (array) : Evaluation of new population.

    Returns:
        (array) : Population after succession.
    """
    zipped_old = list(zip(e, p))
    zipped_old.sort()
    zipped_new = list(zip(new_e, new_p))
    zipped_new.sort()

    for number in range(k):
        zipped_new[-(number+1)] = copy.deepcopy(zipped_old[number])

    return np.array([zipped_new[n][1] for n in range(len(zipped_new))])


def evolution_algorithm(q: Callable, p0: npt.ArrayLike, k: int,
                        ms: float, tmax: int) -> npt.ArrayLike:
    """
    First applies the given permutation, then splits x into partitions given
    the percentages.

    Args:
        q (Callable) : Function.
        p0 (array) : Starting population.
        k (int) : Elite size.
        ms (float) : Mutation strength.
        tmax (int) : Iterations.

    Returns:
        (array) : Best object with value
    """
    # starting parameters 
    curr_p = p0
    curr_evaluation = evaluation_function(q, curr_p)
    curr_best = find_best(curr_p, curr_evaluation)

    for _ in range(tmax):
        # make new population 
        new_p = tournament_selection(curr_p, curr_evaluation)
        new_p = mutation(new_p, ms)
        new_evaluation = evaluation_function(q, new_p)
        new_best = find_best(new_p, new_evaluation)

        # compare best objects
        if new_best[1] < curr_best[1]: curr_best = new_best

        # replace current population with new one
        curr_p = elite_succession(curr_p, new_p, k,
                                  curr_evaluation, new_evaluation)
        curr_evaluation = evaluation_function(q, curr_p)
        
    return curr_best


p = np.array([np.random.uniform(-100, 100, 2) for _ in range(50)])

print(evolution_algorithm(f4, p, 1, 0.55, 10000))
