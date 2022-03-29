
from numpy import mean
from skopt.space import Integer, Real
from skopt import gp_minimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import rand
from benderopt import minimize
import main



def bayesian_optimizer(func, search_space, n_iter):
    search_space = np.array(search_space)
    n_params = search_space.shape[0]
    search_area = []
    for i in range(n_params):
        search_area.append(Real(search_space[i][0], search_space[i][1]))
    result = gp_minimize(func, search_area, n_calls=n_iter, n_initial_points=30)
    return result


def evolution_strategy(func, search_space, n_iter, step_size = 0.1, mu = 5, lam = 20):
    search_space = np.array(search_space)
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
	# initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not is_in_bounds(candidate, search_space):
            candidate = search_space[:, 0] + rand(len(search_space)) * (search_space[:, 1] - search_space[:, 0])
        population.append(candidate)
	# perform the search
    for epoch in range(n_iter):
        scores = [func(c) for c in population]
        ranks = np.argsort(np.argsort(scores))
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
        children = list()
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            children.append(population[i])
            for _ in range(n_children):
                child = None
                while child is None or not is_in_bounds(child, search_space):
                        child = population[i] + randn(len(search_space)) * step_size
                children.append(child)
		# replace population with children
        population = children
    return [best, best_eval]
    
def parzen_estimator(func, search_space, n_iter):
    search_space = np.array(search_space)
    params = []
    for i, area in enumerate(search_space):
        params.append({"name": "p" + str(i+1),
                       "category": "uniform",
                       "search_space":{"low": float(area[0]), "high": float(area[1])}})
    best_params = minimize(func, params, number_of_evaluation=50)
    return best_params


def is_in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True




