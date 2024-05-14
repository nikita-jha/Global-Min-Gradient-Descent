import math
import numpy


def objective_function(individual):
    global evaluation_count
    evaluation_count += 1
    x, y = individual
    return (abs(x) + abs(y)) * (1 + abs(math.sin(3 * abs(x) * math.pi)) + abs(math.sin(3 * abs(y) * math.pi))),

from deap import base, creator, tools, algorithms
import random

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -60, 40) #for x
toolbox.register("attr_float", random.uniform, -30, 70) #for y
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=9)
toolbox.register("evaluate", objective_function)


def run_ga():
    POPULATION_SIZE = 50  # Reduced population size
    N_GENERATIONS = 20     # Increased number of generations
    
    cxpb = 0.7  # Higher crossover probability
    mutpb = 0.3  # Lower mutation probability
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    for gen in range(N_GENERATIONS):
        offspring = algorithms.varOr(pop, toolbox, lambda_=100, cxpb=cxpb, mutpb=mutpb) 
        fits = toolbox.map(toolbox.evaluate, offspring)
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        hof.update(offspring)
        pop[:] = tools.selBest(offspring, len(pop))
        
        print(f"Gen: {gen} Min {hof.items[0].fitness.values[0]}")
        
        cxpb *= 0.99  
        mutpb *= 0.99  
    
    return stats, hof[0].fitness.values, hof[0]


for run in range(10):
    evaluation_count = 0
    print(f"\nRun {run+1}")
    logbook, fitness, individual = run_ga()
    print(f"Total evaluations in run {run+1}: {evaluation_count}")
    
    print(f"Best Individual for Run {run+1}:")
    print(f"Fitness: {fitness[0]}")
    print(f"Values: {individual}")  















