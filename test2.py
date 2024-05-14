import math
import numpy
from deap import base, creator, tools, algorithms

import random

#Fitness Function
def ackley(individual):
    global evaluation_count
    evaluation_count += 1
    n = len(individual)
    sum1 = sum( x**2 for x in individual)
    sum2 = sum(math.cos(math.pi * x * 2) for x in individual)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e,

#Step 2: Setup toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('attr_float', random.uniform, -30, 30)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", ackley)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=9)

def run_ga():
    POPULATION_SIZE = 50
    N_GENERATIONS = 500
    
    cxpb = 0.7
    mutpb = 0.3
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POPULATION_SIZE, lambda_=399, 
                              cxpb=cxpb, mutpb=mutpb, stats=stats, 
                              ngen=N_GENERATIONS, halloffame=hof, 
                              verbose=True)
    
    return hof[0]

for run in range(10):
    evaluation_count = 0
    best_individual = run_ga()
    print(f"\nRun {run+1}")
    print(f"Total evaluations in run {run+1}: {evaluation_count}")
    print(f"Best Individual for Run {run+1}:")
    print(f"Fitness: {best_individual.fitness.values[0]}")
    print(f"Values: {best_individual}")  
