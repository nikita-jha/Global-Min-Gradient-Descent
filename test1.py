import math
import numpy
from deap import base, creator, tools, algorithms
import random
import numpy as np

def fitness_function(individual):
    global evaluation_count
    evaluation_count += 1
    x, y = individual
    return (abs(x) + abs(y)) * (1 + abs( math.sin (abs(x) * math.pi)) + abs (math.sin(abs(y) * math.pi))),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float_x", random.uniform, -60, 40)
toolbox.register("attr_float_y", random.uniform, -30, 70)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float_x, toolbox.attr_float_y), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)

def calculate_diversity(population):
    distances = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            distances.append(math.dist(population[i], population[j]))
    return sum(distances) / len(distances)

def consistency(best_fitnesses):
    return np.std(best_fitnesses) 

def run_ga():
    POPULATION_SIZE = 50
    N_GENERATIONS = 39
    
    cxpb = 0.7
    mutpb = 0.3
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POPULATION_SIZE, lambda_=50, 
                              cxpb=cxpb, mutpb=mutpb, stats=stats, 
                              ngen=N_GENERATIONS, halloffame=hof, 
                              verbose=True)
    
    diversity = calculate_diversity([ind for ind in pop])
    
    return hof[0], diversity

best_fitnesses = []
diversities = []

for run in range(10):
    evaluation_count = 0
    best_individual, diversity = run_ga()
    best_fitnesses.append(best_individual.fitness.values[0])
    diversities.append(diversity)
    print(f"\nRun {run+1}")
    print(f"Total evaluations in run {run+1}: {evaluation_count}")
    print(f"Best Individual for Run {run+1}:")
    print(f"Fitness: {best_individual.fitness.values[0]}")
    print(f"Values: {best_individual}")  


print(f"Average Best Fitness: {numpy.mean(best_fitnesses)}")
print(f"Average Diversity: {numpy.mean(diversities)}")
print(f"Consistency (Standard Deviation of Best Fitnesses): {np.std(best_fitnesses)}")
