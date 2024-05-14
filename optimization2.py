import math
from deap import base, creator, tools, algorithms
import random
import numpy

#Fitness Function
def ackley(individual):
    global counter
    counter += 1
    n = len(individual)
    sum1 = sum(x**2 for x in individual)
    sum2 = sum(math.cos(2*math.pi*x) for x in individual)
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
    POPULATION_SIZE = 200
    N_GENERATIONS = 100
    
    cxpb = 0.7  # Initial crossover probability
    mutpb = 0.3  # Initial mutation probability
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    for gen in range(N_GENERATIONS):
        offspring = algorithms.varOr(pop, toolbox, lambda_=2000, cxpb=cxpb, mutpb=mutpb)
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
    counter = 0
    print(f"\nRun {run+1}")
    logbook, fitness, individual = run_ga()
    print(f"Total evaluations in run {run+1}: {counter}")
    
    print(f"Best Individual for Run {run+1}:")
    print(f"Fitness: {fitness[0]}")
    print(f"Values: {individual}")  # Since we have 30 dimensions, we'll just print the whole individual

    
