import multiprocessing
from operator import mul
import numpy as np
from multiprocessing import Pool
from itertools import product
import random
import time

class EvolutionaryOptimizer:
    def __init__(self, NDEG, FitnessFunction, n_processes=4, bounds=None):
        self.fitFunc = FitnessFunction
        self.N = NDEG
        self.n_processes = n_processes

        self.scale_mins = np.zeros(NDEG)
        self.scale_maxs = np.ones(NDEG)

        if bounds is not None and len(bounds) == NDEG:
            for i, b in zip(range(len(bounds)), bounds):
                self.scale_mins[i] = b[0]
                self.scale_maxs[i] = b[1]
        
    #generates a uniformly random population
    def generateInitialPopulation(self, size):
        parameters = np.random.rand(size, self.N)

        return self.__scale_parameters__(parameters)

    def __scale_parameters__(self, parameters):
        parameters = parameters + self.scale_mins[np.newaxis, :]
        parameters = parameters * (self.scale_maxs[np.newaxis, :] - self.scale_mins[np.newaxis, :])
        return parameters

    def __clip__parameters(self, parameters):
        for i in range(parameters.shape[0]):
            if parameters[i] < self.scale_mins[i]:
                parameters[i] = self.scale_mins[i]
                continue
            elif parameters[i] > self.scale_maxs[i]:
                parameters[i] = self.scale_maxs[i]
                continue
        return parameters

    #returns fitness for a single specimen
    def getSingleFitness(self, specimen):
        return self.fitFunc(specimen)

    #return fitness for whole population
    def getPopulationFitness(self, population):
        with Pool(self.n_processes) as pool:
            pop_list = [p for p in population]

            scores = pool.map(self.fitFunc, pop_list)
            
        
        return np.array(scores)

    def breed(self, N, population, best_percentage = 0.05, randomness = 0.1, last_fitness = None,
             mutation_probability=0.5):
        if last_fitness is None:
            scores = self.getPopulationFitness(population)
        else:
            scores = last_fitness

        srt = np.argsort(scores)

        scores = scores[srt]
        population = population[srt]

        Nsel = int(np.ceil(len(scores)*best_percentage))

        sel_pop = population[0:Nsel]
        sel_scores = scores[0:Nsel]

        indices = np.arange(sel_pop.shape[0])

        new_pop = []

        for i in range(N):
            if random.random() < mutation_probability and randomness > 0:
                rvec = np.random.normal(0,randomness, size=self.N)#(np.random.random(self.N)-0.5)*2* randomness
            else:
                rvec = np.zeros(self.N)

            choice = np.random.choice(indices, 2)

            choice = sel_pop[[choice[0], choice[1]]]
            
            offspring = np.mean(choice, axis=0) + rvec

            new_pop.append(offspring)

        return np.array(new_pop)

    def breed_loop(self, N, epochs, best_percentage = 0.05, randomness = 0.1, verbose=False,
                   stop_below=None, mutation_probability=0.5):
        pop = self.generateInitialPopulation(N)
        
        best_score = 1e20
        best_specimen = None

        best_pop_score = 1e12
        best_pop = None

        fitness_values = []
        fitness_values.append(self.getPopulationFitness(pop))
        
        for i in range(epochs):
            t0 = time.time()
            pop = self.breed(N, pop, best_percentage, randomness, last_fitness=fitness_values[-1], mutation_probability=mutation_probability)
            fitness = self.getPopulationFitness(pop)

            fitness_values.append(fitness)

            best_fitness = np.min(fitness)


            if best_fitness < best_score:
                best_score = best_fitness
                best_specimen = pop[np.argmin(fitness)]

                if verbose:
                    print("New best found: "+str(best_score))

                if stop_below is not None and best_score < stop_below:
                    break

            if np.mean(fitness) < best_pop_score:
                best_pop_score = np.mean(fitness)
                best_pop = pop


            if verbose:
                print(str(i) + "\t" + str(np.mean(fitness)) + "\t" + str(np.round(time.time()-t0,6))+"s")

        return fitness_values, best_specimen, best_pop

