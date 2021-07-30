from EvolOpt import EvolutionaryOptimizer
import numpy as np

f1 = np.random.randint(0,2,20)


def ExampleFitness(parameters):
    x = np.linspace(-10,10,500)
    f2 = np.array(parameters)

    return np.mean(np.abs(f1 - f2))

if __name__ == "__main__":
    
    #initialize the optimizer
    opt = EvolutionaryOptimizer(100, ExampleFitness, n_processes=4)

    #run the optimizer
    fitness_values, best_specimen, best_population = opt.breed_loop(4000, 100, verbose=True, stop_below=0.1, randomness=0.05, best_percentage=0.02)

    #output
    print("Best specimen:", best_specimen)
