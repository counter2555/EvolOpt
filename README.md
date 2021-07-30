# EvolOpt
EvolOpt is a simple to use, multiprocessing, evolutionary optimizer for python.

## Usage

At first you have to import the "EvolutionaryOptimizer" class.

```python
from EvolOpt import EvolutionaryOptimizer
```

To initialize the optimizer an instance of the class has to be created.

```python
opt = EvolutionaryOptimizer(100, FitnessFunction, n_processes=4)
```

Thereby the first parameter (here 100) stands for the number of degrees of freedoms of the problem, also known as the number of free parameters. The FitnessFunction is the function, which evaluates your problem and returns a score to be minimized. Within this function you can for example solve an equation, or any other problem that can be reduced to a zero dimensional metric. n_processes defines the number of processes that are being used on your machine during the optimization runs.

Finally you have to start the optimizer by calling the breed_loop method.

```python
fitness_values, best_specimen, best_population = opt.breed_loop(4000, 100, verbose=True, stop_below=0.1, randomness=0.05, best_percentage=0.02)
```

Thereby the parameters are:

* N - the number of specimens in each generation
* epochs - the number of generations that are generated
* best_percentage - the quantile that is used for reproduction (e.g. 0.05 for the best 5%)
* randomness - a measure for how random the mutations are
* verbose - defines whether a line is printed for each generation
* stop_below - a cutoff at which metric value the optimizer stops
* mutation_probability - the probability of mutation
