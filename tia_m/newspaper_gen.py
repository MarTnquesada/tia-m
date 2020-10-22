import argparse
import random
from deap import base, creator, tools, algorithms
import numpy
import matplotlib.pyplot as plt
import networkx

"""
NEWS READERS MATRIX
- each row corresponds with a reader (row 0 with reader 1 and so on)
- each reader has the time cost of reading the newspapered that corresponds with the position 
(pos 0 represents newspaper 1 and so on)
"""
NEWS_READERS = [[10, 15, 25, 12, 22, 66, 25, 35, 50, 12],
                [18, 20, 15, 30, 15, 10, 14, 50, 30, 15],
                [20, 30, 10, 30, 55, 12, 40, 15, 12, 18],
                [45, 45, 44, 15, 18, 15, 25, 10, 25, 20],
                [15, 12, 15, 22, 20, 20, 28, 18, 22, 20],
                [20, 10, 8, 11, 20, 12, 22, 20, 14, 45],
                [12, 12, 6, 12, 10, 18, 40, 20, 12, 70],
                [15, 18, 22, 55, 10, 20, 20, 22, 30, 22],
                [70, 30, 12, 18, 12, 20, 21, 12, 40, 30],
                [15, 15, 40, 20, 15, 15, 30, 6, 20, 15]]


def generate_individual_from_ro(relative_order):
    """
    Each individual is represented by a list of readers, in which the position of each reader relates to its identity.
    Every reader is represented by a sequence of newspapers, where the position of each one corresponds with its
    identity, and their values tell which is the starting time of reading for that newspaper and reader
    :param relative_order:
    :return:
    """
    shape = relative_order.shape
    individual = numpy.full((shape[0], shape[1]), numpy.nan)
    # create final timeline, randomizing initial priority
    for order in range(0, shape[1]):
        for consumer in numpy.random.permutation([c for c in range(0, shape[0])]):
            resource = relative_order[consumer][order]
            previous_resource = None
            if order > 0:
                previous_resource = relative_order[consumer][order - 1]

            projected_start_time = 0 if previous_resource is None else individual[consumer][previous_resource] + NEWS_READERS[consumer][previous_resource]
            projected_end_time = projected_start_time + NEWS_READERS[consumer][resource]
            while True:
                search_conflict = False
                for c2 in range(0, shape[0]):
                    if not numpy.isnan(individual[c2][resource]):
                        if individual[c2][resource] <= projected_start_time < individual[c2][resource] + NEWS_READERS[c2][resource] or \
                                individual[c2][resource] < projected_end_time <= individual[c2][resource] + NEWS_READERS[c2][resource] or \
                                individual[c2][resource] >= projected_start_time and projected_end_time >= individual[c2][resource] + NEWS_READERS[c2][resource]:
                            projected_start_time = individual[c2][resource] + NEWS_READERS[c2][resource]
                            search_conflict = True
                            break
                if not search_conflict:
                    break

            individual[consumer][resource] = projected_start_time

    return individual


def get_individual_ro(individual):
    relative_order = numpy.full((individual.shape[0], individual.shape[1]), -1, dtype=int)
    for consumer in range(0, individual.shape[0]):
        sorted_consumer = sorted(individual[consumer])
        relative_order[consumer] = [sorted_consumer.index(individual[consumer][i]) for i in range(0, individual.shape[1])]
    return relative_order


def init_individual(icls, shape):
    """
    Each individual is represented by a list of readers, in which the position of each reader relates to its identity.
    Every reader is represented by a sequence of newspapers, where the position of each one corresponds with its
    identity, and their values tell which is the starting time of reading for that newspaper and reader
    :param icls:
    :param shape:
    :return:
    """
    relative_order = numpy.array([numpy.random.permutation(range(0, shape[1])) for i in range(0, shape[0])])
    individual = generate_individual_from_ro(relative_order)
    return icls(individual)


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def mutate(individual):
    return individual


def evaluate(individual):
    max_time = -1
    for consumer in range(0, individual.shape[0]):
        for resource in range(0, individual.shape[1]):
            if individual[consumer, resource] + NEWS_READERS[consumer][resource] > max_time:
                max_time = individual[consumer, resource] + NEWS_READERS[consumer][resource]
    return max_time,


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population_size', default=50)
    parser.add_argument('--hof_size', default=1)
    parser.add_argument('--ngen', default=40)
    parser.add_argument('--mutpb', default=0.2)
    parser.add_argument('--cxpb', default=0.5)
    args = parser.parse_args()

    # create types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

    # initialize bag population containing random (valid) individuals
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, shape=(10, 10))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # initialize operators and evaluation function
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # create history object, which is used to get nice prints of the overall evolution process
    history = tools.History()
    # decorate the variation operator so that they can be used to retrieve a history
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # create an initial population
    pop = toolbox.population(n=args.population_size)
    history.update(pop)
    p = pop[0]
    s = get_individual_ro(pop[0])
    # create a list of statistics to be retrieved during the evolution process (they are shown in the logbook)
    stats = tools.Statistics()
    stats.register('min', min)
    # create a hall of fame, which contains the best individual/s that ever lived in the population during the evolution
    hof = tools.HallOfFame(maxsize=args.hof_size, similar=numpy.array_equal)

    # simplest evolutionary algorithm as presented in chapter 7 of Back, Fogel and Michalewicz, “Evolutionary Computation 1 : Basic Algorithms and Operators”, 2000.
    final_population, logbook = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=args.cxpb, mutpb=args.mutpb,
                                                    ngen=args.ngen, stats=stats, halloffame=hof, verbose=True)

    # output results of the evolutionary algorithm
    print('*' * 100)
    print('FINAL POPULATION\n')
    print(final_population)
    print('*' * 100)
    print('HALL OF FAME\n')
    print(hof)
    print('*' * 100)
    print('BEST INDIVIDUAL')
    print(hof[0])
    print('\nEVALUATION')
    print(evaluate(hof[0]))


if __name__ == '__main__':
    main()