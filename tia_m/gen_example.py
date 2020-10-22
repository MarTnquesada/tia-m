import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import networkx

IND_SIZE = 10  # size of the individual, in this case it is a list so this will be the length of the list
POB_SIZE = 50  # population size
HOF_SIZE = 10 # hall of fame size
CXPB, MUTPB, NGEN = 0.5, 0.2, 40 #


def evaluate(individual):
    return sum(individual),


def main():
    # create types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # initialize population containing random individuals
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # initialize operators and evaluation function
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # create history object, which is used to get nice prints of the overall evolution process
    history = tools.History()
    # decorate the variation operator so that they can be used to retrieve a history
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # create an initial population
    pop = toolbox.population(n=POB_SIZE)
    history.update(pop)
    # create a list of statistics to be retrieved during the evolution process (they are shown in the logbook)
    stats = tools.Statistics()
    stats.register('min', min)
    # create a hall of fame, which contains the best individual/s that ever lived in the population during the evolution
    hof = tools.HallOfFame(maxsize=HOF_SIZE)

    # simplest evolutionary algorithm as presented in chapter 7 of Back, Fogel and Michalewicz, “Evolutionary Computation 1 : Basic Algorithms and Operators”, 2000.
    final_population, logbook = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                                    stats=stats, halloffame=hof, verbose=True)
    # output results of the evolutionary algorithm
    print('*'*100)
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

    # draw a digraph representing the population evolution during that has taken place
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the graph top-down
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.show()


if __name__ == '__main__':
    main()
