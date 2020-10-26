import argparse
import numpy
import random
import matplotlib.pyplot as plt
from tia_m.newspaper_gen import generate_individual_from_ro


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


def random_state(shape=(10, 10)):
    relative_order = numpy.array([numpy.random.permutation(range(0, shape[1])) for i in range(0, shape[0])])
    state = generate_individual_from_ro(relative_order)
    return state


def cost_function(state):
    max_time = -1
    for consumer in range(0, state.shape[0]):
        for resource in range(0, state.shape[1]):
            if state[consumer][resource] + NEWS_READERS[consumer][resource] > max_time:
                max_time = state[consumer, resource] + NEWS_READERS[consumer][resource]
    return max_time


def get_random_neighbour(state):
    consumer = random.randint(0, len(state) - 1)
    size = len(state[consumer])
    randpos1 = random.randint(0, size - 1)
    randpos2 = random.randint(0, size - 1)
    if size > 1:
        while randpos1 == randpos2:
            randpos2 = random.randint(0, size - 1)
    new_state = state.copy()
    new_state[consumer][randpos1], new_state[consumer][randpos2] = \
        new_state[consumer][randpos2].copy(), new_state[consumer][randpos1].copy()
    return new_state


def annealing(initial_t, annealing_type, t_annealing_constant, maxsteps=1000):
    t = initial_t
    state = random_state()
    best_state = state
    cost = cost_function(state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        if annealing_type == 'lineal':
            t = initial_t - step * t_annealing_constant
        elif annealing_type == 'lineal':
            t = t * t_annealing_constant
        elif annealing_type == 'exponential':
            t = t / (1 + t * t_annealing_constant)
        else:
            print('Error, no correct annealing type selected')
            return None
        new_state = get_random_neighbour(state)
        new_cost = cost_function(new_state)
        cost_delta = cost - new_cost
        if cost_delta > 0:
            if cost_function(best_state) > cost_function(new_state):
                best_state = new_state
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
        else:
            if numpy.exp(- (new_cost - cost) / t) > numpy.random.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
    return best_state, cost_function(best_state), states, costs


def plot_annealing(costs):
    plt.figure()
    plt.suptitle("Evolution of costs of the simulated annealing")
    plt.subplot(122)
    plt.plot(costs, 'a')
    plt.title("Costs")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_t', default=50)
    parser.add_argument('--annealing_type', default='lineal', choices=['lineal', 'multiplicative', 'exponential'])
    parser.add_argument('--t_annealing_constant', default=0.05, help='Range: [0, 1]')
    parser.add_argument('--maxsteps', default=1000)
    args = parser.parse_args()

    best_state, best_state_cost, states, costs = annealing(args.initial_t, args.annealing_type,
                                                           args.t_annealing_constant, args.maxsteps)
    plot_annealing(costs)


if __name__ == '__main__':
    main()