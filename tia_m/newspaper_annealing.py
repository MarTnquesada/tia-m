import argparse
import numpy
import matplotlib.pyplot as plt
from tia_m.newspaper_gen import generate_individual_from_ro, evaluate


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


def random_start(shape=(10, 10)):
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


def plot_annealing(states, costs):
    plt.figure()
    plt.suptitle("Evolution of states and costs of the simulated annealing")
    plt.subplot(121)
    plt.plot(states, 'r')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'b')
    plt.title("Costs")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population_size', default=50)
    parser.add_argument('--hof_size', default=5)
    parser.add_argument('--ngen', default=40)
    parser.add_argument('--mutpb', default=0.2)
    parser.add_argument('--cxpb', default=0.5)
    args = parser.parse_args()


    def annealing(random_start, cost_function, random_neighbour, acceptance, temperature, maxsteps=1000, debug=True):
        """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
        state = random_start()
        cost = cost_function(state)
        states, costs = [state], [cost]
        for step in range(maxsteps):
            fraction = step / float(maxsteps)
            T = temperature(fraction)
            new_state = random_neighbour(state, fraction)
            new_cost = cost_function(new_state)
            if debug: print(
                "Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(
                    step, maxsteps, T, state, cost, new_state, new_cost))
            if get_acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
                # print("  ==> Accept it!")
            # else:
            #    print("  ==> Reject it...")
        return state, cost_function(state), states, costs


if __name__ == '__main__':
    main()