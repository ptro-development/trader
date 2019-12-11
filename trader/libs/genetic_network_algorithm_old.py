import random
import pprint
import pickle
import math
import copy
import numpy as np
import neurolab as nl

from trader.libs.trade_sequence_tree_trails import TreeKeysTrailsCollection
from trader.libs.trade_sequence_tree import get_tree_layers_counts


def init_random_layer_bases_and_wigths(layer):
    layer.np['w'][:] = np.random.uniform(-0.5, 0.5, layer.np['w'].shape)
    layer.np['b'][:] = np.random.uniform(-0.5, 0.5, layer.np['b'].shape)


def fitnes_function(network, outputs, targets):
    error = targets - outputs
    return -0.5 * np.sum(np.square(error))


def get_random_member_index(fitness, fitness_sum, population_size):
    cumulative_sum = 0.0
    member_index = 0
    fitness_point = -random.random() * fitness_sum
    for index in range(population_size):
        cumulative_sum += fitness[index]
        if cumulative_sum > fitness_point:
            member_index = index
            break
    return member_index


def get_random_member_index_by_rank(fitness):
    indexes = []
    for i in range(0, len(fitness)):
        for j in range(0, i+1):
            indexes.append(i)
    index_finess = [(index, value) for index, value in enumerate(fitness)]
    index_finess.sort(key=lambda x: x[1])
    index = random.choice(indexes)
    return index_finess[index][0]


def xrandom_arryas_cross_over(first, second, prob=0.5):
    assert len(first) == len(second)
    for index in range(len(first)):
        if random.random() > prob:
            yield first[index]
        else:
            yield second[index]


def random_arryas_cross_over(first, second, prob=0.5):
    return list(xrandom_arryas_cross_over(first, second, prob))


def numpy_random_arryas_cross_over(first, second, prob=0.5):
    return np.array(random_arryas_cross_over(first, second, prob))


def random_members_cross_over(
        old_member_first, old_member_second,
        new_member_first, new_member_second):
    if random.random() < 0.5:
        print "copy"
        # only copy
        new_member_first = old_member_first
        new_member_second = old_member_second
    else:
        print "crossover"
        # cross over
        new_member_first = old_member_second  # noqa
        new_member_second = old_member_first  # noqa


def random_value_mutation(value, mutation_rate, min_value, max_value):
    if random.random() < mutation_rate:
        random_value = 2.0 * math.sqrt(-2 * math.log(random.random())) * \
            math.sin(2 * math.pi * random.random())
        value += random_value
        # limiting extremes
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value
    return value


def xrandom_values_mutation(values, mutation_rate, min_value, max_value):
    for value in values:
        yield random_value_mutation(value, mutation_rate, min_value, max_value)


def random_values_mutation(values, mutation_rate, min_value, max_value):
    return list(
        xrandom_values_mutation(values, mutation_rate, min_value, max_value))


def numpy_random_values_mutation(values, mutation_rate, min_value, max_value):
    return np.array(
        random_values_mutation(values, mutation_rate, min_value, max_value))


def xget_layers_indexes(layer_count, only_layer=None):
    if only_layer is None:
        start = 0
        end = layer_count
    else:
        if only_layer > (layer_count-1):
            raise ValueError("only_layer > layer_count - 1 is not true")
        start = only_layer
        end = only_layer + 1
    return xrange(start, end)


class NetworksPopulation(object):

    def __init__(
            self, size, network_layers_layout,
            input_values_ranges, min_max_base_value_limits,
            min_max_weight_value_limits, tree, trade_sequence,
            trade_data, samples):
        assert len(min_max_base_value_limits) == 2 and \
            isinstance(min_max_base_value_limits, list)
        assert len(min_max_weight_value_limits) == 2 and \
            isinstance(min_max_weight_value_limits, list)
        self.samples = samples
        self.min_base, self.max_base = min_max_base_value_limits
        self.min_weight, self.max_weight = min_max_weight_value_limits
        self.input_values_ranges = input_values_ranges
        self.network_layers_layout = network_layers_layout
        self.size = size
        self.tree = copy.deepcopy(tree)
        self.trade_sequence = trade_sequence
        self.trade_data = trade_data
        self.tree_depth = len(get_tree_layers_counts(self.tree))
        self._init_population()
        self.tree_trails = TreeKeysTrailsCollection(self.tree)
        # self._init_population_trails()

    def get_samples(self):
        return self.samples

    def trade(self):
        self._compute_probability_and_delete_key()
        self._init_population_trails()
        self._compute_amount_to_buy()
        self._normalize_amount_to_buy()

    def get_tree(self):
        return self.tree

    def get_tree_depth(self):
        return self.tree_depth

    def get_trade_sequence(self):
        return self.trade_sequence

    def get_trade_data(self):
        return self.trade_data

    def _init_population(self):
        self.population = [
            nl.net.newff(self.input_values_ranges, self.network_layers_layout)
            for i in range(self.size)]

    """
    def _init_population_trails(self):
        tree_trails = TreeKeysTrailsCollection(self.tree)
        self.population_tree_trails = [
            copy.deepcopy(tree_trails) for i in range(self.size)]
    """

    def _compute_probability_and_delete_key(self):
        for bit in self.trade_sequence:
            self.tree_trails.append(bit[1])
            self.tree_trails.compute_probability_delete_euclidean_closest_key_extended(  # noqa
                bit[1], bit[0], self.samples)

    def _init_population_trails(self):
        self.population_tree_trails = [
            copy.deepcopy(self.tree_trails) for i in range(self.size)]

    '''
    def _compute_probability_and_delete_key(self):
        for bit in self.trade_sequence:
            for trails in self.population_tree_trails:
                trails.append(bit[1])
                trails.compute_probability_delete_euclidean_closest_key(
                    bit[1], bit[0])
    '''

    def _compute_amount_to_buy(self):
        for index, network in enumerate(self.population):
            self.population_tree_trails[index].compute_amount_to_buy(network)

    def _normalize_amount_to_buy(self):
        for trails in self.population_tree_trails:
            trails.normalize_amount_to_buy()

    def get_size(self):
        return self.size

    def get_network_layers_layout(self):
        return self.network_layers_layout

    def get_input_values_ranges(self):
        return self.input_values_ranges

    def get_min_max_base_value_limits(self):
        return [self.min_base, self.max_base]

    def get_min_max_weight_value_limits(self):
        return [self.min_weight, self.max_weight]

    def get_min_base_limit(self):
        return self.min_base

    def get_max_base_limit(self):
        return self.max_base

    def get_min_weight_limit(self):
        return self.min_weight

    def get_max_weight_limit(self):
        return self.max_weight

    def get_population(self):
        return self.population

    def get_population_member(self, index):
        assert index < self.get_size()
        return self.population[index]

    def set_population_member(self, index, member):
        assert index < self.get_size()
        self.population[index] = member

    def get_network_layers_count(self):
        return len(self.population[0].layers)

    def set_network_layer_bases(self, population_index, layer_index, values):
        self.population[population_index].layers[layer_index].np['b'] = values

    def get_network_layer_bases(self, pupulation_index, layer_index):
        return self.population[pupulation_index].layers[layer_index].np['b']

    def set_network_layer_weights(
            self, pupulation_index, layer_index, inner_index, values):
        self.population[pupulation_index].layers[layer_index].np['w'][inner_index] = values  # noqa

    def get_network_layer_weights(
            self, pupulation_index, layer_index, inner_index):
        return self.population[pupulation_index].layers[layer_index].np['w'][inner_index]  # noqa

    def get_network_weights_layer_count(self, population_index, layer_index):
        return len(self.population[population_index].layers[layer_index].np['w'])  # noqa

    def copy_in_network_layer_properties(
            self, population_index, layer_index, properties):
        self.population[population_index].layers[layer_index].np = copy.deepcopy(properties)  # noqa

    def get_network_layer_properties(
            self, population_index, layer_index):
        return self.population[population_index].layers[layer_index].np

    def copy_in_population(self, population):
        self.population[:] = population

    def _compute_population_fitness(self):
        self.fitness = []
        self.loss = []
        self.investment = []
        self.partial_investments = []
        self.no_limit_investment = []
        self.no_limit_gain = []
        self.no_limit_loss = []
        self.best_fitness = -1.0
        self.worst_fitness = 1000000000.0
        self.best_fitness_index = 0

        for index, trails in enumerate(self.population_tree_trails):
            gain, loss, investment, partial_investments, no_limit_gain, no_limit_loss, no_limit_investment = \
                trails.get_overall_gain_for_estimated_trades_amounts_extended(
                    self.trade_data)
            self.fitness.append(gain)
            self.loss.append(loss)
            self.investment.append(investment)
            self.partial_investments.append(partial_investments)
            self.no_limit_gain.append(no_limit_gain)
            self.no_limit_loss.append(no_limit_loss)
            self.no_limit_investment.append(no_limit_investment)
            """
            if self.fitness[-1] > self.best_fitness:
                self.best_fitness = self.fitness[-1]
                self.best_fitness_index = index
            if self.fitness[-1] < self.worst_fitness:
                self.worst_fitness = self.fitness[-1]
            """
            gain = self.fitness[-1] + self.loss[-1]
            if gain > self.best_fitness:
                self.best_fitness = gain
                self.best_fitness_index = index
            if gain < self.worst_fitness:
                self.worst_fitness = gain

    def get_population_fitness(self):
        self._compute_population_fitness()
        return {
            "best_fitness": self.best_fitness,
            "best_fitness_index": self.best_fitness_index,
            "worst_fitness": self.worst_fitness,
            "fitness": self.fitness, "loss": self.loss,
            "investment": self.investment,
            "partial_investments": self.partial_investments,
            "no_limit_gain": self.no_limit_gain,
            "no_limit_loss": self.no_limit_loss,
            "no_limit_investment": self.no_limit_investment}

    def save_population(self, population_save_file_path):
        print "Saving population into %s" % population_save_file_path
        with open(population_save_file_path, "w") as fd:
            pickle.dump(self, fd)


class NetworksPopulationEvolution(object):

    def __init__(self, population, cross_rate, mutation_rate):
        self.population = population
        self.new_population = None
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def get_population(self):
        return self.population

    def _adjust_fitness(self, fitness, worst_fitness):
        # modify fitness values for easier selection to make
        # sure that member with worst fitness is not going
        # to be chosen
        fitness_sum = 0.0
        for index in range(self.population.get_size()):
            fitness[index] -= worst_fitness
            fitness_sum -= fitness[index]
        return fitness_sum

    def evolve_all_neurons_at_once(self):
        data = self.evolve_one_cycle()
        self._print_report(data)
        return data["best_fitness"]

    def evolve_neurons_backwards_from_last_layer(self):
        data = {}
        for layer in reversed(xrange(
                0, self.population.get_network_layers_count())):
            data = self.evolve_one_cycle(layer)
        self._print_report(data)
        return data["best_fitness"]

    def evolve_neurons_forward_from_first_layer(self):
        data = {}
        for layer in xrange(
                0, self.population.get_network_layers_count()):
            data = self.evolve_one_cycle(layer)
        self._print_report(data)
        return data["best_fitness"]

    def evolve_one_cycle(self, only_layer=None):
        self.population.trade()
        data = self.population.get_population_fitness()  # noqa
        fitness_sum = self._adjust_fitness(
            data["fitness"], data["worst_fitness"])
        self.evolve(
            data["fitness"], fitness_sum,
            data["best_fitness_index"], only_layer)
        return data

    def _print_report(self, data):
        best_fitness_index = data["best_fitness_index"]
        diff_investment = data["investment"][best_fitness_index] - data["no_limit_investment"][best_fitness_index]  # noqa
        diff_gain = data["best_fitness"] - data["no_limit_gain"][best_fitness_index]  # noqa
        print "Population_size: " + \
            str(self.population.get_size()) + " Gain,loss_from_limited_investment: %s,%s" % (data["best_fitness"], data["loss"][best_fitness_index]) + \
            " Limited_investment: " + str(data["investment"][best_fitness_index]) + \
            " Gain,loss_from_no_limit_investment: %s,%s" % (data["no_limit_gain"][best_fitness_index], data["no_limit_loss"][best_fitness_index]) + \
            " No_limit_investment: " + str(data["no_limit_investment"][best_fitness_index]) + \
            " Difference_between_limited_and_no_limited_investment: " + str(diff_investment) + \
            " Difference_between_limited_gain_and_no_limited_gain:" + str(
                diff_gain)
        # print "Sorted investments:", partial_investments[best_fitness_index]  # noqa

    def evolve(
            self, fitness, fitness_sum, best_fitness_index, only_layer=None):
        # generate new population, the values of new generation are
        # not important as they are going to be overwritten by cross
        # over and mutation from previous population
        self.new_population = NetworksPopulation(
            self.population.get_size(),
            self.population.get_network_layers_layout(),
            self.population.get_input_values_ranges(),
            self.population.get_min_max_base_value_limits(),
            self.population.get_min_max_weight_value_limits(),
            self.population.get_tree(),
            self.population.get_trade_sequence(),
            self.population.get_trade_data(),
            self.population.get_samples())
        # print "new_population", self.new_population.population_tree_trails

        for index in range(0, self.population.get_size(), 2):
            """
            first_member_index = get_random_member_index(
                fitness, fitness_sum, self.population.get_size())
            second_member_index = get_random_member_index(
                fitness, fitness_sum, self.population.get_size())
            """
            first_member_index = get_random_member_index_by_rank(fitness)
            second_member_index = get_random_member_index_by_rank(fitness)

            # do cross over of bases and weights
            if random.random() < self.cross_rate:

                self.cross_over_bases(
                    index, first_member_index, second_member_index, only_layer)
                self.mutate_bases(index, only_layer)
                self.cross_over_weights(
                    index, first_member_index, second_member_index, only_layer)
                self.mutate_weights(index, only_layer)

            else:
                # complete copy, no cross over
                for layer_index in xget_layers_indexes(
                        self.population.get_network_layers_count(), only_layer):  # noqa
                    self.new_population.copy_in_network_layer_properties(
                        index, layer_index,
                        self.population.get_network_layer_properties(
                            first_member_index, layer_index))
                    self.new_population.copy_in_network_layer_properties(
                        index+1, layer_index,
                        self.population.get_network_layer_properties(
                            second_member_index, layer_index))

        # elitism, keeping the best member
        self.new_population.set_population_member(
            0, self.population.get_population_member(best_fitness_index))
        # self.population.copy_in_population(
        #    self.new_population.get_population())
        self.population = copy.deepcopy(self.new_population)

    def cross_over_bases(
            self, index, first_member_index,
            second_member_index, only_layer=None):
        for layer_index in xget_layers_indexes(
                self.population.get_network_layers_count(), only_layer):  # noqa
            self.new_population.set_network_layer_bases(
                index, layer_index,
                numpy_random_arryas_cross_over(
                    self.population.get_network_layer_bases(
                        first_member_index, layer_index),
                    self.population.get_network_layer_bases(
                        second_member_index, layer_index)))
            self.new_population.set_network_layer_bases(
                index+1, layer_index,
                numpy_random_arryas_cross_over(
                    self.population.get_network_layer_bases(
                        first_member_index, layer_index),
                    self.population.get_network_layer_bases(
                        second_member_index, layer_index)))

    def mutate_bases(self, index, only_layer=None):
        for child in range(2):
            for layer_index in xget_layers_indexes(
                    self.population.get_network_layers_count(), only_layer):  # noqa
                self.new_population.set_network_layer_bases(
                    index+child, layer_index,
                    numpy_random_values_mutation(
                        self.new_population.get_network_layer_bases(
                            index+child, layer_index),
                        self.mutation_rate,
                        self.population.get_min_base_limit(),
                        self.population.get_max_base_limit()))

    def cross_over_weights(
            self, index, first_member_index,
            second_member_index, only_layer=None):
        for layer_index in xget_layers_indexes(
                self.population.get_network_layers_count(), only_layer):  # noqa
            for inner_index in range(
                    self.population.get_network_weights_layer_count(index, layer_index)):  # noqa
                self.new_population.set_network_layer_weights(
                    index, layer_index, inner_index,
                    numpy_random_arryas_cross_over(
                        self.population.get_network_layer_weights(
                            first_member_index, layer_index, inner_index),
                        self.population.get_network_layer_weights(
                            second_member_index, layer_index, inner_index)))
                self.new_population.set_network_layer_weights(
                    index+1, layer_index, inner_index,
                    numpy_random_arryas_cross_over(
                        self.population.get_network_layer_weights(
                            first_member_index, layer_index, inner_index),
                        self.population.get_network_layer_weights(
                            second_member_index, layer_index, inner_index)))

    def mutate_weights(self, index, only_layer=None):
        for child in range(2):
            for layer_index in xget_layers_indexes(
                    self.population.get_network_layers_count(), only_layer):  # noqa
                for inner_index in range(
                        self.population.get_network_weights_layer_count(index, layer_index)):  # noqa
                    self.new_population.set_network_layer_weights(
                        index+child, layer_index, inner_index,
                        numpy_random_values_mutation(
                            self.new_population.get_network_layer_weights(
                                index+child, layer_index, inner_index),
                            self.mutation_rate,
                            self.population.get_min_weight_limit(),
                            self.population.get_max_weight_limit()))


class GenetationsEvolution(object):

    def __init__(
            self, networks_population_evolution,
            evolution_cycles, report_every=10):
        self.evolution = networks_population_evolution
        self.evolution_cycles = evolution_cycles
        self.report_every = report_every
        self._init_evolution_strategies()

    def _init_evolution_strategies(self, switch_strategy_threshold=3):
        self.switch_strategy_threshold = switch_strategy_threshold
        self.last_strategy_index = 0
        self.last_strategy_change_at = 0
        self.strategies_router = [
            self.evolution.evolve_neurons_backwards_from_last_layer,
            self.evolution.evolve_neurons_forward_from_first_layer,
            self.evolution.evolve_all_neurons_at_once]
        self.strategies_stats = {}
        for index, function in enumerate(self.strategies_router):
            self.strategies_stats[index] = {
                "name": function.func_name, "iterations": 0}

    def _update_evolution_strategies_stats(self, run):
        iterations = self.strategies_stats[self.last_strategy_index][
            "iterations"]
        new_iterations = iterations + run - self.last_strategy_change_at
        self.strategies_stats[self.last_strategy_index].update(
            {"iterations": new_iterations})

    def _switch_evolution_strategies(self, run, without_progress_counter):
        if not without_progress_counter % self.switch_strategy_threshold:
            self._update_evolution_strategies_stats(run)
            self.last_strategy_change_at = run
            if self.last_strategy_index + 1 < len(self.strategies_router):
                self.last_strategy_index += 1
            else:
                self.last_strategy_index = 0

    def run(self, population_save_file_path="population.pickle"):
        last_best_fitness = -10000000
        without_progress_counter = 0

        for run in range(self.evolution_cycles):
            print "Generation:", run, self.strategies_router[
                self.last_strategy_index]
            best_fitness = self.strategies_router[self.last_strategy_index]()
            if best_fitness > last_best_fitness:
                last_best_fitness = best_fitness
            else:
                without_progress_counter += 1
            self.evolution.get_population().save_population(
                population_save_file_path)
            self._switch_evolution_strategies(run, without_progress_counter)

        self._switch_evolution_strategies(run, without_progress_counter)
        print "Strategies stats:"
        pprint.pprint(self.strategies_stats)
