import sys
import time
import random
import operator
import pydot
import pickle
import json
# import copy
import csv
import datetime

import neurolab as nl

from optparse import OptionParser

from trader.libs.samples import load_samples_library, \
    analyse_sample_attributes_extended, get_key_count_map
from trader.libs.relations import CorrelationFact, CorrelationFactsRelation, \
    get_correlation_facts_relation_key_dict
# from trader.libs.trade_sequence_tree import update_top_leafs_counts, \
from trader.libs.trade_sequence_tree import get_tree_layers_counts
from trader.libs.trade_sequence_tree import get_tree_layers_counts_extended
# from trader.libs.utils import close_index_closed_end_interval
from trader.libs.utils import close_index, portions, get_chunks_indexes
# from trader.sample_matching_tasks import find_sample_correlations_no_limits
from trader.gpu_sample_matching_tasks import load_program, find_sample_correlations_no_limits
# from trader.sample_matching_tasks import find_first_sample_correlations
# from celery import group
from trader.libs.trade_sequence_tree_trails import TreeKeysTrailsCollection
from trader.libs.trade_sequence_tree import increment_leaf_key_count
# from trader.libs.trade_sequence_tree import generate_tree_chart
from trader.libs.genetic_network_algorithm import NetworksPopulation, \
    NetworksPopulationEvolution, GenetationsEvolution
#    compute_probability_and_delete_key
from trader.libs.trade_sequence_tree import reset_tree_values, \
    normalise_tree_volatilities
from trader.libs.linear_normalisation import LinearNormalisation
from trader.libs.volatilities import get_price_volatilities


def build_correlation_facts(samples, trade_data, min_appearance_count=1):
    """ To build correlation facts based on samples """
    correlation_facts = []
    for sample in samples:
        correlation_positions_counter = len(sample["+correlation_positions"])
        if correlation_positions_counter >= min_appearance_count:  # noqa
            for correlation_position in sample["+correlation_positions"]:
                sample_size = len(sample["sample_data"])
                correlation_facts.append(
                    CorrelationFact(
                        correlation_position,
                        correlation_positions_counter,
                        sample["sample_epoch"],
                        True,
                        sample["sample_attributes"]["status"],
                        trade_data.prices[correlation_position],
                        trade_data.prices[correlation_position + sample_size - 1])  # noqa
                )
        correlation_positions_counter = len(sample["-correlation_positions"])
        if correlation_positions_counter >= min_appearance_count:  # noqa
            for correlation_position in sample["-correlation_positions"]:
                sample_size = len(sample["sample_data"])
                correlation_facts.append(
                    CorrelationFact(
                        correlation_position,
                        correlation_positions_counter,
                        sample["sample_epoch"],
                        False,
                        sample["sample_attributes"]["status"],
                        trade_data.prices[correlation_position],
                        trade_data.prices[correlation_position + sample_size - 1])  # noqa
                )
    return correlation_facts


def build_plus_correlation_facts(samples, trade_data, min_appearance_count=1):
    """ To build correlation facts based on samples """
    correlation_facts = []
    for sample in samples:
        correlation_positions_counter = len(sample["+correlation_positions"])
        if correlation_positions_counter >= min_appearance_count:  # noqa
            for correlation_position in sample["+correlation_positions"]:
                sample_size = len(sample["sample_data"])
                correlation_facts.append(
                    CorrelationFact(
                        correlation_position,
                        correlation_positions_counter,
                        sample["sample_epoch"],
                        True,
                        sample["sample_attributes"]["status"],
                        trade_data.prices[correlation_position],
                        trade_data.prices[correlation_position + sample_size - 1])  # noqa
                )
    return correlation_facts


def get_all_samples_epochs(facts_time_correlation_map):
    return set([
        facts_time_correlation_map[cor_position].get_sample_epoch()
        for cor_position in facts_time_correlation_map.keys()])


def get_samples_mixin(facts_time_correlation_map, sample_size):
    """ To get list of ordered elements

        [([correlated_sample_position_start, correlated_sample_position_end], index_in_correlated_positions, sample_start_price, sample_end_price), ... ]  # noqa
    """
    return [(
        [p, p + sample_size],
        facts_time_correlation_map[p].get_sample_epoch(),
        facts_time_correlation_map[p].start_price,
        facts_time_correlation_map[p].end_price,
        facts_time_correlation_map[p].get_sample_status())
        for p in sorted(facts_time_correlation_map.keys())]


def get_samples_correlation_chains(
        mixin, trade_data, chain_length, sample_epoch):
    """ To get list of correlation chains based on
        sample epoch and distance
    """
    chain = []
    chains = []
    for m in mixin:
        if m[1] == sample_epoch and len(chain) == 0:
            chain = [m]
            continue
        if len(chain) > 0 and m[0][0] < chain[0][0][0] + chain_length:
            chain.append(m)
        else:
            if len(chain) > 0:
                # price_diff = chain[-1][3] - chain[0][2]
                # trade_amount = sum(
                #    trade_data.trades[tick] for tick in range(
                #        chain[0][0], chain[-1][0] + 1))
                # print price_diff * trade_amount * 0.15, price_diff, trade_amount, chain, "\n"  # noqa
                chains.append(chain)
                chain = []
    return chains


def get_time_correlation_map(correlation_facts):
    """ To build samples which have correlation in time line
        { position_1: fact_1, position_2: fact_2 ...}
    """
    time_correlation_map = {}
    for fact in correlation_facts:
        position = fact.get_correlation_position()
        if position in time_correlation_map:
            if time_correlation_map[position].get_correlation_positions_counter() < fact.get_correlation_positions_counter():  # noqa
                time_correlation_map[position] = fact
        else:
            time_correlation_map[position] = fact
    return time_correlation_map


# this one does limit negative relation which should be bad :(
def get_time_correlation_map_2(correlation_facts, sample_size):
    """ To build samples which have correlation in time line
        { position_1: fact_1, position_2: fact_2 ...}
    """
    time_correlation_map = {}
    positions = []
    sorted_correlation_facts = sorted(
        correlation_facts,
        key=lambda x: [
            x.get_correlation_positions_counter(), x.get_price_delta()],
        reverse=True)
    for fact in sorted_correlation_facts:
        position = fact.get_correlation_position()
        if position not in time_correlation_map:
            if not close_index(position, positions, sample_size):
                time_correlation_map[position] = fact
                positions.append(position)
    return time_correlation_map


# this one does not limits negative relation which should be good :)
def get_time_correlation_map_3(correlation_facts, sample_size):
    """ To build samples which have correlation in time line
        { position_1: fact_1, position_2: fact_2 ...}
    """
    time_correlation_map = {}
    positions = []
    sorted_correlation_facts = sorted(
        correlation_facts,
        key=lambda x: [
            x.get_correlation_positions_counter()],
        reverse=True)
    print "sorted_correlation_facts count:", len(sorted_correlation_facts)
    for fact in sorted_correlation_facts:
        position = fact.get_correlation_position()
        if position not in time_correlation_map:
            # only samples which do not overlap
            # I may not need this ... !!!
            if not close_index(position, positions, sample_size):
                time_correlation_map[position] = fact
                positions.append(position)
    return time_correlation_map


# this one does not make sure that facts overlap
def get_time_correlation_map_4(correlation_facts, sample_size):
    """ To build samples which have correlation in time line
        { position_1: fact_1, position_2: fact_2 ...}
    """
    time_correlation_map = {}
    sorted_correlation_facts = sorted(
        correlation_facts,
        key=lambda x: [
            x.get_correlation_positions_counter()],
        reverse=True)
    for fact in sorted_correlation_facts:
        position = fact.get_correlation_position()
        time_correlation_map[position] = fact
    return time_correlation_map


def count_coverage(time_correlation_map, sample_size, all_count):
    """ Count not overlapping samples """
    covered = len(time_correlation_map.keys()) * sample_size
    return portions(covered, all_count-covered)


def build_facts_relations(facts, max_mapping_trials=10):
    """ To map facts relations by random sampling """
    rel_map = []
    unique = {}
    for i in range(0, max_mapping_trials):
        relation = None
        first_index = random.randint(0, len(facts)-1)
        second_index = random.randint(0, len(facts)-1)
        if first_index != second_index:
            relation = CorrelationFactsRelation(
                facts[first_index], facts[second_index])
            str_relation = relation.get_signature()
            if str_relation not in unique:
                unique.update({str_relation: None})
                rel_map.append(relation)
    return rel_map


def count_relations(facts_rel_map, focus=25):
    " To count facts relations within specified focus distance "
    counted_ralations = {}
    for fact_relation in facts_rel_map:
        key = fact_relation.get_fact_relation_key_string(focus)
        if key not in counted_ralations:
            counted_ralations.update({key: 0})
        else:
            counted_ralations[key] += 1
    return counted_ralations


def build_relation_chart(counted_ralations, focus):
    " To relations chart "
    graph = pydot.Dot(graph_type='graph')
    colors = {
        1: "violet", 2: "indigo", 3: "blue",
        4: "green", 5: "yellow", 6: "orange",
        7: "red"}
    for key_string, value in sorted(counted_ralations.items(), key=operator.itemgetter(1), reverse=True):  # noqa
        if value != 0:
            key_dict = get_correlation_facts_relation_key_dict(key_string)
            # if key_dict["order"] == "True" and key_dict["second_fact_sample_up_or_down"] == "up":  # noqa
            if key_dict["order"] == "True":  # noqa
                color = "black"
                if value in colors:
                    color = colors[value]
                edge = pydot.Edge(
                    str(key_dict["first_fact_correlation_sing"]) + str(key_dict["first_fact_sample_epoch"]) +  # noqa
                        "_" + key_dict["first_fact_sample_up_or_down"],
                    str(key_dict["second_fact_correlation_sing"]) + str(key_dict["second_fact_sample_epoch"]) +  # noqa
                        "_" + key_dict["second_fact_sample_up_or_down"],
                    label=key_string.rstrip("_True") + "_" + str(value),
                    labelfontcolor="#009933",
                    fontsize="10.0",
                    color=color)
                graph.add_edge(edge)
                print key_string, value
    graph.write_png("market_%s.png" % focus)


def fill_interval_gap(diff, sample_size, empty_keys, start_position):
    addition = []
    end_position = 0
    if diff > 0:
        for index in range(0, diff // sample_size):
            new_start_position = start_position + sample_size * index
            end_position = new_start_position + sample_size
            addition.append([
                [new_start_position, end_position], empty_keys[-1][1]])
        extra = diff % sample_size
        if extra > 0:
            if end_position == 0:
                new_start_position = start_position
            else:
                new_start_position = end_position
            end_position = new_start_position + extra
            addition.append([
                [new_start_position, end_position], empty_keys[extra-1][1]])
    return addition


def fill_gaps_extended(
        start_position, end_position, sequence, empty_keys):
    sample_size = len(empty_keys)
    new_sequence = []
    for index, event in enumerate(sequence[:-1]):
        new_sequence.append(event)
        # start of next sample - end of sample
        diff = sequence[index+1][0][0] - event[0][1]
        new_sequence.extend(
            fill_interval_gap(diff, sample_size, empty_keys, event[0][1]))
    # deal with known end of sequence
    new_sequence.append(sequence[-1])
    # deal with start of sequence empty spaces if any
    diff = sequence[0][0][0] - start_position
    start_sequence = fill_interval_gap(diff, sample_size, empty_keys, 0)
    # deal with end of sequence empty spaces if any
    diff = end_position - sequence[-1][0][1]
    new_sequence.extend(
        fill_interval_gap(diff, sample_size, empty_keys, sequence[-1][0][1]))
    return start_sequence + new_sequence


def add_start_and_end_empty_keys(root_leaf, sequence):
    for key in sequence:
        if isinstance(key[1], str):
            next_leaf = root_leaf.get_leafs((key[1],))
            if not next_leaf:
                root_leaf.add_leaf(key[1], {"count": 1})
        else:
            break


class TradeSampleChain(object):

    def __init__(
            self, matched_samples, start_position,
            end_position, trade_data, sample_size):
        self.sample_size = sample_size
        self.trade_data = trade_data
        print "Build correlation facts ..."
        facts = build_correlation_facts(matched_samples, trade_data)
        # facts = build_plus_correlation_facts(matched_samples, trade_data)

        print "Build facts' time correlation map with samples which do not overlap and most occurrent samples considered first."  # noqa
        facts_time_cor_map = get_time_correlation_map_3(facts, sample_size)

        # compute coverage here
        print "TradeSampleChain Coverage:", count_coverage(
            facts_time_cor_map, sample_size, len(trade_data.counters))

        clean_chain = []
        for position in sorted(facts_time_cor_map.keys()):
            clean_chain.append([
                [position, position + sample_size],
                facts_time_cor_map[position].get_sample_epoch(),
                facts_time_cor_map[position].get_price_delta()])

        self.empty_keys = [(i, "N" + str(i)) for i in range(0, sample_size)]
        self.chain = fill_gaps_extended(
            start_position, end_position,
            clean_chain, self.empty_keys)

    def get_chain(self):
        return self.chain


def next_trade_data_analyze(
        rescaled_trade_data_pickle_path, trained_network=None):
    # get rescale period
    trade_data = load_pickle(rescaled_trade_data_pickle_path)
    rescale_period = trade_data.get_rescale_period()
    previous_gain_tree = load_pickle("previous_gain_tree.pickle")
    previous_loss_tree = load_pickle("previous_loss_tree.pickle")

    network = None
    if not trained_network:
        population = load_pickle("population.pickle")
        network = population.get_population()[0]
    else:
        network = trained_network

    previous_gain_samples = load_json("previous_gain_samples.json")
    previous_loss_samples = load_json("previous_loss_samples.json")

    # new_file_path = "testing_data_examples/set-1-22.6-days.json"
    # new_file_path = "testing_data_examples/set-1_2-22.6-days.json"
    # new_file_path = "testing_data_examples/set-2-22.6-days.json"
    # new_file_path = "testing_data_examples/set-3-22.6-days.json"
    # new_file_path = "testing_data_examples/data-1"
    # new_file_path = "testing_data_examples/data-2"
    # new_file_path = "testing_data_examples/data-3"
    new_file_path = "testing_data_examples/data-4"
    # new_file_path = "testing_data_examples/data-6"
    # new_file_path = "testing_data_examples/bitstamp-json-3-months.log"
    print "Loading new data from " + str(new_file_path)
    from trader.libs.trade_data import TradeData, fill_empty_gaps
    new_trade_data = TradeData(new_file_path, rescale_period)

    print "Filling price gaps."
    fill_empty_gaps(new_trade_data.prices, new_trade_data.counters)

    sample_size = len(previous_gain_samples[0]["sample_data"])
    required_correlation = previous_gain_samples[0]["required_correlation"]
    # print "sample_size", sample_size, "required_correlation", required_correlation  # noqa
    # required_correlation = 0.85
    # required_correlation = 0.80

    matched_gain_samples = get_matched_samples(
        new_trade_data, previous_gain_samples, sample_size, required_correlation)  # noqa
    matched_loss_samples = get_matched_samples(
        new_trade_data, previous_loss_samples, sample_size, required_correlation)  # noqa

    save_pickle("next_rescaled_trade_data.pickle", new_trade_data)
    save_json("next_matched_gain_samples.json", matched_gain_samples)
    save_json("next_matched_loss_samples.json", matched_loss_samples)

    print "Build new sample chains."
    gain_trade_chain = TradeSampleChain(
        matched_gain_samples,
        start_position=0,
        end_position=len(new_trade_data.prices)-1,
        trade_data=new_trade_data,
        sample_size=sample_size)
    new_gain_trade_sequence = gain_trade_chain.get_chain()
    loss_trade_chain = TradeSampleChain(
        matched_loss_samples,
        start_position=0,
        end_position=len(new_trade_data.prices)-1,
        trade_data=new_trade_data,
        sample_size=sample_size)
    new_loss_trade_sequence = loss_trade_chain.get_chain()

    """
    print "New chain " + str(new_trade_sample_chain.get_chain())
    count = len(new_trade_sample_chain.get_chain())
    not_match = len(
        filter(
            lambda x: isinstance(x[1], str),
            new_trade_sample_chain.get_chain()))
    print "New chain match %s and not match %s" % (count-not_match, not_match)
    print portions(count-not_match, not_match)
    """

    # create network population to compute profit of new data
    # TODO: this logic should be somewhere else obviously
    print "Running new trade sequence on old skeleton tree."
    networks_population = NetworksPopulation(
        size=1,
        network_layers_layout=None,
        input_values_ranges=None,
        min_max_base_value_limits=[None, None],
        min_max_weight_value_limits=[None, None],
        gain_loss_trees=(previous_gain_tree, previous_loss_tree),
        gain_loss_trade_sequences=(
            new_gain_trade_sequence, new_loss_trade_sequence),
        trade_data=new_trade_data,
        gain_loss_samples=(previous_gain_samples, previous_loss_samples),
        init_population=False)
    networks_population.population = [network]
    # networks_population._compute_probability_and_delete_key()
    networks_population._compute_stats_and_delete_key()
    # networks_population._normalise_volatilities()
    networks_population._compute_amount_to_buy()
    data = networks_population.get_population_fitness()
    # TODO: this logic should be somewhere else obviously
    evolution = NetworksPopulationEvolution(
        networks_population, cross_rate=None, mutation_rate=None)
    evolution._print_report(data)

    print "Gain:"
    show_tree_coverage_in_percentage(
        previous_gain_tree,
        networks_population.gain_loss_tree_trails[0].get_tree())
    print "Loss:"
    show_tree_coverage_in_percentage(
        previous_loss_tree,
        networks_population.gain_loss_tree_trails[1].get_tree())
    print "Gain extended:"
    show_tree_coverage_in_percentage_extended(
        previous_gain_tree,
        networks_population.gain_loss_tree_trails[0].get_tree())
    print "Loss extended:"
    show_tree_coverage_in_percentage_extended(
        previous_loss_tree,
        networks_population.gain_loss_tree_trails[1].get_tree())


def show_tree_coverage_in_percentage(original_tree, modified_tree):
    original_tree_counts = get_tree_layers_counts(original_tree)
    modified_tree_counts = get_tree_layers_counts(modified_tree)
    print "Previous tree layer counts:", original_tree_counts
    print "Modified previous tree layer counts by new trade sequence:", modified_tree_counts  # noqa
    coverage = []
    for index in range(0, len(original_tree_counts)):
        if modified_tree_counts[index] != 0.0:
            coverage.append(
                100 - (modified_tree_counts[index] / (original_tree_counts[index] / 100.0)))  # noqa
        else:
            coverage.append(0.0)
    print "Coverage in percentage:", coverage


def show_tree_coverage_in_percentage_extended(original_tree, modified_tree):
    known, unknown = get_tree_layers_counts_extended(original_tree)
    modified_known, modified_unknown = get_tree_layers_counts_extended(
        modified_tree)
    print "Previous tree layer counts:", known, unknown
    print "Modified previous tree layer counts by new trade sequence:", modified_known, modified_unknown  # noqa
    coverage_known = []
    for i in range(0, len(known)):
        if modified_known[i] != 0.0:
            coverage_known.append(
                100 - (modified_known[i] / (known[i] / 100.0)))
        else:
            coverage_known.append(0.0)
    coverage_unknown = []
    for i in range(0, len(unknown)):
        if modified_unknown[i] != 0.0:
            coverage_unknown.append(
                100 - (modified_unknown[i] / (unknown[i] / 100.0)))
        else:
            coverage_unknown.append(0.0)
    print "Coverage in percentage known:", coverage_known
    print "Coverage in percentage unknown:", coverage_unknown

    """
    gain = past_tree_trails.get_overall_gain(
        new_trade_data, gain_percentage=0.10)
    print "Theoretical gain with sell at the end of sequence for new tree is:", gain  # noqa
    """


def build_gain_chains(
        trade_data, mixin, all_samples_epochs,
        chain_length=10, minimum_chain_length=3):
    print "Build chains of maximum length %s and gain ..." % chain_length
    print """
    This considers only chains where the end price is always bigger against previous sample in chain.
    Drop of price in the middle of the chain is acceptable if end gain is bigger. This can by very dangerous as
    whole chain does not have to be always be achieved and spikes in chain can make it very unpredictable and
    therefore gain might vanish.

    This could be improved to stop chain if there were two consequent drops in price.
    """  # noqa
    interesting_chains = []
    # if small you will find sooner than later which good
    for sample_epoch in list(all_samples_epochs):
        chains = get_samples_correlation_chains(
            mixin, trade_data, chain_length, sample_epoch)
        for ch in chains:
            # print ch
            if len(ch) > minimum_chain_length:
                last_gain = 0
                last_gain_index = 0
                gain = 0.0
                loss = 0.0
                # do not take first element into consideration
                # as we have to start from somewhere
                for index, sample in enumerate(ch[1:]):
                    # [([correlated_sample_position_start, correlated_sample_position_end], index_in_correlated_positions, sample_start_price, sample_end_price), ... ]  # noqa
                    price_diff = sample[3] - sample[2]
                    trade_amount = sum(
                        trade_data.trades[tick] for tick in range(
                            sample[0][0], sample[0][1]))
                    current_gain = price_diff * trade_amount * 0.10
                    # print current_gain, last_gain_index, price_diff, trade_amount, "\n"  # noqa
                    if current_gain > 0:
                        gain += current_gain
                    else:
                        loss += current_gain
                    if (gain + loss) > last_gain:
                        last_gain = gain + loss
                        last_gain_index = index
                if last_gain > 0:
                    interesting_chains.append(
                        ([last_gain, gain, loss], ch[0: last_gain_index+1]))
    return interesting_chains


def build_loss_chains(
        trade_data, mixin, all_samples_epochs,
        chain_length=10, minimum_chain_length=3):
    print "Build chains of maximum length %s and loss ..." % chain_length
    print """
    This considers only chains where the end price is always smaller against previous sample in chain.
    Jump of price in the middle of the chain is acceptable if end gain is smaller. The whole chain does
    not have to be always be achieved and spikes in chain can make it very unpredictable and therefore full
    loss might not achieved.
    """  # noqa
    interesting_chains = []
    # if small you will find sooner than later which good
    for sample_epoch in list(all_samples_epochs):
        chains = get_samples_correlation_chains(
            mixin, trade_data, chain_length, sample_epoch)
        for ch in chains:
            # print ch
            if len(ch) > minimum_chain_length:
                last_gain = 0
                last_gain_index = 0
                gain = 0.0
                loss = 0.0
                # do not take first element into consideration
                # as we have to start from somewhere
                for index, sample in enumerate(ch[1:]):
                    # [([correlated_sample_position_start, correlated_sample_position_end], index_in_correlated_positions, sample_start_price, sample_end_price), ... ]  # noqa
                    price_diff = sample[3] - sample[2]
                    trade_amount = sum(
                        trade_data.trades[tick] for tick in range(
                            sample[0][0], sample[0][1]))
                    current_gain = price_diff * trade_amount * 0.10
                    # print current_gain, last_gain_index, price_diff, trade_amount, "\n"  # noqa
                    if current_gain > 0:
                        gain += current_gain
                    else:
                        loss += current_gain
                    if (gain + loss) < last_gain:
                        last_gain = gain + loss
                        last_gain_index = index
                if last_gain < 0:
                    interesting_chains.append(
                        ([last_gain, gain, loss], ch[0: last_gain_index+1]))
    return interesting_chains


def get_sorted_flattened_filtered_chains(filtered_chains):
    flattened_filtered_chains = []
    for ch in filtered_chains:
        for ch_m in ch[1]:
            flattened_filtered_chains.append(ch_m)
    flattened_filtered_chains.sort(key=lambda x: x[0][0])
    return flattened_filtered_chains


def build_sequence_from_filtered_chains(
        flattened_filtered_chains, empty_keys, count):
    new_sequence = fill_gaps_extended(
        0, count-1, flattened_filtered_chains, empty_keys)
    print "Trading sequence:", new_sequence
    return new_sequence


def build_tree(tree, filtered_chains, sample_size, empty_keys):
    for mixin in filtered_chains:
        leaf = tree.get_root()
        for index, step in enumerate(mixin[1][:-1]):
            # add key if needed
            next_leaf = leaf.get_leafs((step[1],))
            if next_leaf:
                leaf = next_leaf[0]
                increment_leaf_key_count(leaf)
            else:
                leaf = leaf.add_leaf(step[1], {"count": 1})
            # add gaps after key if any
            # start of next sample - end of sample
            diff = mixin[1][index+1][0][0] - step[0][1]
            keys = fill_interval_gap(
                diff, sample_size, empty_keys, step[0][1])
            for key in keys:
                next_leaf = leaf.get_leafs((key[1],))
                if next_leaf:
                    leaf = next_leaf[0]
                    increment_leaf_key_count(leaf)
                else:
                    leaf = leaf.add_leaf(key[1], {"count": 1})
        # add last key
        next_leaf = leaf.get_leafs((mixin[1][-1][1],))
        if next_leaf:
            leaf = next_leaf[0]
            increment_leaf_key_count(leaf)
        else:
            leaf.add_leaf(mixin[1][-1][1], {"count": 1})


def get_empty_keys(sample_size):
    return [(i, "N" + str(i)) for i in range(0, sample_size)]


def initiate_tree(
        facts_time_cor_map, all_samples_epochs, sample_size, empty_keys):
    print "Initiate tree ..."
    from trader.libs.trade_sequence_tree import TradeSequenceTree
    common_leaf_attributes = {}
    for position in facts_time_cor_map:
        common_leaf_attributes.update({
            facts_time_cor_map[position].get_sample_epoch():
            facts_time_cor_map[position].get_sample_status()})
    map(
        lambda x: common_leaf_attributes.update({x[1]: "unknown"}),
        empty_keys)
    # print common_leaf_attributes
    tree = TradeSequenceTree(
        found_sample_keys=list(all_samples_epochs),
        not_found_sample_keys=[key[1] for key in empty_keys],
        common_leafs_atributes=common_leaf_attributes,
        common_leafs_atributes_key_index_map=get_key_count_map())
    # print tree
    return tree


def get_clean_samples(samples, previous_samples_epochs):
    print "Delete previous sample matches."
    previous_samples = []
    for sample in samples:
        if sample["sample_epoch"] in previous_samples_epochs:
            previous_samples.append(sample)
    map(
        lambda x: x.update({
            "+correlation_positions": [],
            "+correlation_positions_epochs": [],
            "-correlation_positions": [],
            "-correlation_positions_epochs": []}),
        previous_samples)
    return previous_samples


def save_pickle(file_name, data):
    print "Saving into %s" % file_name
    with open(file_name, "w") as fd:
        pickle.dump(data, fd)


def load_pickle(file_name):
    print "Loading from %s" % file_name
    data = None
    with open(file_name, "r") as fd:
        data = pickle.load(fd)
    return data


def save_json(file_name, data):
    print "Saving into %s" % file_name
    with open(file_name, "w") as fd:
        json.dump(data, fd)


def load_json(file_name):
    print "Loading from %s" % file_name
    data = []
    with open(file_name, "r") as fd:
        data = json.load(fd)
    return data


"""
def get_matched_samples(
        trade_data, previous_samples, sample_size, required_correlation):
    print "Running sample matching."
    g = group(
        # find_first_sample_correlations.s(
        # find_sample_correlations.s(  # noqa
        find_sample_correlations_no_limits.s(  # noqa
            trade_data.prices, previous_samples[ch[0]:ch[1]],
            sample_size,
            required_correlation) for ch in get_chunks_indexes(
                len(previous_samples), 20))
    matched_samples = []
    for results in g().get():
        for result in results:
            matched_samples.append(result)
    return matched_samples
"""

def get_matched_samples(
        trade_data, previous_samples, sample_size, required_correlation):
    print "Running sample matching."
    load_program(sample_size)
    step = 1024;
    updated_samples = []
    for index in xrange(0, len(previous_samples), step):
        parson_start = time.time()
        updated_samples.extend(
            find_sample_correlations_no_limits(
                trade_data.prices, previous_samples[index:index + step],
                sample_size, required_correlation))
        print "Matching of " + str(step) + " took " + str(time.time() - parson_start) + " seconds for index " + str(index)
    return updated_samples


def get_populated_tree_with_sequence(
        tree, tree_depth, trade_sample_chain, volatility_upper_norm_limit=10.0):
    """ Populate a skeleton tree from trading sequence.

        up_normalisation_multiplier:
            is used to give some extra room for normalisation of future values
    """
    whole_sequence_past_tree_trails = TreeKeysTrailsCollection(tree)
    tree = whole_sequence_past_tree_trails.get_tree()
    # Examples of bit:
    #   start_end_position, sample_epoch, price_delta
    #   [[8623, 8638], 1426915910.0, -1.3507142857143037]
    #   [[8638, 8640], 'N1']
    # max_value = None
    for bit in trade_sample_chain.get_chain():
        volatilities = get_price_volatilities(
            trade_sample_chain.sample_size, bit[0][0], trade_sample_chain.trade_data.prices)
        # last_max_value = max(volatilities)
        # if max_value is None or last_max_value > max_value:
        #    max_value = last_max_value
        # print "volatilities ", volatilities, "bit ", bit
        whole_sequence_past_tree_trails.append_only_if_inactive(bit[1])
        whole_sequence_past_tree_trails.add_key(
            bit[1], bit[0], tree_depth, volatilities)
            # bit[1], bit[0], tree_depth)
    # print "max_volatility_value: " + str(max_value)
    # whole_sequence_past_tree_trails.normalise_volatilities(min_value, max_value)
    linear_normalisation = LinearNormalisation(
        0.0, volatility_upper_norm_limit, 0.0, 1.0)
    normalise_tree_volatilities(tree.get_root(), linear_normalisation)
    return tree


def get_start_and_end_date(trade_data):
    start_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_start)).strftime('%d-%m-%Y %H:%M:%S')
    end_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_end)).strftime('%d-%m-%Y %H:%M:%S')
    return start_date, end_date


def recalculate_samples_attributes(data):
    print "Recalculate samples attributes, this is not necessary if incoming data is correct !!!"  # noqa
    map(lambda x: x.update({"sample_attributes": analyse_sample_attributes_extended(x["sample_data"])}),  # noqa
        data)


def print_trade_data_stats(trade_data, rescale_period):
    start_date, end_date = get_start_and_end_date(trade_data)
    print "\nLog start from %s to %s" % (start_date, end_date)
    print "Data rescaling of %s records done with rescale_period %s seconds resulting into %s records. %s empty data records. Overall %s %% of empty records" % (  # noqa
        trade_data.line_counter,
        rescale_period,
        len(trade_data.times),
        trade_data.counters.count(0),
        trade_data.counters.count(0)/(len(trade_data.times)/100.0))


def get_samples_epochs(chains):
    previous_samples_epochs = set([])
    for ch in chains:
        previous_samples_epochs.add(ch[1])
    return previous_samples_epochs


def relation(
        samples_library_path,
        samples_library_focused_relations_path,
        rescaled_trade_data_pickle_path):

    trade_data = load_pickle(rescaled_trade_data_pickle_path)
    data = load_samples_library(samples_library_path)

    one_sample = data[0]
    sample_size = len(one_sample["sample_data"])
    required_correlation = one_sample["required_correlation"]
    rescale_period = trade_data.get_rescale_period()
    print_trade_data_stats(trade_data, rescale_period)
    recalculate_samples_attributes(data)

    min_appearance = 1
    facts = build_correlation_facts(
        data, trade_data, min_appearance)
    print "Build facts' time correlation map with samples which do not overlap and most occurrent samples considered first."  # noqa
    facts_time_cor_map = get_time_correlation_map_3(facts, sample_size)
    print "get_all_samples_epochs", time.time()
    all_samples_epochs = get_all_samples_epochs(facts_time_cor_map)
    print "get_samples_mixin", time.time()
    mixin = get_samples_mixin(facts_time_cor_map, sample_size)

    chain_length = sample_size * 10
    # chain_length = sample_size * 6
    # chain_length = sample_size * 15
    # chain_length = sample_size * 20
    # chain_length = sample_size * 6 * 2

    print "build_gain_chains", time.time()
    gain_chains = build_gain_chains(
        trade_data, mixin, all_samples_epochs, chain_length)
    loss_chains = build_loss_chains(
        trade_data, mixin, all_samples_epochs, chain_length)

    empty_keys = get_empty_keys(sample_size)
    print "get_sroted_flattened_filtered_chains", time.time()
    flattened_gain_filtered_chains = get_sorted_flattened_filtered_chains(
        gain_chains)
    flattened_loss_filtered_chains = get_sorted_flattened_filtered_chains(
        loss_chains)

    print "build_sequence_from_filtered_chains", time.time()
    new_gain_sequence = build_sequence_from_filtered_chains(
        flattened_gain_filtered_chains, empty_keys, len(trade_data.prices))
    new_loss_sequence = build_sequence_from_filtered_chains(
        flattened_loss_filtered_chains, empty_keys, len(trade_data.prices))

    print "initate_tree", time.time()
    gain_tree = initiate_tree(
        facts_time_cor_map, all_samples_epochs, sample_size, empty_keys)
    loss_tree = initiate_tree(
        facts_time_cor_map, all_samples_epochs, sample_size, empty_keys)

    print "build_tree", time.time()
    build_tree(gain_tree, gain_chains, sample_size, empty_keys)
    build_tree(loss_tree, loss_chains, sample_size, empty_keys)

    save_pickle("skeleton_gain_tree.pickle", gain_tree)
    save_pickle("skeleton_loss_tree.pickle", loss_tree)
    reset_tree_values(gain_tree.get_root())
    reset_tree_values(loss_tree.get_root())

    previous_gain_samples_epochs = get_samples_epochs(
        flattened_gain_filtered_chains)
    previous_loss_samples_epochs = get_samples_epochs(
        flattened_loss_filtered_chains)

    clean_gain_samples = get_clean_samples(
        data, previous_gain_samples_epochs)
    clean_loss_samples = get_clean_samples(
        data, previous_loss_samples_epochs)
    save_json("previous_gain_samples.json", clean_gain_samples)
    save_json("previous_loss_samples.json", clean_loss_samples)

    matched_gain_samples = get_matched_samples(
        trade_data, clean_gain_samples, sample_size, required_correlation)
    matched_loss_samples = get_matched_samples(
        trade_data, clean_loss_samples, sample_size, required_correlation)

    print "Building new chains."
    new_gain_chain = TradeSampleChain(
        matched_gain_samples,
        start_position=0,
        end_position=len(trade_data.prices)-1,
        trade_data=trade_data,
        sample_size=sample_size)
    new_gain_sequence = new_gain_chain.get_chain()
    save_pickle("previous_trade_gain_sequence.pickle", new_gain_sequence)
    new_loss_chain = TradeSampleChain(
        matched_loss_samples,
        start_position=0,
        end_position=len(trade_data.prices)-1,
        trade_data=trade_data,
        sample_size=sample_size)
    new_loss_sequence = new_loss_chain.get_chain()
    save_pickle("previous_trade_loss_sequence.pickle", new_loss_sequence)

    print "Populate skeleton tree with new sequence."
    gain_tree_depth = len(get_tree_layers_counts(gain_tree))
    save_pickle(
        "previous_gain_tree.pickle",
        get_populated_tree_with_sequence(
            gain_tree, gain_tree_depth, new_gain_chain))
    loss_tree_depth = len(get_tree_layers_counts(loss_tree))
    save_pickle(
        "previous_loss_tree.pickle",
        get_populated_tree_with_sequence(
            loss_tree, loss_tree_depth, new_loss_chain))


def train_network(
        sequence_gain_tree_file_path,
        sequence_loss_tree_file_path,
        trade_data_file_path,
        previous_gain_tree_file_path,
        previous_loss_tree_file_path,
        previous_gain_samples_file_path,
        previous_loss_samples_file_path):

    trade_data = load_pickle(trade_data_file_path)
    gain_tree = load_pickle(previous_gain_tree_file_path)
    gain_trade_sequence = load_pickle(sequence_gain_tree_file_path)
    gain_previous_samples = load_json(previous_gain_samples_file_path)

    loss_tree = load_pickle(previous_loss_tree_file_path)
    loss_trade_sequence = load_pickle(sequence_loss_tree_file_path)
    loss_previous_samples = load_json(previous_loss_samples_file_path)

    gain_layers_counts = get_tree_layers_counts(gain_tree)
    gain_layers_counts_extended = get_tree_layers_counts_extended(gain_tree)
    gain_tree_depth = len(gain_layers_counts)
    loss_layers_counts = get_tree_layers_counts(loss_tree)
    loss_layers_counts_extended = get_tree_layers_counts_extended(loss_tree)
    loss_tree_depth = len(loss_layers_counts)
    print "Gain tree layers counts (all, [known, unknown])", gain_layers_counts, gain_layers_counts_extended  # noqa
    print "Loss tree layers counts (all, [known, unknown])", loss_layers_counts, loss_layers_counts_extended  # noqa

    print "Build train networks population ... "
    records_count = len(trade_data.counters)-1
    depth = gain_tree_depth
    if gain_tree_depth < loss_tree_depth:
        depth = loss_tree_depth
    """
    Inputs:

    -1 at down of interval represent missing value

    1. Probability of hitting leaf in tree
        1 1.1 - know sample
        2 1.2 - unknown
    3. Layer number / index (where we are traid from beginning)
    4. Current position in whole sequence
    5. Starting position of traid
    6. Normalized volatilities of intervals [0.3, 0.7, 1.6, 3]
        6 6.1
        7 6.2
        8 6.3
        9 6.4
    """
    networks_population = NetworksPopulation(
        size=6,
        # network_layers_layout=[10, 15, 7, 5, 1],
        network_layers_layout=[10, 7, 1],
        input_values_ranges=[
            [-1, 1], [-1, 1], [-1, depth-1], [-1, records_count], [-1, records_count],  # noqa
            [-1, 1], [-1, 1], [-1, 1], [-1, 1],
            [-1, 1], [-1, 1], [-1, depth-1], [-1, records_count], [-1, records_count],  # noqa
            [-1, 1], [-1, 1], [-1, 1], [-1, 1]],
        # [0, depth-1], [0, records_count]],
        min_max_base_value_limits=[-30.0, 30.0],
        min_max_weight_value_limits=[-30.0, 30.0],
        gain_loss_trees=(gain_tree, loss_tree),
        gain_loss_trade_sequences=(gain_trade_sequence, loss_trade_sequence),
        trade_data=trade_data,
        gain_loss_samples=(gain_previous_samples, loss_previous_samples))
    networks_population_evolution = NetworksPopulationEvolution(
        networks_population, cross_rate=0.95, mutation_rate=0.08)
    generations_evolution = GenetationsEvolution(
        networks_population_evolution=networks_population_evolution,
        evolution_cycles=20,
        report_every=10)

    generations_evolution.run()

"""
def delete_tree_update(tree, whole_sequence_past_tree_trails, new_sequence):

    # delete tree counters as you walk through sequence
    # whole_sequence_past_tree_trails.init_layers_counts()
    for bit in new_sequence:
        print "Probability for key:%s to land in tree" % bit[1], \
            whole_sequence_past_tree_trails.get_in_layer_match_probabilities()
        update_top_leafs_counts_2(
            whole_sequence_past_tree_trails, bit[0], bit[1], False, 20)
        print "\n"

    generate_tree_chart("trading_sequence_tree_3.png", tree)
    sys.exit(0)
"""

"""
    print "Build facts relations by random sampling ..."
    facts_rel_map = build_facts_relations(facts, max_mapping_trials=100000)

    focused_relations = {}
    print "Build facts relation charts ..."
    for focus in xrange(1, 5, 1):
        print "Collapse all sampled relations to count most often appearing with focus %s" % focus  # noqa
        counted_relations = count_relations(facts_rel_map, focus)
        focused_relations.update({focus: counted_relations})
        print "Build facts relation chart ..."
        build_relation_chart(counted_relations, focus)
        print

    save_facts_relations(
        samples_library_focused_relations_path, focused_relations)
    print "Focused facts relations saved into " + \
        str(samples_library_focused_relations_path)
"""


def train_network_2(
        sequence_tree_file_path,
        trade_data_file_path, previous_tree_file_path,
        previous_samples_file_path):

    trade_data = load_pickle(trade_data_file_path)
    tree = load_pickle(previous_tree_file_path)
    trade_sequence = load_pickle(sequence_tree_file_path)
    previous_samples = load_json(previous_samples_file_path)

    layers_counts = get_tree_layers_counts(tree)
    layers_counts_extended = get_tree_layers_counts_extended(tree)
    print "Tree layers counts", layers_counts, layers_counts_extended

    print "Generate train data ..."
    tree_trails = TreeKeysTrailsCollection(tree=tree, write_train_data=True)
    # for bit in trade_sequence[0:1000]:
    for bit in trade_sequence:
        # tree_trails.append(bit[1])
        tree_trails.append_only_if_inactive(bit[1])
        tree_trails.compute_probability_delete_euclidean_closest_key_extended(  # noqa
            bit[1], bit[0], previous_samples)
    tree_trails.record_train_data(
        trade_data=trade_data, min_length=3, gain_percentage=0.1)
    tree_trails.end_train_data()

    print "Load train data ..."
    inputs = []
    targets = []
    with open("train_data.csv", "r") as fd:
        reader = csv.DictReader(fd, tree_trails.get_train_data_order())
        for line in reader:
            data = [float(line[f]) for f in reader.fieldnames]
            prob_known, prob_unkown, layer, index, amount_to_buy = data
            inputs.append([prob_unkown, prob_unkown, layer])
            targets.append((data[-1],))

    print "Build train network ... "
    network_layers_layout = [4, 7, 5, 1]
    network_layers_layout = [7, 5, 1]
    # records_count = len(trade_data.counters)-1
    tree_depth = len(layers_counts)
    """
    Inputs:

    1. Probability of hitting leaf in tree
        1.1 - know sample
        1.2 - unknown
    2. Layer number
    3. Position in trade sequence
    """
    input_values_ranges = [
        # [0, 1], [0, 1], [0, tree_depth-1], [0, records_count]]
        [0, 1], [0, 1], [0, tree_depth-1]]

    network = nl.net.newff(
        input_values_ranges, network_layers_layout)
    # network.trainf = nl.train.train_gdx
    # network.trainf = nl.train.train_gd
    # network.trainf = nl.train.train_gda
    network.trainf = nl.train.train_rprop
    # network.trainf = nl.train.train_bfgs
    # network.trainf = nl.train.train_cg
    # network.trainf = nl.train.train_cg
    network.train(inputs, targets, epochs=100, show=1)
    # print "Error", err

    network_file_path = "network.pickle"
    print "Saving trained network into " + str(network_file_path)
    with open(network_file_path, "w") as fd:
        pickle.dump(network, fd)


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-s", "--samples_library_path",
        dest="samples_library_path",
        default="samples_library.json",
        help="file path to load samples library from library")
    parser.add_option(
        "-f", "--samples_library_focused_relations_path",
        dest="samples_library_focused_relations_path",
        default="focused_facts_relations.json",
        help="file path to save samples library focused relations")
    parser.add_option(
        "-r", "--rescaled_trade_data_pickle_path",
        dest="rescaled_trade_data_pickle_path",
        default="rescaled_trade_data.pickle",
        help="file path to rescaled trade data pickle file")
    parser.add_option(
        "-t", "--train_network",
        dest="train_network",
        default=False,
        action="store_true",
        help="start training network")
    parser.add_option(
        "-p", "--past",
        dest="past",
        default=False,
        action="store_true",
        help="compute future")

    return parser.parse_args(argv)


def main():
    options, args = parse_options(sys.argv[1:])

    if options.train_network:
        train_network(
            # train_network_2(
            sequence_gain_tree_file_path="previous_trade_gain_sequence.pickle",
            sequence_loss_tree_file_path="previous_trade_loss_sequence.pickle",
            trade_data_file_path=options.rescaled_trade_data_pickle_path,
            previous_gain_tree_file_path="previous_gain_tree.pickle",
            previous_loss_tree_file_path="previous_loss_tree.pickle",
            previous_gain_samples_file_path="previous_gain_samples.json",
            previous_loss_samples_file_path="previous_loss_samples.json")

    elif not options.past:
        print "Future estimation:"
        traning_two = False
        network = None
        if traning_two:
            network_file_path = "network.pickle"
            print "Loading network from " + str(network_file_path)
            with open(network_file_path, "r") as fd:
                network = pickle.load(fd)
        next_trade_data_analyze(
            options.rescaled_trade_data_pickle_path, network)
    else:
        print "Past analysis:"
        relation(
            options.samples_library_path,
            options.samples_library_focused_relations_path,
            options.rescaled_trade_data_pickle_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())
