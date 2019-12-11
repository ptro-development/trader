import random
import datetime
import json
from utils import close_index

from scipy.stats.stats import pearsonr


def eliminate_samples(samples, acceptable_correlation):
    new_samples = []
    new_samples.append(samples.pop())
    while(samples):
        sample = samples.pop()
        for n_sample in new_samples:
            cor, other = pearsonr(
                sample["sample_data"], n_sample["sample_data"])
            if cor >= acceptable_correlation or cor <= -acceptable_correlation:
                continue
            else:
                new_samples.append(sample)
                break
    return new_samples


def load_samples_library(file_path):
    samples = None
    with open(file_path, "r") as fd:
        samples = json.load(fd)
    return samples


def save_samples_library(file_path, sorted_samples_plus, sorted_samples_minus):
    with open(file_path, "w") as fd:
        json.dump(
            {
                "sorted_samples_plus": sorted_samples_plus,
                "sorted_samples_minus": sorted_samples_minus
            },
            fd)


def remove_samples_min_correlation(
        samples_library,
        min_number_correlations_positions=2):
    mncs = min_number_correlations_positions
    reduced_samples = [s for s in samples_library["sorted_samples_plus"]
                       if len(s["+correlation_positions"]) >= mncs or
                       len(s["-correlation_positions"]) >= mncs]
    reduced_samples += [s for s in samples_library["sorted_samples_minus"]
                        if len(s["+correlation_positions"]) >= mncs or
                        len(s["-correlation_positions"]) >= mncs]
    return reduced_samples


def analyse_sample_attributes(sample):
    """ To find sample attributes

    Attributes:
    min_position_value - [index, value]
    max_position_value - [index, value]
    up_or_down         - up
                       - down
                       - variates
    """
    attributes = {
        "min_position_value": [0, sample[1]],
        "max_position_value": [0, sample[1]],
        "up_or_down": -1,
    }
    for index, value in enumerate(sample):
        if value < attributes["min_position_value"][1]:
            attributes["min_position_value"] = [index, value]
        if value > attributes["max_position_value"][1]:
            attributes["max_position_value"] = [index, value]
    if attributes["min_position_value"][0] == 0 and attributes["max_position_value"][0] == len(sample)-1:  # noqa
        attributes["up_or_down"] = "up"
    elif attributes["min_position_value"][0] == len(sample)-1 and attributes["max_position_value"][0] == 0:  # noqa
        attributes["up_or_down"] = "down"
    else:
        attributes["up_or_down"] = "variates"
    return attributes


def get_key_count_map():
    return {
        "down": -2, "down_variates": -1,
        "same": 0, "same_variates": 0,
        "up_variates": 1, "up": 2,
        "unknown": 10}


def get_leaf_atrribute_number(key, common_leaf_attributes):
    return get_key_count_map()[common_leaf_attributes[key]]


def analyse_sample_attributes_extended(sample):
    """ To find sample attributes

    Attributes:
    min_position_value - [index, value]
    max_position_value - [index, value]
    status             - up, up_variates, down, down_variates, same, same_variates  # noqa
    """
    attributes = {
        "min_position_value": [0, sample[1]],
        "max_position_value": [0, sample[1]],
        "status": -1,
    }
    for index, value in enumerate(sample):
        if value < attributes["min_position_value"][1]:
            attributes["min_position_value"] = [index, value]
        if value > attributes["max_position_value"][1]:
            attributes["max_position_value"] = [index, value]
    if attributes["min_position_value"][0] == 0 and attributes["max_position_value"][0] == len(sample)-1:  # noqa
        attributes["status"] = "up"
    elif attributes["min_position_value"][0] == len(sample)-1 and attributes["max_position_value"][0] == 0:  # noqa
        attributes["status"] = "down"
    else:
        if sample[0] < sample[-1]:
            attributes["status"] = "up_variates"
        elif sample[0] > sample[-1]:
            attributes["status"] = "down_variates"
        else:
            if attributes["min_position_value"][0] == attributes["max_position_value"][0]:  # noqa
                attributes["status"] = "same"
            else:
                attributes["status"] = "same_variates"
    return attributes


def add_sample_attributes(samples):
    for index, sample in enumerate(samples):
        samples[index]["sample_attributes"] = analyse_sample_attributes(
            sample["sample_data"])


def aceptable_min_sample_value_variation(sample_data, min_sample_variation):
    return abs(max(sample_data) - min(sample_data)) >= min_sample_variation


def get_empty_sample():
    return {
        "sample_position": None,
        "sample_data": [],
        "+correlation_positions": [],
        "-correlation_positions": [],
    }


def get_random_samples(
        data, sample_size,
        sample_count, min_sample_variation):
    samples = []
    indexes = []
    for i in range(0, sample_count):
        start_of_sample = random.randint(0, len(data) - sample_size - 1)
        if start_of_sample not in indexes and \
            aceptable_min_sample_value_variation(
                data[start_of_sample: start_of_sample + sample_size],
                min_sample_variation):
            indexes.append(start_of_sample)
            s = get_empty_sample()
            s["sample_position"] = start_of_sample
            s["sample_data"] = data[
                start_of_sample: start_of_sample + sample_size]
            samples.append(s)
    return samples


def get_correlations_gaps_trades_times(
        positions, gaps, trades, times, sample_size):
    c_gaps = []
    c_trades = []
    c_times = []
    for index in positions:
        c_gaps.append(gaps[index:index+sample_size].count(True))
        c_trades.append(sum(trades[index:index+sample_size]))
        c_times.append(
            datetime.datetime.fromtimestamp(
                times[index]).strftime('%d-%m-%Y %H:%M:%S'))
    return c_gaps, c_trades, c_times


def print_price_sample(times, sample, gaps, trades):
    index = sample["sample_position"]
    sample_size = len(sample["sample_data"])
    print "sample_position : " + str(index)
    print "required_correlation : " + str(sample["required_correlation"])
    print "sample_start_time : " + datetime.datetime.fromtimestamp(
        times[index]).strftime('%d-%m-%Y %H:%M:%S')
    print "sample_data : " + str(sample["sample_data"])
    print "sample_trade_amount : " + str(sum(trades[index:index+sample_size]))

    print "+correlations_counter : " + str(
        len(sample["+correlation_positions"]))
    print "+correlations_positions : " + str(sample["+correlation_positions"])
    c_gaps, c_trades, c_times = get_correlations_gaps_trades_times(
        sample["+correlation_positions"], gaps, trades, times, sample_size)
    print "+correlations_gaps :" + str(c_gaps)
    print "+correlations_trades_amount :" + str(c_trades)
    print "+correlations_dates :" + str(c_times)

    print "-correlations_counter : " + str(
        len(sample["-correlation_positions"]))
    print "-correlations_positions : " + str(sample["-correlation_positions"])
    c_gaps, c_trades, c_times = get_correlations_gaps_trades_times(
        sample["-correlation_positions"], gaps, trades, times, sample_size)
    print "-correlations_gaps :" + str(c_gaps)
    print "-correlations_trades_amount :" + str(c_trades)
    print "-correlations_dates :" + str(c_times)


def remove_close_samples(sorted_samples, window_size=10):
    indexes = []
    reduced_samples = []
    for sample in sorted_samples:
        index = sample['sample_position']
        if not close_index(index, indexes, window_size):
            indexes.append(index)
            reduced_samples.append(sample)
    return reduced_samples


def get_key_sample_data(key, samples):
    data = None
    for sample in samples:
        if sample["sample_epoch"] == key:
            data = sample["sample_data"]
            break
    return data
