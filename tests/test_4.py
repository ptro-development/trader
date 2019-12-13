# from libs import samples
# from libs.utils import get_window_size, get_chunks_indexes

a = [252.5388888888889, 253.17199999999997, 253.76124999999996, 253.88818181818186, 253.7075, 253.9708333333333, 254.13857142857142, 254.28142857142853, 254.85, 255.1824242424242, 256.49529411764706, 258.1256451612904, 259.27476190476193, 260.6963736263738, 261.64125000000007, 260.7168965517241, 260.9781818181818, 261.68379310344835, 261.44913043478266, 261.0037037037037, 261.57000000000005, 261.4607692307693, 261.3786666666667, 261.645, 261.9057142857143]  # noqa

b = [234, 234, 3, 323, 32]
c = [6, 5, 4, 3, 2, 1]

d = [{'+correlation_positions': [462, 784, 1115, 1561, 3872, 4682, 6329], 'sample_position': 2114, '-correlation_positions': [4350, 5010], 'sample_data': [252.5388888888889, 253.17199999999997, 253.76124999999996, 253.88818181818186, 253.7075, 253.9708333333333, 254.13857142857142, 254.28142857142853, 254.85, 255.1824242424242, 256.49529411764706, 258.1256451612904, 259.27476190476193, 260.6963736263738, 261.64125000000007, 260.7168965517241, 260.9781818181818, 261.68379310344835, 261.44913043478266, 261.0037037037037, 261.57000000000005, 261.4607692307693, 261.3786666666667, 261.645, 261.9057142857143]}, {'+correlation_positions': [462, 1115, 5416, 5997], 'sample_position': 2283, '-correlation_positions': [], 'sample_data': [254.3402083333334, 254.29148148148138, 253.18038461538467, 254.3511111111111, 254.59673913043483, 254.81911111111117, 256.51474999999994, 257.0913207547169, 256.8608333333333, 256.6642857142857, 256.2736, 256.28999999999996, 259.43878504672904, 263.0657236842102, 263.50942857142843, 262.9005263157895, 263.43833333333333, 262.6224137931035, 262.2471428571429, 263.1981578947368, 263.3591304347826, 263.36449999999996, 262.3446153846154, 261.99071428571426, 263.0395454545455]}]  # noqa

# print samples.analyse_sample(a)
# print samples.analyse_sample(b)
# print samples.analyse_sample(c)

# samples.add_sample_attributes(d)
# print d

"""
g = [23]
print get_window_size(len(b), 0.45)
print get_window_size(len(a), 0.30)
print get_window_size(len(g))

print get_chunks_indexes(5, 2)
print get_chunks_indexes(5, 5)
print get_chunks_indexes(5, 1)
# print get_chunks_indexes(5, 6)
"""

from scipy.stats.stats import pearsonr


def test_sample_match_2(
        samples_library, test_sample,
        actual_position, expected_correlation=0.975):
    test_sample_size = len(test_sample)
    for sample in samples_library:
        for index in reversed(range(len(test_sample))):
            cor, other = pearsonr(
                sample[:test_sample_size - index],
                test_sample[index:])
            print cor, sample[:test_sample_size - index], test_sample[index:]
            if cor >= expected_correlation:
                    # if sample['sample_position'] == 2113 and actual_position == 2138:  # noqa
                    #    print test_sample[index:], "\n", sample['sample_data'][:test_sample_size - index]  # noqa
                    print "Sample %s match at %s with correlation %s and cross section percentage %s up or down %s" % (  # noqa
                        0,
                        actual_position - test_sample_size,
                        expected_correlation,
                        (test_sample_size - index) / (float(test_sample_size) / 100.0),  # noqa
                        True)
"""
samples_library = [252.58555555555554, 252.5388888888889, 253.17199999999997, 253.76124999999996, 253.88818181818186, 253.7075, 253.9708333333333, 254.13857142857142, 254.28142857142853, 254.85, 255.1824242424242, 256.49529411764706, 258.1256451612904, 259.27476190476193, 260.6963736263738, 261.64125000000007, 260.7168965517241, 260.9781818181818, 261.68379310344835, 261.44913043478266, 261.0037037037037, 261.57000000000005, 261.4607692307693, 261.3786666666667, 261.645]  # noqa
test_sample = [252.58555555555554, 252.5388888888889, 253.17199999999997, 253.76124999999996, 253.88818181818186, 253.7075, 253.9708333333333, 254.13857142857142, 254.28142857142853, 254.85, 255.1824242424242, 256.49529411764706, 258.1256451612904, 259.27476190476193, 260.6963736263738, 261.64125000000007, 260.7168965517241, 260.9781818181818, 261.68379310344835, 261.44913043478266, 261.0037037037037, 261.57000000000005, 261.4607692307693, 261.3786666666667, 261.645]  # noqa
test_sample_match_2(
    [samples_library],
    test_sample,
    1000,
    0.975)
"""

"""
data.log_start
    1426110410
data.rescale_period
    300
"""

import time


def get_closest_epoch(last_epoch, period):
    """
    New_epoch is smaller than period.
    >>> previous_epoch = int(time.time()-200);
    >>> new_epoch = get_closest_epoch(previous_epoch, 300)
    >>> new_epoch - previous_epoch
    300

    New_epoch is bigger than period.
    >>> previous_epoch = int(time.time()-700);
    >>> new_epoch = get_closest_epoch(previous_epoch, 300)
    >>> new_epoch - previous_epoch
    600

    New_epoch is the same as last_epoch.
    >>> previous_epoch = int(time.time());
    >>> new_epoch = get_closest_epoch(previous_epoch, 300)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 22, in get_closest_epoch
    ValueError: Current epoch is <= to last_epoch.
    """
    closest_epoch = None
    delta = int(time.time()) - int(last_epoch)
    if delta <= 0:
        raise ValueError("Current epoch is <= to last_epoch.")
    if delta <= period:
        closest_epoch = last_epoch + period
    else:
        closest_epoch = last_epoch + (delta // period) * period
    return closest_epoch

# import doctest
# doctest.testmod()

import pickle
with open("rescaled_trade_data.pickle", "r") as f:
    trade_data = pickle.load(f)
with open("samples_library.pickle", "r") as f:
    lib = pickle.load(f)

sorted_samples_plus = lib["sorted_samples_plus"]
sorted_samples_minus = lib["sorted_samples_minus"]

print "Convert samples positions and their matches into epochs"
for samples in sorted_samples_plus, sorted_samples_minus:
    # for i in samples:
    #    for j in i["+correlation_positions"]:
    #        print j
    #        print trade_data.times[j]

    map(
        lambda x: x.update({
            "-correlation_positions_epochs": [trade_data.times[i] for i in x["-correlation_positions"]]}),  # noqa
        samples
    )
    map(
        lambda x: x.update({
            "+correlation_positions_epochs": [trade_data.times[i] for i in x["+correlation_positions"]]}),  # noqa
        samples
    )
print sorted_samples_plus[0]
