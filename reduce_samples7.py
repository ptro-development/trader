import json
import sys
import time
import multiprocessing
from os import getpid

from trader.libs.samples import save_simple_samples_library
from trader.libs.utils import get_chunks_indexes
from trader.libs.file_utils import load_json, \
    convert_samples_file_to_line_records, m_sort_line_records, \
    get_line_records, save_samples_to_line_records, \
    convert_line_records_file_to_samples_file


def reduce_sample(sorted_samples, sample_index, common_set, positions):
    remove_indexes = []
    for common in common_set:
        remove_indexes.append(
                sorted_samples[sample_index][positions].index(common))
    data_positions = []
    data_epochs = []
    for i, value in enumerate(sorted_samples[sample_index][positions]):
        if i not in remove_indexes:
            data_positions.append(value)
            data_epochs.append(sorted_samples[sample_index][positions + '_epochs'][i])
    #if len(remove_indexes):
    #    print positions, len(data_epochs)
    sorted_samples[sample_index][positions] = data_positions
    sorted_samples[sample_index][positions + '_epochs'] = data_epochs


def reduce_samples(inputs):
    start_index, stop_index, file_path, max_window, positions = inputs
    # sorted_samples need to be sorted in reverse order, so the most appearing samples first
    sorted_samples, count = get_line_records(file_path, start_index, stop_index)
    print getpid(), "- Samples count to reduce:", count
    intervals = [int(i * 0.1 * count) for i in range(1, 10)]
    intervals.append(count)
    intervals_index = 0
    ss_time = time.time()
    s_time = time.time()
    for index in xrange(0, count, max_window):
        if index > intervals[intervals_index]:
            print getpid(), "- Done:", str(intervals[intervals_index]), "in", time.time() - s_time
            intervals_index += 1
            s_time = time.time()
        walk_index = 0
        for inner_index in xrange(index + 1, len(sorted_samples)):
            if walk_index < max_window and inner_index < stop_index:
                walk_index += 1
                operator = '-' if positions[0] == "+" else '+'
                for cor_position in [positions, operator + positions[1:]]:
                    a = set(sorted_samples[index][cor_position])
                    b = set(sorted_samples[inner_index][cor_position])
                    common_set = list(a & b)
                    if len(a) >= len(b):
                        reduce_sample(sorted_samples, inner_index, common_set, cor_position)
                    else:
                        reduce_sample(sorted_samples, index, common_set, cor_position)
            else:
                break
    print getpid(), "- Chunk done in:", time.time() - ss_time
    new_file_path = "fun_reduce_samples_" + str(start_index) + "_" + str(stop_index) + "_" + file_path
    save_samples_to_line_records(sorted_samples, new_file_path)
    return (start_index, stop_index, new_file_path)


def remove_empty_samples(samples):
    # ((start, stop, file_path), ...)
    new_samples = []
    for sample in samples:
        with open(sample[2]) as fd:
            for line in fd:
                data = json.loads(line)
                if len(data["+correlation_positions"]) and len(data["-correlation_positions"]):
                    new_samples.append(data)
    return save_samples_to_line_records(new_samples, "removed_empty_samples.json")


def reduce_correlations(file_path, lines_count, depth, positions="+correlation_positions"):
    s_time = time.time()
    print "Reducing %s. Samples count:" % positions, lines_count
    chunks = get_chunks_indexes(lines_count, int(lines_count/(lines_count * 0.005)))
    jobs = [[chunk[0], chunk[1], file_path, depth, positions] for chunk in chunks]
    results = []
    for (start, stop, file_path) in pool.map(reduce_samples, jobs):
        results.append((start, stop, file_path))
    print "It took:", time.time() - s_time
    return results


def t_remove_empty_samples(results):
    s_time = time.time()
    print "Removing empty samples."
    file_path, lines_count = one_pool.map(remove_empty_samples, [results])[0]
    print "It took:", time.time() - s_time, "samples count:", lines_count
    return file_path, lines_count


def sort_correlations(file_path, positions="+correlation_positions"):
    s_time = time.time()
    print "Sorting %s." % positions
    file_path, lines_count = one_pool.map(m_sort_line_records, [[file_path, positions]])[0]
    print "It took:", str(time.time() - s_time)
    return file_path, lines_count


def clean_samples(file_path, lines_count, depth=10):
    file_path, lines_count = sort_correlations(file_path, "+correlation_positions")
    results = reduce_correlations(file_path, lines_count, depth, "+correlation_positions")
    file_path, lines_count = t_remove_empty_samples(results)
    file_path, lines_count = sort_correlations(file_path, "-correlation_positions")
    results = reduce_correlations(file_path, lines_count, depth, "-correlation_positions")
    file_path, lines_count = t_remove_empty_samples(results)
    return file_path, lines_count


def reduce_in_depths(file_path, lines_count, depths=[1, 2, 3, 5, 8, 13, 21, 34]):
    for i in depths:
        depth = 2 * i
        print "Cleaning samples in depth:", depth
        file_path, lines_count = clean_samples(file_path, lines_count, depth)
        print "Reduced to:", lines_count, " samples."
        new_file_path, lines_count = convert_line_records_file_to_samples_file(file_path)
        print "Result saved into %s" % new_file_path
        print


# DO reducing ...
one_pool = multiprocessing.Pool(1)
pool = multiprocessing.Pool(1)
file_path, lines_count = one_pool.map(convert_samples_file_to_line_records, [sys.argv[1]])[0]
reduce_in_depths(file_path, lines_count)
