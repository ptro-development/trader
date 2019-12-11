import time
import random

from scipy.spatial import distance


def portions(x, y):
    return x, float(x)/(x+y), y, float(y)/(x+y)


def get_window_size(data_size, percentage=0.10):
    """ To get size of window based on required percentage. """
    result = None
    part = data_size * percentage
    if part < 1:
        result = 1
    else:
        result = int(part)
    return result


def get_euclidean_closest_element(base_element, elements):
    """
        element  [element, x_1, x_2, ... x_n]
        elements [[element_1, x_1, x_2, ... x_n], ....]
    """
    assert isinstance(base_element, tuple) and len(base_element) != 0
    assert isinstance(elements, list) and len(elements) != 0
    smalest_distance = distance.euclidean(base_element[1:], elements[0][1:])
    smalest_distance_elements = [elements[0]]
    if len(elements) > 1:
        for index, element in enumerate(elements[1:]):
            last_distance = distance.euclidean(base_element[1:], element[1:])
            if last_distance < smalest_distance:
                smalest_distance = last_distance
                smalest_distance_elements = [element]
            elif last_distance == smalest_distance:
                smalest_distance_elements.append(element)
    closest_element = None
    if len(smalest_distance_elements) > 1:
        closest_element = random.choice(smalest_distance_elements)
    else:
        closest_element = smalest_distance_elements[0]
    return closest_element, smalest_distance


def get_chunks_indexes(interval_size, count):
    """ To generate count amount of chunks in interval <0, interval_size>

    >>> get_chunks_indexes(10, 3)
    [(0, 3), (3, 6), (6, 10)]

    >>> get_chunks_indexes(3, 3)
    [(0, 1), (1, 2), (2, 3)]

    >>> get_chunks_indexes(2, 3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 10, in get_chunks_indexes
    ValueError: Error interval_size < count
    """
    if interval_size < count:
        raise ValueError("Error interval_size < count")
    n = interval_size / count
    i = (count - interval_size % count) * n
    idx = range(n, i, n) + range(i, interval_size, n+1)
    idx1 = [0] + idx
    idx2 = idx + [interval_size]
    return zip(idx1, idx2)


def close_index(index, indexes, window_size=10):
    not_allowed = [index - i for i in range(1, window_size)]
    not_allowed.extend([index + i for i in range(1, window_size)])
    found = False
    for na in not_allowed:
        if na in indexes:
            found = True
            break
    return found


def close_index_closed_end_interval(index, indexes, window_size=10):
    return close_index(index, indexes, window_size+1)


def get_closest_epoch(last_epoch, period):
    """
    To get closest_epoch to current_epoch from
    last_epoch where

    closest_epoch = last_epoch + (X * period)

    where X is such a number that condition

    closest_epoch > current_epoch is True.

    current_epoch is smaller than period
    >>> previous_epoch = int(time.time()-200);
    >>> new_epoch = get_closest_epoch(previous_epoch, 300)
    >>> new_epoch - previous_epoch
    300

    current_epoch is bigger than period
    >>> previous_epoch = int(time.time()-700);
    >>> new_epoch = get_closest_epoch(previous_epoch, 300)
    >>> new_epoch - previous_epoch
    900

    current_epoch is the same as last_epoch
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
        closest_epoch = last_epoch + (delta // period) * period + period
    return closest_epoch


def get_first_and_last_log_lines(log_path):
    first, last = None, None
    with open(log_path, "rb") as f:
        first = f.readline()
        f.seek(-2, 2)
        while f.read(1) != "\n":
            f.seek(-2, 1)
        last = f.readline()
    return first, last


def get_start_and_end_trade_log_epochs(log_path):
    return map(
        lambda x: float(x.split(" ", 1)[0]),
        get_first_and_last_log_lines(log_path)
    )


def connect_arrays(first, second):
    len_first = len(first)
    len_second = len(second)
    max_size = len_first
    if len_second > max_size:
        max_size = len_second
    new_first = max_size * [-1]
    new_second = max_size * [-1]
    new_first[0:len_first] = first[:]
    new_second[0:len_second] = second[:]
    return new_first + new_second
