from __future__ import absolute_import
from trader.celery import app
from scipy.stats.stats import pearsonr


@app.task
def close_index(index, indexes, window_size=10):
    not_allowed = [index - i for i in range(1, window_size)]
    not_allowed.extend([index + i for i in range(1, window_size)])
    found = False
    for na in not_allowed:
        if na in indexes:
            found = True
            break
    return found


@app.task
def find_sample_correlations(
        data, samples, sample_size, acceptable_correlation):
    for d_index in range(0, len(data) - sample_size):
        for s_index, sample in enumerate(samples):
            # to avoid correlation to itself by not testing
            # in window of size 2 * sample_size with
            # sample["sample_position"] in the middle
            if d_index < sample["sample_position"] - sample_size or d_index > sample["sample_position"] + sample_size:  # noqa
                cor, other = pearsonr(
                    sample["sample_data"],
                    data[d_index: d_index + sample_size])
                if cor > acceptable_correlation:
                    # to avoid multiple close correlation matches
                    if not close_index(d_index, samples[s_index]["+correlation_positions"]):  # noqa
                        samples[s_index]["+correlation_positions"].append(
                            d_index)
                elif cor < -acceptable_correlation:
                    # to avoid multiple close correlation matches
                    if not close_index(d_index, samples[s_index]["-correlation_positions"]):  # noqa
                        samples[s_index]["-correlation_positions"].append(
                            d_index)
    return samples


@app.task
def find_first_sample_correlations(
        data, samples, sample_size, acceptable_correlation):
    for d_index in range(0, len(data) - sample_size):
        for s_index, sample in enumerate(samples):
            # to avoid correlation to itself by not testing
            # in window of size 2 * sample_size with
            # sample["sample_position"] in the middle
            if d_index < sample["sample_position"] - sample_size or d_index > sample["sample_position"] + sample_size:  # noqa
                cor, other = pearsonr(
                    sample["sample_data"],
                    data[d_index: d_index + sample_size])
                if cor > acceptable_correlation:
                    # to avoid multiple close correlation matches
                    if not close_index(d_index, samples[s_index]["+correlation_positions"]):  # noqa
                        samples[s_index]["+correlation_positions"].append(
                            d_index)
                        break
                elif cor < -acceptable_correlation:
                    # to avoid multiple close correlation matches
                    if not close_index(d_index, samples[s_index]["-correlation_positions"]):  # noqa
                        samples[s_index]["-correlation_positions"].append(
                            d_index)
                        break
    return samples


@app.task
def find_sample_correlations_no_limits(
        data, samples, sample_size, acceptable_correlation):
    for d_index in range(0, len(data) - sample_size):
        for s_index, sample in enumerate(samples):
            cor, other = pearsonr(
                sample["sample_data"],
                data[d_index: d_index + sample_size])
            if cor > acceptable_correlation:
                samples[s_index]["+correlation_positions"].append(
                    d_index)
            elif cor < -acceptable_correlation:
                samples[s_index]["-correlation_positions"].append(
                    d_index)
    return samples
