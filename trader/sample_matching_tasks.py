from __future__ import absolute_import

from trader.celery import app
from trader.libs.event import Event
from trader.event_consumer_single_tasks import process_incoming_event

from scipy.stats.stats import pearsonr


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


@app.task
def filter_period_event(event, min_percentage_match=5.0):
    if event is not None:
        if event.is_min_required_percentage(min_percentage_match) is True:
            print "Filtered events with %s%% match." % min_percentage_match
            e = event.get_dict()
            process_incoming_event.apply_async((e,))
            print "percentage %s position %s relative_match_position %s correlation %s" % (  # noqa
                e["percentage"],
                e["sample"]["sample_epoch"],
                e["relative_match_position"],
                e["correlation"])


@app.task
def test_sample_library_match(
        library_sample, incoming_price_data, incoming_trade_data,
        incoming_data_position, expected_correlation=0.975):
    """ To test if incoming data correlates to sample from library.
        Function returns event with expected_correlation or higher.
        In case it does not find one it returns None.

        Make sure that:
        - incoming data has resolution as samples from library
        - samples and incoming_data have the same size

        Note: The oldest data is always at beginning of arrays
              incoming_price_data and incoming_trade_data
    """
    incoming_data_size = len(incoming_price_data)
    event = None
    for index in reversed(range(len(incoming_price_data))):
        cor, other = pearsonr(
            library_sample['sample_data'][:incoming_data_size - index],
            incoming_price_data[index:])
        if cor >= expected_correlation or cor <= -expected_correlation:
                percentage = (incoming_data_size - index) / \
                    (float(incoming_data_size) / 100.0)
                if (event is not None and event.get_percentage() < percentage) or \
                        event is None:
                    event = Event(
                        relative_match_position=incoming_data_position,
                        sample=library_sample,
                        incoming_price_data=incoming_price_data,
                        incoming_trade_data=incoming_trade_data,
                        correlation=cor,
                        percentage=percentage)
    return event
