from __future__ import absolute_import
from celery import chain

from trader.celery import app
from trader.consumer_logger_tasks import write_log
from trader.sample_matching_tasks import test_sample_library_match, \
    filter_period_event
from celery.signals import celeryd_after_setup
from trader.libs.utils import get_closest_epoch
from trader.libs.trade_data import SampleTradeDataBuffer, fill_empty_gaps
from trader.libs.samples import load_samples_library, \
    remove_samples_min_correlation

import datetime
import time

samples_library = None
sample_trade_data_buffer = None
expected_correlation = None
periods_counter = 0


@celeryd_after_setup.connect
def init_consumer(sender, instance, **kwargs):
    if not sender.startswith("consumer@"):
        return
    global samples_library, \
        sample_trade_data_buffer, expected_correlation
    samples_library = load_samples_library(app.conf.SAMPLES_LIBRARY_FILE)
    print "Samples were loaded from " + str(app.conf.SAMPLES_LIBRARY_FILE)
    samples_library = remove_samples_min_correlation(
        samples_library,
        app.conf.MIN_NUMBER_CORRELATIONS_POSITIONS)
    print "Removing samples which do not have at least %s correlation matches." % \
        app.conf.MIN_NUMBER_CORRELATIONS_POSITIONS
    rescale_period = samples_library[0]["rescale_period"]
    print "Rescale period " + str(rescale_period)
    closest_epoch = None
    if app.conf.REPLAY_MODE:
        print "Replay trade data mode active."
        closest_epoch = time.time() + rescale_period
    else:
        print "Running in trade data listen mode."
        closest_epoch = get_closest_epoch(
            samples_library[0]["sample_epoch"],
            samples_library[0]["rescale_period"])
    print "Closest epoch for rescaling calculated as " + \
        datetime.datetime.fromtimestamp(
            closest_epoch).strftime('%d-%m-%Y %H:%M:%S') + " " + \
        str(closest_epoch)
    sample_size = len(samples_library[0]["sample_data"])
    sample_trade_data_buffer = SampleTradeDataBuffer(
        sample_size,
        rescale_period,
        closest_epoch
    )
    print "Initialised incoming data buffer of size " + str(sample_size)
    expected_correlation = samples_library[0]["required_correlation"]
    # expected_correlation = 0.95
    print "Expected sample correlation " + str(expected_correlation)


@app.task
def process_incoming_collapsed_trade_log(
        incoming_prices, incoming_trades, position):
    global samples_library, expected_correlation
    for library_sample in samples_library:
        ch = chain(
            test_sample_library_match.s(
                library_sample,
                incoming_prices,
                incoming_trades,
                position,
                expected_correlation
            ),
            filter_period_event.s()
        )
        ch.apply_async()


@app.task
def process_incoming_trade_log(line):
    global sample_trade_data_buffer, samples_library, \
        expected_correlation, periods_counter
    write_log.apply_async((line,))
    new_periods = sample_trade_data_buffer.update(line)
    periods_counter += new_periods
    if new_periods > 0 and sample_trade_data_buffer.is_buffer_full():  # noqa
        if periods_counter % 20 == 0:
            print "Period counter: " + str(periods_counter)
        incoming_prices = sample_trade_data_buffer.get_prices()
        incoming_trades = sample_trade_data_buffer.get_trades()
        incoming_records_counter = sample_trade_data_buffer.get_records_counter()  # noqa
        fill_empty_gaps(
            incoming_prices,
            incoming_records_counter)
        """
        Do not fill these gaps as they need to be without modification.
        """
        # fill_empty_gaps(
        #    incoming_trades,
        #    incoming_records_counter)
        for library_sample in samples_library:
            ch = chain(
                test_sample_library_match.s(
                    library_sample,
                    incoming_prices[1:],
                    incoming_trades[1:],
                    sample_trade_data_buffer.get_epochs()[0],
                    expected_correlation
                ),
                filter_period_event.s()
            )
            ch.apply_async()
