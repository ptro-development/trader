import sys
import time
import datetime
import pickle

from celery import group

from trader.sample_matching_tasks import find_sample_correlations
from trader.sample_matching_tasks import find_sample_correlations_no_limits
from trader.sample_matching_tasks import find_first_sample_correlations
from trader.libs.trade_data import TradeData, fill_empty_gaps
# from trader.libs.trade_data import get_up_patterns_stats_3
# from trader.libs.trade_data import get_grow_sample_percentage_stats_3
# from trader.libs.trade_data import gain_up_estimate
from trader.libs.samples import get_random_samples, print_price_sample, \
    remove_close_samples, add_sample_attributes, save_samples_library, \
    eliminate_samples
from trader.libs.utils import get_chunks_indexes
# from trader.libs.plot import plot_plus_correlations_for_sample

from optparse import OptionParser
# from pylab import show, rcParams


def plot_best_correlation(data, sample, limit=6):
    rcParams['figure.figsize'] = 15, 10
    subplot_num = limit * 100 + 11
    plot_plus_correlations_for_sample(data, sample, subplot_num, limit)


def print_xy(x_label, y_label, data):
    print x_label + "," + y_label
    for x, y in data:
        print str(x) + "," + str(y)


def build_samples_library(
        incoming_log,
        samples_library,
        rescaled_data_log,
        sample_count,
        rescale_period,
        sample_size,
        min_sample_variation,
        acceptable_correlation_coef,
        min_samples_distance):

    trade_data = TradeData(
        log_path=incoming_log,
        rescale_period=rescale_period)

    start_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_start)).strftime('%d-%m-%Y %H:%M:%S')
    end_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_end)).strftime('%d-%m-%Y %H:%M:%S')

    print "\nLog start from %s to %s" % (start_date, end_date)
    print "Data rescaling of %s records done with rescale_period %s seconds resulting into %s records. %s empty data records. Overall %s %% of empty records" % (  # noqa
        trade_data.line_counter,
        rescale_period,
        len(trade_data.times),
        trade_data.counters.count(0),
        trade_data.counters.count(0)/(len(trade_data.times)/100.0))

    """
    sample_sizes, grow_sizes, trade_sizes = get_up_patterns_stats_3(
        trade_data.prices, trade_data.trades,
        trade_data.counters, grow_rounding=4)
    cut_percentage = 0.15
    trade_percentage = 0.25
    gains = gain_up_estimate(
        grow_sizes, trade_sizes, cut_percentage, trade_percentage)
    print "Sample sizes:"
    print_xy(
        "sample_size", "count",
        sorted(sample_sizes.items(), key=lambda x: x[1], reverse=True))
    print "Grow sizes:"
    print_xy(
        "grow_size", "count",
        sorted(grow_sizes.items(), key=lambda x: x[0], reverse=True))
    print "Trade sizes:"
    print_xy(
        "grow_size", "trade_amount",
        sorted(trade_sizes.items(), key=lambda x: x[0], reverse=True))
    print "Gains sizes for cut_percentage %s and trade_percentage %s" % (cut_percentage, trade_percentage)  # noqa
    print_xy(
        "grow_size", "gain",
        sorted(gains.items(), key=lambda x: x[1], reverse=True))
    print "100%% correct estimation, %s rescale period, overall gain %s" % (rescale_period, sum(gains.values())) + \
        " all_up_trades_count " + str(sum(grow_sizes.values()))
    print "Empty data gaps were filled with average of two values around gap " + \
        "or previous value if not present."
    """

    fill_empty_gaps(trade_data.prices, trade_data.counters)

    # There is need to keep this exact as trading is going to be based on it,
    # so keep it commented.
    # fill_empty_gaps(trade_data.trades, trade_data.counters)

    """
    stats_less, stats_same, stats_grow, stats_grow_less = get_grow_sample_percentage_stats_3(  # noqa
        trade_data.prices,
        trade_data.counters, sample_size)
    print_xy(
        "count_less", "count",
        sorted(stats_less.items(), key=lambda x: x[0], reverse=False))
    print_xy(
        "count_same", "count",
        sorted(stats_same.items(), key=lambda x: x[0], reverse=False))
    print_xy(
        "count_grow", "count",
        sorted(stats_grow.items(), key=lambda x: x[0], reverse=False))
    print_xy(
        "stats_grow_less", "count",
        sorted(stats_grow_less.items(), key=lambda x: x[0], reverse=False))
    sys.exit()
    """

    print "Saving rescaled data into %s" % rescaled_data_log
    with open(rescaled_data_log, "w") as fd:
        pickle.dump(trade_data, fd)
    samples = get_random_samples(
        trade_data.prices,
        sample_size, sample_count, min_sample_variation)
    print "Data sampled for sample_count %s and was able to get sample_count %s for sample_size %s (%s minutes) min_sample_variation %s" % (  # noqa
        sample_count, len(samples),
        sample_size,
        sample_size * rescale_period / 60,
        min_sample_variation)
    samples = eliminate_samples(samples, acceptable_correlation_coef)
    print "Similar samples were eliminated, new sample_count is " + str(
        len(samples))

    start_time = time.time()
    # order of samples pased to find_first_sample_correlations
    # does not mather in this case
    g = group(
        # find_first_sample_correlations.s(
        # find_sample_correlations.s(  # noqa
        find_sample_correlations_no_limits.s(  # noqa
            trade_data.prices, samples[ch[0]:ch[1]],
            sample_size,
            acceptable_correlation_coef) for ch in get_chunks_indexes(
                len(samples), 40))
    updated_samples = []
    for results in g().get():
        for result in results:
            updated_samples.append(result)
    print "Sample matching took % seconds" % (time.time() - start_time)

    print "Looking for %s%% correlations finished ...\n" % \
        acceptable_correlation_coef

    sorted_samples_plus = sorted(
        updated_samples,
        key=lambda sample: len(sample["+correlation_positions"]), reverse=True)
    sorted_samples_minus = sorted(
        updated_samples,
        key=lambda sample: len(sample["-correlation_positions"]), reverse=True)

    # actual_window_size = 2 * rescale_period * min_samples_distance
    """
    actual_window_size = rescale_period * sample_size
    print "Removing close samples within window %s (%s minutes)" % \
        (actual_window_size, actual_window_size / 60)
    sorted_samples_plus = remove_close_samples(
        sorted_samples_plus, min_samples_distance)
    sorted_samples_minus = remove_close_samples(
        sorted_samples_minus, min_samples_distance)
    """

    print "Add to samples acceptable_correlation_coef, sample_epoch and rescale_period."  # noqa
    for samples in sorted_samples_plus, sorted_samples_minus:
        map(lambda x: x.update({
            "rescale_period": rescale_period,
            "sample_epoch": trade_data.times[x["sample_position"]],
            "required_correlation": acceptable_correlation_coef}
            ),
            samples)
    print "Enrich samples with its attributes ..."
    add_sample_attributes(sorted_samples_plus)
    add_sample_attributes(sorted_samples_minus)

    print "Add samples positions matches as epochs"
    for samples in sorted_samples_plus, sorted_samples_minus:
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

    print "Add trades amount to samples"
    for samples in sorted_samples_plus, sorted_samples_minus:
        map(
            lambda x: x.update({"sample_data_trades_amount": trade_data.trades[x["sample_position"]: x["sample_position"] + sample_size]}),  # noqa
            samples
        )

    print "Saving result into %s" % samples_library
    save_samples_library(
        samples_library, sorted_samples_plus, sorted_samples_minus)

    print "\nBest prices +correlations"
    for i in range(0, 5):
        print_price_sample(
            trade_data.times,
            sorted_samples_plus[i],
            trade_data.gaps,
            trade_data.trades)
        print

    print "\nBest prices -correlations"
    for i in range(0, 5):
        print_price_sample(
            trade_data.times,
            sorted_samples_minus[i],
            trade_data.gaps,
            trade_data.trades)
        print

    # print "\nPlotting best +correlations"
    # plot_best_correlation(trade_data.prices, sorted_samples_plus[0], limit=6)

    # print "\nPlotting best -correlations"
    # plot_best_correlation(
    #    trade_data.prices, sorted_samples_minus[0], limit=6)

    # show()
    # time.sleep(10)


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-i", "--incoming_trade_log",
        dest="incoming_log", help="file path to load trade data log")
    parser.add_option(
        "-p", "--samples_library",
        default="samples_library.json", dest="samples_library",
        help="file path to save samples library")
    parser.add_option(
        "-l", "--rescaled_data_log",
        default="rescaled_trade_data.pickle", dest="rescaled_data_log",
        help="file path to save re-scaled trade data")
    parser.add_option(
        "-c", "--sample_count",
        default="30000", dest="sample_count", help="sample count")
    parser.add_option(
        "-r", "--rescale_period",
        default="300", dest="rescale_period", help="rescale period in seconds")
    parser.add_option(
        "-s", "--sample_size",
        default="25", dest="sample_size", help="sample size")
    parser.add_option(
        "-v", "--min_sample_variation",
        default="8", dest="min_sample_variation",
        help="minimal sample variation")
    parser.add_option(
        "-a", "--acceptable_correlation_coef",
        default="0.975", dest="acceptable_correlation_coef",
        help="acceptable correlation coefficient")
    parser.add_option(
        "-m", "--min_samples_distance",
        default="10", dest="min_samples_distance",
        help="minimal distance of samples")
    return parser.parse_args(argv)


def main():
    options, args = parse_options(sys.argv[1:])
    if not options.incoming_log:
        sys.stderr.write("Not provided path to trade data log file.\n")
        return 1
    build_samples_library(
        options.incoming_log,
        options.samples_library,
        options.rescaled_data_log,
        int(options.sample_count),
        float(options.rescale_period),
        int(options.sample_size),
        float(options.min_sample_variation),
        float(options.acceptable_correlation_coef),
        int(options.min_samples_distance)
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
