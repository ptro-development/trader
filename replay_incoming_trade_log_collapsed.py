import sys
import datetime

from optparse import OptionParser

from trader.celery import app
from trader.libs.trade_data import TradeData, fill_empty_gaps


def replay(log_path, celery_queue, rescale_period=60 * 5, sample_size=25):
    trade_data = TradeData(
        log_path=log_path,
        rescale_period=rescale_period)
    start_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_start)).strftime('%d-%m-%Y %H:%M:%S')
    end_date = datetime.datetime.fromtimestamp(
        int(trade_data.log_end)).strftime('%d-%m-%Y %H:%M:%S')
    print "\nLog start from %s to %s" % (start_date, end_date)
    print "Data rescaling of %s records done with rescale_period %s seconds resulting into %s records. %s empty data records." % (  # noqa
        trade_data.line_counter,
        rescale_period,
        len(trade_data.times),
        trade_data.counters.count(0))
    print "Empty data gaps were filled with average of two values around gap " + \
        "or previous value if not present."
    fill_empty_gaps(trade_data.prices, trade_data.counters)
    # do not fill these gaps as these need to be without modification
    # fill_empty_gaps(trade_data.trades, trade_data.counters)
    for index in range(0, len(trade_data.counters)-sample_size):
        print "Collapsed data position " + str(index) + \
            " was sent to queue " + str(celery_queue)
        app.send_task(
            'trader.consumer_tasks.process_incoming_collapsed_trade_log',
            args=(
                trade_data.prices[index: index + sample_size],
                trade_data.trades[index: index + sample_size],
                trade_data.times[index]),
            queue=celery_queue)


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-l", "--log",
        dest="log", help="file path to trade data log file")
    parser.add_option(
        "-q", "--queue",
        default="incoming_trade_data", dest="queue", help="celery queue name")
    parser.add_option(
        "-r", "--rescale_period",
        default="300", dest="rescale_period", help="rescale period in seconds")
    parser.add_option(
        "-s", "--sample_size",
        default="25", dest="sample_size", help="sample size")
    return parser.parse_args(argv)


def main():
    options, args = parse_options(sys.argv[1:])
    if not options.log:
        sys.stderr.write("Not provided path to log file.\n")
        return 1
    replay(
        options.log, options.queue,
        int(options.rescale_period), int(options.sample_size))
    return 0

if __name__ == "__main__":
    sys.exit(main())
