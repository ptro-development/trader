import sys
import time

from optparse import OptionParser

from trader.celery import app


def replay(log_path, celery_queue, line_numbers):
    trade_data = None
    delta = 0
    with open(log_path, "r") as fd:
        trade_data = fd.readlines()
    for index, line in enumerate(trade_data[0:line_numbers]):
        print "Line " + str(index) + " was sent to queue " + str(celery_queue)
        time_r, data = line.split(" ", 1)
        time_r = float(time_r)
        if index == 0:
            delta = time.time() - time_r
        app.send_task(
            'trader.consumer_tasks.process_incoming_trade_log',
            args=("%s %s" % (time_r + delta, data), ),
            queue=celery_queue)


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-l", "--log",
        dest="log", help="file path to trade log file")
    parser.add_option(
        "-q", "--queue",
        default="incoming_trade_data", dest="queue", help="celery queue name")
    parser.add_option(
        "-c", "--line_numbers",
        dest="line_numbers", help="amount of lines to replay from log")
    return parser.parse_args(argv)


def main():
    options, args = parse_options(sys.argv[1:])
    if not options.log:
        sys.stderr.write("Not provided path to log file.\n")
        return 1
    if not options.line_numbers:
        sys.stderr.write("Not provided numbers of lines to replay.\n")
        return 1
    replay(options.log, options.queue, int(options.line_numbers))
    return 0

if __name__ == "__main__":
    sys.exit(main())
