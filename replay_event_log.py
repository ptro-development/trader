import sys
import json

from optparse import OptionParser
from trader.celery import app


def is_min_required_percentage(event, percentage):
    return percentage <= event["percentage"]


def send_to_process_incoming_event(line, percentage):
    log_line = json.loads(line.rstrip())
    send_flag = False
    if is_min_required_percentage(log_line["event"], percentage):
        app.send_task(
            "trader.event_consumer_single_tasks.process_incoming_event",
            args=(log_line["event"], ),
            queue="incoming_trade_events")
        send_flag = True
    return send_flag


def send_to_process_trade_candidate(line):
    send_flag = False
    start, first_event_str, second_event_str, found_relation_key = \
        line.rstrip().split("|")
    first_event = json.loads(first_event_str)
    second_event = json.loads(second_event_str)
    app.send_task(
        "trader.event_consumer_multi_tasks.process_filtered_trade_candidate",
        args=(first_event, second_event, found_relation_key),
        queue="trade_events_multi")
    send_flag = True
    return send_flag


def replay(
        log_path, log_type_key,
        line_numbers, start_position, percentage):
    data = None
    sent_to_queue_counter = 0
    with open(log_path, "r") as fd:
        data = fd.readlines()
    if line_numbers == -1:
        line_numbers = len(data)
    for index, line in enumerate(data[start_position: line_numbers]):
        try:
            send_flag = False
            if log_type_key == "1":
                send_flag = send_to_process_incoming_event(line, percentage)
            elif log_type_key == "2":
                send_flag = send_to_process_trade_candidate(line)
            if send_flag:
                sent_to_queue_counter += 1
            print "Processed " + str(start_position + index + 1) + " lines where " + \
                str(sent_to_queue_counter) + " were sent to celery."
        except Exception, e:
            print line.rstrip()
            print e


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-l", "--log",
        dest="log", help="file path to event log file")
    parser.add_option(
        "-k", "--log_type_key",
        dest="log_type_key",
        default="1",
        help="log type 1 = trader.event_consumer_single_tasks.process_incoming_event 2 = trader.event_consumer_multi_tasks.process_trade_candidate")  # noqa
    parser.add_option(
        "-c", "--line_numbers",
        default="-1",
        dest="line_numbers",
        help="amount of lines to replay from log, default is -1 to replay all lines")  # noqa
    parser.add_option(
        "-s", "--log_line_start_position",
        default="0",
        dest="start_position",
        help="line start position in log")
    parser.add_option(
        "-p", "--min_required_percentage",
        default="40.0",
        dest="min_required_percentage",
        help="only lines with minimal required percentage are going to be sent for processing")  # noqa
    return parser.parse_args(argv)


def main():
    options, args = parse_options(sys.argv[1:])
    if not options.log:
        sys.stderr.write("Not provided path to log file.\n")
        return 1
    if not options.line_numbers:
        sys.stderr.write("Not provided numbers of lines to replay.\n")
        return 1
    replay(
        options.log,
        options.log_type_key,
        int(options.line_numbers),
        int(options.start_position),
        float(options.min_required_percentage))
    return 0

if __name__ == "__main__":
    sys.exit(main())
