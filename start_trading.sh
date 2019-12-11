#!/bin/bash
set -e

rm -f *.log
rm -f /tmp/trading.log
rm -f /tmp/events.log
rm -f celerybeat-schedule

celery -A trader purge -f

celery multi start consumer consumer_logger sample_matching event_logger event_consumer_single event_consumer_multi -l info -Q:1 incoming_trade_data -c:consumer 1 -Q:2 incoming_trade_data_logger -c:consumer_logger 1 -Q:3 incoming_trade_data_sample_matching -c:sample_matching 10 -Q:4 event_logger -c:event_logger 1 -Q:5 incoming_trade_events -c:event_consumer_single 1 -Q:6 trade_events_multi -c:event_consumer_multi 4 -A trader -B:2,4,6
