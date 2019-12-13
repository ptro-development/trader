import os
from datetime import timedelta

# non celery configuration options
LOG_PATH = "/tmp/trading.log"
EVENT_LOG_PATH = "/tmp/events.log"
SAMPLES_LIBRARY_FILE = "samples_library.json"
SAMPLES_LIBRARY_FOCUSED_RELATIONS_FILE = "focused_facts_relations.json"  # noqa
REPLAY_MODE = True

# To reduce samples in library
MIN_NUMBER_CORRELATIONS_POSITIONS = 2
MIN_NUMBER_RELATION_APPEARANCES = 1

# To control processing flow, keys are functions names
SHOULD_TERMINATE = {
    "process_incoming_event": False
}

# Trade heart beat in seconds
TRADE_HEART_BEAT = 30.0

CELERY_INCLUDE = (
    'trader.consumer_tasks',
    'trader.consumer_logger_tasks',
    'trader.sample_matching_tasks',
    'trader.event_logger_tasks',
    'trader.event_consumer_single_tasks',
    'trader.event_consumer_multi_tasks',
    'trader.sample_library_tasks'
)

"""
BROKER_URL = 'amqp://guest@localhost//'
CELERY_RESULT_BACKEND = 'amqp://guest@localhost//'
"""
# for redis
BROKER_TRANSPORT_OPTIONS = {'fanout_patterns': True}
master_trader_active = os.environ.get('MASTER_TRADER_ACTIVE')

if master_trader_active:
    BROKER_URL = 'redis://192.168.0.16:6379/0'
    CELERY_RESULT_BACKEND = 'redis://192.168.0.16:6379/0'
else:
    BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

CELERY_TASK_RESULT_EXPIRES = 3600
# CELERY_TIMEZONE = 'Europe/London'
cELERY_TIMEZONE = 'UTC'

# Default queue
CELERY_QUEUES = {
    'celery': {
        "exchange": "default",
        "binding_key": "celery",
    }
}

# Consuming of incoming trade data
# WORKER: consumer
CELERY_QUEUES.update({
    'incoming_trade_data': {
        'exchange': 'incoming_trade_data',
        'routing_key': 'incoming_trade_data',
    }
})

CELERY_ROUTES = {
    'trader.consumer.process_incoming_trade_log': {
        'queue': 'incoming_trade_data',
        'routing_key': 'incoming_trade_data'
    },
    'trader.consumer.process_incoming_collapsed_trade_log': {
        'queue': 'incoming_trade_data',
        'routing_key': 'incoming_trade_data'
    }
}

# Logging of incoming trade data
# WORKER: consumer_logger
CELERY_QUEUES.update({
    'incoming_trade_data_logger': {
        'exchange': 'incoming_trade_data_logger',
        'routing_key': 'incoming_trade_data_logger',
    }
})

CELERY_ROUTES.update({
    'trader.consumer_logger_tasks.flush_log': {
        'queue': 'incoming_trade_data_logger',
        'routing_key': 'incoming_trade_data_logger'
    },
    'trader.consumer_logger_tasks.write_log': {
        'queue': 'incoming_trade_data_logger',
        'routing_key': 'incoming_trade_data_logger'
    }
})

CELERYBEAT_SCHEDULE = {
    'store_incoming_trade_data': {
        'task': 'trader.consumer_logger_tasks.flush_log',
        'schedule': timedelta(seconds=30),
        'args': (),
        'options': {'queue': 'incoming_trade_data_logger'}
    },
}

# Logging of events from sample matching
# WORKER: event_logger
CELERY_QUEUES.update({
    'event_logger': {
        'exchange': 'event_logger',
        'routing_key': 'event_logger',
    }
})

CELERY_ROUTES.update({
    'trader.event_logger_tasks.flush_log': {
        'queue': 'event_logger',
        'routing_key': 'event_logger'
    },
    'trader.event_logger_tasks.write_log': {
        'queue': 'event_logger',
        'routing_key': 'event_logger'
    }
})

CELERYBEAT_SCHEDULE.update({
    'store_events_data': {
        'task': 'trader.event_logger_tasks.flush_log',
        'schedule': timedelta(seconds=30),
        'args': (),
        'options': {'queue': 'event_logger'}
    },
})


# Processing incoming event only one at the time
# WORKER: event_consumer_single
CELERY_QUEUES.update({
    'incoming_trade_events': {
        'exchange': 'incoming_trade_events',
        'routing_key': 'incoming_trade_events',
    }
})

CELERY_ROUTES.update({
    'trader.event_consumer_single_tasks.process_incoming_event': {
        'queue': 'incoming_trade_events',
        'routing_key': 'incoming_trade_events'
    },
    'trader.event_consumer_single_tasks.collect_second_event_match_status': {
        'queue': 'incoming_trade_events',
        'routing_key': 'incoming_trade_events'
    },
    'trader.event_consumer_single_tasks.update_second_event_match_status': {
        'queue': 'incoming_trade_events',
        'routing_key': 'incoming_trade_events'
    }
})

# Processing multiple events at the time
# WORKER: event_consumer_multi
CELERY_QUEUES.update({
    'trade_events_multi': {
        'exchange': 'trade_events_multi',
        'routing_key': 'trade_events_multi',
    }
})

CELERY_ROUTES.update({
    'trader.event_consumer_multi_tasks.process_filtered_trade_candidate': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.process_trade_candidate': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.event_initial_match': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.register_trade': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.test_event_in_facts_relations': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.trades_heart_beat': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    },
    'trader.event_consumer_multi_tasks.update_registered_trades': {
        'queue': 'trade_events_multi',
        'routing_key': 'trade_events_multi'
    }
})

CELERYBEAT_SCHEDULE.update({
    'trades_heart_beat': {
        'task': 'trader.event_consumer_multi_tasks.trades_heart_beat',
        'schedule': timedelta(seconds=TRADE_HEART_BEAT),
        'args': (),
        'options': {'queue': 'trade_events_multi'}
    },
})

# Samples matching engine
# WORKER: sample_matching
CELERY_QUEUES.update({
    'incoming_trade_data_sample_matching': {
        'exchange': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching',
    }
})

CELERY_ROUTES.update({
    'trader.sample_matching_tasks.filter_period_event': {
        'queue': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching'
    },
    'trader.sample_matching_tasks.test_sample_library_match': {
        'queue': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching'
    },
    'trader.sample_matching_tasks.find_sample_correlations': {
        'queue': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching'
    },
    'trader.sample_matching_tasks.find_first_sample_correlations': {
        'queue': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching'
    },
    'trader.sample_matching_tasks.find_sample_correlations_no_limits': {
        'queue': 'incoming_trade_data_sample_matching',
        'routing_key': 'incoming_trade_data_sample_matching'
    },
})
