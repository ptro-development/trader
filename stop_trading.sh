#!/bin/bash
set -e

celery multi stop consumer consumer_logger sample_matching event_logger event_consumer_single event_consumer_multi -A trader
