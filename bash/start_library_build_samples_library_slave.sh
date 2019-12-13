#!/bin/bash
set -e

export C_FORCE_ROOT=TRUE
export MASTER_TRADER_ACTIVE=TRUE
celery -A trader purge -f
celery -A trader worker -l info -c 2
