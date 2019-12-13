#!/bin/bash
set -e

export C_FORCE_ROOT=TRUE
celery -A trader purge -f
celery -A trader worker -l info -c 8
