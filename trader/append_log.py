from __future__ import absolute_import
from trader.celery import app
from celery.signals import worker_shutdown

LOG_FD = None


def init_log_file_descriptor(log_path):
    global LOG_FD
    if not LOG_FD:
        LOG_FD = open(log_path, "a")


@worker_shutdown.connect
def clean_up(**kwargs):
    global LOG_FD
    print "Cleaning ..."
    if LOG_FD:
        LOG_FD.close()


@app.task(ignore_result=True)
def flush_log():
    global LOG_FD
    if LOG_FD:
        LOG_FD.flush()


@app.task(ignore_result=True)
def write_log(line):
    global LOG_FD
    if LOG_FD:
        LOG_FD.write(line)
