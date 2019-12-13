from __future__ import absolute_import
from celery.signals import celeryd_after_setup
from celery.signals import worker_shutdown

from trader.celery import app

LOG_FD = None


@celeryd_after_setup.connect
def init_logger(sender, instance, **kwargs):
    global LOG_FD
    if sender.startswith("event_logger@"):
        init_log_file_descriptor(app.conf.EVENT_LOG_PATH)
        print "Log %s initialised." % app.conf.EVENT_LOG_PATH


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
