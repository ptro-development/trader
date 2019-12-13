from __future__ import absolute_import
from celery.signals import celeryd_after_setup

from celery import chord, chain

from trader.celery import app
from trader.event_logger_tasks import write_log
from trader.event_consumer_multi_tasks import process_trade_candidate
from trader.libs.event import dict_to_event
from trader.flow_control import should_terminate
from trader.libs.relations import remove_facts_relations, load_facts_relations

import json
import time

trade_candidates = []
focus_keys = []


@celeryd_after_setup.connect
def init_trader(sender, instance, **kwargs):
    global focus_keys
    if not sender.startswith("event_consumer_single@"):
        return
    facts_relations = load_facts_relations(
        app.conf.SAMPLES_LIBRARY_FOCUSED_RELATIONS_FILE)
    print "Loaded samples relations from %s." % \
        app.conf.SAMPLES_LIBRARY_FOCUSED_RELATIONS_FILE
    remove_facts_relations(
        facts_relations,
        True,
        app.conf.MIN_NUMBER_RELATION_APPEARANCES)
    print "Removing samples relations which do not have at least %s appearances." % \
        app.conf.MIN_NUMBER_RELATION_APPEARANCES
    focus_keys = [int(key) for key in facts_relations.keys()]
    focus_keys.sort()


def is_first_event_match(
        trade_candidates, event, min_required_percentage=100.0):
    result = False
    match = event.is_min_required_percentage(min_required_percentage)
    if match:
        result = True
        for trade in trade_candidates:
            if trade["first_event"].get_dict() == event.get_dict():
                result = False
                break
    return result


def prepare_second_event_matches_tasks(
        trade_candidates, event,
        focus_key, min_relation_appearance,
        min_required_correlation_percentage):
    tasks = []
    for trade in trade_candidates:
        if trade["first_event"].get_dict() != event \
                and not trade["second_event_found"] in ["expired"]:
            tasks.append(
                process_trade_candidate.s(
                    trade["first_event"].get_dict(),
                    event,
                    focus_key,
                    min_relation_appearance,
                    min_required_correlation_percentage))
    return tasks


@app.task(bind=True)
def update_second_event_match_status(self, status, trade_candidates, index):
    if status.ready():
        print "status.result: " + str(status.result)
        trade_candidates[index]["second_event_found"] = str(status.result)
        return
    else:
        print "Not ready yet ... " + str(status)
    raise self.retry(countdown=0.5, max_retries=2)


@app.task
def collect_second_event_match_status(results, trade_candidates):
    statuses = results
    # print "match :" + str(results)
    # print "match results:" + str(type(results))
    if not isinstance(results, list):
        statuses = [results]
    print "run update: " + str(len(statuses)) + " " + str(statuses)
    for index, status in enumerate(statuses):
        ch = chain(
            update_second_event_match_status.s(
                status, trade_candidates, index))
        ch.apply_async()


def mark_expired_trade_candidates(
        trade_candidates, current_event_epoch, max_age_limit):
    for index, trade in enumerate(trade_candidates):
        if current_event_epoch - trade["first_event"].get_relative_match_position() > max_age_limit:  # noqa
            trade_candidates[index]["second_event_found"] = "expired"


@app.task(ignore_result=True)
def process_incoming_event(event):
    global trade_candidates, focus_keys
    line = json.dumps({"time": time.time(), "event": event}) + "\n"
    write_log.apply_async((line,))

    if should_terminate("process_incoming_event"):
        return

    event_inst = dict_to_event(event)

    # remove old tasks
    max_distance = 500
    rescale_period = 60 * 5
    min_relation_appearance = 1
    max_focus = 80
    min_required_correlation_percentage = 35.0
    limit_in_seconds = max_distance * rescale_period
    mark_expired_trade_candidates(
        trade_candidates,
        event_inst.get_relative_match_position(),
        limit_in_seconds)

    if is_first_event_match(
            trade_candidates, event_inst, min_required_percentage=100.0):
        trade_candidates.append({
            "first_event": event_inst,
            "second_event_found": "inactive"})
        active_trades = len(
            [i for i in trade_candidates if i["second_event_found"] == "active"])  # noqa
        expired_trades = len(
            [i for i in trade_candidates if i["second_event_found"] == "expired"])  # noqa
        inactive_trades = len(
            [i for i in trade_candidates if i["second_event_found"] == "inactive"])  # noqa
        print "Current trades inactive:%s active:%s expired:%s" % \
            (inactive_trades, active_trades, expired_trades)

    for focus_key in focus_keys:
        if focus_key <= max_focus:
            tasks = prepare_second_event_matches_tasks(
                trade_candidates, event, focus_key,
                min_required_correlation_percentage, min_relation_appearance)
    if len(tasks) != 0:
        print "Tasks amount: " + str(len(tasks))
        chord(tasks)(collect_second_event_match_status.s(trade_candidates))
