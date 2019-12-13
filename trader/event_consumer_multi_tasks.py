from __future__ import absolute_import
from celery.signals import celeryd_after_setup

from trader.celery import app
from trader.libs.samples import load_samples_library, \
    remove_samples_min_correlation
# from trader.libs.relations import get_closest_focus_index, get_correlation_facts_relation_key_dict, \  # noqa
from trader.libs.relations import get_correlation_facts_relation_key_dict, \
    remove_facts_relations, load_facts_relations
from trader.libs.event import dict_to_event, \
    get_events_correlation_facts_relation_key
# from trader.libs.trade_data import get_closest_period_index_down
from trader.libs.single_task import REDIS_CLIENT, only_one
from trader.libs.trade import Trade, get_registered_trades, \
    set_registered_trades, delete_registered_trades

import json

facts_relations = None
samples_library = None
rescale_period = None


@celeryd_after_setup.connect
def init_trader(sender, instance, **kwargs):
    global facts_relations, samples_library, rescale_period
    if not sender.startswith("event_consumer_multi@"):
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
    samples_library = load_samples_library(app.conf.SAMPLES_LIBRARY_FILE)
    print "Samples were loaded from " + str(app.conf.SAMPLES_LIBRARY_FILE)
    samples_library = remove_samples_min_correlation(
        samples_library,
        app.conf.MIN_NUMBER_CORRELATIONS_POSITIONS)
    print "Removing samples which do not have at least %s correlation matches." % \
        app.conf.MIN_NUMBER_CORRELATIONS_POSITIONS
    rescale_period = samples_library[0]["rescale_period"]
    print "Rescale period " + str(rescale_period)
    delete_registered_trades(REDIS_CLIENT)
    print "Deleted registered_trades from redis."


""""
def test_events_exist_in_facts_relations(first_event, second_event, relation):
    first_event_sample_position_str = first_event.get_correlation_sign() + \
        str(first_event.get_sample_epoch())
    second_event_sample_position_str = second_event.get_correlation_sign() + \
        str(second_event.get_sample_epoch())
    return relation["first_fact_sample_position"] == first_event_sample_position_str \
        and relation["second_fact_sample_position"] == second_event_sample_position_str  # noqa
"""


def test_events_exist_in_facts_relations(events_relation_key, relation_key):
    return events_relation_key == relation_key


def test_events_happened_min_relation_times(
        relation_value, min_relation_appearance):
    return relation_value >= min_relation_appearance


def test_events_left_to_right_order(first_event, second_event):
    return first_event.get_relative_match_position() < second_event.get_relative_match_position()  # noqa


def test_correlation_position_relation_left_to_right_order(relation):
    return relation["order"] == "True"


def test_events_same_max_focus_distance(relation, distance_str):
    return relation["distance"] == distance_str


@app.task
def test_event_in_facts_relations(
        facts_relations,
        rescale_period,
        first_event, second_event,
        min_relation_appearance, focus_key):
    """ To test
            - sample events exist in facts_relations
            - sample events happened in past at least min_relation_appearance times  # noqa
            - sample events are in right time order (from left to right)
            - sample events happened within max_focus distance
    """
    found = False
    found_relation_key = None
    for relation_key, relation_value in facts_relations[str(focus_key)].items():  # noqa
        relation_dict = get_correlation_facts_relation_key_dict(
            relation_key)
        events_relation_key = get_events_correlation_facts_relation_key(  # noqa
            first_event, second_event, focus_key)
        tests = []
        # test sample events exist in facts_relations
        tests.append(
            test_events_exist_in_facts_relations(
                # first_event, second_event, relation))
                events_relation_key, relation_key))
        print "A: " + str(tests) + " " + events_relation_key + \
            "_" + str(second_event.get_percentage())
        if all(tests):
            # test sample events happened in past at least min_relation_appearance times  # noqa
            tests.append(
                test_events_happened_min_relation_times(
                    relation_value, min_relation_appearance))
        else:
            continue
        print "AA: " + str(tests) + " " + events_relation_key + \
            "_" + str(second_event.get_percentage())
        if all(tests):
            # test sample events are in right time order (from left to right)  # noqa
            tests.append(
                test_events_left_to_right_order(
                    first_event, second_event))
        else:
            continue
        print "AAA: " + str(tests) + " " + events_relation_key + \
            "_" + str(second_event.get_percentage())
        if all(tests):
            # test correlation position relation is from left to right  # noqa
            # this might need review later as it can be very limiting  # noqa
            tests.append(
                test_correlation_position_relation_left_to_right_order(  # noqa
                    relation_dict))
        else:
            continue
        print "AAAA: " + str(tests) + " " + events_relation_key + \
            "_" + str(second_event.get_percentage())
        """
        if all(tests):
            # test sample events happened at same max_focus distance  # noqa
            distance = get_closest_period_index_down(
                first_event.get_relative_match_position(),
                second_event.get_relative_match_position(),
                rescale_period)
            distance_str = str(get_closest_focus_index(
                distance, focus_key))
            tests.append(
                test_events_same_max_focus_distance(
                    relation_dict, distance_str))
        else:
            continue
        print "AAAAA: " + str(tests) + " " + events_relation_key
        """
        if all(tests):
            found = True
            found_relation_key = relation_key
            break
    return found, found_relation_key


@app.task
def event_initial_match(
        facts_relations, rescale_period,
        first_event, second_event, min_relation_appearance,
        focus_key, min_required_correlation_percentage):
    found_relation_key = None
    match = "inactive"
    # events_relation_key = get_events_correlation_facts_relation_key(
    #    first_event, second_event, 25) + \
    #    "_" + str(second_event.get_percentage())
    # TODO
    # second_sample_match_allowed_gaps(self, event):
    # second_sample_test_minimum_allowed_trading_amount(self, event):
    tests = []
    tests.append(
        second_event.is_min_required_percentage(
            min_required_correlation_percentage))
    # print "C: " + str(events_relation_key)
    if all(tests):
        tests.append(second_event.is_sample_attribute_up())
        # print "CC: " + str(events_relation_key)
        if all(tests):
            found, found_relation_key = test_event_in_facts_relations(
                facts_relations,
                rescale_period,
                first_event, second_event,
                min_relation_appearance, focus_key)
            tests.append(found)
            # print "CCC: " + str(events_relation_key)
            print "second_tests: " + str(tests)
            if all(tests):
                match = "active"
                # print "CCCC: " + str(events_relation_key)
                process_filtered_trade_candidate(
                    first_event,
                    second_event,
                    found_relation_key)
    print "match: " + str(match)
    return match


@app.task(ignore_result=True, name='trader.event_consumer_multi_tasks.trades_heart_beat')  # noqa
@only_one(key="SingleTask", timeout=20)
def trades_heart_beat():
    registered_trades = get_registered_trades(REDIS_CLIENT)
    for index, trade in enumerate(registered_trades):
        registered_trades[index].run_trade_heart_beat()
    set_registered_trades(REDIS_CLIENT, registered_trades)


def print_trade_candidate(
        prefix, first_event, second_event, found_relation_key):
    print str(prefix) + "|trade candidate|" + str(first_event) + "|" + \
        str(second_event) + "|" + str(found_relation_key)


def update_registered_trades(
        registered_trades, first_event, second_event, found_relation_key):
    updated_flag = False
    for index, trade in enumerate(registered_trades):
        if trade.is_new_event_valid_for_addition(first_event, second_event):
            updated_flag = True
            print "Trade has been already registered ..."
            print_trade_candidate(
                "save_last_event", first_event,
                second_event, found_relation_key)
            registered_trades[index].save_last_event(second_event)
            break
    return updated_flag


def register_trade(
        registered_trades, first_event, second_event, found_relation_key):
    flag = True
    for trade in registered_trades:
        if not trade.is_new_event_valid_for_registration(
                first_event, second_event):
            flag = False
            break
    if flag:
        trade = Trade(
            first_event, second_event,
            found_relation_key, app.conf.TRADE_HEART_BEAT)
        registered_trades.append(trade)
        print_trade_candidate(
            "Register trade", first_event,
            second_event, found_relation_key)
        trade.print_second_events("register_trade")
        print "Registered trades count %s" % len(registered_trades)
    return flag


@app.task(ignore_result=True, name='trader.event_consumer_multi_tasks.process_filtered_trade_candidate')  # noqa
@only_one(key="SingleTask", timeout=20)
def process_filtered_trade_candidate(
        first_event, second_event, found_relation_key):
    if isinstance(first_event, dict):
        first_event = dict_to_event(first_event)
    if isinstance(second_event, dict):
        second_event = dict_to_event(second_event)
    print "process_filtered_trade_candidate|" + json.dumps(first_event.get_dict()) + "|" + \
        json.dumps(second_event.get_dict()) + "|" + str(found_relation_key)
    registered_trades = get_registered_trades(REDIS_CLIENT)
    updated_flag = update_registered_trades(
        registered_trades, first_event, second_event, found_relation_key)
    if not updated_flag:
        register_trade(
            registered_trades, first_event, second_event, found_relation_key)
    set_registered_trades(REDIS_CLIENT, registered_trades)


@app.task
def process_trade_candidate(
        first_event, second_event, focus_key,
        min_relation_appearance, min_required_correlation_percentage=35.0):
    global facts_relations, rescale_period, registered_trades
    first_event_inst = dict_to_event(first_event)
    second_event_inst = dict_to_event(second_event)
    return event_initial_match.apply_async((
        facts_relations,
        rescale_period,
        first_event_inst,
        second_event_inst,
        min_relation_appearance,
        focus_key,
        min_required_correlation_percentage,))
