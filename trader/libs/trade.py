from __future__ import absolute_import

from trader.libs.event import dict_to_event


import time
import json
import random


class Trade(object):

    trades_states = {
        "new": "trade registered",
        "unexpected": "trade is not progressing as planed",
        "expected": "trade is progressing as planed",
        "finished": "trade finished",
        "wrong": "trade should not start"
    }

    price_states = {
        "start": "starting value",
        "missing": "last value was not provided",
        "stale": "last value did not change",
        "up": "last value was bigger than previous",
        "down": "last value was smaller than previous",
    }

    alert_states = {
        "ok": "expected progress of trade",
        "warning": "unexpected progress of trade in allowed range",
        "critical": "unexpected progress of trade above allowed range",
    }

    def __init__(
            self, first_event, second_event,
            found_relation_key, trade_heart_beat, json_data=None):
        if not json_data:
            self.randint = random.randint(0, 1000000)
            self.first_event = first_event
            self.next_events = [second_event]
            self.found_relation_key = found_relation_key
            self.trade_heart_beat = trade_heart_beat
            self.buy_sell_flag = True
            self.state = {
                "trade_state": "new",
                "price_state": "start",
                "alert_state": "ok"}
            init_time = time.time()
            self.event_save_time = [init_time]
            self.status_check_ticks = [init_time]
            """
            One buy record
                {
                    "amount": amount_number,
                    "time:" epoch_time,
                    "price": price_when_buying
                }
            """
            """
            One buy record
                {
                    "amount": amount_number,
                    "time:" epoch_time,
                    "price": price_when_selling
                }
            """
            self.buy_data = {
                "records": [],
                "asked_price": None,
                "start": None,
                "stop": None,
                "target_amount": None,
                "target_amount_per_heart_beat": None
            }
            self.sell_data = {
                "records": [],
                "start": None,
                "stop": None,
                "target_amount": None,
            }
            self._compute_buy_sell_attributes(second_event, 0.2)
        else:
            self.randint = json_data["randint"]
            self.first_event = dict_to_event(json_data["first_event"]["data"])
            self.next_events = []
            for event in json_data["next_events"]:
                self.next_events.append(
                    dict_to_event(event["data"]))
            self.found_relation_key = json_data["found_relation_key"]
            self.trade_heart_beat = json_data["trade_heart_beat"]
            self.buy_sell_flag = json_data["buy_sell_flag"]
            self.state = json_data["state"]
            self.event_save_time = json_data["event_save_time"]
            self.status_check_ticks = json_data["status_check_ticks"]
            self.buy_data = json_data["buy_data"]
            self.sell_data = json_data["sell_data"]

    def _compute_buy_sell_attributes(
            self, event, buy_target_amount_trade_percentage=0.20):
        percentage = event.get_dict()["percentage"]
        buy_portion = 100.0 - percentage
        if buy_portion == 0.0:
            self.state["trade_state"] = "wrong"
        else:
            sample_lenght = len(event.get_sample()["sample_data"])
            whole_sample_time = sample_lenght * self.get_rescale_period()

            # BUY attributes
            self.buy_data["start"] = time.time()
            self.buy_data["end"] = self.buy_data["start"] + \
                (buy_portion / 100.0 * whole_sample_time / 2.0)

            start_index = int(percentage / 100.0 * sample_lenght)
            end_index = (sample_lenght - start_index) / 2

            self.buy_data["target_amount"] = buy_target_amount_trade_percentage * sum(  # noqa
                event.get_sample()["sample_data_trades_amount"][start_index:end_index])  # noqa
            buy_steps = (self.buy_data["end"] - self.buy_data["start"]) / self.trade_heart_beat  # noqa
            self.buy_data["target_amount_per_heart_beat"] = self.buy_data["target_amount"] / buy_steps  # noqa
            self.buy_data["asked_price"] = self.get_last_price()

            # SELL attributes
            self.sell_data["start"] = self.buy_data["end"]
            self.sell_data["end"] = self.sell_data["start"] + \
                (buy_portion / 100.0 * whole_sample_time / 2.0)

            print "_compute_buy_sell_attributes:" + str(self.get_start_events_samples_signatures()) + "|" + \
                str(self.buy_data) + "|" + str(self.sell_data)

    def get_last_price(self):
        # TODO: get last market price not averaged and combine it together
        # with averaged to come with "right" price
        return self.next_events[-1].get_dict()["incoming_price_data"][-1]

    def _update_sell_atrributes(self):
        if self.sell_data["target_amount"] is None:
            self.sell_data["target_amount"] = self.buy_data["target_amount"]

    def run_trade_heart_beat(self):
        if self.state["trade_state"] in ["new", "expected"]:
            self.update_status()
        if self.state["trade_state"] in ["finished", "wrong"]:
            # do not do anything ...
            return
        elif self.state["trade_state"] in ["expected", "new"]:
            current_time = time.time()
            if current_time >= self.buy_data["start"] and current_time <= self.buy_data["end"] \
                    and self.buy_data["target_amount"] > 0.0:  # noqa
                # buy as planned :)
                self.buy()
            elif current_time >= self.sell_data["start"] and current_time <= self.sell_data["end"] \
                    and self.sell_data["target_amount"] > 0.0:  # noqa
                # sell as planned :)
                self._update_sell_atrributes()
                self.sell()
            else:
                # TODO
                pass
        elif self.state["trade_state"] == "unexpected":
            # something went badly start selling :(
            if self.sell_data["target_amount"] > 0.0:
                self._update_sell_atrributes()
                self.sell()
        else:
            # do not do anything ...
            pass

    def buy(self):
        current_time = time.time()
        amount = self.buy_data["target_amount"]
        if self.buy_data["target_amount"] >= self.buy_data["target_amount_per_heart_beat"]:  # noqa
            amount = self.buy_data["target_amount_per_heart_beat"]
        self.buy_data["records"].append({
            "time": current_time,
            "amount": amount,
            "price": self.buy_data["asked_price"],   # noqa TODO: based on actual market price
        })
        print "buy:Buying " + str(amount)
        self.buy_data["target_amount"] -= amount

    def sell(self):
        current_time = time.time()
        index = len(self.sell_data["records"])
        price = self.sell_data["records"][index]["price"]
        amount = self.sell_data["records"][index]["amount"]
        self.buy_data["records"].append({
            "time": current_time,
            "amount": amount,
            "price": price,  # noqa TODO: based on actual market price
        })
        print "sell:Selling " + str("amount")
        self.sell_data["target_amount"] -= amount

    def is_event_saved_already(self, new_event):
        present_flag = False
        new_event_dict = new_event.get_dict()
        for event in self.next_events:
            if new_event_dict == event.get_dict():
                present_flag = True
                break
        return present_flag

    def is_initial_events_sample_signature_match(
            self, first_event, second_event):
        return first_event.get_sample_signature() == self.first_event.get_sample_signature() \
            and second_event.get_sample_signature() == self.next_events[0].get_sample_signature()  # noqa

    def is_last_second_event_match(self, new_event):
        return new_event.get_dict() == self.next_events[-1].get_dict()

    def is_last_second_event_too_far_in_past(self, new_event, max_limit=2.5):
        return new_event.get_dict()["relative_match_position"] - \
            self.next_events[-1].get_dict()["relative_match_position"] > \
            max_limit * self.get_rescale_period()

    def is_new_event_valid_for_addition(self, first_event, second_event):
        return self.is_initial_events_sample_signature_match(first_event, second_event) and \
            not self.is_last_second_event_too_far_in_past(second_event) and \
            not self.is_event_saved_already(second_event)

    def is_events_overlap(self, second_event):
        sample_lenght = len(self.first_event.get_sample()["sample_data"])
        new_trade = [
            int(second_event.get_dict()["relative_match_position"]),
            int(second_event.get_dict()["relative_match_position"] + self.get_rescale_period() * sample_lenght)]  # noqa
        current_trade = [
            int(self.next_events[0].get_dict()["relative_match_position"]),
            int(self.next_events[0].get_dict()["relative_match_position"] + self.get_rescale_period() * sample_lenght)]  # noqa
        return max(0, min(new_trade[1], current_trade[1]) - max(new_trade[0], current_trade[0])) > 0  # noqa

    def is_new_event_valid_for_registration(self, first_event, second_event):
        return not self.is_initial_events_sample_signature_match(first_event, second_event) or \
            (not self.is_events_overlap(second_event) and
                self.is_initial_events_sample_signature_match(
                    first_event, second_event))

    def get_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def get_initial_second_event_sample_signature(self):
        return self.next_events[0].get_sample_signature()

    def get_start_events_samples_signatures(self):
        return (
            self.first_event.get_sample_signature(),
            self.get_initial_second_event_sample_signature())

    def get_rescale_period(self):
        return self.next_events[0].get_rescale_period()

    def print_second_events(self, prefix):
        trade_str = prefix + "|" + str(self.get_start_events_samples_signatures()) + "|" \
            + str(self.found_relation_key) + "|" + str(self.randint) + "|"
        for e in self.next_events:
            trade_str += str(e.get_dict()["relative_match_position"]) + \
                "|" + str(e.get_dict()["percentage"]) + "|"
        print trade_str

    # this should run as single task
    def save_last_event(self, event):
        self.next_events.append(event)
        self.event_save_time.append(time.time())
        last_price = event.get_dict()["incoming_price_data"][-1]
        previous_price = self.next_events[-1].get_dict()["incoming_price_data"][-1]  # noqa
        price_state = trade_state = alert_state = None
        if last_price > previous_price:
            price_state = "up"
            trade_state = "expected"
            alert_state = "ok"
        elif last_price < previous_price:
            price_state = "down"
            price_state = "unexpected"
            alert_state = "critical"
        else:
            price_state = "stale"
            price_state = "unexpected"
            alert_state = "critical"
        self.state = {
            "trade_state": trade_state,
            "price_state": price_state,
            "alert_state": alert_state}
        print "save_last_event: trade %s last event status %s" % (
            str(self.get_start_events_samples_signatures()), self.state)
        self.print_second_events("save_last_event:all")

    def update_status(self, warning_limit=1.0, critical_limit=2.0):
        """ To flag out problems with trade and update its state
            otherwise just keep the status unchanged.
        """
        update_flag = False
        self.status_check_ticks.append(time.time())
        period = self.get_rescale_period()
        if self.event_save_time[-1] < (self.status_check_ticks[-1] - period * critical_limit):  # noqa
            update_flag = True
            self.state = {
                "trade_state": "unexpected",
                "price_state": "missing",
                "alert_state": "critical"}
        elif self.event_save_time[-1] < (self.status_check_ticks[-1] - period * warning_limit):  # noqa
            update_flag = True
            self.state = {
                "trade_state": "unexpected",
                "price_state": "missing",
                "alert_state": "warning"}
        if update_flag:
            print "update_status: trade %s status %s" % (
                str(self.get_start_events_samples_signatures()), self.state)
            self.print_second_events("update_status:all")


def get_registered_trades(redis_client, name="registered_trades"):
    trades_json_str = redis_client.get(name)
    trades = []
    if trades_json_str is not None:
        for trade in json.loads(trades_json_str):
            print "get_registered_trades:" + str(trade)
            trades.append(
                Trade(None, None, None, None, trade))
    return trades


def set_registered_trades(redis_client, trades, name="registered_trades"):
    data = []
    for trade in trades:
        json_trade = trade.get_json()
        json_dict = json.loads(json_trade)
        data.append(json_dict)
    return redis_client.set(name, json.dumps(data))


def delete_registered_trades(redis_client, name="registered_trades"):
    return redis_client.delete(name)
