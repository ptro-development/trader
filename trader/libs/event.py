import json

from trader.libs.relations import get_closest_focus_index, \
    CorrelationFactsRelationKey
from trader.libs.trade_data import get_closest_period_index_down


class Event(object):

    def __init__(
            self, relative_match_position, sample,
            incoming_price_data, incoming_trade_data, correlation, percentage):
        self.data = {
            "relative_match_position": relative_match_position,
            "sample": sample,
            "incoming_price_data": incoming_price_data,
            "incoming_trade_data": incoming_trade_data,
            "correlation": correlation,
            "percentage": percentage,
            "sample_signature": self._compute_sample_signature(sample),
        }

    def get_rescale_period(self):
        return self.data["sample"]["rescale_period"]

    def _compute_sample_signature(self, sample):
        return hash(
            json.dumps(sample, sort_keys=True))

    def get_sample_signature(self):
        return self.data["sample_signature"]

    def get_sample_attribute_up_or_down(self):
        return self.data["sample"]["sample_attributes"]["up_or_down"]

    def is_sample_attribute_up(self):
        return self.data["sample"]["sample_attributes"]["up_or_down"] == "up"

    def is_min_required_percentage(self, percentage):
        return percentage <= self.data["percentage"]

    def __str__(self):
        return str(self.data)

    def get_dict(self):
        return self.data

    def get_sample(self):
        return self.data["sample"]

    def get_sample_position(self):
        return self.data["sample"]["sample_position"]

    def get_sample_epoch(self):
        return self.data["sample"]["sample_epoch"]

    def get_percentage(self):
        return self.data["percentage"]

    def get_correlation_sign(self):
        sign = "-"
        if self.data["correlation"] > 0:
            sign = "+"
        return sign

    def get_relative_match_position(self):
        return self.data["relative_match_position"]


def dict_to_event(event_dict):
    return Event(
        relative_match_position=event_dict["relative_match_position"],
        sample=event_dict["sample"],
        incoming_price_data=event_dict["incoming_price_data"],
        incoming_trade_data=event_dict["incoming_trade_data"],
        correlation=event_dict["correlation"],
        percentage=event_dict["percentage"],
    )


def filter_events(events, percentage=45.0):
    return [e for e in events
            if e.is_min_required_percentage(percentage) is True]


def get_events_correlation_facts_relation_key(
        first_event, second_event, focus):
    distance = get_closest_period_index_down(
        first_event.get_relative_match_position(),
        second_event.get_relative_match_position(),
        first_event.get_rescale_period())
    focus_distance = get_closest_focus_index(distance, focus)
    key = CorrelationFactsRelationKey(
        first_fact_correlation_sing=first_event.get_correlation_sign(),
        first_fact_sample_epoch=first_event.get_sample_epoch(),
        first_fact_sample_up_or_down=first_event.get_sample_attribute_up_or_down(),  # noqa
        second_fact_correlation_sing=second_event.get_correlation_sign(),
        second_fact_sample_epoch=second_event.get_sample_epoch(),
        second_fact_sample_up_or_down=second_event.get_sample_attribute_up_or_down(),  # noqa
        facts_distance=focus_distance,
        order=first_event.get_relative_match_position() < second_event.get_relative_match_position()  # noqa
    )
    return key.get_key_string()
