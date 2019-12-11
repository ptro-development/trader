import json


def load_facts_relations(file_path):
    relations = None
    with open(file_path, "r") as fd:
        relations = json.load(fd)
    return relations


def save_facts_relations(file_path, relations):
    with open(file_path, "w") as fd:
        relations = json.dump(relations, fd)


def get_closest_focus_index(number, step=25):
    """ To get closest index with rounding up

    >>> get_closest_focus_index(10, 3)
    12
    >>> get_closest_focus_index(9, 3)
    9
    >>> get_closest_focus_index(1, 3)
    3
    """
    result = (number / step)
    result *= step
    if number % step != 0:
        result += step
    return result


def remove_facts_relations(
        facts_relation,
        keep_only_right_to_left=True,
        min_number_relation_appearances=1):
    for focus_key in facts_relation.keys():
        for relation_key, relation_value in facts_relation[focus_key].items():  # noqa
            if keep_only_right_to_left and "False" in relation_key:
                del facts_relation[focus_key][relation_key]
                continue
            if relation_value < min_number_relation_appearances:
                del facts_relation[focus_key][relation_key]


class CorrelationFact(object):

    def __init__(
            self, correlation_position, correlation_positions_counter,
            sample_epoch, correlation_sign,
            sample_status, start_price, end_price):
        """ Correlation fact represents one sample correlation position
            with sample position it correlates to.
        """
        # Sample other samples correlate to.
        # In wider context this can be in multiple CorrelationFact.
        self.correlation_position = correlation_position
        self.correlation_positions_counter = correlation_positions_counter
        # Tested sample.
        # In wider context this can be only in one CorrelationFact.
        self.sample_epoch = sample_epoch
        self.correlation_sign = correlation_sign  # noqa
        self.sample_status = sample_status
        self.start_price = start_price
        self.end_price = end_price

    def get_price_delta(self):
        return self.end_price - self.start_price

    def get_correlation_sign(self):
        sign = "-"
        if self.correlation_sign:
            sign = "+"
        return sign

    def get_sample_status(self):
        return self.sample_status

    def get_sample_epoch(self):
        return self.sample_epoch

    def get_correlation_position(self):
        return self.correlation_position

    def get_correlation_positions_counter(self):
        return self.correlation_positions_counter

    def __str__(self):
        return str(self.correlation_position) + "_" + \
            str(self.sample_epoch) + "_" + \
            str(self.correlation_sign) + "_" + \
            self.sample_status


class CorrelationFactsRelation(object):

    def __init__(self, first_fact, second_fact):
        """ Represent relations between two facts.

            Computed relations are:
                - distance between to correlations (integer)
                - next to each other (True, False)
                - equality of samples which they correlate to (True, False)
                - order in correlations in sequence:
                    left to right (True) or right to left (False)
        """
        # rescaled values are used like correlation_position
        self.data = {
            "first_fact": first_fact,
            "second_fact": second_fact,
            "distance": abs(first_fact.correlation_position - second_fact.correlation_position),  # noqa
            "repetition": abs(first_fact.correlation_position - second_fact.correlation_position) == 1,  # noqa
            "equality": first_fact.sample_epoch == second_fact.sample_epoch,  # noqa
            "order": first_fact.correlation_position < second_fact.correlation_position,  # noqa
        }

    def __str__(self):
        return str(self.data)

    def get_dict(self):
        return self.data

    def get_signature(self):
        data = [
            str(self.data["first_fact"]),
            str(self.data["second_fact"]),
            self.data["distance"],
            self.data["repetition"],
            self.data["equality"],
            self.data["order"]]
        return "_".join(map(lambda x: str(x), data))

    def get_fact_relation_key_string(self, focus):
        key = CorrelationFactsRelationKey(
            self.data["first_fact"].get_correlation_sign(),
            self.data["first_fact"].get_sample_epoch(),
            self.data["first_fact"].get_sample_status(),
            self.data["second_fact"].get_correlation_sign(),
            self.data["second_fact"].get_sample_epoch(),
            self.data["second_fact"].get_sample_status(),
            get_closest_focus_index(self.data["distance"], focus),
            self.data["order"])
        return key.get_key_string()


class CorrelationFactsRelationKey(object):

    keys_order = [
        "first_fact_correlation_sing",
        "first_fact_sample_epoch",
        "first_fact_sample_status",
        "second_fact_correlation_sing",
        "second_fact_sample_epoch",
        "second_fact_sample_status",
        "distance",
        "order"
    ]

    def __init__(
            self, first_fact_correlation_sing,
            first_fact_sample_epoch, first_fact_sample_status,
            second_fact_correlation_sing, second_fact_sample_epoch,
            second_fact_sample_status, facts_distance, order):
        self.data = {
            "first_fact_correlation_sing": first_fact_correlation_sing,
            "first_fact_sample_epoch": first_fact_sample_epoch,
            "first_fact_sample_status": first_fact_sample_status,
            "second_fact_correlation_sing": second_fact_correlation_sing,
            "second_fact_sample_epoch": second_fact_sample_epoch,
            "second_fact_sample_status": second_fact_sample_status,
            "distance": facts_distance,
            "order": order,
        }

    def get_key_string(self):
        key_str = ""
        for key in self.keys_order:
            key_str += str(self.data[key]) + "_"
        return key_str.rstrip("_")

    def get_dict(self):
        return self.data


def get_correlation_facts_relation_key_dict(key_string):
    values = key_string.split("_")
    return dict(
        zip(CorrelationFactsRelationKey.keys_order, values))
