import copy
import csv
import random

# from trade_sequence_tree import report_extended_leaf_next_probabilities
from trader.libs.trade_sequence_tree import increment_leaf_key_count, \
    decrement_leaf_key_count, get_in_layer_match_probability, \
    get_tree_layers_counts_extended, get_next_not_found_leafs_extended, \
    get_next_found_leafs_extended, get_in_layer_match_probability_extended
from trader.libs.linear_normalisation import LinearNormalisation
from trader.libs.utils import get_euclidean_closest_element
from trader.libs.samples import get_leaf_atrribute_number


class TreeKeyTrail(object):

    def __init__(self, key, start_leaf):
        self.active = True
        self.key = key
        self.leafs_history = []
        # inputs
        self.in_layer_match_probability = []
        # outputs
        self.amount_to_buy = []
        # [[start_position, end_position], ...]
        self.positions = []
        self.leafs_history.append(start_leaf)

    def has_started(self):
        return len(self.leafs_history) > 1 and self.active

    def update(
            self, key, start_end_positions, tree,
            positive_or_negative, depth=10):
        """ Update key trail with new key against
            imprinted past in the tree
        """
        if self.active:
            found_leaf = self.leafs_history[-1].get_leafs((key,))
            if found_leaf:
                leaf = found_leaf[0]
                if positive_or_negative:
                    increment_leaf_key_count(leaf)
                else:
                    decrement_leaf_key_count(leaf)
                self.positions.append(start_end_positions)
                if not leaf.is_last_leaf():
                    self.leafs_history.append(leaf)
                else:
                    pass
                    # add finished trail into results
                    # for accounting
            else:
                # this trail did progress in past so
                # ending it now
                self.active = False

    def add_key(
            self, key, start_end_positions, depth=10):
        """ Add key to trail if it appeared previously """
        found = False
        if self.active:
            found_leaf = self.leafs_history[-1].get_leafs((key,))
            if found_leaf:
                leaf = found_leaf[0]
                increment_leaf_key_count(leaf)
                self.positions.append(start_end_positions)
                if not leaf.is_last_leaf():
                    # this one should be probably still added
                    self.leafs_history.append(leaf)
                else:
                    self.active = False
                found = True
            else:
                # this trail did progress in past so
                # ending it now
                self.active = False
        return found

    def delete_key(
            self, key, start_end_positions, depth=10):
        """ Delete key from trail if it appeared previously """
        found = False
        if self.active:
            found_leaf = self.leafs_history[-1].get_leafs((key,))
            if found_leaf:
                leaf = found_leaf[0]
                decrement_leaf_key_count(leaf)
                self.positions.append(start_end_positions)
                if not leaf.is_last_leaf():
                    self.leafs_history.append(leaf)
                else:
                    self.active = False
                found = True
            else:
                # this trail did progress in past so
                # ending it now
                self.active = False
        return found

    def _get_keys_in_range(self, key, limit, sample_size):
        keys = [key]
        if isinstance(key, basestring) and key.startswith("N"):
            length = int(key[1:])
            lower_limit = length-limit
            if lower_limit < 0:
                lower_limit = 0
            upper_limit = length+limit+1
            if upper_limit > sample_size:
                upper_limit = sample_size
            keys = ["N"+str(i) for i in range(lower_limit, upper_limit)]
        return keys

    def delete_key_within_range(
            self, key, start_end_positions, depth, allowed_range, sample_size):
        """ To delete keys if they are known unknown.
            e.g.: N4 with range limit 2 would delete key
            key if input key was (N2, N3, N4, N5, N6)

            which key to delete from allowed range if there are multiple keys
            available ?

            Find all keys which are available in range and then choose
            one at random to delete.
        """
        found = False
        if self.active:
            keys = self._get_keys_in_range(key, allowed_range, sample_size)
            if len(keys) > 1:
                new_key = random.choice(keys)
                keys = [new_key]
            found_leaf = self.leafs_history[-1].get_leafs(keys)
            if found_leaf:
                leaf = found_leaf[0]
                decrement_leaf_key_count(leaf)
                self.positions.append(start_end_positions)
                if not leaf.is_last_leaf():
                    self.leafs_history.append(leaf)
                else:
                    self.active = False
                found = True
            else:
                # this trail did progress in past so
                # ending it now
                self.active = False
        return found

    def delete_euclidean_closest_key(
            self, key, start_end_positions,
            not_found_keys, found_keys,
            common_leaf_attributes, samples):
        found = False
        found_leaf = None
        last_leaf = self.leafs_history[-1]
        if self.active:
            # print "F last_leaf",  last_leaf
            # print "F last_leaf all", last_leaf.get_next_leafs()
            found_leaf = last_leaf.get_leafs((key,))
            if not found_leaf:
                if isinstance(key, basestring) and key.startswith("N"):
                    # print "not_found_keys", not_found_keys
                    leafs = get_next_not_found_leafs_extended(
                        last_leaf, not_found_keys)
                    # print "A leafs", leafs
                    if leafs:
                        found_leaf = (get_euclidean_closest_element(
                            # x1: length of unknown
                            # x2: probability of happening
                            (key, int(key[1:]), 1.0),
                            # (key, int(key[1:])),
                            leafs)[0][0],)
                    # print "A key", key, "found_leaf", found_leaf
                else:
                    leafs = get_next_found_leafs_extended(
                        key, last_leaf, found_keys,
                        common_leaf_attributes, samples)
                    # print "B leafs", leafs
                    if leafs:
                        attribute = get_leaf_atrribute_number(
                            key, common_leaf_attributes)
                        # print "attribute", attribute
                        # x1: sample attribute number (up, down, stale ...)
                        # x2: probability of happening
                        # x3: correlation number
                        found_leaf = (get_euclidean_closest_element(
                            (key, attribute, 1.0, 1.0),
                            # (key, attribute),
                            leafs)[0][0],)
                    # print "B key", key, "found_leaf", found_leaf
            if found_leaf:
                leaf = found_leaf[0]
                # print "C leaf", leaf
                decrement_leaf_key_count(leaf)
                self.positions.append(start_end_positions)
                """
                if not leaf.is_last_leaf():
                    self.leafs_history.append(leaf)
                else:
                    self.active = False
                """
                self.leafs_history.append(leaf)
                if leaf.is_last_leaf():
                    # print "Last leaf :("
                    self.active = False
                found = True
            else:
                # print "End of trial", self.positions
                # this trail did progress in past so
                # ending it now
                self.active = False
        return found

    def get_gain_after_first(self, trade_data, gain_percentage=0.10):
        gain = 0.0
        loss = 0.0
        if len(self.positions) >= 2:
            for index, (start_position, end_position) in enumerate(
                    self.positions[1:]):
                price_diff = trade_data.prices[end_position] - \
                    trade_data.prices[start_position]
                trade_amount = sum(
                    trade_data.trades[tick] for tick in range(
                        start_position, end_position))
                # this does not distinguish between (+/-) gain
                res = price_diff * trade_amount * gain_percentage
                if res > 0.0:
                    gain += res
                else:
                    loss += res
        return gain, loss

    def get_overall_gain_for_estimated_trades_amounts(
            self, trade_data, gain_percentage=0.10):
        gain = 0.0
        investment = 0.0
        no_limit_gain = 0.0
        no_limit_investment = 0.0
        if len(self.positions) >= 2:
            # skip first position
            for index, (start_position, end_position) in enumerate(
                    self.positions[1:]):
                price_diff = trade_data.prices[end_position] - \
                    trade_data.prices[start_position]
                trade_amount = sum(
                    trade_data.trades[tick] for tick in range(
                        start_position, end_position))
                no_limit_investment += trade_data.prices[start_position] * trade_amount * \
                    gain_percentage
                no_limit_gain += price_diff * trade_amount * \
                    gain_percentage
                # index + 1 compensates for not starting 0 element
                investment += trade_data.prices[start_position] * trade_amount * \
                    self.amount_to_buy[index+1] * gain_percentage
                gain += price_diff * trade_amount * \
                    self.amount_to_buy[index+1] * gain_percentage
                # print investment, gain

        return gain, investment, no_limit_gain, no_limit_investment

    def get_overall_gain_for_estimated_trades_amounts_extended(
            self, trade_data, gain_percentage=0.10):
        gain = 0.0
        loss = 0.0
        investment = 0.0
        no_limit_gain = 0.0
        no_limit_loss = 0.0
        no_limit_investment = 0.0
        if len(self.positions) >= 2:
            # skip first position
            for index, (start_position, end_position) in enumerate(
                    self.positions[1:]):
                price_diff = trade_data.prices[end_position] - \
                    trade_data.prices[start_position]
                trade_amount = sum(
                    trade_data.trades[tick] for tick in range(
                        start_position, end_position))
                no_limit_investment += trade_data.prices[start_position] * trade_amount * \
                    gain_percentage
                last_no_limit_gain = price_diff * trade_amount * \
                    gain_percentage
                if last_no_limit_gain > 0:
                    no_limit_gain += last_no_limit_gain
                else:
                    no_limit_loss += last_no_limit_gain
                # index + 1 compensates for not starting 0 element
                investment += trade_data.prices[start_position] * trade_amount * \
                    self.amount_to_buy[index+1] * gain_percentage
                last_gain = price_diff * trade_amount * \
                    self.amount_to_buy[index+1] * gain_percentage
                if last_gain > 0:
                    gain += last_gain
                else:
                    loss += last_gain
                # print investment, gain

        return gain, loss, investment, no_limit_gain, \
            no_limit_loss, no_limit_investment

    def get_min_max_amount_to_buy(self):
        return (min(self.amount_to_buy), max(self.amount_to_buy))

    def set_amount_to_buy(self, network):
        for index, probability in enumerate(
                self.in_layer_match_probability):
            if index < len(self.positions):
                """
                1. Probability of hitting leaf in tree
                    - know sample
                    - unknown
                2. Layer number
                3. Position in trade sequence
                """
                """
                to_buy = network.sim([
                    [
                        probability[0],
                        probability[1],
                        index,
                        self.positions[index][0]]])[0][0]
                """
                to_buy = network.sim([[
                    index,
                    self.positions[index][0]]])[0][0]

                self.amount_to_buy.append(to_buy)
                if to_buy < 0.0:
                    self.amount_to_buy[-1] = 0.0

    def get_training_data(self):
        data = {}
        # start_position = None
        starting_position = None
        if self.in_layer_match_probability:
            starting_position = self.positions[0][0]
        for index, probability in enumerate(
                self.in_layer_match_probability):
            if index < len(self.positions):
                data.update({
                    self.positions[index][0]: {
                        "known_probability": probability[0],
                        "unknown_probability": probability[1],
                        "layer_index": index,
                        "starting_position": starting_position,
                        "position": self.positions[index][0]}})
        # if data:
        #    start_position = self.positions[0][0]
        return data

    def record_train_data(
            self, train_data_csv, trade_data,
            min_length=3, gain_percentage=0.1):
        if len(self.in_layer_match_probability) >= min_length:
            for index, probability in enumerate(
                    self.in_layer_match_probability):
                if index < len(self.positions):
                    start_position, end_position = self.positions[index]
                    price_diff = trade_data.prices[end_position] - \
                        trade_data.prices[start_position]
                    trade_amount = sum(
                        trade_data.trades[tick] for tick in range(
                            start_position, end_position))
                    # this does not distinguish between (+/-) gain
                    amount_to_buy = price_diff * trade_amount * gain_percentage
                    if amount_to_buy < 0.0:
                        amount_to_buy = 0.0
                    data = dict(zip(
                        train_data_csv.fieldnames, (
                            probability[0], probability[1],
                            index, self.positions[index][0],
                            amount_to_buy)))
                    train_data_csv.writerow(data)

    def normalize_amount_to_buy(self, linear_normalisation):
        self.amount_to_buy = linear_normalisation.nomalise_array(
            self.amount_to_buy)

    def get_in_layer_match_probability(self, tree, starting_layers_counts):
        return get_in_layer_match_probability(
            tree, len(self.leafs_history)-1, starting_layers_counts)

    def get_in_layer_match_probability_extended(
            self, tree,
            starting_layers_counts_known, starting_layers_counts_unknown):
        return get_in_layer_match_probability_extended(
            tree, len(self.leafs_history)-1,
            starting_layers_counts_known, starting_layers_counts_unknown)

    def set_in_layer_match_probability(self, tree, starting_layers_counts):
        if self.active:
            self.in_layer_match_probability.append(
                self.get_in_layer_match_probability(
                    tree, starting_layers_counts))

    def set_in_layer_match_probability_extended(
            self, tree,
            starting_layers_counts_known, starting_layers_counts_unknown):
        if self.active:
            self.in_layer_match_probability.append(
                self.get_in_layer_match_probability_extended(
                    tree, starting_layers_counts_known,
                    starting_layers_counts_unknown))


class TreeKeysTrailsCollection(object):

    def __init__(self, tree, write_train_data=False):
        self.tree = copy.deepcopy(tree)
        # self.starting_layers_counts = get_tree_layers_counts(self.tree)
        self.starting_layers_counts_known, self.starting_layers_counts_unknown = get_tree_layers_counts_extended(  # noqa
            tree)
        # starting sequence keys in the past tree
        self.known_keys = [
            l.get_key() for l in self.tree.get_root().get_next_leafs()]
        self.keys_trails = {}
        for key in self.known_keys:
            self.keys_trails[key] = []
        """
        self.all_keys_appearance_counter = {}
        for key in self.tree.get_all_sample_keys():
            self.all_keys_appearance_counter[key] = 0
        """
        self.train_data_order = [
            "probability_known_sample",
            "probability_unknown_sample",
            "tree_layer_number",
            "trade_sequence_position",
            "amount_to_buy"]
        if write_train_data:
            self.train_data_fd = open("train_data.csv", "w")
            self.train_data_csv = csv.DictWriter(
                self.train_data_fd, self.train_data_order)

    def get_train_data_order(self):
        return self.train_data_order

    def record_train_data(
            self, trade_data, min_length, gain_percentage=0.1):
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].record_train_data(
                    self.train_data_csv, trade_data,
                    min_length, gain_percentage)

    def end_train_data(self):
        self.train_data_fd.close()

    def get_tree(self):
        return self.tree

    """
    def update_appearance_counter(self, key):
        if key in self.all_keys_appearance_counter:
            self.all_keys_appearance_counter[key] += 1

    def get_appearance_counter(self, key):
        counter = 0
        if key in self.all_keys_appearance_counter:
            counter = self.all_keys_appearance_counter[key]
        return counter
    """

    def append(self, new_key):
        """ Start a new possible future trail for new_key """
        if new_key in self.keys_trails:
            self.keys_trails[new_key].append(
                TreeKeyTrail(new_key, self.tree.get_root()))

    def append_only_if_inactive(self, new_key):
        """ Start a new possible future trail for new_key """
        # is there an active trade ?
        active_trade = False
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                if self.keys_trails[key][index].active:
                    active_trade = True
        if not active_trade:
            if new_key in self.keys_trails:
                self.keys_trails[new_key].append(
                    TreeKeyTrail(new_key, self.tree.get_root()))

    def update(
            self, new_key, start_end_positions,
            positive_or_negative, depth):
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].update(
                    new_key, start_end_positions, self.tree,
                    positive_or_negative, depth)

    def add_key(
            self, new_key, start_end_positions, depth):
        """ Add a new key to all trails if it appeared previously """
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].add_key(
                    new_key, start_end_positions, depth)

    def delete_key(
            self, new_key, start_end_positions, depth):
        """ Delete a new key from all trails if it appeared previously """
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].delete_key(
                    new_key, start_end_positions, depth)

    def compute_probability_delete_key_within_range(
            self, new_key, start_end_positions,
            depth, allowed_range, sample_size):
        """ Compute in layer match probability followed by delete of new key from
            all trails if it appeared previously """
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].set_in_layer_match_probability(  # noqa
                    self.tree, self.starting_layers_counts)
                self.keys_trails[key][index].delete_key_within_range(
                    new_key, start_end_positions,
                    depth, allowed_range, sample_size)

    def compute_probability_delete_euclidean_closest_key(
            self, new_key, start_end_positions, samples):
        """ Compute in layer match probability followed by delete of new key from
            all trails if it appeared previously """
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].set_in_layer_match_probability(  # noqa
                    self.tree, self.starting_layers_counts)
                self.keys_trails[key][index].delete_euclidean_closest_key(
                    new_key, start_end_positions,
                    self.tree.get_not_found_sample_keys(),
                    self.tree.get_found_sample_keys(),
                    self.tree.get_common_leafs_atributes(),
                    samples)

    def compute_probability_delete_euclidean_closest_key_extended(
            self, new_key, start_end_positions, samples):
        """ Compute in layer match probability followed by delete of new key from
            all trails if it appeared previously """
        # counter = 0
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                # counter += 1
                self.keys_trails[key][index].set_in_layer_match_probability_extended(  # noqa
                    self.tree,
                    self.starting_layers_counts_known,
                    self.starting_layers_counts_unknown)
                self.keys_trails[key][index].delete_euclidean_closest_key(
                    new_key, start_end_positions,
                    self.tree.get_not_found_sample_keys(),
                    self.tree.get_found_sample_keys(),
                    self.tree.get_common_leafs_atributes(),
                    samples)
        # print "==============", counter

    def compute_probability_and_delete_key(
            self, new_key, start_end_positions, depth):
        """ Compute in layer match probability followed by delete of new key from
            all trails if it appeared previously """
        for key in self.known_keys:
            for index, trails in enumerate(self.keys_trails[key]):
                self.keys_trails[key][index].set_in_layer_match_probability(  # noqa
                    self.tree, self.starting_layers_counts)
                self.keys_trails[key][index].delete_key(
                    new_key, start_end_positions, depth)

    def compute_amount_to_buy(self, network):
        for key in self.known_keys:
            for trails in self.keys_trails[key]:
                trails.set_amount_to_buy(network)

    def get_training_data(self):
        data = {}
        for key in self.known_keys:
            for trails in self.keys_trails[key]:
                trail_data = trails.get_training_data()
                if trail_data:
                    data.update(trail_data)
        return data

    def get_overall_gain_for_estimated_trades_amounts(self, trade_data):
        gain = 0.0
        investment = 0.0
        no_limit_investment = 0.0
        no_limit_gain = 0.0
        partial_investments = []
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                last_gain, last_investment, last_no_limit_gain, last_no_limit_investment = \
                    trail.get_overall_gain_for_estimated_trades_amounts(
                        trade_data)
                gain += last_gain
                partial_investments.append(last_investment)
                investment += last_investment
                no_limit_investment += last_no_limit_investment
                no_limit_gain += last_no_limit_gain
        return gain, investment, partial_investments, no_limit_gain, \
            no_limit_investment

    def get_overall_gain_for_estimated_trades_amounts_extended(
            self, trade_data):
        gain = 0.0
        loss = 0.0
        investment = 0.0
        no_limit_investment = 0.0
        no_limit_gain = 0.0
        no_limit_loss = 0.0
        partial_investments = []
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                last_gain, last_loss, last_investment, last_no_limit_gain, last_no_limit_loss, last_no_limit_investment = \
                    trail.get_overall_gain_for_estimated_trades_amounts_extended(  # noqa
                        trade_data)
                gain += last_gain
                loss += last_loss
                partial_investments.append(last_investment)
                investment += last_investment
                no_limit_investment += last_no_limit_investment
                no_limit_gain += last_no_limit_gain
                no_limit_loss += last_no_limit_loss
        return gain, loss, investment, partial_investments, no_limit_gain, \
            no_limit_loss, no_limit_investment

    def get_min_max_amount_to_buy(self):
        min_max = []
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                min_max.extend(trail.get_min_max_amount_to_buy())
        return (min(min_max), max(min_max))

    def normalize_amount_to_buy(self):
        min_value, max_value = self.get_min_max_amount_to_buy()
        linear_normalisation = LinearNormalisation(
            min_value, max_value, 0.0, 1.0)
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                trail.normalize_amount_to_buy(linear_normalisation)

    def get_overall_gain(self, trade_data, gain_percentage=0.10):
        gain = 0.0
        loss = 0.0
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                t_gain, t_loss = trail.get_gain_after_first(
                    trade_data, gain_percentage)
                gain += t_gain
                loss += t_loss
        return gain, loss

    def get_in_layer_match_probabilities(self):
        probabilities = []
        for key in self.known_keys:
            for trail in self.keys_trails[key]:
                if trail.has_started():
                    probabilities.append(
                        trail.get_in_layer_match_probability(
                            self.tree, self.starting_layers_counts))
        return probabilities
