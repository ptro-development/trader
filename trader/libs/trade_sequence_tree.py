import pydot
import numpy

from scipy.stats.stats import pearsonr

from trader.libs.samples import get_leaf_atrribute_number, \
    get_key_sample_data


class Leaf(object):

    def __init__(self, key, data, previous_leaf):
        self.key = key
        self.data = data
        self.previous_leaf = previous_leaf
        self.next_leafs = []

    def __str__(self):
        return "<key:" + str(self.key) + " data:" + str(self.data) + \
            " next_leafs count:" + str(len(self.next_leafs)) + ">"

    def is_last_leaf(self):
        return len(self.get_next_leafs()) == 0

    def get_key(self):
        return self.key

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_next_leafs(self):
        return self.next_leafs

    def get_previous_leaf(self):
        return self.previous_leaf

    def add_leaf(self, key, data):
        new_leaf = Leaf(key, data, self)
        self.next_leafs.append(new_leaf)
        return new_leaf

    def get_leafs(self, keys):
        return filter(
            lambda x: x.get_key() in keys,
            self.next_leafs
        )


class TradeSequenceTree(object):

    def __init__(
            self, found_sample_keys,
            not_found_sample_keys,
            common_leafs_atributes,
            common_leafs_atributes_key_index_map):
        self.found_sample_keys = found_sample_keys
        self.not_found_sample_keys = not_found_sample_keys
        self.common_leafs_atributes = common_leafs_atributes
        self.common_leafs_atributes_key_index_map = common_leafs_atributes_key_index_map  # noqa
        self.first_leaf = Leaf(None, None, None)

    def get_root(self):
        return self.first_leaf

    def get_found_sample_keys(self):
        return self.found_sample_keys

    def get_not_found_sample_keys(self):
        return self.not_found_sample_keys

    def get_all_sample_keys(self):
        return self.found_sample_keys + self.not_found_sample_key

    def get_common_leafs_atributes(self):
        return self.common_leafs_atributes

    def get_common_leafs_atributes_key_index_map(self):
        return self.common_leafs_atributes_key_index_map


def get_next_not_found_leafs_extended(last_leaf, not_found_keys):
    assert isinstance(last_leaf, Leaf) and last_leaf is not None
    assert isinstance(not_found_keys, list) and len(not_found_keys) != 0
    # [[leaf, length, probability], ...]
    data = []
    overall_count = 0
    for n_leaf in last_leaf.get_leafs(not_found_keys):
        count = n_leaf.get_data()["count"]
        if count > 0:
            overall_count += count
            data.append([n_leaf, int(n_leaf.get_key()[1:]), count])
            # data.append([n_leaf, int(n_leaf.get_key()[1:])])
    # compute probability
    for index, bit in enumerate(data):
        data[index][2] = float(bit[2]) / overall_count
        # pass
    return data


def get_next_found_leafs_extended(
        key, last_leaf, found_keys, common_leaf_attributes, samples):
    assert isinstance(last_leaf, Leaf) and last_leaf is not None
    assert isinstance(found_keys, list) and len(found_keys) != 0
    assert isinstance(common_leaf_attributes, dict) and len(common_leaf_attributes) != 0  # noqa
    assert isinstance(samples, list) and len(found_keys) != 0
    # [[leaf, attribute_number, probability, correlation], ...]
    data = []
    overall_count = 0
    key_sample_data = get_key_sample_data(key, samples)
    for n_leaf in last_leaf.get_leafs(found_keys):
        count = n_leaf.get_data()["count"]
        if count > 0:
            overall_count += count
            n_attribute = get_leaf_atrribute_number(
                n_leaf.get_key(), common_leaf_attributes)
            n_key_sample_data_b = get_key_sample_data(
                n_leaf.get_key(), samples)
            # print "key_sample_data", key_sample_data, "n_key_sample_data_b", n_key_sample_data_b  # noqa
            correlation, other = pearsonr(key_sample_data, n_key_sample_data_b)
            data.append([n_leaf, n_attribute, count, correlation])
            # data.append([n_leaf, n_attribute])
    # compute probability
    for index, bit in enumerate(data):
        data[index][2] = float(bit[2]) / overall_count
        # pass
    return data


def get_leafs_chain(tree, leafs_key_sequence):
    success = True
    chain = []
    chain.append(tree.get_root())
    for index, key in enumerate(leafs_key_sequence):
        tmp_leaf = chain[-1].get_leafs([key])
        if tmp_leaf:
            chain.append(tmp_leaf[0])
        else:
            success = False
            break
    return (success, chain)


def get_last_leaf_in_leafs_chain(tree, leafs_key_sequence):
    success = True
    last_leaf = tree.get_root()
    for key in leafs_key_sequence:
        tmp_leaf = last_leaf.get_leafs([key])
        if tmp_leaf:
            last_leaf = tmp_leaf
        else:
            success = False
            break
    if success:
        return last_leaf
    else:
        return None


def get_leaf_samples_count(leaf, keys):
    return sum(
        leaf.get_data()["count"] for leaf in leaf.get_leafs(keys))


def get_leafs_chain_probability_report(tree, leafs_key_sequence):
    success, chain = get_leafs_chain(
        tree, leafs_key_sequence)
    if success:
        print "Whole sample probability chain:"
    else:
        print "Not whole sample probability chain was found."
    data = []
    for leaf in chain:
        found_probability, not_found_probability = get_leaf_next_probabilities(
            leaf, tree)
        data.append(
            leaf.get_key(),
            found_probability, not_found_probability)
    print data


def get_last_leafs(leaf, depth):
    if depth != 0:
        depth -= 1
        for leaf in leaf.get_next_leafs():
            if leaf.is_last_leaf():
                yield leaf
            else:
                for n_leaf in get_last_leafs(leaf, depth):
                    yield n_leaf
    else:
        yield leaf


def get_leafs_samples_chain(leaf, root_leaf):
    yield leaf
    if not leaf.get_previous_leaf() == root_leaf:
        for n_leaf in get_leafs_samples_chain(
                leaf.get_previous_leaf(), root_leaf):
            yield n_leaf


def reset_tree_counters(root_leaf):
    for leaf in root_leaf.get_next_leafs():
        data = leaf.get_data()
        data["count"] = 0
        leaf.set_data(data)
        if not leaf.is_last_leaf():
            reset_tree_counters(leaf)


def get_leafs_layers(leafs, depth=0):
    next_leafs = []
    for leaf in leafs:
        next_leafs.extend(
            leaf.get_next_leafs())
    if next_leafs:
        yield next_leafs
        if depth > 0:
            for layer in get_leafs_layers(next_leafs, depth-1):
                yield layer


def get_next_leafs_extended(leaf):
    known = []
    unknown = []
    for n_leaf in leaf.get_next_leafs():
        if isinstance(n_leaf.get_key(), basestring):
            unknown.append(n_leaf)
        else:
            known.append(n_leaf)
    return known, unknown


def get_leafs_layers_extended(leafs, depth=0):
    known_next = []
    unknown_next = []
    for leaf in leafs:
        known, unknown = get_next_leafs_extended(leaf)
        known_next.extend(known)
        unknown_next.extend(unknown)
    if known_next or unknown_next:
        yield known_next, unknown_next
        if depth > 0:
            for known, unknown in get_leafs_layers_extended(
                    known_next + unknown_next, depth-1):
                yield known, unknown


def get_leafs_probabilities(leafs, tree):
    counts = numpy.array((9 * [0.0]))
    key_count_map = tree.get_common_leafs_atributes_key_index_map()
    atributes = tree.get_common_leafs_atributes()
    all_count = 0.0
    for leaf in leafs:
        count = leaf.get_data()["count"]
        counts[
            key_count_map[
                atributes[leaf.get_key()]]] += count
        all_count += count
    if all_count > 0:
        counts /= all_count
    return counts


def get_in_layer_match_probability(
        tree, layer_position, starting_layers_counts):
    current_layers_counts = get_tree_layers_counts(tree, layer_position)
    probability = 0.0
    if layer_position > -1 and layer_position < len(current_layers_counts):
        current_count = float(current_layers_counts[layer_position])
        if current_count > 0:
            probability += current_count / starting_layers_counts[
                layer_position]
    return probability


def get_in_layer_match_probability_extended(
        tree, layer_position,
        starting_layers_counts_known, starting_layers_counts_unknown):
    assert isinstance(starting_layers_counts_known, list) and \
        len(starting_layers_counts_known) != 0
    assert isinstance(starting_layers_counts_unknown, list) and \
        len(starting_layers_counts_unknown) != 0
    assert isinstance(layer_position, int)
    assert isinstance(tree, TradeSequenceTree)
    current_layers_counts_known, current_layers_counts_unknown = get_tree_layers_counts_extended(  # noqa
        tree, layer_position)
    # [know, unknown]
    probability = [0.0, 0.0]
    starting_layers = (
        starting_layers_counts_known, starting_layers_counts_unknown)
    for index, current_layers in enumerate((
            current_layers_counts_known, current_layers_counts_unknown)):
        if layer_position > -1 and layer_position < len(current_layers):
            current_count = float(current_layers[layer_position])
            if current_count > 0:
                probability[index] += current_count / starting_layers[index][
                    layer_position]
    return probability


def get_leafs_count(leafs):
    count = 0.0
    for leaf in leafs:
        count += leaf.get_data()["count"]
    return count


def get_leaf_next_probabilities(leaf, tree):
    # counts array
    # [found_count, not_found_count]
    counts = [0, 0]
    counts[0] = get_leaf_samples_count(
        leaf, tree.get_found_sample_keys())
    counts[1] = get_leaf_samples_count(
        leaf, tree.get_not_found_sample_keys())
    all_count = sum(counts)
    probabilities = 2 * [-1.0]
    for index, count in enumerate(counts):
        if count > 0:
            probabilities[index] = count / float(all_count)
    return probabilities


def get_extended_leaf_next_probabilities(leaf, tree, depth=0):
    counts = numpy.array((9 * [0.0]))
    key_count_map = tree.get_common_leafs_atributes_key_index_map()
    atributes = tree.get_common_leafs_atributes()
    all_count = 0.0
    for next_leaf in leaf.get_next_leafs():
        count = next_leaf.get_data()["count"]
        counts[
            key_count_map[
                atributes[next_leaf.get_key()]]] += count
        all_count += count
    if all_count > 0:
        counts /= all_count
    print "depth A", depth
    yield counts
    if depth > 0:
        print "depth B", depth
        counts_next = numpy.array((9 * [0.0]))
        for next_leaf in leaf.get_next_leafs():
            if not next_leaf.is_last_leaf():
                for prob in get_extended_leaf_next_probabilities(
                        next_leaf, tree, depth-1):
                    counts_next += prob
                all_counts = sum(counts_next)
                if all_counts > 0:
                    counts_next /= float(all_counts)
        yield counts_next


def report_extended_leaf_next_probabilities_old(leaf, tree, depth=0):
    key_count_map = tree.get_common_leafs_atributes_key_index_map()
    prob = [p for p in get_extended_leaf_next_probabilities(
        leaf, tree, depth)]
    new_chain = reversed(
        list(get_leafs_samples_chain(leaf, tree.get_root())))
    common_leaf_atributes = tree.get_common_leafs_atributes()
    print "Chain: " + "->".join(
        [str(l.get_key()) + "(%s)" % common_leaf_atributes[l.get_key()] for l in new_chain])  # noqa
    for index, p in enumerate(prob):
        data = zip(
            sorted(key_count_map.items(), key=lambda x: x[1]), p)
        report = [str(d[0][0]) + ":" + "%.3f" % d[1] for d in data]
        print "Level " + str(index) + " " + " ".join(report)


def report_extended_leaf_next_probabilities(leaf, tree, depth=0):
    key_count_map = tree.get_common_leafs_atributes_key_index_map()
    new_chain = reversed(
        list(get_leafs_samples_chain(leaf, tree.get_root())))
    common_leaf_atributes = tree.get_common_leafs_atributes()
    print "Chain: " + "->".join(
        [str(l.get_key()) + "(%s)" % common_leaf_atributes[l.get_key()] for l in new_chain])  # noqa
    for index, layer in enumerate(get_leafs_layers([leaf], depth)):
        prob = get_leafs_probabilities(layer, tree)
        data = zip(
            sorted(key_count_map.items(), key=lambda x: x[1]), prob)
        report = [str(d[0][0]) + ":" + "%.3f" % d[1] for d in data]
        print "Level " + str(index) + " " + " ".join(report)


def get_next_leafs_chains_report(tree, leafs_key_sequence, depth):
    success, chain = get_leafs_chain(
        tree, leafs_key_sequence)
    if success:
        found_chains = []
        not_found_chains = []
        for leaf in get_last_leafs(chain[-1], depth):
            new_chain = reversed(
                list(get_leafs_samples_chain(leaf, tree.get_root())))
            if leaf.get_key() in tree.get_found_sample_keys():
                found_chains.append(new_chain)
            else:
                not_found_chains.append(new_chain)
        probabilities = get_leaf_next_probabilities(
            chain[-1], tree)
        print "For sequence:%s depth:%s is found_chains:%s not_found_chains:%s found_probability:%.3f not_found_probability:%.3f" % (  # noqa
            leafs_key_sequence, depth,
            len(found_chains), len(not_found_chains),
            probabilities[0], probabilities[1])


def update_top_leaf_counts(
        tree, future_trails, top_leaf_key,
        next_leaf_key, positive_or_negative,
        report_probabilities=False, depth=0):
    tmp = []
    while(future_trails[top_leaf_key]):
        leaf = future_trails[top_leaf_key].pop()
        found_leaf = leaf.get_leafs([next_leaf_key])
        if found_leaf:
            leaf = found_leaf[0]
            data = leaf.get_data()
            if positive_or_negative:
                data["count"] += 1
            else:
                data["count"] -= 1
            leaf.set_data(data)
            if report_probabilities:
                report_extended_leaf_next_probabilities(
                    leaf, tree, depth)
            if not leaf.is_last_leaf():
                tmp.append(leaf)
            else:
                pass
                # add finished trail into results
                # for accounting
        else:
            pass
            """ This scenario should not be covered
            as there is supposed to be always the first
            match and next is not guarantied.
            """
    if tmp:
        future_trails[top_leaf_key] = tmp


def update_top_leafs_counts(
        future_trails, incoming_sequence_key,
        tree, top_leafs_keys, positive_or_negative=True,
        report_probabilities=False, depth=0):
    if incoming_sequence_key in top_leafs_keys:
        root_leaf = tree.get_root()
        future_trails[incoming_sequence_key].append(root_leaf)
    # do processing of new key for existing future trail
    for top_leaf_key in top_leafs_keys:
        update_top_leaf_counts(
            tree, future_trails, top_leaf_key,
            incoming_sequence_key, positive_or_negative,
            report_probabilities, depth)


def update_top_leafs_counts_2(
        tree_trails, start_end_positions,
        incoming_sequence_key, positive_or_negative,
        depth=10):
    tree_trails.append(incoming_sequence_key)
    tree_trails.update(
        incoming_sequence_key, start_end_positions,
        positive_or_negative, depth)


def generate_tree_edge(leaf, graph, depth=0, color="black", common_leafs=True):
    if leaf.get_previous_leaf():
        key = str(leaf.get_key()) + "_" + str(depth)
        if not common_leafs:
            key += "_" + str(leaf.get_previous_leaf().get_key())
    else:
        key = str(leaf.get_key()) + "_" + str(depth)
    depth += 1
    for n_leaf in leaf.get_next_leafs():
        key_next = str(n_leaf.get_key()) + "_" + str(depth)
        if not common_leafs:
            key_next += "_" + str(leaf.get_key())
        edge = pydot.Edge(
            key, key_next,
            label=str(n_leaf.get_data()["count"]),
            labelfontcolor="#009933",
            fontsize="10.0",
            color=color)
        graph.add_edge(edge)
        generate_tree_edge(n_leaf, graph, depth, color)


def generate_tree_chart(file_path, tree, size=None):
    if size:
        graph = pydot.Dot(graph_type="graph", size=size)
    else:
        graph = pydot.Dot(graph_type="graph", overlap="False")
    generate_tree_edge(tree.get_root(), graph)
    graph.write_png(file_path, prog="dot")


def get_tree_layers_counts(tree, max_depth=50):
    layer_counts = []
    for layer in get_leafs_layers([tree.get_root()], max_depth):
        layer_counts.append(get_leafs_count(layer))
    return layer_counts


def get_tree_layers_counts_extended(tree, max_depth=50):
    known_layer_counts = []
    unknown_layer_counts = []
    for known, unknown in get_leafs_layers_extended(
            [tree.get_root()], max_depth):
        known_layer_counts.append(get_leafs_count(known))
        unknown_layer_counts.append(get_leafs_count(unknown))
    return known_layer_counts, unknown_layer_counts


def increment_leaf_key_count(leaf, data_key="count"):
    if data_key in leaf.data:
        data = leaf.get_data()
        data[data_key] += 1
        leaf.set_data(data)


def decrement_leaf_key_count(leaf, data_key="count"):
    if data_key in leaf.data:
        data = leaf.get_data()
        data[data_key] -= 1
        leaf.set_data(data)
