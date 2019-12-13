from trader.libs.trade_sequence_tree import TradeSequenceTree
from trader.libs.trade_sequence_tree import get_leafs_chain
from trader.libs.trade_sequence_tree import get_next_leafs_chains_report
# from trader.libs.trade_sequence_tree import get_leafs_chain_probability_report  # noqa

found_keys = ["a", "b", "c"]
not_found_keys = ["N"]
common_leaf_atributes = {
    "a": "up", "b": "down_variates", "c": "up_variates", "N": "unknown"}
key_count_map = {
    "up": 0, "up_variates": 1,
    "down": 2, "down_variates": 3,
    "equal": 4, "equal_variates": 5,
    "same": 6, "same_variates": 7,
    "unknown": 8}
tree = TradeSequenceTree(
    found_keys, not_found_keys,
    common_leaf_atributes, key_count_map)
root = tree.get_root()
a_1 = root.add_leaf("a", {"count": 3})
b_1 = root.add_leaf("b", {"count": 2})

n_1 = root.add_leaf("N", {"count": 3})

c_2 = a_1.add_leaf("c", {"count": 2})
a_1.add_leaf("a", {"count": 23})
a_1.add_leaf("N", {"count": 18})

b_1.add_leaf("b", {"count": 6})
b_1.add_leaf("a", {"count": 332})
b_1.add_leaf("N", {"count": 48})

b_3 = c_2.add_leaf("b", {"count": 2})
c_3 = c_2.add_leaf("c", {"count": 45})
a_3 = c_2.add_leaf("a", {"count": 12})
c_2.add_leaf("N", {"count": 2})

b_3.add_leaf("a", {"count": 2})
b_3.add_leaf("b", {"count": 22})
b_3.add_leaf("N", {"count": 4})

a_3.add_leaf("b", {"count": 12})
a_3.add_leaf("N", {"count": 42})
a_3.add_leaf("c", {"count": 2})

c_3.add_leaf("a", {"count": 3})
c_3.add_leaf("c", {"count": 10})
c_3.add_leaf("N", {"count": 2})

chain = get_leafs_chain(tree, ["a", "c"])
# print chain
# print leaf[-1].get_data()
# get_leafs_chain_probability_report(tree, ["a", "c"])

import sys
sys.setrecursionlimit(10000)
get_next_leafs_chains_report(tree, ["a", "c"], 1)
get_next_leafs_chains_report(tree, ["a"], 2)
from trader.libs.trade_sequence_tree import generate_tree_chart
from trader.libs.trade_sequence_tree import reset_tree_counters
generate_tree_chart("sample_sequence_tree_0.png", tree)
reset_tree_counters(tree.get_root())
# generate_tree_chart(tree)

from trader.libs.trade_sequence_tree import get_last_leafs
from trader.libs.trade_sequence_tree import get_leafs_samples_chain
last_leafs = [i for i in get_last_leafs(tree.get_root(), depth=5)]
print "last_leafs", [i.get_key() for i in last_leafs]
chains = []
for last_leaf in last_leafs:
    ch = [i.get_key() for i in get_leafs_samples_chain(
        last_leaf, tree.get_root())]
    ch.reverse()
    chains.append(ch)
print "chains", chains

import random
random.shuffle(chains)
sequence = []
for ch in chains:
    sequence.extend(ch)
    num = random.randint(0, 10)
    for i in range(num):
        choice = random.choice(["a", "b", "c", "N", "N"])
        sequence.extend(choice)

sequence = ['a', 'c', 'b', 'N', 'a', 'c', 'c', 'a', 'a', 'N', 'N', 'N', 'c', 'N', 'N', 'N', 'a', 'a', 'N', 'b', 'N', 'a', 'c', 'c', 'N', 'a', 'N', 'N', 'b', 'b', 'a', 'c', 'b', 'b', 'b', 'N', 'N', 'a', 'b', 'a', 'N', 'N', 'N', 'N', 'c', 'N', 'a', 'a', 'c', 'a', 'b', 'a', 'a', 'b', 'N', 'c', 'N', 'b', 'a', 'a', 'c', 'a', 'c', 'c', 'a', 'c', 'N', 'c', 'N', 'N', 'N', 'b', 'c', 'c', 'b', 'b', 'N', 'c', 'c', 'a', 'N', 'a', 'N', 'N', 'N', 'a', 'N', 'a', 'c', 'b', 'b', 'c', 'N', 'a', 'c', 'a', 'N', 'b', 'c', 'a', 'N', 'b', 'N', 'a', 'b', 'N', 'N', 'N', 'a', 'b', 'a', 'a', 'N', 'a', 'c', 'b', 'a', 'N', 'N', 'b', 'a', 'c', 'a', 'a', 'N', 'c', 'a', 'a', 'a', 'N', 'c', 'b', 'a', 'c', 'c', 'c', 'a', 'N', 'N', 'a', 'N', 'a']  # noqa
print sequence

from trader.libs.trade_sequence_tree import update_top_leafs_counts

# build tree from skeleton and sequence up
top_leafs_keys = [leaf.get_key() for leaf in tree.get_root().get_next_leafs()]

future_trails = {}
for key in top_leafs_keys:
    future_trails[key] = []

for incoming_sequence_key in sequence:
    update_top_leafs_counts(
        future_trails, incoming_sequence_key,
        tree, top_leafs_keys, True)

generate_tree_chart("sample_sequence_tree_1.png", tree)

"""
from trader.libs.trade_sequence_tree import \
    get_leafs_layers, get_leafs_probabilities
#    get_extended_leaf_next_probabilities
# print list(get_extended_leaf_next_probabilities(a_1, tree, depth=3))
for layer in get_leafs_layers([a_1], depth=3):
    print layer
    print "prob", get_leafs_probabilities(layer, tree)
    for i in layer:
        print i.get_key()
"""

"""
from trader.libs.trade_sequence_tree import \
    report_extended_leaf_next_probabilities
report_extended_leaf_next_probabilities(
    a_1, tree, 3)
"""

# delete tree counters as you walk through sequence
future_trails = {}
for key in top_leafs_keys:
    future_trails[key] = []

for incoming_sequence_key in sequence:
    print "Incoming key:", incoming_sequence_key
    # report_extended_leaf_next_probabilities(
    #    incoming_sequence_key, tree, 2)
    update_top_leafs_counts(
        future_trails, incoming_sequence_key,
        tree, top_leafs_keys, False, True, 3)
    print "\n"

generate_tree_chart("sample_sequence_tree_2.png", tree)
"""
import pydot

graph = pydot.Dot(graph_type='graph', size='100 80')

for i in range(3):
    edge = pydot.Edge("king", "lord%d" % i)
    graph.add_edge(edge)

vassal_num = 0
for i in range(3):
    for j in range(2):
        edge = pydot.Edge("lord%d" % i, "vassal%d" % vassal_num)
        graph.add_edge(edge)
        vassal_num += 1

graph.write_png('example1_graph.png', prog="fdp")
"""
