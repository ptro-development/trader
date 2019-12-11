import unittest

from libs.event import dict_to_event
from mock import patch, call, ANY, MagicMock
from trader.event_consumer_multi_tasks import test_event_in_facts_relations, \
    event_initial_match, process_second_event

event_dict = {'sample': {'+correlation_positions': [460, 1113, 5414, 5995], 'sample_position': 2281, 'sample_data': [253.99358490566024, 254.6093055555557, 254.3870000000001, 253.6676470588235, 253.75169230769237, 254.48589743589736, 254.625, 255.67019607843145, 257.05054545454544, 256.8244444444444, 256.6986666666666, 256.62055555555554, 256.0177777777777, 257.0244, 261.6499415204678, 263.0266379310346, 263.0508695652174, 263.3938888888888, 262.84647058823526, 262.3411999999999, 262.86206896551727, 263.56928571428574, 263.22999999999996, 263.09590909090906, 262.04600000000005], 'sample_epoch': 1426795010, '+correlation_positions_epochs': [1426248710, 1426444610, 1427734910, 1427909210], 'rescale_period': 300, '-correlation_positions': [4429], 'required_correlation': 0.975, '-correlation_positions_epochs': [1427439410], 'sample_attributes': {'min_position_value': [3, 253.6676470588235], 'max_position_value': [21, 263.56928571428574], 'up_or_down': 'variates'}}, 'percentage': 8.0, 'relative_match_position': 1426117310, 'incoming_data': [296.5728571428572, 296.6625, 296.505, 296.66, 296.1533333333333, 296.305, 296.2331818181819, 296.2033333333333, 296.40500000000003, 296.1914285714286, 296.2, 296.2, 296.72, 296.11555555555555, 296.49285714285713, 295.94199999999995, 295.16295454545457, 294.4082758620688, 294.2005, 293.9911111111112, 294.62923076923073, 294.90250000000003, 294.89, 294.91, 294.9128571428572], 'correlation': 1.0}  # noqa

facts_relations = {
    u"25": {u'-1427048810_-1427049110_25_True': 3, u'+1427048810_+1427049110_25_True': 2, u'+1426794410_+1426795010_25_True': 2},  # noqa
    u"50": {u'+1427048810_+1427049110_50_True': 2, u'-1427048810_-1427049110_50_True': 3, u'+1426794410_+1426795010_50_True': 2},  # noqa
    u"100": {u'+1427048810_+1427049110_100_True': 2, u'-1427048810_-1427049110_100_True': 3, u'+1426794410_+1426795010_100_True': 2},  # noqa
    u"125": {u'+1426794410_+1426795010_125_True': 2, u'-1426743110_-1426743110_1500_True': 2, u'-1427048810_-1427049110_125_True': 3, u'+1426745210_-1426743110_125_True': 2, u'+1427048810_+1427049110_125_True': 2},  # noqa
    u"150": {u'-1426743110_-1426743110_1500_True': 3, u'-1427199710_-1426743110_150_True': 2, u'+1426795010_+1426745210_450_True': 3, u'+1426745210_-1426743110_3450_True': 2, u'+1426794410_+1426795010_150_True': 2, u'-1427201510_-1426743110_150_True': 2, u'-1426743110_-1427201510_1350_True': 2, u'-1427048810_-1427049110_150_True': 3, u'+1426745210_-1426743110_150_True': 2, u'+1427048810_+1427049110_150_True': 2, u'-1426743110_-1427199710_1350_True': 2},  # noqa
    u"175": {u'-1427199710_-1426743110_175_True': 2, u'-1427201510_-1426743110_175_True': 2, u'-1426743110_+1426745210_1575_True': 2, u'+1426745210_-1426743110_4900_True': 2, u'-1426743110_-1427201510_1400_True': 2, u'+1427048810_+1427049110_175_True': 2, u'+1426745210_-1426743110_175_True': 2, u'+1426745210_-1426743110_3500_True': 2, u'-1427048810_-1427049110_175_True': 3, u'+1426794410_+1426795010_175_True': 2, u'-1426743110_-1427199710_1400_True': 2, u'+1426745210_+1427048810_1575_True': 2},  # noqa
}

# grep "\"percentage\": 100.0" | grep "\"sample_epoch\": 1427048810"
first_event_dict = {"sample": {"+correlation_positions": [2548, 3063, 5363, 6436], "-correlation_positions": [805, 1707, 2910, 5523], "rescale_period": 300, "-correlation_positions_epochs": [1426352210, 1426622810, 1426983710, 1427767610], "required_correlation": 0.975, "sample_position": 3127, "sample_epoch": 1427048810, "+correlation_positions_epochs": [1426875110, 1427029610, 1427719610, 1428041510], "sample_data": [260.42333333333335, 260.25909090909096, 260.51, 260.47249999999997, 260.50363636363636, 260.5166666666666, 260.68666666666667, 260.90722222222223, 260.99333333333334, 261.5811764705882, 261.9155555555556, 263.22021052631595, 263.7484375, 264.25535714285724, 264.91275862068983, 265.46508196721317, 265.10930232558144, 265.32771428571425, 267.80144444444466, 268.1751612903226, 268.0549999999998, 268.49826086956523, 267.84909090909093, 268.1871428571429, 268.20842105263154], "sample_attributes": {"min_position_value": [0, 260.25909090909096], "max_position_value": [21, 268.49826086956523], "up_or_down": "variates"}}, "percentage": 100.0, "correlation": -0.97572865764670513, "incoming_data": [287.178, 287.01545454545465, 286.99, 286.992, 287.16785714285714, 286.98, 286.89666666666665, 286.532, 286.00115384615384, 286.45000000000005, 285.48777777777775, 285.1862068965518, 285.18966666666677, 285.3532352941176, 285.4045, 284.78783783783786, 284.3145454545454, 284.17833333333334, 283.3264179104478, 282.8413333333334, 283.80857142857144, 283.13391304347823, 282.96833333333336, 282.9633333333333, 282.70125], "relative_match_position": 1426352210}  # noqa

# grep "\"percentage\": 60.0" | grep "\"sample_epoch\": 1427049110"
second_event_dict = {"sample": {"+correlation_positions": [2549, 3064, 6437], "-correlation_positions": [806, 1708, 2911, 5524], "rescale_period": 300, "-correlation_positions_epochs": [1426352510, 1426623110, 1426984010, 1427767910], "required_correlation": 0.975, "sample_position": 3128, "sample_epoch": 1427049110, "+correlation_positions_epochs": [1426875410, 1427029910, 1428041810], "sample_data": [260.25909090909096, 260.51, 260.47249999999997, 260.50363636363636, 260.5166666666666, 260.68666666666667, 260.90722222222223, 260.99333333333334, 261.5811764705882, 261.9155555555556, 263.22021052631595, 263.7484375, 264.25535714285724, 264.91275862068983, 265.46508196721317, 265.10930232558144, 265.32771428571425, 267.80144444444466, 268.1751612903226, 268.0549999999998, 268.49826086956523, 267.84909090909093, 268.1871428571429, 268.20842105263154, 267.7508], "sample_attributes": {"min_position_value": [0, 260.25909090909096], "max_position_value": [20, 268.49826086956523], "up_or_down": "variates"}}, "percentage": 60.0, "correlation": -0.97907939875608263, "incoming_data": [296.71166666666664, 297.36965517241373, 296.918, 296.9866666666667, 296.590625, 296.73100000000005, 296.973076923077, 296.8107692307692, 296.24375, 296.535, 296.91400000000004, 296.8085714285714, 296.86571428571426, 296.72437500000007, 296.70000000000005, 296.6038461538462, 296.4492857142856, 296.73894736842107, 296.42923076923074, 296.27222222222224, 296.19692307692316, 295.9408571428573, 295.7312500000001, 295.6351515151515, 295.3762500000001], "relative_match_position": 1426189610}  # noqa


class TestEventInFactsRelations(unittest.TestCase):

    global event_dict
    global facts_relations

    @patch("trader.event_consumer_multi_tasks.get_fact_relation_key_dict")  # noqa
    def test_test_event_in_facts_relations_max_focus(
            self, get_fact_relation_key_dict_mock):
        first_vent = second_event = dict_to_event(event_dict)
        test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        self.assertEqual(
            [
                get_fact_relation_key_dict_mock.mock_calls[0],
                get_fact_relation_key_dict_mock.mock_calls[3],
                get_fact_relation_key_dict_mock.mock_calls[6]
            ],
            [
                call(u'+1426794410_+1426795010_25_True'),
                call(u'+1427048810_+1427049110_25_True'),
                call(u'-1427048810_-1427049110_25_True')
            ]
        )
        self.assertEqual(
            len(get_fact_relation_key_dict_mock.mock_calls), 9)

    def test_test_event_in_facts_relations_success(
            self):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        self.assertEqual(
            (found, relation_key),
            (True, "-1427048810_-1427049110_25_True")
        )

    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_sample_events_exist_in_facts_relations_success(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, True]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        expected_calls = [
            call(ANY, ANY, {'first_fact_sample_position': u'+1426794410', 'facts_order': u'True', 'second_fact_sample_position': u'+1426795010', 'facts_distance': u'25'}),  # noqa
            call(ANY, ANY, {'first_fact_sample_position': u'+1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'+1427049110', 'facts_distance': u'25'}),  # noqa
            call(ANY, ANY, {'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'})]  # noqa
        test_events_exist_in_facts_relations_mock.assert_has_calls(
            expected_calls)
        test_events_happened_min_relation_times_mock.assert_has_calls(
            [call(3, 1)])

    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_sample_events_exist_in_facts_relations_fail(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, False]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        expected_calls = [
            call(ANY, ANY, {'first_fact_sample_position': u'+1426794410', 'facts_order': u'True', 'second_fact_sample_position': u'+1426795010', 'facts_distance': u'25'}),  # noqa
            call(ANY, ANY, {'first_fact_sample_position': u'+1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'+1427049110', 'facts_distance': u'25'}),  # noqa
            call(ANY, ANY, {'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'})]  # noqa
        test_events_exist_in_facts_relations_mock.assert_has_calls(
            expected_calls)
        test_events_happened_min_relation_times_mock.assert_has_calls([])

    @patch("trader.event_consumer_multi_tasks.test_events_right_to_left_order")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_test_events_happened_min_relation_times_success(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock,
            test_events_right_to_left_order_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, True]
        test_events_happened_min_relation_times_mock.side_effect = [True]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        test_events_happened_min_relation_times_mock.assert_has_calls(
            [call(3, 1)])
        test_events_right_to_left_order_mock.assert_has_calls(
            [call(ANY, ANY, {'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'})])  # noqa

    @patch("trader.event_consumer_multi_tasks.test_events_right_to_left_order")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_test_events_happened_min_relation_times_fail(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock,
            test_events_right_to_left_order_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, True]
        test_events_happened_min_relation_times_mock.side_effect = [False]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        test_events_happened_min_relation_times_mock.assert_has_calls(
            [call(3, 1)])
        test_events_right_to_left_order_mock.assert_has_calls([])

    @patch("trader.event_consumer_multi_tasks.test_events_same_max_focus_distance")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_right_to_left_order")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_test_events_right_to_left_order_success(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock,
            test_events_right_to_left_order_mock,
            test_events_same_max_focus_distance_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, True]
        test_events_happened_min_relation_times_mock.side_effect = [True]
        test_events_right_to_left_order_mock.side_effect = [True]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        test_events_right_to_left_order_mock.assert_has_calls(
            [call(ANY, ANY, {'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'})])  # noqa
        test_events_same_max_focus_distance_mock.assert_has_calls(
            [call({'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'}, '25')])  # noqa

    @patch("trader.event_consumer_multi_tasks.test_events_same_max_focus_distance")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_right_to_left_order")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_happened_min_relation_times")  # noqa
    @patch("trader.event_consumer_multi_tasks.test_events_exist_in_facts_relations")  # noqa
    def test_test_event_in_facts_relations_test_events_right_to_left_order_fail(  # noqa
            self,
            test_events_exist_in_facts_relations_mock,
            test_events_happened_min_relation_times_mock,
            test_events_right_to_left_order_mock,
            test_events_same_max_focus_distance_mock):
        first_vent = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        test_events_exist_in_facts_relations_mock.side_effect = [
            False, False, True]
        test_events_happened_min_relation_times_mock.side_effect = [True]
        test_events_right_to_left_order_mock.side_effect = [False]
        found, relation_key = test_event_in_facts_relations(
            facts_relations,
            first_vent,
            second_event,
            min_relation_appearance=1,
            max_focus=35)
        test_events_right_to_left_order_mock.assert_has_calls(
            [call(ANY, ANY, {'first_fact_sample_position': u'-1427048810', 'facts_order': u'True', 'second_fact_sample_position': u'-1427049110', 'facts_distance': u'25'})])  # noqa
        test_events_same_max_focus_distance_mock.assert_has_calls([])


class TestEventInitialMatch(unittest.TestCase):

    global event_dict
    global facts_relations

    def test_event_initial_match_is_min_required_percentage_success(
            self):
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.return_value = False
        match = event_initial_match(
            facts_relations=None,
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance=None,
            max_focus=None,
            min_required_correlation_percentage=45.0)
        self.assertTrue(second_event.is_sample_attribute_up.called)
        self.assertFalse(match)

    def test_event_initial_match_is_min_required_percentage_fail(
            self):
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.side_effect = False
        match = event_initial_match(
            facts_relations=None,
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance=None,
            max_focus=None,
            min_required_correlation_percentage=80.0)
        self.assertFalse(second_event.is_sample_attribute_up.called)
        self.assertFalse(match)

    @patch("trader.event_consumer_multi_tasks.test_event_in_facts_relations")
    def test_event_initial_match_is_sample_attribute_up_success(
            self,
            test_event_in_facts_relations_mock):
        test_event_in_facts_relations_mock.return_value = (False, "some_key")
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.return_value = True
        match = event_initial_match(
            facts_relations="fact_relations",
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance="min_relation_appearance",
            max_focus="max_focus",
            min_required_correlation_percentage=45.0)
        test_event_in_facts_relations_mock.assert_called_once_with(
            "fact_relations", first_event, second_event,
            "min_relation_appearance", "max_focus")
        self.assertFalse(match)

    @patch("trader.event_consumer_multi_tasks.test_event_in_facts_relations")
    def test_event_initial_match_is_sample_attribute_up_fail(
            self,
            test_event_in_facts_relations_mock):
        test_event_in_facts_relations_mock.return_value = (False, "some_key")
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.return_value = False
        match = event_initial_match(
            facts_relations="fact_relations",
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance="min_relation_appearance",
            max_focus="max_focus",
            min_required_correlation_percentage=45.0)
        self.assertFalse(test_event_in_facts_relations_mock.called)
        self.assertFalse(match)

    @patch("trader.event_consumer_multi_tasks.buy")
    @patch("trader.event_consumer_multi_tasks.test_event_in_facts_relations")
    def test_event_initial_match_test_event_in_facts_relations_success(
            self,
            test_event_in_facts_relations_mock,
            buy_mock):
        test_event_in_facts_relations_mock.return_value = (True, "some_key")
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.return_value = True
        match = event_initial_match(
            facts_relations="fact_relations",
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance="min_relation_appearance",
            max_focus="max_focus",
            min_required_correlation_percentage=45.0)
        buy_mock.assert_called_with(first_event, second_event, "some_key")
        self.assertTrue(match)

    @patch("trader.event_consumer_multi_tasks.buy")
    @patch("trader.event_consumer_multi_tasks.test_event_in_facts_relations")
    def test_event_initial_match_test_event_in_facts_relations_fail(
            self,
            test_event_in_facts_relations_mock,
            buy_mock):
        test_event_in_facts_relations_mock.return_value = (False, "some_key")
        first_event = dict_to_event(first_event_dict)
        second_event = dict_to_event(second_event_dict)
        second_event.is_sample_attribute_up = MagicMock()
        second_event.is_sample_attribute_up.return_value = True
        match = event_initial_match(
            facts_relations="fact_relations",
            first_event=first_event,
            second_event=second_event,
            min_relation_appearance="min_relation_appearance",
            max_focus="max_focus",
            min_required_correlation_percentage=45.0)
        self.assertFalse(buy_mock.called)
        self.assertFalse(match)


class TestProcessSecondEvent(unittest.TestCase):

    global event_dict
    global facts_relations

    @patch("trader.event_consumer_multi_tasks.event_initial_match.apply_async")
    @patch("trader.event_consumer_multi_tasks.dict_to_event")
    def test_process_second_event(
            self,
            dict_to_event_mock,
            apply_async_mock):
        dict_to_event_mock.side_effect = [
            "first_event", "second_effect"]
        process_second_event(
            first_event_dict,
            second_event_dict,
            facts_relations)
        self.assertEqual(
            dict_to_event_mock.mock_calls,
            [call(first_event_dict), call(second_event_dict)])
        apply_async_mock.assert_called_once_with(
            ('first_event', 'second_effect', ANY, ANY, ANY))
