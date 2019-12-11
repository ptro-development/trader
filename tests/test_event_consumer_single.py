import unittest
import json
from libs.event import dict_to_event
from mock import patch, MagicMock

from trader.event_consumer_single_tasks import is_first_event_match, \
    prepare_second_event_matches_tasks, update_second_event_match_status, \
    process_incoming_event

from trader.event_consumer_single_tasks import expected_trades

event_dict = {'sample': {'+correlation_positions': [460, 1113, 5414, 5995], 'sample_position': 2281, 'sample_data': [253.99358490566024, 254.6093055555557, 254.3870000000001, 253.6676470588235, 253.75169230769237, 254.48589743589736, 254.625, 255.67019607843145, 257.05054545454544, 256.8244444444444, 256.6986666666666, 256.62055555555554, 256.0177777777777, 257.0244, 261.6499415204678, 263.0266379310346, 263.0508695652174, 263.3938888888888, 262.84647058823526, 262.3411999999999, 262.86206896551727, 263.56928571428574, 263.22999999999996, 263.09590909090906, 262.04600000000005], 'sample_epoch': 1426795010, '+correlation_positions_epochs': [1426248710, 1426444610, 1427734910, 1427909210], 'rescale_period': 300, '-correlation_positions': [4429], 'required_correlation': 0.975, '-correlation_positions_epochs': [1427439410], 'sample_attributes': {'min_position_value': [3, 253.6676470588235], 'max_position_value': [21, 263.56928571428574], 'up_or_down': 'variates'}}, 'percentage': 8.0, 'relative_match_position': 1426117310, 'incoming_data': [296.5728571428572, 296.6625, 296.505, 296.66, 296.1533333333333, 296.305, 296.2331818181819, 296.2033333333333, 296.40500000000003, 296.1914285714286, 296.2, 296.2, 296.72, 296.11555555555555, 296.49285714285713, 295.94199999999995, 295.16295454545457, 294.4082758620688, 294.2005, 293.9911111111112, 294.62923076923073, 294.90250000000003, 294.89, 294.91, 294.9128571428572], 'correlation': 1.0}  # noqa


class TestProcessIncomingEvent(unittest.TestCase):

    global event_dict

    @patch("trader.event_consumer_single_tasks.is_first_event_match")
    @patch("trader.event_consumer_single_tasks.prepare_second_event_matches_tasks")  # noqa
    @patch("trader.event_consumer_single_tasks.write_log.apply_async")
    @patch("trader.event_consumer_single_tasks.time")
    def test_process_incoming_event_write_log_triggered(
            self,
            time_mock,
            log_mock,
            prepare_second_event_matches_tasks_mock,
            is_first_event_match_mock):
        is_first_event_match_mock.return_value = False
        prepare_second_event_matches_tasks_mock.return_value = []
        time_mock.time.return_value = "sometimes_afternoon"
        line = json.dumps(
            {"time": "sometimes_afternoon", "event": event_dict}) + "\n"
        process_incoming_event(event_dict, [])
        log_mock.assert_called_once_with((line,))

    @patch("trader.event_consumer_single_tasks.write_log.apply_async")
    @patch("trader.event_consumer_single_tasks.dict_to_event")
    @patch("trader.event_consumer_single_tasks.is_first_event_match")
    @patch("trader.event_consumer_single_tasks.prepare_second_event_matches_tasks")  # noqa
    def test_process_incoming_event_is_first_event_success(
            self,
            prepare_second_event_matches_tasks_mock,
            is_first_event_match_mock,
            dict_to_event_mock,
            log_mock):
        dict_to_event_mock.return_value = MagicMock()
        expected_trades_loc = [{
            'first_event': dict_to_event_mock.return_value,
            'second_event_found': False}]
        is_first_event_match_mock.return_value = True
        prepare_second_event_matches_tasks_mock.return_value = []
        process_incoming_event(event_dict, [])
        prepare_second_event_matches_tasks_mock.assert_called_once_with(
            expected_trades_loc,
            event_dict)

    @patch("trader.event_consumer_single_tasks.write_log.apply_async")
    @patch("trader.event_consumer_single_tasks.dict_to_event")
    @patch("trader.event_consumer_single_tasks.is_first_event_match")
    @patch("trader.event_consumer_single_tasks.prepare_second_event_matches_tasks")  # noqa
    def test_process_incoming_event_is_first_event_false(
            self,
            prepare_second_event_matches_tasks_mock,
            is_first_event_match_mock,
            dict_to_event_mock,
            log_mock):
        dict_to_event_mock.return_value = MagicMock()
        is_first_event_match_mock.return_value = False
        prepare_second_event_matches_tasks_mock.return_value = []
        process_incoming_event(event_dict, [])
        prepare_second_event_matches_tasks_mock.assert_called_once_with(
            [],
            event_dict)

    @patch("trader.event_consumer_single_tasks.update_second_event_match_status")  # noqa
    @patch("trader.event_consumer_single_tasks.write_log.apply_async")
    @patch("trader.event_consumer_single_tasks.dict_to_event")
    @patch("trader.event_consumer_single_tasks.is_first_event_match")
    @patch("trader.event_consumer_single_tasks.prepare_second_event_matches_tasks")  # noqa
    def test_process_incoming_event_tasks_available(
            self,
            prepare_second_event_matches_tasks_mock,
            is_first_event_match_mock,
            dict_to_event_mock,
            log_mock,
            update_second_event_match_status_mock):
        dict_to_event_mock.return_value = MagicMock()
        is_first_event_match_mock.return_value = True
        prepare_second_event_matches_tasks_mock.return_value = ["something"]
        expected_trades_loc = [{
            'first_event': dict_to_event_mock.return_value,
            'second_event_found': False}]
        print "expected_trades", expected_trades
        process_incoming_event(event_dict, [])
        prepare_second_event_matches_tasks_mock.assert_called_once_with(
            expected_trades_loc,
            event_dict)
        update_second_event_match_status_mock.assert_called_once_with(
            expected_trades_loc,
            ["something"]
        )

    @patch("trader.event_consumer_single_tasks.update_second_event_match_status")  # noqa
    @patch("trader.event_consumer_single_tasks.write_log.apply_async")
    @patch("trader.event_consumer_single_tasks.dict_to_event")
    @patch("trader.event_consumer_single_tasks.is_first_event_match")
    @patch("trader.event_consumer_single_tasks.prepare_second_event_matches_tasks")  # noqa
    def test_process_incoming_event_tasks_not_available(
            self,
            prepare_second_event_matches_tasks_mock,
            is_first_event_match_mock,
            dict_to_event_mock,
            log_mock,
            update_second_event_match_status_mock):
        dict_to_event_mock.return_value = MagicMock()
        is_first_event_match_mock.return_value = True
        prepare_second_event_matches_tasks_mock.return_value = []
        expected_trades_loc = [{
            'first_event': dict_to_event_mock.return_value,
            'second_event_found': False}]
        print "expected_trades", expected_trades
        process_incoming_event(event_dict, [])
        prepare_second_event_matches_tasks_mock.assert_called_once_with(
            expected_trades_loc,
            event_dict)
        self.assertFalse(update_second_event_match_status_mock.called)


class TestIsFirstEventMatch(unittest.TestCase):

    global event_dict

    def setUp(self):
        self.event_dict_55_percent = event_dict.copy()
        self.event_dict_55_percent.update({"percentage": 55})
        self.event_55_percent = dict_to_event(self.event_dict_55_percent)

    def test_is_first_event_match_min_required_percentage_success(self):
        self.assertTrue(
            is_first_event_match(
                [], self.event_55_percent, 10))
        self.assertTrue(
            is_first_event_match(
                [], self.event_55_percent, 55))

    def test_is_first_event_match_min_required_percentage_fail(self):
        self.assertFalse(
            is_first_event_match(
                [], self.event_55_percent, 70))

    def test_is_first_event_match_min_required_trade_registered(self):
        new_event_dict = event_dict.copy()
        new_event_dict.update({"correlation": 0.45})
        expected_trades = [
            {
                "first_event": dict_to_event(new_event_dict),
                "second_event_found": False
            },
            {
                "first_event": self.event_55_percent,
                "second_event_found": False
            }
        ]
        self.assertFalse(
            is_first_event_match(
                expected_trades, self.event_55_percent, 10))

    def test_is_first_event_match_min_required_trade_not_registered(self):
        new_event_dict = event_dict.copy()
        new_event_dict.update({"correlation": 0.45})
        new_event_dict_another = event_dict.copy()
        new_event_dict_another.update({"correlation": 0.55})
        expected_trades = [
            {
                "first_event": dict_to_event(new_event_dict),
                "second_event_found": False
            },
            {
                "first_event": dict_to_event(new_event_dict_another),
                "second_event_found": False
            }
        ]
        self.assertTrue(
            is_first_event_match(
                expected_trades, self.event_55_percent, 10))


class TestPrepareSecondEventMatchesTasks(unittest.TestCase):

    @patch("trader.event_consumer_single_tasks.process_second_event.s")
    def test_prepare_second_event_matches_tasks_one_trade_ready(
            self, process_second_event_s_mock):
        new_event_dict = event_dict.copy()
        new_event_dict.update({"correlation": 0.45})
        new_event_dict_another = event_dict.copy()
        new_event_dict_another.update({"correlation": 0.55})
        second_event_dict = event_dict.copy()
        second_event_dict.update({"correlation": 0.25})
        expected_trades = [
            {
                "first_event": dict_to_event(new_event_dict),
                "second_event_found": True
            },
            {
                "first_event": dict_to_event(new_event_dict_another),
                "second_event_found": False
            }
        ]
        process_second_event_s_mock.return_value = "Something"
        result = prepare_second_event_matches_tasks(
            expected_trades, second_event_dict)
        process_second_event_s_mock.assert_called_once_with(
            new_event_dict_another, second_event_dict)
        self.assertEqual(result, ["Something"])

    @patch("trader.event_consumer_single_tasks.process_second_event.s")
    def test_prepare_second_event_matches_tasks_no_trade_ready(
            self, process_second_event_s_mock):
        new_event_dict = event_dict.copy()
        new_event_dict.update({"correlation": 0.45})
        new_event_dict_another = event_dict.copy()
        new_event_dict_another.update({"correlation": 0.55})
        second_event_dict = event_dict.copy()
        second_event_dict.update({"correlation": 0.25})
        expected_trades = [
            {
                "first_event": dict_to_event(new_event_dict),
                "second_event_found": True
            },
            {
                "first_event": dict_to_event(new_event_dict_another),
                "second_event_found": True
            }
        ]
        result = prepare_second_event_matches_tasks(
            expected_trades, second_event_dict)
        self.assertFalse(process_second_event_s_mock.called)
        self.assertEqual(result, [])

    @patch("trader.event_consumer_single_tasks.process_second_event.s")
    def test_prepare_second_event_matches_tasks_two_trades_ready(
            self, process_second_event_s_mock):
        second_event_dict = "new_2"
        expected_trades = [
            {
                "first_event": MagicMock(),
                "second_event_found": False
            },
            {
                "first_event": MagicMock(),
                "second_event_found": False
            }
        ]
        expected_trades[0]["first_event"].get_dict.return_value = "new_0"
        expected_trades[1]["first_event"].get_dict.return_value = "new_1"
        result = prepare_second_event_matches_tasks(
            expected_trades, second_event_dict)
        self.assertEqual(result, [
            process_second_event_s_mock.return_value,
            process_second_event_s_mock.return_value])
        expected_trades[0]["first_event"].get_dict.assert_called_once()
        expected_trades[1]["first_event"].get_dict.assert_called_once()


class TestUpdateSecondventMatchStatus(unittest.TestCase):

    @patch("trader.event_consumer_single_tasks.group")
    def test_update_second_event_match_status_no_status_change(
            self, group_mock):
        expected_trades = [
            {"second_event_found": False},
            {"second_event_found": False}]
        tasks = [MagicMock(), MagicMock()]
        results = [MagicMock(), MagicMock()]
        results[0].get.return_value = False
        results[1].get.return_value = False
        group_mock.return_value.apply_async.return_value.get.return_value = results  # noqa
        update_second_event_match_status(expected_trades, tasks)
        group_mock.assert_called_once_with(tasks)
        results[0].get.assert_called_once()
        results[1].get.assert_called_once()
        self.assertFalse(expected_trades[0]["second_event_found"])
        self.assertFalse(expected_trades[1]["second_event_found"])

    @patch("trader.event_consumer_single_tasks.group")
    def test_update_second_event_match_status_one_status_change(
            self, group_mock):
        expected_trades = [
            {"second_event_found": False},
            {"second_event_found": False}]
        tasks = [MagicMock(), MagicMock()]
        results = [MagicMock(), MagicMock()]
        results[0].get.return_value = False
        results[1].get.return_value = True
        group_mock.return_value.apply_async.return_value.get.return_value = results  # noqa
        update_second_event_match_status(expected_trades, tasks)
        group_mock.assert_called_once_with(tasks)
        results[0].get.assert_called_once()
        results[1].get.assert_called_once()
        self.assertFalse(expected_trades[0]["second_event_found"])
        self.assertTrue(expected_trades[1]["second_event_found"])

    @patch("trader.event_consumer_single_tasks.group")
    def test_update_second_event_match_status_two_status_change(
            self, group_mock):
        expected_trades = [
            {"second_event_found": False},
            {"second_event_found": False}]
        tasks = [MagicMock(), MagicMock()]
        results = [MagicMock(), MagicMock()]
        results[0].get.return_value = True
        results[1].get.return_value = True
        group_mock.return_value.apply_async.return_value.get.return_value = results  # noqa
        update_second_event_match_status(expected_trades, tasks)
        group_mock.assert_called_once_with(tasks)
        results[0].get.assert_called_once()
        results[1].get.assert_called_once()
        self.assertTrue(expected_trades[0]["second_event_found"])
        self.assertTrue(expected_trades[1]["second_event_found"])
