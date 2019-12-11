import json

import numpy as np
from collections import deque
from trader.libs.utils import get_start_and_end_trade_log_epochs


def get_closest_index(start_value, find_value, period):
    i = (float(find_value) - float(start_value)) / float(period)
    return int(round(i))


def get_closest_period_index_down(start_value, find_value, period):
    i = (float(find_value) - float(start_value)) / float(period)
    return int(i)


def average_data(data, counters):
    for index, value in enumerate(data):
        try:
            data[index] /= counters[index]
        except ZeroDivisionError:
            pass


class TradeData(object):

    def __init__(self, log_path, rescale_period):
        self.log_path = log_path
        self.rescale_period = rescale_period
        self.log_start, self.log_end = get_start_and_end_trade_log_epochs(
            log_path)
        # print self.rescale_period, self.log_start, self.log_end
        self.times = np.arange(
            float(self.log_start) + self.rescale_period,
            float(self.log_end) + 2 * self.rescale_period,
            self.rescale_period)
        self.counters = [0] * len(self.times)
        self.prices = [0.0] * len(self.times)
        self.trades = [0.0] * len(self.times)
        self.prices_times_trades = [0.0] * len(self.times)
        self.line_counter = self._rescale_data()
        self.gaps = [False] * len(self.counters)

    def get_rescale_period(self):
        return self.rescale_period

    def _rescale_data(self, average=True):
        price = 0.0
        line_counter = 0
        with open(self.log_path, "r") as fd:
            for line in fd:
                line_counter += 1
                time_r, data = line.split(" ", 1)
                time_r = float(time_r)
                data_hash = json.loads(data.strip())
                index = get_closest_period_index_down(
                    self.log_start,
                    time_r,
                    self.rescale_period)
                price = float(data_hash["price"])
                self.counters[index] += 1
                self.prices[index] += price
                self.trades[index] += float(data_hash["amount"])
                self.prices_times_trades[index] += \
                    price * float(data_hash["amount"])
                # trade_id = int(data_hash["id"])
            if average:
                average_data(self.prices, self.counters)
                average_data(self.trades, self.counters)
                average_data(self.prices_times_trades, self.counters)
            return line_counter


def get_up_patterns_stats(data, grow_rounding=2):
    sample_sizes = {}
    grow_sizes = {}
    last_sample_size = 0
    grow_size = 0
    for index, value in enumerate(data):
        if index != 0 and value > data[index-1]:
            last_sample_size += 1
            grow_size += value - data[index-1]
        else:
            if last_sample_size != 0:
                if last_sample_size not in sample_sizes:
                    sample_sizes.update({last_sample_size: 1})
                else:
                    sample_sizes[last_sample_size] += 1
                last_sample_size = 0
            if grow_size != 0:
                grow_size_rounded = round(grow_size, grow_rounding)
                if grow_size_rounded not in grow_sizes:
                    grow_sizes.update({grow_size_rounded: 1})
                else:
                    grow_sizes[grow_size_rounded] += 1
                grow_size = 0
    return sample_sizes, grow_sizes


def get_up_patterns_stats_2(prices, trades, grow_rounding=2):
    sample_sizes = {}
    grow_sizes = {}
    trade_sizes = {}
    last_sample_size = 0
    grow_size = 0
    trade_size = 0
    for index, value in enumerate(prices):
        if index != 0 and value > prices[index-1]:
            last_sample_size += 1
            grow_size += value - prices[index-1]
            trade_size += trades[index]
        else:
            if last_sample_size != 0:
                if last_sample_size not in sample_sizes:
                    sample_sizes.update({last_sample_size: 1})
                else:
                    sample_sizes[last_sample_size] += 1
                last_sample_size = 0
            if grow_size != 0:
                grow_size_rounded = round(grow_size, grow_rounding)
                if grow_size_rounded not in grow_sizes:
                    grow_sizes.update({grow_size_rounded: 1})
                    trade_sizes.update({grow_size_rounded: trade_size})
                else:
                    grow_sizes[grow_size_rounded] += 1
                    trade_sizes[grow_size_rounded] += trade_size
                grow_size = 0
                trade_size = 0
    return sample_sizes, grow_sizes, trade_sizes


def get_up_patterns_stats_3(prices, trades, counters, grow_rounding=2):
    sample_sizes = {}
    grow_sizes = {}
    trade_sizes = {}
    last_sample_size = 0
    grow_size = 0
    trade_size = 0
    for index, value in enumerate(prices):
        if counters[index] != 0 and prices[index-1] != 0.0:
            if index != 0 and value > prices[index-1]:
                last_sample_size += 1
                grow_size += value - prices[index-1]
                trade_size += trades[index]
            else:
                if last_sample_size != 0:
                    if last_sample_size not in sample_sizes:
                        sample_sizes.update({last_sample_size: 1})
                    else:
                        sample_sizes[last_sample_size] += 1
                    last_sample_size = 0
                if grow_size != 0:
                    grow_size_rounded = round(grow_size, grow_rounding)
                    if grow_size_rounded not in grow_sizes:
                        grow_sizes.update({grow_size_rounded: 1})
                        trade_sizes.update({grow_size_rounded: trade_size})
                    else:
                        grow_sizes[grow_size_rounded] += 1
                        trade_sizes[grow_size_rounded] += trade_size
                    grow_size = 0
                    trade_size = 0
    return sample_sizes, grow_sizes, trade_sizes


def get_grow_sample_percentage_stats(
        prices, counters, sample_size=5):
    last_sample_size = 0
    grow_percentage = {}
    for index, value in enumerate(prices):
        if counters[index] != 0:
            if index != 0:
                if prices[index-1] != 0.0:
                    if value > prices[index-1]:
                        last_sample_size += 1
                        if last_sample_size <= sample_size:
                            last_sample_size_percentage = int(
                                last_sample_size / (sample_size / 100.0))
                            if last_sample_size_percentage not in grow_percentage:  # noqa
                                grow_percentage.update(
                                    {last_sample_size_percentage: 1})
                            else:
                                grow_percentage[last_sample_size_percentage] += 1  # noqa
                        else:
                            # new interval
                            # print "D"
                            last_sample_size = 1
                    else:
                        # new value smaller than previous, so reset
                        # print "C"
                        last_sample_size = 0
                else:
                    # it looks like gap, so reset
                    # print "B"
                    last_sample_size = 0
            else:
                # first value
                # print "A"
                last_sample_size = 1
        else:
            # gap detected, so reset
            last_sample_size = 0
    return grow_percentage


def get_grow_sample_percentage_stats_2(
        prices, counters, sample_size=5):
    last_sample_size = 0
    grow_percentage = {}
    for index, value in enumerate(prices):
        if counters[index] != 0:
            if index != 0:
                if prices[index-1] != 0.0:
                    if value >= prices[index-1]:
                        last_sample_size += 1
                        if last_sample_size <= sample_size:
                            last_sample_size_percentage = int(
                                last_sample_size / (sample_size / 100.0))
                            if last_sample_size_percentage not in grow_percentage:  # noqa
                                grow_percentage.update(
                                    {last_sample_size_percentage: 1})
                            else:
                                grow_percentage[last_sample_size_percentage] += 1  # noqa
                        else:
                            # new interval
                            # print "D"
                            last_sample_size = 1
                    else:
                        # new value smaller than previous, so reset
                        # print "C"
                        last_sample_size = 0
                else:
                    # it looks like gap, so reset
                    # print "B"
                    last_sample_size = 0
            else:
                # first value
                # print "A"
                last_sample_size = 1
        else:
            # gap detected, so reset
            last_sample_size = 0
    return grow_percentage


def get_grow_sample_percentage_stats_3(
        prices, counters, rescale_period):
    grow_counters = {}
    same_counters = {}
    less_counters = {}
    grow_less_counters = {}
    index = 0
    main_flag = True
    rescale_period_int = int(rescale_period)
    while main_flag:
        if index + rescale_period_int >= len(prices):
            main_flag = False
        else:
            if prices[index + rescale_period_int] > prices[index]:
                grow_counter = 0
                same_counter = 0
                less_counter = 0
                grow_less_counter = 0
                for i in range(rescale_period_int):
                    if prices[index + i + 1] > prices[index + i]:
                        grow_counter += 1
                        grow_less_counter += 1
                    elif prices[index + i + 1] == prices[index + i]:
                        same_counter += 1
                    else:
                        grow_less_counter += 1
                        less_counter += 1
                if grow_counter not in grow_counters:
                    grow_counters.update({grow_counter: 1})
                else:
                    grow_counters[grow_counter] += 1
                if same_counter not in same_counters:
                    same_counters.update({same_counter: 1})
                else:
                    same_counters[same_counter] += 1
                if less_counter not in less_counters:
                    less_counters.update({less_counter: 1})
                else:
                    less_counters[less_counter] += 1
                if grow_less_counter not in grow_less_counters:
                    grow_less_counters.update({grow_less_counter: 1})
                else:
                    grow_less_counters[grow_less_counter] += 1
                index += rescale_period_int
            else:
                index += 1
    return less_counters, same_counters, grow_counters, grow_less_counters


def gain_up_estimate(
        grow_sizes, trade_sizes, cut_percentage=0.1, trade_percentage=0.25):
    gains = {}
    for grow_size, grow_count in grow_sizes.items():
        gain = grow_size * grow_count * trade_sizes[grow_size] * \
            cut_percentage * trade_percentage
        gains.update({grow_size: gain})
    return gains


class SampleTradeDataBuffer(object):

    def __init__(self, sample_size, rescale_period, initial_epoch):
        # extra size for averaging of the gaps at start of queue
        self.sample_size = sample_size
        extra_sample_size = sample_size + 1
        # the oldest data is always at beginning of buffers
        self.sample_price_buffer = deque([None] * extra_sample_size)
        self.sample_trade_buffer = deque([None] * extra_sample_size)
        self.sample_record_counter = deque([None] * extra_sample_size)
        self.sample_epoch_buffer = deque([None] * extra_sample_size)
        self.current_period_price = 0.0
        self.current_period_trade = 0.0
        self.current_period_record = 0
        self.rescale_period = rescale_period
        self.epoch = initial_epoch

    def _push_values_to_buffers(self):
        try:
            self.sample_price_buffer.append(
                self.current_period_price / self.current_period_record)
            self.sample_price_buffer.popleft()
            self.sample_trade_buffer.append(
                self.current_period_trade / self.current_period_record)
            self.sample_trade_buffer.popleft()
            self.sample_record_counter.append(
                self.current_period_record)
            self.sample_record_counter.popleft()
            self.sample_epoch_buffer.append(self.epoch)
            self.sample_epoch_buffer.popleft()
        except ZeroDivisionError, e:
            print e

    def _reset_period_values(self):
        self.current_period_price = 0.0
        self.current_period_trade = 0.0
        self.current_period_record = 0

    def _update_period_values(self, data):
        try:
            data_hash = json.loads(data.strip())
            self.current_period_price += float(data_hash["price"])
            self.current_period_trade += float(data_hash["amount"])
            self.current_period_record += 1
        except Exception, e:
            print "Problem to parse data " + str(data)
            print e

    def update(self, line):
        periods_counter = 0
        time_r, data = line.split(" ", 1)
        time_r = float(time_r)
        # is record in period ?
        if time_r < self.epoch:
            self._update_period_values(data)
        else:
            # was there gap without data ?
            delta = time_r - self.epoch
            # push accumulated data into buffers
            self._push_values_to_buffers()
            self._reset_period_values()
            if delta > self.rescale_period:
                # push gaps into buffers
                gaps_counter = int(delta // self.rescale_period)
                for current_buffer in self.sample_price_buffer, self.sample_trade_buffer, self.sample_record_counter:  # noqa
                    for i in xrange(gaps_counter):
                        current_buffer.append(0.0)
                        current_buffer.popleft()
                for i in xrange(gaps_counter):
                    self.epoch += self.rescale_period
                    periods_counter += 1
                    self.sample_epoch_buffer.append(self.epoch)
                    self.sample_epoch_buffer.popleft()
                if delta % self.rescale_period > 0:
                    self.epoch += self.rescale_period
                    periods_counter += 1
                # first value in next period
                self._update_period_values(data)
            else:
                # first value in next period
                self.epoch += self.rescale_period
                periods_counter += 1
                self._update_period_values(data)
        return periods_counter

    def is_buffer_full(self):
        return self.sample_price_buffer.count(None) == 0

    def get_prices(self):
        return [x for x in self.sample_price_buffer if x is not None]

    def get_epochs(self):
        return [x for x in self.sample_epoch_buffer if x is not None]

    def get_trades(self):
        return [x for x in self.sample_trade_buffer if x is not None]

    def get_records_counter(self):
        return [x for x in self.sample_record_counter if x is not None]


def fill_empty_gaps(data, record_counters):
    for index, value in enumerate(record_counters):
        if index != 0 and value == 0:
            if index+1 < len(record_counters):
                if data[index+1] > 0 and data[index-1] > 0:
                    data[index] = (data[index-1] + data[index+1]) / 2
                else:
                    data[index] = data[index-1]
            else:
                data[index] = data[index-1]
