def get_price_volatility_of_interval(start_position, interval, prices):
    volatility = 0.0
    before_start_position = start_position - interval
    if before_start_position >= 0 and before_start_position < len(prices):
        for index in range(before_start_position, start_position, 1):
            # always price delta against previous value
            volatility += abs(prices[index] - prices[index + 1])
        # calculate average
        volatility /= interval
    # In case there was not enough history calculate volatility from
    # what is available.
    #
    # TODO:
    #   It would be probably better to signal somehow that volatility can not be
    #   calculated.
    elif start_position > 0:
        for index in range(0, start_position, 1):
            volatility += abs(prices[index] - prices[index + 1])
        volatility /= start_position
    return volatility


def get_price_volatilities(
        sample_size, start_position_in_sequence, prices, intervals=[0.3, 0.7, 1.6, 3]):
    volatilities = []
    volatilities_intervals = map(
        lambda x:int(x * sample_size),
        intervals)
    for interval in volatilities_intervals:
        volatilities.append(
            get_price_volatility_of_interval(start_position_in_sequence, interval, prices))
    return volatilities
