"""
class Event:

    def __init__(
            self, relative_match_position, sample,
            incomming_data, correlation, percentage):
        self.data = {
            "relative_match_position": relative_match_position,
            "sample": sample,
            "incoming_data": incomming_data,
            "correlation": correlation,
            "percentage": percentage,
        }

    def is_percentage_equal_or_bigger(self, percentage):
        return percentage <= self.data["percentage"]

    def __str__(self):
        return str(self.data)

    def get(self):
        return self.data
"""
"""
def filter_events(events, percentage=45.0):
    return [e for e in events
            if e.is_percentage_equal_or_bigger(percentage) is True]
"""
