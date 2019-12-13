class LinearNormalisation(object):

    def __init__(self, min_value, max_value, lower_limit=0.0, upper_limit=1.0):
        self.min_value = min_value
        self.max_value = max_value
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        # y = ax + b
        self.a, self.b = self._linear(
            self.min_value, self.max_value, self.lower_limit, self.upper_limit)

    def _linear(self, a, b, c, d):
        """
        Returns coefficients of linear equation a, b for:
            y = ax + b
        from range (a,b) to (c,d)
        """
        """
        if b == a: raise ValueError(
          "Mapping not possible due to equal limits
          ")
        """
        if b == a:
            c1 = 0.0
            c2 = (c + d) / 2.
        else:
            c1 = (d - c) / (b - a)
            c2 = c - a * c1
        return c1, c2

    def normalise_value(self, value):
        return self.a * value + self.b

    def xnormalise_array(self, array):
        for value in array:
            yield self.normalise_value(value)

    def normalise_array(self, array):
        return list(self.xnormalise_array(array))
