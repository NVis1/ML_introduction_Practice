import numpy as np


class Percent:
    val: int | float

    def __init__(self, val: int | float | str):
        if isinstance(val, str):
            if not val.endswith('%'):
                raise ValueError("Given value is a non-percentage string.")

            val = val.rstrip('%')
            if not val.replace(".", "", 1).isdigit():
                raise ValueError("Given value is a non-percentage string.")

            val = float(val)

        self.val = val

    def __repr__(self):
        return self.val

    def __str__(self):
        return str(self.val) + "%"

    def __int__(self):
        return int(self.val * 100)

    def __float__(self):
        return float(self.val * 100)

    def __add__(self, other: int | float | np.number):
        if isinstance(other, Percent):
            return Percent(other.val + self.val)
        elif isinstance(other, (int, float, np.number)):
            return other + other*float(self.val)
        else:
            raise ValueError

    def __sub__(self, other: int | float | np.number):
        if isinstance(other, Percent):
            return Percent(other.val - self.val)
        elif isinstance(other, (int, float, np.number)):
            return other - other*float(self.val)
        else:
            raise ValueError
