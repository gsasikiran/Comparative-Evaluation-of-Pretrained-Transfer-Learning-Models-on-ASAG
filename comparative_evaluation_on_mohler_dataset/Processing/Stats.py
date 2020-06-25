from math import sqrt

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np


class Metrics:
    def __init__(self, x, y):
        self.x = self.__check_nan(np.asarray(x))
        self.y = self.__check_nan(np.asarray([round(i*2)/2  for i in y]))

    def __check_nan(self, array):

        NaNs_index = np.isnan(array)
        array[NaNs_index] = 0

        return array
    def rmse(self):
        for val in self.y:
            if np.isnan(val) or not np.isfinite(val):
                print(val)
        return sqrt(mean_squared_error(self.x, self.y))

    def pearson_correlation(self):
        mean_x = sum(self.x) / len(self.x)
        mean_y = sum(self.y) / len(self.y)
        cov = sum((a - mean_x) * (b - mean_y) for (a, b) in zip(self.x, self.y)) / len(self.x)

        std_x, std_y = np.std(self.x), np.std(self.y)

        p = cov / (std_x * std_y)

        return float(p)

    def spearman_correlation(self):
        return spearmanr(self.x, self.y)
