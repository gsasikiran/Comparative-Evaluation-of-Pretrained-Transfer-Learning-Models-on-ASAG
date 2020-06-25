from sklearn.linear_model import LinearRegression,Ridge
from sklearn.isotonic import IsotonicRegression
import numpy as np


class RegressionAnalysis:
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def __check_nan(self, array):

        NaNs_index = np.isnan(array)
        array[NaNs_index] = 0

        return array

    def linear(self):
        clf = LinearRegression()

        train_x = self.__check_nan(self.train_x.to_numpy().reshape(-1,1))
        train_y = self.__check_nan(self.train_y.to_numpy().reshape(-1, 1))
        test_x = self.__check_nan(self.test_x.to_numpy().reshape(-1, 1))

        for val in train_x:
            if np.isnan(val) or not np.isfinite(val):
                print(val)

        clf.fit(train_x, train_y)
        test_y_pred = clf.predict(test_x)
        return test_y_pred

    def ridge(self):
        clf = Ridge()

        train_x = self.__check_nan(self.train_x.to_numpy().reshape(-1, 1))
        train_y = self.__check_nan(self.train_y.to_numpy().reshape(-1, 1))
        test_x = self.__check_nan(self.test_x.to_numpy().reshape(-1, 1))

        clf.fit(train_x, train_y)
        test_y_pred = clf.predict(test_x)
        return test_y_pred


    def isotonic(self):

        clf = IsotonicRegression()
        train_x = self.train_x.to_list()
        train_y = self.train_y.to_list()
        test_x = self.test_x.to_list()
        clf.fit(train_x, train_y)
        test_y_pred = clf.predict(test_x)
        return test_y_pred
