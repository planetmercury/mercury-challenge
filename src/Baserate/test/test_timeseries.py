import unittest
import sys
sys.path.append("..")
from main.timeseries import *
import os
import json

import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
from dateutil.parser import parse


class TestTimeseries(unittest.TestCase):

    test_wed_index = pd.DatetimeIndex(start="2018-01-03", end="2018-07-04", freq="W-WED")
    test_monthly_index = pd.DatetimeIndex(start="2015-01-01", end="2018-06-01", freq="MS")
    test_daily_index = pd.DatetimeIndex(start="2018-06-01", end="2018-06-30", freq="D")

    def test_ewma_predict(self):
        """
        Tests timeseries.ewma_predict
        :return:
        """
        start_date = "2018-07-01"
        hist_values = [1, 4, 3, 5, 7, 8, 5, 6, 3]
        ewm_avg = pd.Series(hist_values).ewm(halflife=1).mean().iloc[-1]
        history_dates = pd.date_range(start=parse(start_date), periods=len(hist_values))
        history_ser = pd.Series(hist_values, index=history_dates)
        future_start = parse(start_date) + datetime.timedelta(days=len(hist_values))
        future_start = future_start.strftime("%Y-%m-%d")
        expected = pd.Series([ewm_avg],
                              index=pd.date_range(start=future_start, periods=1))
        expected = expected.apply(lambda x: round(x, 0))
        result = ewma_predict(history_ser, halflife=1, n_ahead=1)["Predictions"]
        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        # Multiple days ahead
        expected = pd.Series([ewm_avg]*3,
                              index=pd.date_range(start=future_start, periods=3))
        expected = expected.apply(lambda x: round(x, 0))
        result = ewma_predict(history_ser, halflife=1, n_ahead=3)["Predictions"]
        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_arma_predict(self):
        """
        Tests timeseries.arma_predict
        :return:
        """
        start_date = "2018-07-01"
        hist_values = [1, 4, 3, 5, 7, 8, 5, 6, 3]
        history_dates = pd.date_range(start=parse(start_date), periods=len(hist_values))
        history_ser = pd.Series(hist_values, index=history_dates)
        future_start = parse(start_date) + datetime.timedelta(days=len(hist_values))
        future_start = future_start.strftime("%Y-%m-%d")
        test_order = (1,0,0)
        model = sm.tsa.ARMA(history_ser, order=test_order).fit()
        future_end = parse(start_date) + datetime.timedelta(days=len(hist_values) + 1)
        future_end = future_end.strftime("%Y-%m-%d")
        expected = model.predict(first_date=future_start, last_date=future_end)
        args_ = {"order": test_order}
        result = arma_predict(history_ser, **args_)
        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_future_dates(self):
        """
        Tests timeseries.future_dates
        :return:
        """
        # Weekly
        expected = pd.date_range(start="2018-07-04", end="2018-07-25", freq="W-WED")
        result = future_dates(self.test_wed_index, 3)
        try:
            pd.testing.assert_index_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        #Monthly
        expected = pd.date_range(start="2018-06-01", end="2018-12-31", freq="MS")
        result = future_dates(self.test_monthly_index, n_ahead=6)
        try:
            pd.testing.assert_index_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        #Daily
        expected = pd.date_range(start="2018-06-30", end="2018-07-07")
        result = future_dates(self.test_daily_index, n_ahead=7)
        try:
            pd.testing.assert_index_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_status_quo_predict(self):
        """
        Tests timeseries.status_quo_predict
        :return:
        """
        values = range(6)
        history_dates = pd.date_range(start="2018-07-01", periods=6)
        history = pd.Series(values, index=history_dates)
        future_dates = pd.date_range(start="2018-07-07", periods=2)
        predictions = [5, 5]
        expected = pd.Series(predictions, index=future_dates)*1.0
        result = status_quo_predict(history, n_ahead=2)["Predictions"]
        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_naive_predict(self):
        """
        Tests timeseries.naive_predict
        :return:
        """
        values = range(6)
        history_dates = pd.date_range(start="2018-07-01", periods=6)
        history = pd.Series(values, index=history_dates)
        future_dates = pd.date_range(start="2018-07-07", periods=2)
        predictions = [0, 0]
        expected = pd.Series(predictions, index=future_dates)*1.0
        result = naive_predict(history, level=0, n_ahead=2)["Predictions"]
        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_hist_avg_predict(self):
        """
        Tests timeseries.hist_avg_predict
        :return:
        """
        values = [1,2,3,5,8,9,6]
        history_dates = pd.date_range(start="2018-07-01", periods=7)
        history = pd.Series(values, index=history_dates)
        future_dates = pd.date_range(start="2018-07-08", periods=2)
        predictions = [10.0, 12.0]
        expected = pd.Series(predictions, index=future_dates)
        result = hist_avg_predict(history, n_ahead=2, freq=3)["Predictions"]

        try:
            pd.testing.assert_series_equal(expected, result)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        self.assertRaises(ValueError, hist_avg_predict, history, 4, 3)
        self.assertRaises(ValueError, hist_avg_predict, history[:2], 4, 3)


    def test_extrapolate(self):
        """
        Tests timeseries.extrapolate
        :return:
        """
        pass



    def test_prev_day_of_week(self):
        """
        Tests timeseries.prev_day_of_week method
        :return:
        """
        test_day = datetime.date(2018, 7, 4)
        expected = datetime.date(2018, 7, 2)
        result = prev_day_of_week(test_day, 0)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 6, 30)
        result = prev_day_of_week(test_day, 5)
        self.assertEqual(expected, result)


    def test_next_day_of_week(self):
        """
        Tests timeseries.next_day_of_week method
        :return:
        """
        test_day = datetime.date(2018, 7, 2)
        expected = datetime.date(2018, 7, 4)
        result = next_day_of_week(test_day)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 7, 7)
        result = next_day_of_week(test_day, 5)
        self.assertEqual(expected, result)


    def test_iso_year_start(self):
            """
            Tests timeseries.iso_year_start method
            :return:
            """
            yy = 2018
            expected = parse("2018-01-01").date()
            result = iso_year_start(yy)
            self.assertEqual(result, expected)
            yy = 2019
            expected = parse("2018-12-31").date()
            result = iso_year_start(yy)
            self.assertEqual(result, expected)


    def test_iso_to_gregorian(self):
        """
        Tests timeseries.iso_to_gregorian method
        :return:
        """

        expected = datetime.date(2018, 1, 1)
        result = iso_to_gregorian(2018,1,1)
        self.assertEqual(expected, result)

        expected = datetime.date(2018, 12, 31)
        result = iso_to_gregorian(2019, 1, 1)
        self.assertEqual(expected, result)

        expected = datetime.date(2018, 7, 4)
        result = iso_to_gregorian(2018, 27, 3)
        self.assertEqual(expected, result)

    def test_isoweek_to_gregorian(self):
        """
        Tests timeseries.isoweek_to_gregorian method
        :return:
        """
        expected = datetime.date(2018, 1, 1)
        test_iso = (2018, 1, 1)
        result = isoweek_to_gregorian(test_iso)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 12, 31)
        test_iso = (2019, 1, 1)
        result = isoweek_to_gregorian(test_iso)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 7, 4)
        test_iso = (2018, 27, 3)
        result = isoweek_to_gregorian(test_iso)
        self.assertEqual(expected, result)


    def test_convert_to_refday(self):
        """
        Tests timeseries.convert_to_refday
        :return:
        """

        expected = datetime.date(2018, 1, 3)
        test_dd = parse("2018-01-01")
        result = convert_to_refday(test_dd)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 1, 3)
        test_dd = parse("2018-01-02")
        result = convert_to_refday(test_dd)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 1, 3)
        test_dd = parse("2018-01-05")
        result = convert_to_refday(test_dd)
        self.assertEqual(expected, result)
        expected = datetime.date(2019, 1, 2)
        test_dd = parse("2018-12-31")
        result = convert_to_refday(test_dd)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 7, 4)
        test_dd = parse("2018-07-02")
        result = convert_to_refday(test_dd)
        self.assertEqual(expected, result)
        expected = datetime.date(2018, 7, 3)
        test_dd = parse("2018-07-02")
        result = convert_to_refday(test_dd, 2)
        self.assertEqual(expected, result)
