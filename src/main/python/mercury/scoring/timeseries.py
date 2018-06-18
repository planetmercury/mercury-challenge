# timeseries.py
# Code for time series manipulations and predictions.
__author__ = "Pete Haglich, peter.haglich@iarpa.gov"

import os
import json

import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
from dateutil.parser import parse


icews_prop_path = os.path.join(
  os.getcwd(), "..", "resources", "scoring", "icews_properties.json"
  )
with open(icews_prop_path, "r") as f:
    icews_props = json.loads(f.read())
# The REFERENCE_WEEKDAY is used to anchor ISO Weeks.
# The value in icews_properties.json, 2, corresponds to Wednesday.
REFERENCE_WEEKDAY = icews_props["REFERENCE_WEEKDAY"]


def ewma_predict(history_ser, halflife=1, n_ahead=1):
    """
    Fits an exponentially weighted moving average of the given order
    :param n_ahead: How far ahead to predict
    :param history_ser: Series of past values
    :param halflife: Halflife in EWMA calculation
    :returns: Non-negative integer prediction
    """
    ewma_ser = pd.ewma(history_ser, halflife=halflife)
    predict = ewma_ser[-1]
    predict = round(predict, 0)
    predict = max(predict, 0.0)

    pred_dates = future_dates(history_ser.index, n_ahead)[1:]
    predict_ser = pd.Series(n_ahead*[int(predict)],
                            index=pred_dates)
    predict_ser = predict_ser.apply(lambda x: 1.0*x)

    return {"Predictions":predict_ser, "Model": "EWMA", "Model_Params": "Halflife {0}".format(halflife)}


def arma_predict(history_ser, verbose=False,
                 n_ahead=1, dynamic=False,
                 **arma_args):
    """
    Fits an ARMA model of the given order
    :param dynamic: Argument for the predict method of ARMA
    :param n_ahead: how far ahead to predict
    :param history_ser: Series of past values
    :param order: Order of the ARMA model
    :param verbose: Print some details
    :returns: Non-negative integer prediction
    """
    if not arma_args:
        arma_args = {"trend": "nc",
                     "max_ar": 4, "max_ma": 4}
    else:
        if "trend" not in arma_args:
            arma_args["trend"] = "nc"
    if "order" not in arma_args:
        result = sm.tsa.stattools.arma_order_select_ic(history_ser,
                                                       **arma_args)
        arma_args["order"] = result.bic_min_order
    if verbose:
        print("Fitting ARMA with order {0}".format(arma_args["order"]))
    model = sm.tsa.ARMA(history_ser,
                        order=arma_args["order"]).fit()
    if verbose:
        print(model.summary())

    pred_dates = future_dates(history_ser.index, n_ahead)
    first_date = str(pred_dates[0].date())
    last_date = str(pred_dates[-1].date())
    predict_ser = model.predict(first_date, last_date, dynamic=dynamic)
    predict_ser = predict_ser.apply(lambda x: round(x, 0))
    predict_ser = predict_ser.apply(lambda x: max(x, 0.0))

    return {"Predictions": predict_ser.iloc[1:], "Model": "ARMA",
            "Model_Params": "Order ({0}, {1})".format(*arma_args["order"])}


def future_dates(dt_index, n_ahead=1):
    """
    Compute n_ahead dates using the dt_index; start with the last value
    of the index
    """

    first_date = dt_index[-1]

    if dt_index.freqstr[0] == "W":
        last_date = first_date + datetime.timedelta(7 * n_ahead)

    else:
        last_date = first_date + datetime.timedelta(n_ahead)

    future_dates = pd.date_range(first_date, last_date,
                                 freq=dt_index.freq)
    return future_dates


def status_quo_predict(history_ser, n_ahead=1):
    """
    Fits a constant model
    :param history_ser: Series of past values
    :returns: Non-negative integer prediction
    """

    pred_dates = future_dates(history_ser.index, n_ahead)[1:]
    predict_ser = pd.Series(n_ahead * [history_ser.values[-1]],
                            index=pred_dates)
    predict_ser = predict_ser.apply(lambda x: 1.0*x)

    return {"Predictions":predict_ser, "Model": "Status Quo", "Model_Params": "-"}


def naive_predict(history_ser, n_ahead=1, level=0):
    """
    Fits a constant model
    :param n_ahead: How many predictions to generate
    :param history_ser: Series of past values
    :param level: The constant value
    :returns: Non-negative integer prediction
    """

    pred_dates = future_dates(history_ser.index, n_ahead)[1:]
    predict_ser = pd.Series(n_ahead * [level],
                            index=pred_dates)
    predict_ser = predict_ser.apply(lambda x: 1.0*x)

    return {"Predictions":predict_ser, "Model": "Naive", "Model_Params": "Constant Level {0}".format(level)}


def hist_avg_predict(history_ser, halflife=1, n_ahead=1, freq=52):
    """
    Computes the historical averages for a series modulo the frequency
    :param in_ser: A list of values
    :param halflife: Default halflife to be used in EWMA computation, default 1
    :param n_ahead: How many predictions to generate
    :param freq: The length of the season to be studied
    :returns: List of historical averages
    """
    if len(history_ser) < freq:
        # No history to use, just take the EWMA
         prediction = n_ahead * [pd.ewma(history_ser, halflife=halflife).iloc[-1]]
    elif freq <= len(history_ser) < 2 * freq:
        # Then there's only one possible history data point.
        curr_level = pd.ewma(history_ser, halflife=halflife).iloc[-1]
        old_hist_ser = history_ser.iloc[:(len(history_ser) - (freq - 1))]
        old_hist_values = old_hist_ser.iloc[:n_ahead]
        old_hist_values = list(old_hist_values)
        old_hist_level = pd.ewma(old_hist_ser, halflife=halflife).iloc[-2]
        scale = curr_level / old_hist_level
        prediction = [scale * hv for hv in old_hist_values]
    else:
        # Do the legitimate averaging
        old_hist_ser = history_ser.iloc[:(len(history_ser) - (freq - 1))]
        old_hist_values = list(old_hist_ser.values)
        curr_level = pd.ewma(pd.Series(history_ser), halflife=halflife).iloc[-1]
        # print("Current level", curr_level)
        rev_hist_ser = [old_hist_values.pop() for x in old_hist_values.copy()]
        # print("Reverse History Series", rev_hist_ser)
        rev_avg_ser = []
        for k in range(freq):
            # print(k)
            k_values = [rev_hist_ser[i] for i in range(len(rev_hist_ser)) if (i) % freq == k]
            # print(k_values)
            rev_avg_ser.append(np.mean(k_values))
        # print("Reverse Average Series", rev_avg_ser)
        avg_ser = [rev_avg_ser.pop() for x in rev_avg_ser.copy()]
        # print("Historical average series", avg_ser)
        hist_level = pd.ewma(pd.Series(avg_ser), halflife=halflife).iloc[-2]
        # print("Historical level for comparison", hist_level)
        hist_values = [avg_ser[-1]] + avg_ser[:n_ahead - 1]
        # print("Historical values for the period of interest", hist_values)
        scale = curr_level / hist_level
        # print("Scale Factor", scale)
        prediction = [scale * hv for hv in hist_values]

    prediction = pd.Series([round(p, 0) for p in prediction])
    pred_dates = future_dates(history_ser.index, n_ahead)[1:]
    prediction.index=pred_dates
    out_dict = {"Predictions": prediction, "Model": "Historical Average Weighted By Recent Trend",
                "Model_Params": "Period {0}, Halflife {1}".format(freq, halflife)}
    return out_dict


def extrapolate(history_ser, predict_method, n_ahead=1, **predict_args):
    value_ser = history_ser.copy()
    if predict_method == arma_predict and "order" not in predict_args:
        # Fit the existing data and determine the best order
        result = sm.tsa.stattools.arma_order_select_ic(value_ser,
                                                       **predict_args)
        predict_args["order"] = result.bic_min_order

    try:
        for i in range(n_ahead):
            next_value = predict_method(value_ser, **predict_args)
            # print(next_value)
            value_ser = np.append(value_ser, next_value)
            # print(value_ser)
    except ValueError:  # Give it one more try with a reduced order
        if predict_args["order"][1] > 0:
            predict_args["order"] = (predict_args["order"][0],
                                     predict_args["order"][1] - 1)
            for i in range(n_ahead):
                next_value = predict_method(value_ser, **predict_args)
                # print(next_value)
                value_ser = np.append(value_ser, next_value)
                # print(value_ser)

    history_end_date = history_ser.index[-1]
    warn_period_start = history_end_date + datetime.timedelta(1)
    warn_period_end = history_end_date + datetime.timedelta(n_ahead * 7 + 1)
    warn_dates = pd.date_range(warn_period_start, warn_period_end,
                               freq="W-WED")
    warn_ser = pd.Series(value_ser[-1 * n_ahead:],
                         index=warn_dates)

    return warn_ser


def next_day_of_week(ref_date, day_of_week=2):
    """
    Finds the next occurrence of the given day of the week on or after the ref_date.
    We use a default of 2 for Wednesday.
    :param ref_date: The anchor date
    :param day_of_week: Which day of the week to find?  Monday = 0, etc.
    :return: date
    """
    if isinstance(ref_date, str):
        ref_date = parse(ref_date)
    days_ahead = day_of_week - ref_date.weekday()
    if days_ahead < 0:  # Target date already happened
        days_ahead += 7
    return ref_date + datetime.timedelta(days_ahead)


def prev_day_of_week(ref_date, day_of_week=2):
    """
    Finds the previous occurrence of the given day of the week on or before the ref_date
    :param ref_date: The anchor date
    :param day_of_week: Which day of the week to find?  Monday = 0, etc.
    :return: date
    """
    if isinstance(ref_date, str):
        ref_date = parse(ref_date)
    days_behind = ref_date.weekday() - day_of_week
    if days_behind < 0:  # Target date already happened
        days_behind += 7
    return ref_date - datetime.timedelta(days_behind)

def iso_year_start(iso_year):
    """The gregorian calendar date of the first day of the given ISO year
    :param iso_year: The ISO year for which we are trying to find the starting date
    :return: First day of the ISO year.
    """
    fourth_jan = datetime.date(iso_year, 1, 4)
    delta = datetime.timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta


def iso_to_gregorian(iso_year, iso_week, iso_day):
    """Gregorian calendar date for the given ISO year, week and day
    :param iso_year: ISO Year
    :param iso_week: ISO Week
    :param iso_day: ISO Day
    :return: Gregorian date for the given ISO Year/Week/Day
    """
    year_start = iso_year_start(iso_year)
    return year_start + datetime.timedelta(days=iso_day-1, weeks=iso_week-1)


def isoweek_to_gregorian(iso_week):
    """Takes (year, week, day) object and calls iso_to_gregorian"""
    yy = iso_week[0]
    ww = iso_week[1]
    dd = iso_week[2]
    return iso_to_gregorian(yy, ww, dd)


def convert_to_refday(dd, refday=REFERENCE_WEEKDAY):
    """
    Determines the reference day for the given ISO day
    :param dd: Date of interest
    :param refday: Reference day for ISO weeks.  Default is REFERENCE_WEEKDAY
    :return: Gregorian date for reference day for the given ISO day
    """
    dd_iso = dd.isocalendar()
    ref_iso = (dd_iso[0], dd_iso[1], refday)
    ref_date = iso_to_gregorian(*ref_iso)
    return ref_date

