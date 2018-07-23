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
from sklearn.metrics import mean_squared_error

import sys
sys.path.append("../..")
from ExpressScore.main.express_score import (
    CaseCountScorer,
    Defaults
)


def mean_qs_error(predicted, actual, accuracy_denominator=Defaults.ACCURACY_DENOMINATOR):
    """
    Computes an error based on the CaseCountScorer.quality_score metric
    :param predicted: List of predicted values
    :param actual: List of actual values
    :param accuracy_denominator: Scaling factor
    :return: Mean of 1 - QS
    """

    if len(predicted) != len(actual):
        raise ValueError("We need the same number of predicted and actual values")
    else:
        value_zip = zip(predicted, actual)
        errors = [(1-CaseCountScorer.quality_score(p, a, accuracy_denominator=accuracy_denominator))
                   for (p,a) in value_zip]
        mean_error = np.mean(errors)

    return mean_error


class TsDefaults:
    """
    Stores default values
    """

    ERROR_FUNCTION = mean_qs_error
    P_MAX = 4
    D_MAX = 1
    Q_MAX = 4


# The REFERENCE_WEEKDAY is used to anchor ISO Weeks.
# The value in icews_properties.json, 2, corresponds to Wednesday.
REFERENCE_WEEKDAY = 3

def ewma_predict(history_ser, halflife=1, n_ahead=1):
    """
    Fits an exponentially weighted moving average of the given order
    :param n_ahead: How far ahead to predict
    :param history_ser: Series of past values
    :param halflife: Halflife in EWMA calculation
    :returns: Non-negative integer prediction
    """

    ewma_ser = history_ser.ewm(halflife=halflife).mean()
    predict = ewma_ser.iloc[-1]
    predict = round(predict, 0)
    predict = max(predict, 0.0)

    pred_dates = future_dates(history_ser.index, n_ahead)[1:]
    predict_ser = pd.Series(n_ahead*[int(predict)],
                            index=pred_dates)
    predict_ser = predict_ser.apply(lambda x: 1.0*x)

    return {"Predictions":predict_ser, "Model": "EWMA", "Model_Params": "Halflife {0}".format(halflife)}


def arima_predict(history_ser, order, verbose=False,
                  n_ahead=1, dynamic=True):
    """
    Fits an ARMA model of the given order
    :param history_ser: Series of past values
    :param order: Order of the ARMA model
    :param dynamic: Argument for the predict method of ARMA, default True
    :param n_ahead: how far ahead to predict
    :param verbose: Print some details
    :returns: Non-negative integer prediction
    """
    model = sm.tsa.ARIMA(history_ser,
                        order=order).fit()
    if verbose:
        print(model.summary())

    pred_dates = future_dates(history_ser.index, n_ahead)
    first_date = str(pred_dates[1].date())
    last_date = str(pred_dates[-1].date())
    predict_ser = model.predict(first_date, last_date, dynamic=dynamic)
    predict_ser = predict_ser.apply(lambda x: round(x, 0))
    predict_ser = predict_ser.apply(lambda x: max(x, 0.0))

    return {"Predictions": predict_ser, "Model": "ARMA",
            "Model_Params": "Order ({0}, {1})".format(*order)}


def future_dates(dt_index, n_ahead=1):
    """
    Compute n_ahead dates using the dt_index; start with the last value
    of the index
    """

    first_date = dt_index[-1]

    if dt_index.freqstr is None:
        # Then we have to infer the time step size
        use_freq = "W-SUN"
        use_delta=7

    elif dt_index.freqstr[0] == "W":
        use_delta = 7
        use_freq = dt_index.freq

    else:
        use_delta = 1
        use_freq = dt_index.freq

    last_date = first_date + datetime.timedelta(use_delta*n_ahead)

    future_dates = pd.date_range(first_date, last_date,
                                 freq=dt_index.freq)
    return pd.DatetimeIndex(start=dt_index[-1], freq=use_freq, periods=n_ahead+1)


def status_quo_predict(history_ser, n_ahead=1):
    """
    Fits a constant model
    :param history_ser: Series of past values
    :param n_ahead: How many steps ahead to predict
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


def hist_avg_predict(history_ser, n_ahead=1, season_freq=52):
    """
    Computes the historical averages for a series modulo the frequency
    :param history_ser: A list of values
    :param n_ahead: How many predictions to generate
    :param season_freq: The length of the season to be studied
    :returns: List of historical averages
    """
    if len(history_ser) < season_freq:
        # No history to use, just take the EWMA
         raise ValueError("Not enough elements in the history")
    elif n_ahead > season_freq:
        # Trying to predict too far ahead
        raise ValueError("Trying to predict farther ahead than the periodicity")
    else:
        # Then there's only one possible history data point.
        curr_level = history_ser.iloc[-1]
        hist_values = list(history_ser.values)
        popable_values = hist_values.copy()
        rev_values = [popable_values.pop() for v in hist_values]
        curr_comp_values = [v for i,v in enumerate(rev_values[1:])
                            if i % (season_freq) == season_freq - 1]
        curr_comp_level = np.mean(curr_comp_values)
        scale = curr_level/curr_comp_level
        predictions = []
        for k in range(1, n_ahead+1):
            modulus = (len(hist_values) + k) % season_freq
            comp_values = [v for i,v in enumerate(hist_values)
                           if (i+1) % season_freq == modulus]
            comp_level = np.mean(comp_values)
            predictions.append(comp_level*scale)

        predictions = pd.Series([round(p, 0) for p in predictions])
        pred_dates = future_dates(history_ser.index, n_ahead)[1:]
        predictions.index=pred_dates
        out_dict = {"Predictions": predictions, "Model": "Historical Average Weighted By Recent Trend",
                    "Model_Params": "Period {0}".format(season_freq)}
        return out_dict


def extrapolate(history_ser, predict_method, n_ahead=1, **predict_args):
    value_ser = history_ser.copy()
    if predict_method == arima_predict and "order" not in predict_args:
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


def antilog_err_func(err_func, predicted, actual, offset=1):
    """
    Applies err_func to predicted and actual after taking antilog
    :param err_func: A function that computes error
    :param predicted: predicted values
    :param actual: actual values
    :param offset: amount to be subtracted to convert to original scale
    :return: Error value
    """
    antilog_predicted = pd.Series(predicted).apply(lambda x: np.exp(x) - offset)
    antilog_predicted = antilog_predicted.apply(lambda x: round(x, 0))
    antilog_predicted = antilog_predicted.apply(lambda x: max(x, 0))
    antilog_actual = pd.Series(actual).apply(lambda x: np.exp(x) - offset)
    antilog_actual = antilog_actual.apply(lambda x: np.exp(x) - offset)
    antilog_actual = antilog_actual.apply(lambda x: np.exp(x) - offset)
    return err_func(antilog_predicted, antilog_actual)


def evaluate_arima_model(X, arima_order, err_func=TsDefaults.ERROR_FUNCTION, verbose=False):
    """
    Evaluate an ARIMA model for given order.  Copied from https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python.html
    :param X: Time series to fit
    :param arima_order: (p,d,q) model parameters
    :param err_func: How do we measure errors?  Default uses the Quality Score deficit
    :return: Mean Squared Error
    """
    out_dict = dict()
    error_ = float("inf")
    out_dict["Model_Params"] = []
    out_dict["Order"] = arima_order
    train_size = int(len(X)*2.0/3)
    train, test = X[:train_size], X[train_size:]
    history = [t for t in train]
    predictions = list()
    try:
        for t in range(len(test)):
            model = sm.tsa.ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            yhat = int(round(yhat[0], 0))
            yhat = max(yhat, 0)
            predictions.append(yhat)
            history.append(test[t])

        # Calculate out of sample error
        error_ = err_func(test, predictions)
        out_dict["Model_Params"] = model_fit.params
    except (ValueError, np.linalg.LinAlgError) as e:
        if verbose:
            print(repr(e))
        else:
            pass
    out_dict["Error"] = error_
    return out_dict


def evaluate_models(dataset, p_values=range(TsDefaults.P_MAX+1), d_values=range(TsDefaults.D_MAX+1),
                    q_values=range(TsDefaults.Q_MAX+1), err_func=TsDefaults.ERROR_FUNCTION,
                    verbose=False):
    """
    Evaluates a collection of ARIMA models across a parameter space.  Copied from https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python.html
    :param dataset: set to be fit
    :param p_values: AR
    :param d_values: Diff
    :param q_values: MA
    :param err_func: Name of the method to be used to compute error
    :param verbose:  How much output to pring
    :return: best fit
    """
    out_dict = dict()
    dataset = dataset.astype("float32")
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                if verbose:
                    print("Order: ", order)
                model_eval = evaluate_arima_model(dataset, order, err_func, verbose=verbose)
                error_ = model_eval["Error"]
                current_params = model_eval["Model_Params"]
                if verbose: print(error_)
                if error_ < best_score:
                    best_score, best_cfg, model_params = error_, order, current_params
                    if isinstance(best_score, list):
                        best_score = best_score[-1]
                    if verbose:
                        print(best_score)
                        print("New leader in the clubhouse: ARIMA ", order, "Mean Error = {:.3f}".format(best_score))
    if verbose:
        print("Final Best ARIMA ", best_cfg, "Mean Error = {:.3f}".format(best_score))
    out_dict["Order"] = best_cfg
    out_dict["Model_Params"] = model_params
    out_dict["Error"] = best_score

    return out_dict


def one_step_ahead_forecast(train, test, order):
    """
    Computes forecasts one step ahead incrementally
    :param train: Training series
    :param test:  Test series
    :param order:  A model specification order
    :return: Series of predictions
    """
    preds = []
    train_ = train.copy()
    for i, t in enumerate(test.index):
        model = sm.tsa.ARIMA(endog=train_, order=order)
        model_fit = model.fit(disp=False)
        predict = model_fit.forecast(steps=1)[0][0]
        preds.append(predict)
        train_ = pd.concat((train_, test[t:t]))
    pred_ser = pd.Series(preds, index=test.index)
    return pred_ser
