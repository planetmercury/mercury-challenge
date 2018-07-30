'''
This module provides the classes that facilitate generation of baserate
warnings.  It is lightly modified from Pete Haglich's original prototype.
'''

import sys
sys.path.append("../..")

import datetime
import re
from datetime import timedelta
from uuid import uuid4

import pandas as pd
from dateutil.parser import parse
from ExpressScore.main.schema import (
  JSONField,
  EventType,
  DiseaseType,
  CountryName,
  Subtype,
  db
  )
from numpy.random import (
  choice,
  poisson,
  uniform
)
from pandas import DataFrame, Series

from .historian import Historian

from .loaddictionaries import (
  REASON_VALUES,
  POPULATION_VALUES,
  TARGET_VALUES,
  STATE_ACTOR_VALUES,
  NON_STATE_ACTOR_VALUES
)
from .scoring import(
  CASE_COUNT,
  EVENT_DATE,
  EVENT_TYPE,
  WARNING_ID
)
from .timeseries import (
    ewma_predict,
    arima_predict,
    naive_predict,
    status_quo_predict,
    hist_avg_predict
  )


def sample_column(event_df, col_name, sample_size, replace=True, value_list=None):
    """
    Samples the history for a specific column
    :param event_df: DataFrame with event history
    :param col_name: The column to be sampled
    :param sample_size: How many samples to draw
    :param replace: Sample with replacement?  Default True
    :param value_list: Is there a list of allowable values?  Default None
    :return: List of samples
    """
    the_sample = choice(event_df[col_name], size=sample_size, replace=replace)
    result = []
    for item in the_sample:
        if not isinstance(item, list):
            item = [item]
        if value_list is not None:
            item = [x for x in item if x in value_list]
            # It's still possible that the selection isn't a legal value if the data being sampled
            # doesn't conform to the value_list.  In that case just take a random element of the legal
            # value list.
            if item == []:
                item = choice(value_list, size=1)

        this_sample = choice(item, size=1)[0]
        result.append(this_sample)
    return result


def sample_location(event_df, sample_size, replace=True):
    """
    Sample the history of State, City, Latitude, Longitude pairs
    :param event_df: DataFrame with event history
    :param sample_size: How large a sample
    :param replace: Should we sample with replacement, default True
    :return: Dict of synchronized lists
    """
    index_sample = choice(event_df.index, size=sample_size, replace=replace)
    location_columns = [JSONField.STATE, JSONField.CITY]
    if JSONField.LATITUDE in event_df.columns:
        location_columns += [JSONField.LATITUDE, JSONField.LONGITUDE]
    loc_sample = [tuple(event_df[location_columns].ix[i]) for i in index_sample]
    loc_sample = [dict(zip(location_columns, loc)) for loc in loc_sample]
    loc_sample = DataFrame(loc_sample)
    return loc_sample


def sample_dates(start_date, end_date, sample_size, replace=True):
    """
    Take a sample from a date range between start_date and end_date.
    :param start_date: The start date of the range
    :param end_date: The end date of the range
    :param sample_size: How large a sample
    :param replace: Sample with replacement?  Default True
    :return: List of dates
    """
    d_range = pd.date_range(start_date, end_date)
    d_range = [str(dd.date()) for dd in d_range]
    sample = choice(d_range, size=sample_size, replace=replace)
    return sample


class Baserate:
    """
    Abstract class for all baserate models
    """
    REQUIRE_LOCALITY = False

    def __init__(self, country, event_type=False, team="Test_Max_Dist_Date_Diff"):
        """
        Test_Max_Dist_Date_Diff model for a specific country and event type
        :param country: The country to be studied
        :param es: An Elasticsearch instance
        :param event_type: The type of event to be studied
        :param team: The team prefix for warnings.
        """

        self.country = country
        self.team = team
        self.event_type = event_type
        self.historian = Historian(country=country)
        self.facets = {}

    def generate_warning_id(self):
        """
        Basis for warning generation
        :return: warning ID
        """
        return self.team + uuid4().hex

    def get_history(self, history_start_date, history_end_date, gsr):
        """
        Retrieves GSR event history using the historian
        :param history_start_date: first date
        :param history_end_date: end date
        :param matchargs: Additional arguments passed to the get_history method
        :returns: DataFrame of GSR Events
        """

        return self.historian.get_history(
          start_date = history_start_date,
          end_date=history_end_date,
          gsr=gsr
          )

    def make_predictions(self, history_start_date, history_end_date,
                         warning_start_date, warning_end_date,
                         leadtime_mean=None,
                         event_rate=None, replace=True, **matchargs):
        """
        Creates Test_Max_Dist_Date_Diff predictions
        :param history_start_date: first date of GSR history
        :param history_end_date: last date of GSR history
        :param warning_start_date: First date of warning period
        :param warning_end_date: Last date of warning period
        :param leadtime_mean: Mean of a Poisson distribution to simulate lead times
        :param event_rate: Predetermined rate at which warnings are produced, default None
        :param replace: Sample with replacement?  Default True
        :param matchargs: Additional arguments to be matched.
        :returns: Dataframe of warnings
        """
        event_df = self.get_history(history_start_date, history_end_date, **matchargs)
        if not event_rate:
            event_count = len(event_df)
            history_date_range = pd.date_range(history_start_date, history_end_date)
            history_date_count = len(history_date_range)
            event_rate = 1. * event_count / history_date_count

        # Count future dates
        n_days = len(pd.date_range(warning_start_date, warning_end_date))

        # How many events to generate?
        n_events = round(n_days * event_rate)
        if n_events > 0:

            # Sample dates
            new_dates = sample_dates(
                start_date=warning_start_date,
                end_date=warning_end_date,
                sample_size=n_events,
                replace=replace
            )
            br_dict = {JSONField.EVENT_DATE: new_dates}
            if leadtime_mean is not None:
                lt_ser = poisson(lam=leadtime_mean, size=n_events)
                lt_ser = Series([timedelta(int(x) + 1, 0, 0) for x in lt_ser])
                ed_ser = Series([parse(ed) for ed in new_dates])
                ts_ser = ed_ser - lt_ser
                ts_ser = ts_ser.apply(lambda x: str(x.date()))

                def pad_timestamp(ts_str):
                    """
                    Adds Hour/Minute/Second/Fraction data to the timestamp
                    :param ts_str: The string in format "%Y-%m-%d"
                    :return: String with format '%Y-%m-%dT%H:%M:%S.%f'
                    """
                    hours = choice(range(24), 1)[0]
                    minutes, seconds = choice(range(60), 2)
                    out_str = "{0}T{1}:{2}:{3}.0".format(ts_str, hours, minutes, seconds)
                    return out_str
                ts_ser = ts_ser.apply(pad_timestamp)
                br_dict[JSONField.TIMESTAMP] = ts_ser

            # Sample columns
            for col_name in self.facets:
                value_list = self.facets[col_name]
                br_dict[col_name] = sample_column(
                                            event_df=event_df,
                                            col_name=col_name,
                                            sample_size=n_events,
                                            replace=replace,
                                            value_list=value_list
                                            )

            # Sample locations
            if self.event_type in [EventType.MILITARY_ACTION]:
                new_locations = sample_location(event_df=event_df,
                                                sample_size=n_events,
                                                replace=replace)
                for col_name in new_locations.columns:
                    br_dict[col_name] = new_locations[col_name]
            else:
                new_countries = sample_column(event_df=event_df,
                                              col_name=JSONField.COUNTRY,
                                              sample_size=n_events,
                                              replace=replace)
                br_dict[JSONField.COUNTRY] = new_countries

            # Put it together

            br_df = pd.DataFrame(br_dict)
            br_df[JSONField.COUNTRY] = self.country
        else:
            br_dict = {JSONField.COUNTRY: [], JSONField.STATE: [], JSONField.CITY: []}
            for col_name in self.facets:
                br_dict[col_name] = []
            br_dict[JSONField.EVENT_DATE] = []
            br_df = pd.DataFrame(br_dict)
        br_df[JSONField.EVENT_TYPE] = self.event_type
        return br_df

    def convert_warnings_to_json(self, warning_df):
        """
        Converts warnings to a JSON ready for indexing and adding to Elasticsearch
        :param warning_df: A data frame of warnings
        :return:
        """
        warn_list = [self.generate_warning_id() for i in warning_df.index]
        warn_ser = Series(warn_list, index=warning_df.index)
        warning_df[JSONField.WARNING_ID] = warn_ser
        warn_json = warning_df.to_json(force_ascii=False, orient="records")
        warn_json = eval(warn_json)
        warn_json = {"performer_id": self.team, "payload": warn_json}
        return warn_json


class MaBaserate(Baserate):

    def __init__(self, country, team="R"):
        self.REQUIRE_LOCALITY = True
        super().__init__(
          country=country,
          event_type=EventType.MILITARY_ACTION,
          team=team
          )
        self.facets = {JSONField.SUBTYPE: [Subtype.ARMED_CONFLICT, Subtype.FORCE_POSTURE],
                       JSONField.ACTOR: STATE_ACTOR_VALUES}

    def convert_warnings_to_json(self, warning_df):
        """
        Converts warnings to a JSON ready for indexing and adding to Elasticsearch
        :param warning_df:
        :return:
        """
        warn_json = super().convert_warnings_to_json(warning_df)
        for item in warn_json:
            item[JSONField.ACTOR] = re.sub("\\\\", "", item[JSONField.ACTOR])
        return warn_json


class CaseCountBaserate(Baserate):

    model_name_dict = {ewma_predict: "EWMA",
                       arima_predict: "ARMA",
                       naive_predict: "Naive",
                       status_quo_predict: "Status Quo",
                       hist_avg_predict: "Historical Averages"}

    def __init__(self, country, event_type, freq="W-SUN", team="R"):

        self.country = country
        self.team = team
        self.event_type = event_type
        self.historian = Historian()
        self.facets = {}
        self.freq = freq

    def get_history(self, history_start_date, history_end_date, **matchargs):
        """
        Retrieves GSR event history using the historian
        :param history_start_date: first date
        :param history_end_date: end date
        :param matchargs: Additional arguments passed to the get_history method
        :returns: DataFrame of GSR Events
        """

        return self.historian.get_history(db.GSR_TYPE, country=self.country,
                                          start_date=history_start_date,
                                          end_date=history_end_date, **matchargs)

    def make_predictions(self, history_start_date, history_end_date,
                         warning_start_date, warning_end_date,
                         leadtime_mean=None,
                         predict_method=ewma_predict,
                         **pm_args):
        """
        Make case count predictions
        :param history_start_date: First date of history period
        :param history_end_date: Last date of history period
        :param warning_start_date: First date for which warnings are desired
        :param warning_end_date: Last date for which warnings are desired
        :param prob: What probability to assign to warnings?  Default = 0.5
        :param predict_method: Which method for predicting?
        :param pm_args: Additional arguments to be passed to predict_method
        :return: DataFrame with warnings
        """
        event_df = self.get_history(history_start_date=history_start_date,
                                    history_end_date=history_end_date)
        count_ser = event_df[CASE_COUNT].copy()
        count_ser.index = event_df[EVENT_DATE].apply(parse)
        comprehensive_date_range = pd.date_range(count_ser.index[0],
                                                 count_ser.index[-1],
                                                 freq=self.freq)
        count_ser = count_ser.reindex(comprehensive_date_range).fillna(0)
        count_ser = count_ser.apply(lambda x: 1.0*x)

        # Determine the appropriate dates for the warnings.  Note that the Mercury GSR
        # uses Sunday for the reference weekday.
        # warn_dates = pd.date_range(warning_start_date, warning_end_date, freq="W-SUN")
        last_history_date = count_ser.index[-1]
        future_dates = pd.date_range(last_history_date, warning_end_date, freq=self.freq)
        n_ahead = len(future_dates) - 1

        predict_results = predict_method(count_ser, n_ahead=n_ahead, **pm_args)
        predict_ser = predict_results["Predictions"]
        predict_ser = predict_ser[warning_start_date:]
        predict_model = predict_results["Model"]
        predict_model_params = predict_results["Model_Params"]

        warning_df = DataFrame({CASE_COUNT: predict_ser,
                                EVENT_DATE: predict_ser.index})
        warning_df[EVENT_TYPE] = self.event_type
        warning_df[JSONField.COUNTRY] = self.country
        if leadtime_mean is not None:
            ts_ser = warning_df[EVENT_DATE] - datetime.timedelta(leadtime_mean)
            ts_ser = ts_ser.apply(lambda x: str(x.date()))

            def pad_timestamp(ts_str):
                """
                Adds Hour/Minute/Second/Fraction data to the timestamp
                :param ts_str: The string in format "%Y-%m-%d"
                :return: String with format '%Y-%m-%dT%H:%M:%S.%f'
                """
                hours = choice(range(24), 1)[0]
                minutes, seconds = choice(range(60), 2)
                out_str = "{0}T{1}:{2}:{3}.0".format(ts_str, hours, minutes, seconds)
                return out_str

            ts_ser = ts_ser.apply(pad_timestamp)
            warning_df[JSONField.TIMESTAMP] = ts_ser
        warning_df[EVENT_DATE] = warning_df[EVENT_DATE].apply(lambda x: str(x.date()))
        warning_df[CASE_COUNT] = warning_df[CASE_COUNT].apply(lambda x: int(x))
        if predict_method in self.model_name_dict:
            warning_df["Model"] = predict_model
            warning_df["Model_Params"] = predict_model_params

        return warning_df

    def convert_warnings_to_json(self, warning_df):
        """
        Converts warnings to a JSON ready for indexing and adding to Elasticsearch
        :param warning_df:
        :return:
        """
        id_ser = warning_df[EVENT_DATE].apply(lambda x:
                                              "{3} {0} {1} {2}".format(self.event_type,
                                                                       self.country, x,
                                                                       self.team))
        warning_df[WARNING_ID] = id_ser
        warn_json = warning_df.to_json(force_ascii=False, orient="records")
        warn_json = eval(warn_json)
        return warn_json


class IcewsBaserate(CaseCountBaserate):

    def __init__(self, country, team="R"):

        self.country = country
        self.team = team
        self.event_type = EventType.ICEWS_PROTEST
        self.historian = IcewsHistorian()
        self.freq = "W-WED"
        self.facets = {}


class CaseCountDiseaseBaserate(CaseCountBaserate):

    country_disease_map = {CountryName.EGYPT: DiseaseType.AVIAN_INFLUENZA,
                           CountryName.SAUDI_ARABIA: DiseaseType.MERS}

    def __init__(self, country, team="R"):

        self.country = country
        self.team = team
        self.event_type = EventType.DISEASE
        self.historian = CaseCountDiseaseHistorian()
        self.freq = "W-SUN"
        self.facets = {}

    def make_predictions(self, history_start_date, history_end_date,
                         warning_start_date, warning_end_date,
                         leadtime_mean=None,
                         predict_method=ewma_predict, **pm_args):
        """
        Uses the CaseCountBaserate method and then adds a disease field to it.
        :param prob: Probability for the warnings, default is 0.5
        :param history_start_date:
        :param history_end_date:
        :param warning_start_date:
        :param warning_end_date:
        :param predict_method:
        :param pm_args:
        :return:
        """

        preds = super().make_predictions(history_start_date=history_start_date,
                                         history_end_date=history_end_date,
                                         warning_start_date=warning_start_date,
                                         warning_end_date=warning_end_date,
                                         leadtime_mean=leadtime_mean,
                                         predict_method=predict_method, **pm_args)
        preds[EventType.DISEASE] = self.country_disease_map[self.country]

        return preds
