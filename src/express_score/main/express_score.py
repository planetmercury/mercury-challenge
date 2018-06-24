'''
Module to provide lightweight scoring functions for Mercury Challenge participants to score their warnings as part of their test and development process.
'''

import sys

from .schema import (
    JSONField,
    ScoreComponents,
    EventType,
    Subtype
)

import datetime
try:
    from ciso8601 import parse_datetime as parse
except ImportError:
    from dateutil.parser import parse
from collections import (
    ChainMap,
    OrderedDict
)

import numpy as np
import pandas as pd
from geopy.distance import distance

# The next few lines determine which matching algorithm implementation we use.
# If the cython-munkres implementation we prefer to use it, otherwise we switch to the scikit-learn version.
try:
    import dlib
    use_dlib = True
except ImportError:
    try:
        from munkres import munkres
    except ImportError:
        from munkres import Munkres as munkres
    use_dlib = False


class Defaults(object):
    """
    Holds default values for scoring system parameters
    """
    ACCURACY_DENOMINATOR = 4
    LS_WEIGHT = 1.0
    DS_WEIGHT = 1.0
    AS_WEIGHT = 1.0
    ESS_WEIGHT = 1.0
    MAX_DIST = 100.0
    DIST_BUFFER = MAX_DIST/6.0
    MAX_DATE_DIFF = 4.0
    DATE_BUFFER = 4.0


class Scorer:
    '''
    Base class for all scoring classes.
    '''

    def __init__(self):
        pass

    def score(self, warn_data, gsr_data):
        '''
        Scores the warning data against the GSR data.
        :param warn_data: List of dicts of warnings
        :param gsr_data: List of dicts of GSR events
        :return: Dict of scoring results
        '''
        match_ = self.match(warn_data, gsr_data)
        out_dict = dict()
        return out_dict

    def score_one(self, warn_, event_):
        """
        Scores a single warning against a single GSR event
        :param warn_: dict with data for a warning.
        :param event_: dict with data for a GSR event
        :return: dict with scoring results
        """
        out_dict = dict()
        return out_dict()

    def match(self, warn_data, gsr_data):
        """
        Matches the warnings to the GSR events
        :param warn_data: List of dicts of warnings
        :param gsr_data: Lists of dicts of GSR events.
        :return: List of matched data
        """
        return []

    @staticmethod
    def slope_score(result, min_value, max_value):
        """
        Provides a score for the result scaled to 0 to 1
        :param result: Value to be scored
        :param min_value: Smallest allowable value
        :param max_value: Largest allowable value
        :return: Score from 0.0 to 1.0
        """
        try:
            slope = max_value - min_value
            result = min(result, max_value)
            result = max(result, min_value)
            score = 1 - (result - min_value)/slope
        except ZeroDivisionError:
            score = 0
        return score

    @staticmethod
    def facet_score(warn_value, gsr_value, wildcards=[]):
        """
        Sees if warning and GSR match
        :param warn_value: Value from the warning
        :param gsr_value: Value from the event
        :param wildcards: A list of GSR values that match any warning value.
        :return: 1 if they match, 0 otherwise.
        """
        if isinstance(gsr_value, (list, tuple)):
            pass
        else:
            gsr_value = [gsr_value]

        if len(set(wildcards).intersection(set(gsr_value))) > 0:
            matched_value = 1
        else:
             matched_value = int(warn_value in gsr_value)

        return matched_value


class CaseCountScorer(Scorer):
    """
    Scores the case count event types
    """
    def __init__(self, event_type):
        """
        Sets the event type
        :param event_type: Event_Type value
        """
        self.event_type = event_type

    def score_one(self, warn_, event_, accuracy_denominator=Defaults.ACCURACY_DENOMINATOR):
        """
        Scores a single warning against a single GSR event
        :param warn_: dict with warning data
        :param event_: dict with event data
        :param accuracy_denominator: Threshold for scaling with small case counts
        :return: dict with scoring results
        """
        predicted = warn_[JSONField.CASE_COUNT]
        actual = event_[JSONField.CASE_COUNT]
        qs = self.quality_score(predicted, actual, accuracy_denominator)
        out_dict = dict()
        out_dict[JSONField.WARNING_ID] = warn_[JSONField.WARNING_ID]
        out_dict[JSONField.EVENT_ID] = event_[JSONField.EVENT_ID]
        out_dict[JSONField.WARN_VALUE] = predicted
        out_dict[JSONField.EVENT_VALUE] = actual
        out_dict[ScoreComponents.QS] = qs
        return out_dict


    @staticmethod
    def quality_score(predicted, actual, accuracy_denominator=Defaults.ACCURACY_DENOMINATOR):
        """
        Computes quality score on a scale of 0.0 to 1.0
        :param predicted: The predicted value
        :param actual: The actual value
        :param accuracy_denominator: The minimum value for scaling differences
        :return: Quality score value
        """
        numerator = abs(predicted-actual)
        denominator = max(predicted, actual, accuracy_denominator)
        qs = 1 - 1.*numerator/denominator
        return qs

    def match(self, warn_data, gsr_data):
        """
        Matches Warning to GSR by event date
        :param warn_data:
        :param gsr_data:
        :return:
        """
        pass

    def score(self, warn_data, gsr_data):
        """
        Perform scoring on ICEWS Protest Events
        :param warn_data: List of dicts of warnings
        :param gsr_data: List of dicts of GSR events
        :return: Dict with scoring results.
        """
        score_dict = dict()
        if self.warnings is None:
            warn_count = 0
        else:
            warn_count = len(self.warnings)
        if self.gsr_events is None:
            evt_count = 0
        else:
            evt_count = len(self.gsr_events)
        score_dict["Warning Count"] = warn_count
        score_dict["Event Count"] = evt_count
        score_dict["Country"] = self.country
        score_dict["Start Date"] = self.start_date
        score_dict["End Date"] = self.end_date
        score_dict["Event_Type"] = self.event_type
        if self.warnings is not None:
            score_dict["Warnings"] = {self.warnings.ix[dd][EVENT_DATE]: int(self.warnings.ix[dd][CASE_COUNT])
                                      for dd in self.warnings.index}
        if self.gsr_events is not None:
            score_dict["GSR"] = {self.gsr_events.ix[dd][EVENT_DATE]: int(self.gsr_events.ix[dd][CASE_COUNT])
                                 for dd in self.gsr_events.index}
        if self.warnings is None or self.gsr_events is None:
            return {'Results': score_dict}
        else:
            warnings_df = self.warnings
            gsr_df = self.gsr_events
            overlap_warnings = warnings_df[warnings_df[EVENT_DATE].isin(gsr_df[EVENT_DATE])]
            overlap_warnings.index = overlap_warnings[EVENT_DATE]
            overlap_gsr = gsr_df[gsr_df[EVENT_DATE].isin(warnings_df[EVENT_DATE])]
            overlap_gsr.index = overlap_gsr[EVENT_DATE]
            score_df = pd.DataFrame({"Predicted": overlap_warnings[CASE_COUNT],
                                     "Actual": overlap_gsr[CASE_COUNT]})
            score_dict["Match Count"] = len(score_df)
            time_df = pd.DataFrame({"timestamp": overlap_warnings[JSONField.TIMESTAMP],
                                    "Event_Date": overlap_gsr[EVENT_DATE]})
            try:
                time_df[JSONField.EARLIEST_REPORTED_DATE] = overlap_gsr[JSONField.EARLIEST_REPORTED_DATE]
            except KeyError:
                pass
            if len(score_df) == 0:
                return {'Results' : score_dict}
            else:

                score_df[ScoreComponents.QS] = score_df.apply(lambda x: self.quality_score(x["Predicted"],
                                                                             x["Actual"])\
                                                                  ["Quality Score"],
                                                axis=1)
                if JSONField.EARLIEST_REPORTED_DATE in time_df.columns:
                    time_df[ScoreComponents.LT] = time_df.apply(lambda x: self.lead_time(x[JSONField.TIMESTAMP],
                                                                                         x[JSONField.EARLIEST_REPORTED_DATE]),
                                                  axis=1)
                else:
                    time_df[ScoreComponents.LT] = time_df.apply(lambda x: self.lead_time(x[JSONField.TIMESTAMP],
                                                                                         x[EVENT_DATE]),
                                                  axis=1)
                    time_df[ScoreComponents.LT] = time_df[ScoreComponents.LT] + 5
                score_dict["Precision"] = score_dict["Match Count"]/score_dict["Warning Count"]
                score_dict["Recall"] = score_dict["Match Count"]/score_dict["Event Count"]
                score_dict[ScoreComponents.QS] = score_df[ScoreComponents.QS].mean()
                score_dict[ScoreComponents.LT] = time_df[ScoreComponents.LT].mean()

                return {'Results' : score_dict}


class MaScorer(Scorer):
    """
    Scorer for Military Action events
    """

    STATE_ACTORS = ["Fred", "Ethel"]
    WILDCARD_ACTORS = ["Unspecified"]
    ACTORS = STATE_ACTORS + WILDCARD_ACTORS
    SUBTYPES = [Subtype.CONFLICT, Subtype.FORCE_POSTURE]

    def __init__(self, country):
        """
        :param country: The country to be scored
        """
        self.event_type = EventType.MA
        self.country = country

    def match(self, warn_data, gsr_data):
        """
        Builds match list
        :param warn_data:
        :param gsr_data:
        :return:
        """
        pass

    @staticmethod
    def score_one(warn_, event_, max_dist=Defaults.MAX_DIST, dist_buffer=Defaults.DIST_BUFFER,
                  max_date_diff=Defaults.MAX_DATE_DIFF, legit_actors=ACTORS, wildcards=WILDCARD_ACTORS,
                  ls_weight=Defaults.LS_WEIGHT, ds_weight=Defaults.DS_WEIGHT, as_weight=Defaults.AS_WEIGHT,
                  ess_weight = Defaults.ESS_WEIGHT):
        """
        Scores a single warning against a single event
        :param warn_: Dict with data for a warning
        :param event_: Dict with data for an event
        :return: Dict with scoring details
        """
        bad_qs = False
        error_list = []
        notice_list = []
        if ls_weight < 0:
            bad_qs = True
            error_list.append("LS Weight must be positive")
        if ds_weight < 0:
            bad_qs = True
            error_list.append("DS Weight must be positive")
        if as_weight < 0:
            bad_qs = True
            error_list.append("AS Weight must be positive")
        if ess_weight < 0:
            bad_qs = True
            error_list.append("ESS Weight must be positive")
        weight_sum = ls_weight + ds_weight + as_weight + ess_weight
        if weight_sum != 4.0:
            notice_list.append("Reweighting so that sum of weights is 4.0")
            ls_weight = 4*ls_weight/weight_sum
            ds_weight = 4*ds_weight/weight_sum
            as_weight = 4*as_weight/weight_sum
            ess_weight = 4*ess_weight/weight_sum

        out_dict = dict()
        # Compute the distance
        out_dict[JSONField.WARNING_ID] = warn_[JSONField.WARNING_ID]
        out_dict[JSONField.EVENT_ID] = event_[JSONField.EVENT_ID]
        if error_list:
            out_dict["Errors"] = error_list
        else:
            warn_lat = warn_[JSONField.LATITUDE]
            warn_long = warn_[JSONField.LONGITUDE]
            event_lat = event_[JSONField.LATITUDE]
            event_long = event_[JSONField.LONGITUDE]
            loc_approx = event_[JSONField.APPROXIMATE_LOCATION]
            out_dict[JSONField.APPROXIMATE_LOCATION] = loc_approx
            dist = distance((warn_lat, warn_long), (event_lat, event_long)).km
            out_dict["Distance"] = dist
            warn_event_date = warn_[JSONField.EVENT_DATE]
            event_event_date = event_[JSONField.EVENT_DATE]
            gsr_d = parse(event_event_date).date()
            warn_d = parse(warn_event_date).date()
            date_diff = (gsr_d - warn_d).days
            date_diff = np.abs(date_diff)
            out_dict["Date Difference"] = date_diff
            ls = MaScorer.location_score(warn_lat, warn_long, event_lat, event_long, loc_approx, max_dist, dist_buffer)
            out_dict[ScoreComponents.LS] = ls
            ds = MaScorer.date_score(warn_event_date, event_event_date, max_date_diff)
            out_dict[ScoreComponents.DS] = ds
            # Event Subtype
            warn_es = warn_[JSONField.SUBTYPE]
            gsr_es = event_[JSONField.SUBTYPE]
            ess = MaScorer.event_subtype_score(warn_es, gsr_es)
            out_dict[ScoreComponents.ESS] = ess
            # Actor Score
            warn_actor = warn_[JSONField.ACTOR]
            event_actor = event_[JSONField.ACTOR]
            _as = MaScorer.actor_score(warn_actor, event_actor, legit_actors, wildcards)
            out_dict[ScoreComponents.AS] = _as
            if min(ls, ds) == 0:
                qs = 0
            else:
                qs = ls_weight*ls + ds_weight*ds + as_weight*_as + ess_weight*ess
            out_dict[ScoreComponents.QS] = qs
        # Quality Score
        if notice_list:
            out_dict["Notices"] = notice_list
        return out_dict

    def score(self, warn_data, gsr_data):
        """
        Scores the warnings against the GSR
        :param warn_data: List of dicts with warning data
        :param gsr_data: List of dicts with event data
        :return: dict with scoring results
        """

        out_dict = dict()

        return out_dict

    @staticmethod
    def location_score(lat1, long1, lat2, long2, is_approximate=False,
                       max_dist=Defaults.MAX_DIST, dist_buffer=Defaults.DIST_BUFFER):
        """
        Computes location score based on the distance provided.
        :param lat1: Latitude of point 1
        :param long1: Longitude of point 1
        :param lat2: Latitude of point 2
        :param long2: Longitude of point 2
        :param is_approximate: Is the reference location approximate?
        :param max_dist: What is the maximum distance allowed?
        :param dist_buffer: For approximate locations, how much do we give credit for?
        :return: score from 0 to 1.0
        """

        dist_km = distance((lat1, long1), (lat2, long2)).km
        if isinstance(is_approximate, str):
            is_approximate = eval(is_approximate)
        is_approximate = int(is_approximate)
        max_dist = max_dist - is_approximate*dist_buffer
        dist_km = dist_km - is_approximate*dist_buffer
        ls = Scorer.slope_score(dist_km, 0, max_dist)
        return ls

    @staticmethod
    def date_score(warn_date, gsr_date,
                   max_date_diff=Defaults.MAX_DATE_DIFF):
        """
        Computes date score based on warning date and event date.
        :param warn_date: Event_Date in the warning
        :param gsr_date: Event_Date in the GSR
        :param max_date_diff: How many days is considered to be too many?
        :return: score from 0 to 1.0
        """
        gsr_d = parse(gsr_date).date()
        warn_d = parse(warn_date).date()
        date_diff = (gsr_d - warn_d).days
        date_diff = np.abs(date_diff)
        ds = Scorer.slope_score(date_diff, 0, max_date_diff)
        return ds

    @staticmethod
    def actor_score(warn_actor, gsr_actor, legit_actors=ACTORS,
                    wildcards=WILDCARD_ACTORS):
        """
        Computes the facet score for Actor
        :param warn_actor: Warning value for Actor
        :param gsr_actor: GSR value for Actor, could be a list
        :param legit_actors: Which actors are legal values for submission
        :param wildcards: Which values in the GSR are assumed to match all legit inputs?
        :return: 1 if it matches, 0 if it doesn't.
        """

        if warn_actor not in legit_actors:
            _as = 0
        else:
            _as = Scorer.facet_score(warn_actor, gsr_actor, wildcards)

        return _as

    @staticmethod
    def event_subtype_score(warn_subtype, gsr_subtype):
        """
        Computes the facet score for Event Subtype
        :param warn_subtype: Warning value for Event Subtype
        :param gsr_subtype: GSR value for Subtype, could be a list
        :return: 1 if it matches, 0 if it doesn't.
        """

        if warn_subtype not in MaScorer.SUBTYPES:
            ess = 0
        else:
            ess = Scorer.facet_score(warn_subtype, gsr_subtype)

        return ess
