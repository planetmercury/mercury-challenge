'''
Module implementing Mercury T&E scoring that is a light modification of Pete
Haglich's original prototype
'''

__all__ = '''
  SCORER
  MAX_DATE_PERIOD
  '''.split()

import datetime
from ciso8601 import parse_datetime_unaware as parse
from collections import (
    ChainMap,
    OrderedDict
)

import numpy as np
import pandas as pd
from geopy.distance import vincenty, great_circle
from mercury.common import db_management as db
from mercury.common.schema import Dictionary
from mercury.common.schema import (
    JSONField,
    EventType,
    Subtype,
    DiseaseType,
    CountryName,
    Wildcards,
    ScoreComponents
)

from .historian import (
  Historian,
  RareDiseaseHistorian,
  CaseCountDiseaseHistorian,
  IcewsHistorian
  )
from .loaddictionaries import (
  UNSPECIFIED_ACTORS
)

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

CASE_COUNT = JSONField.CASE_COUNT
EVENT_ID = JSONField.EVENT_ID
EVENT_DATE = JSONField.EVENT_DATE
EVENT_TYPE = JSONField.EVENT_TYPE
EVENT_SUBTYPE = JSONField.SUBTYPE
WARNING_ID = JSONField.WARNING_ID
TIMESTAMP = JSONField.TIMESTAMP
EARLIEST_REPORTED_DATE = JSONField.EARLIEST_REPORTED_DATE
cu_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
              JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT,
              JSONField.LATITUDE, JSONField.LONGITUDE]
ws_columns = [JSONField.COUNTRY, EVENT_DATE,
              JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT]
nsa_ma_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
                  JSONField.ACTOR, JSONField.TARGETS, EVENT_SUBTYPE,
                  JSONField.LATITUDE, JSONField.LONGITUDE]
rd_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
              JSONField.DISEASE,
              JSONField.LATITUDE, JSONField.LONGITUDE]
case_count_disease_columns = [JSONField.COUNTRY, EVENT_DATE, JSONField.DISEASE, CASE_COUNT]
icews_columns = [JSONField.COUNTRY, EVENT_DATE, CASE_COUNT]
germane_column_dict = {EventType.CIVIL_UNREST : cu_columns,
                       EventType.WIDESPREAD_CIVIL_UNREST : ws_columns,
                       EventType.DISEASE : rd_columns,
                       EventType.NONSTATE_ACTOR: nsa_ma_columns,
                       EventType.MILITARY_ACTION: nsa_ma_columns,
                       "Case Count Disease": case_count_disease_columns,
                       EventType.ICEWS_PROTEST: icews_columns}
index_column_dict = {
  db.GSR_TYPE: [EVENT_ID, JSONField.EARLIEST_REPORTED_DATE],
  db.WARNING_TYPE: [WARNING_ID],
  }

MAX_DATE_DIFF = 7
DEFAULT_DATE_BUFFER = 7
MAX_DATE_PERIOD = 106
DEFAULT_MAX_DIST = 300
DEFAULT_DIST_METHOD = vincenty
DEFAULT_DIST_UNITS = "km"
DEFAULT_LOC_APPROX_BUFFER = 50

def make_combination_mats(row_value_list, column_value_list):
    """
    Creates dual index matrices for use in other computations.  The warning index matrix will
    have one row per warning index element, the GSR index matrix will have one column per GSR index element.
    :return: pair of matrices
    """
    n_row = len(row_value_list)
    n_col = len(column_value_list)
    row_value_array = np.array(n_col * list(row_value_list))
    row_value_array = row_value_array.reshape(n_col, n_row)
    row_value_array = row_value_array.T
    column_value_array = np.array(n_row * list(column_value_list))
    column_value_array = column_value_array.reshape(n_row, n_col)

    return row_value_array, column_value_array


def make_index_mats(row_value_list, column_value_list):
    """
    Creates dual index matrices for use in other computations.  The warning index matrix will
    have one row per warning index element, the GSR index matrix will have one column per GSR index element.
    :return: pair of matrices
    """
    n_row = len(row_value_list)
    n_col = len(column_value_list)
    row_value_i_list = list(range(n_row))
    row_value_i_array = np.array(n_col * row_value_i_list)
    row_value_i_array = row_value_i_array.reshape(n_col, n_row)
    row_value_i_array = row_value_i_array.T
    column_value_i_list = list(range(n_col))
    column_value_i_array = np.array(n_row * column_value_i_list)
    column_value_i_array = column_value_i_array.reshape(n_row, n_col)

    return row_value_i_array, column_value_i_array


# Precompute all possible date differences
def date_score(date_diff, max_diff=MAX_DATE_DIFF):
    """Computes the date score between date1 and date2, with
    a date_diff of max_diff scoring 0
    :param max_diff: Maximum allowable time difference
    :param date_diff: Number of days between actual and warning event dates
    :return: Float
    """

    date_diff = np.abs(date_diff)
    score = max(0, 1 - (date_diff * 1.0 / max_diff))
    return score


warn_dates = np.array(range(MAX_DATE_PERIOD))
gsr_dates = np.array(range(MAX_DATE_PERIOD))
dd_index_mats = make_index_mats(warn_dates, gsr_dates)
all_dd_mat = dd_index_mats[0] - dd_index_mats[1]
ds_vfunc = np.vectorize(date_score)

class Scorer:
    """Scores a specfic country over a specific interval of time"""
    score_components = [ScoreComponents.QS, ScoreComponents.LT]

    def __init__(self, country, start_date, end_date,
                 event_type, max_dist, max_date_diff, dist_buffer, 
                 date_buffer, ls_weight, ds_weight, ps_weight, rs_weight,
                 vs_weight, as_weight, tss_weight, tts_weight, es_weight,
                 performer_id=None, **kwargs):
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Type of event to be scored
        :param date_buffer: How far to either side of the scoring interval that we look for better matching items.
        :return:
        """
        self.event_type = event_type
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.performer_id = performer_id
        self._start_me(country=country, start_date=start_date, end_date=end_date,
                       date_buffer=date_buffer, **kwargs)
        self.principal_score_mats = self.set_principal_score_mats()
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)

    def _start_me(self, country, start_date, end_date, date_buffer,
                  **kwargs):
        """
        Method to group the setting of common items across all subclasses
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Type of event to be scored
        :return:
        """

        self.country = country
        self.start_date = str(parse(start_date).date())
        self.end_date = str(parse(end_date).date())
        self.date_buffer = date_buffer
        self.first_history_date = (parse(self.start_date) - datetime.timedelta(self.date_buffer)).date()
        self.last_history_date = (parse(self.end_date) + datetime.timedelta(self.date_buffer)).date()
        self.historian = Historian()
        params = {
          'event_type' : self.event_type,
          'country' : self.country,
          'start_date' : self.first_history_date,
          'end_date' : self.last_history_date,
          'performer_id' : self.performer_id
          }
        params.update(kwargs)
        self.gsr_events = self.historian.get_history(db.GSR_TYPE, **params)
        self.submitted_warnings = self.historian.get_history(db.WARNING_TYPE, **params)
        if self.submitted_warnings is not None:
            print("Submitted Warnings: {0}".format(len(self.submitted_warnings)))
            self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
            print("Latest Warnings: {0}".format(len(self.warnings)))
            print("Late Warnings: {0}".format(len(self.warnings[self.warnings.timestamp > self.end_date])))
        else:
            self.warnings = None
        self.facet_score_mats = self.set_facet_score_mats()
        self.lead_time_mat = self.make_lead_time_mat()

    def set_principal_score_mats(self):
        """
        Determines the principal score component matrices
        """
        return dict()

    def set_facet_score_mats(self):
        """
        Determines the facet score component matrices
        """
        return dict()

    @staticmethod
    def probability_score(occurred_boolean, probability):
        """Computes the Brier score for actual and predicted values.
        Args:
            occurred_boolean:  1 if the event occurred, 0 otherwise
            probability: forecast probability of occurrence.
        Returns:
            Float value of the score"""
        return 1 - (occurred_boolean - probability)**2

    @staticmethod
    def precision_recall(warn_or_evt_count, match_count):
        if match_count > warn_or_evt_count:
            return False
        else:
            if warn_or_evt_count > 0:
                return match_count/warn_or_evt_count
            else:
                return 0

    @staticmethod
    def fmeasure(warn_count, event_count, match_count, beta=1):
        """
        Computes the F measure with weight beta
        :param warn_count: How many warnings
        :param event_count: How many events
        :param match_count: How many matches were made
        :param beta: Relative importance of recall, default 1
        :return: Float
        """
        prec = Scorer.precision_recall(warn_count, match_count)
        rec = Scorer.precision_recall(event_count, match_count)
        if prec and rec:
            beta_sq = beta*beta
            f = (1+beta_sq)*(prec*rec)/(beta_sq*prec + rec)
            return f
        else:
            return False

    @staticmethod
    def lead_time(submit_datetime, report_date):
        """
        Computes the lead time between warning submission and reported date.
        Computes the time difference in days between submission and report
        :param submit_datetime: When the warning was submitted.  datetime object
        :param report_date: Earliest reported date.  String format.
        :return: Difference in days
        """
        # Strip the time part of the submit_date
        if isinstance(submit_datetime, datetime.datetime):
            submit_datetime = submit_datetime.date()
        elif isinstance(submit_datetime, str):
            submit_datetime = parse(submit_datetime).date()
        report_date = parse(report_date).date()
        lead_time = (report_date - submit_datetime).days
        return lead_time

    @staticmethod
    def match(input_matrix, allow_zero_scores=False):
        """
        Uses the Munkres algorithm to make assignments
        :param allow_zero_scores: Should scores of 0 be allowed?  Default False
        :param input_matrix: Matrix to do the matching
        :return: List of assignments
        """
        if min(input_matrix.shape) == 0:
            assign = []
        else:
            score_matrix = input_matrix.copy(order="C")
            row_count, col_count = score_matrix.shape
            
            k_max = max(score_matrix.shape[0], score_matrix.shape[1])
            score_matrix_square = np.zeros((k_max, k_max))
            score_matrix_square[:score_matrix.shape[0], :score_matrix.shape[1]] = score_matrix
            
            if use_dlib:
                assign = dlib.max_cost_assignment(dlib.matrix(score_matrix_square))
                assign = [(i, assign[i]) for i in range(k_max)]
            else:
                try:
                    assign = munkres(-1 * score_matrix_square)
                except TypeError:
                    m = munkres()
                    assign = m.compute(cost_matrix = 4-score_matrix_square)
                assign = np.nonzero(assign)
                assign = np.array(list(zip(list(assign[0]), list(assign[1]))))

            assign_scores = np.array([score_matrix_square[x[0], x[1]] for x in assign])
            assign = np.array(assign)

            assign = assign[assign_scores > 0]
            
            if not allow_zero_scores:
                assign_scores = np.array([score_matrix_square[x[0], x[1]] for x in assign])
                assign = np.array(assign)
                assign = assign[assign_scores > 0]
            assign = list([tuple(x) for x in assign])
        assign = [(int(x[0]), int(x[1])) for x in assign]
        return assign

    def do_scoring(self, principal_weight_dict, facet_weight_dict, max_score=4,
                   allow_zero_scores=False):
        """
        Performs scoring of warnings against events.
        :param allow_zero_scores: Should scores of 0 be considered in matching?
        :param max_score: Maximum score allowed
        :param principal_weight_dict: Dict {principal: weight}
        :param facet_weight_dict: Dict {facet: weight}
        :return: Dict with scoring results and other details.
        """
        score_dict = dict()
        if self.warnings is None:
            period_warn_count = 0
            all_warn_count = 0
        else:
            all_warn_count = len(self.warnings)
            try:
                period_warnings = self.warnings[self.start_date <= self.warnings.Event_Date]
                period_warnings = period_warnings[period_warnings.Event_Date <= self.end_date]
            except TypeError:
                print(self.start_date, self.start_date.__class__)
                print(self.end_date, self.end_date.__class__)
                print(self.warnings.Event_Date.iloc[0], self.warnings.Event_Date.iloc[0].__class__)
                return "Type Error in date formats"
            if period_warnings is not None:
                period_warn_count = len(period_warnings)
            else:
                period_warn_count = 0
        if self.gsr_events is None:
            period_gsr_count = 0
            all_evt_count = 0
        else:
            all_evt_count = len(self.gsr_events)
            period_gsr = self.gsr_events[self.start_date <= self.gsr_events.Event_Date]
            period_gsr = period_gsr[period_gsr.Event_Date <= self.end_date]
            if period_gsr is not None:
                period_gsr_count = len(period_gsr)
            else:
                period_gsr_count = 0
        score_dict["Event Count"] = all_evt_count
        score_dict["Warning Count"] = all_warn_count
        score_dict["Country"] = self.country
        score_dict["Start Date"] = self.start_date
        score_dict["End Date"] = self.end_date
        score_dict["Event_Type"] = self.event_type
        detail_dict = dict()
        if self.warnings is None or self.gsr_events is None:
            score_dict["Match Count"] = 0
            matches = []
            lead_times = []
            quality_scores = []

        else:
            qs_mat = self.make_qs_mat(principal_weight_dict, facet_weight_dict, max_score)
            lt_mat = self.make_lead_time_mat()
            matches = self.match(qs_mat, allow_zero_scores)
            # debug code
            if False:
                for x in range(len(self.warnings)):
                    try:
                        print(x, self.warnings.iloc[x]["Event_Date"],
                              self.end_date >= self.warnings.iloc[x] >= self.start_date)
                    except TypeError as e:
                        print(repr(e), x, self.warnings.iloc[x])
            period_warn_indices = [x for x in range(len(self.warnings))
                         if self.end_date >= self.warnings.iloc[x][JSONField.EVENT_DATE] >= self.start_date]
            period_gsr_indices = [x for x in range(len(self.gsr_events))
                         if self.end_date >= self.gsr_events.iloc[x][JSONField.EVENT_DATE] >= self.start_date]
            score_dict["Period Warning Count"] = period_warn_count
            score_dict["Period Event Count"] = period_gsr_count
            match_count = len(matches)
            score_dict["Match Count"] = match_count
            period_warn_matches = [m for m in matches if m[0] in period_warn_indices]
            period_gsr_matches = [m for m in matches if m[1] in period_gsr_indices]

            if period_gsr_matches is not None:
                period_gsr_match_count = len(period_gsr_matches)
            else:
                period_gsr_match_count = 0
            score_dict["Period Events Matched"] = period_gsr_match_count
            score_dict["Recall"] = Scorer.precision_recall(period_gsr_count, period_gsr_match_count)
            if period_warn_matches is not None:
                period_warn_match_count = len(period_warn_matches)
                score_dict["Period Warnings Matched"] = period_warn_match_count
                # Precision is the number of period warnings that were matched divided by the number
                # of period warnings.  There is asymmetry from the definition of recall
                score_dict["Precision"] = Scorer.precision_recall(period_warn_count, period_warn_match_count)
                detail_dict = {"Component Scores": dict()}
                quality_scores = [qs_mat[m[0], m[1]] for m in matches]
                #print("Mean QS for all matches: {0}".format(np.mean(quality_scores)))
                period_quality_scores = [qs_mat[m[0], m[1]] for m in period_warn_matches]
                #print("Mean QS for matched warnings in the period: {0}".format(np.mean(period_quality_scores)))
                detail_dict["Quality Scores"] = quality_scores
                score_dict[ScoreComponents.QS] = np.mean(period_quality_scores)
                score_dict["Component Scores"] = dict()

                lead_times = [int(lt_mat[m[0], m[1]]) for m in matches]
                period_lead_times = [int(lt_mat[m[0], m[1]]) for m in period_warn_matches]
                detail_dict["Lead Times"] = lead_times
                score_dict[ScoreComponents.LT] = np.mean(period_lead_times)
                for p in principal_weight_dict.keys():
                    p_mat = self.principal_score_mats[p]
                    p_scores = [p_mat[m[0], m[1]] for m in matches]
                    period_p_scores = [p_mat[m[0], m[1]] for m in period_warn_matches]
                    detail_dict["Component Scores"][p] = [float(ps) for ps in p_scores]
                    score_dict["Component Scores"][p] = np.mean(period_p_scores)

                for p in facet_weight_dict.keys():
                    p_mat = self.facet_score_mats[p]
                    p_scores = [p_mat[m[0], m[1]] for m in matches]
                    period_p_scores = [p_mat[m[0], m[1]] for m in period_warn_matches]
                    detail_dict["Component Scores"][p] = [float(ps) for ps in p_scores]
                    score_dict["Component Scores"][p] = np.mean(period_p_scores)
            else:
                score_dict["Period Warnings Matched"] = 0

        out_dict = dict()
        out_dict["Warnings"] = []
        out_dict["Matches"] = matches
        if self.warnings is not None:
            prob_scores = []
            matched_warn = [m[0] for m in matches]
            for i in range(len(self.warnings)):
                the_warning = self.warnings.iloc[i]
                warn_id = the_warning.Warning_ID
                match_details = dict()
                warn_data = {"Warning": the_warning.to_dict(), "Match": match_details}
                warn_was_matched = i in matched_warn
                warn_versions = self.submitted_warnings[self.submitted_warnings.Warning_ID == warn_id]
                prob_versions = warn_versions.Probability.values
                probm_scores = [self.probability_score(warn_was_matched, p) for p in prob_versions]
                prob_scores.append(np.mean(probm_scores))
                if warn_was_matched:
                    warn_match_index = matched_warn.index(i)
                    gsr_iloc = matches[warn_match_index][1]
                    the_event = self.gsr_events.iloc[gsr_iloc]
                    match_details = {"GSR": the_event.to_dict()}
                    mult_versions_flag = the_warning.sequence > 1
                    if mult_versions_flag:
                        gsr_id = the_event.Event_ID
                        across_version_scores = self._score_across_versions(warn_id, gsr_id)
                        for cs in self.score_components:
                            list_name = "{0}s".format(cs)
                            if list_name in detail_dict:
                                detail_dict[list_name][warn_match_index] = across_version_scores[cs]
                                match_details[cs] = across_version_scores[cs]
                            else:
                                detail_dict["Component Scores"][cs][warn_match_index]\
                                                                     = across_version_scores[cs]
                                match_details[cs] = across_version_scores[cs]

                    for cs in self.score_components:
                        list_name = "{0}s".format(cs)
                        if list_name in detail_dict:
                            scores_list = detail_dict[list_name]
                        else:
                            scores_list = detail_dict["Component Scores"][cs]
                        # Debug code
                        try:
                            match_details[cs] = scores_list[warn_match_index]
                        except IndexError as e:
                            print(repr(e))
                            print("There are {0} matched warnings".format(len(matched_warn)))
                            print(i)
                            print(the_warning)
                            print(cs)
                            print(scores_list)
                            return

                    warn_data["Match"] = match_details
                warn_data["Warning"]["latest"] = str(warn_data["Warning"]["latest"])
                warn_data["Warning"]["sequence"] = int(warn_data["Warning"]["sequence"])
                out_dict["Warnings"].append(warn_data)
            out_dict["Unmatched GSR Events"] = []
            matched_gsr = [m[1] for m in matches]
            if self.gsr_events is not None:
                for gi in range(len(self.gsr_events)):
                    if gi not in matched_gsr:
                        gsr_evt_data = self.gsr_events.iloc[gi].to_dict()
                        out_dict["Unmatched GSR Events"].append(gsr_evt_data)

            detail_dict["Prob-M"] = prob_scores
        else:
            if self.gsr_events is not None:
                out_dict["Unmatched GSR Events"] = [self.gsr_events.iloc[gi].to_dict()
                                                    for gi in range(len(self.gsr_events))]
            else:
                out_dict["Unmatched GSR Events"] = []

        # Recompute the mean metrics
        #if self.warnings is not None and self.gsr_events is not None:
            # TODO 5/12/17:  Figure out how to compute this over just period warnings matched.
            #score_dict["Quality Score"] = np.mean(detail_dict["Quality Scores"])
            #score_dict["Lead Time"] = np.mean(detail_dict["Lead Times"])
            #for cs in self.score_components:
                #if cs not in [ScoreComponents.QS, ScoreComponents.LT]:
                    #score_dict["Component Scores"][cs] = np.mean(detail_dict["Component Scores"][cs])

        out_dict["Results"] = score_dict
        out_dict["Scoring Parameters"] = OrderedDict()
        out_dict["Scoring Parameters"]["Performer ID"] = self.performer_id
        out_dict["Scoring Parameters"]["Country"] = self.country
        out_dict["Scoring Parameters"]["Start Date"] = self.start_date
        out_dict["Scoring Parameters"]["End Date"] = self.end_date
        out_dict["Scoring Parameters"]["Event Type"] = self.event_type
        out_dict["Scoring Parameters"]["Max Distance"] = self.max_dist
        out_dict["Scoring Parameters"]["Max Date Diff"] = self.max_date_diff
        out_dict["Scoring Parameters"]["Distance Buffer"] = self.dist_buffer
        out_dict["Scoring Parameters"]["Date Buffer"] = self.date_buffer
        out_dict["Scoring Parameters"]["Max Score"] = max_score
        out_dict["Scoring Parameters"]["Allow Zero Scores"] = str(allow_zero_scores)
        for p in principal_weight_dict:
            out_dict["Scoring Parameters"][p + " Weight"] = float(principal_weight_dict[p])
        for p in facet_weight_dict:
            out_dict["Scoring Parameters"][p + " Weight"] = float(facet_weight_dict[p])
        out_dict["Details"] = detail_dict
        return out_dict

    def _score_across_versions(self, warn_id, gsr_id):
        """
        Scores all versions of the given warning against the given GSR
        :param warn_id: Warning_ID
        :param gsr_id: GSR Event_ID
        :returns: dict with average scores
        """
        result_dict = dict()
        warn_df = self.submitted_warnings
        gsr_df = self.gsr_events
        germane_warnings = warn_df[warn_df.Warning_ID == warn_id]
        gsr_data = gsr_df[gsr_df.Event_ID == gsr_id].to_dict()
        gsr_data = {k: list(gsr_data[k].values())[0] for k in gsr_data}
        scoring_params = {
            "max_date_diff": self.max_date_diff,
            "max_dist": self.max_dist,
            "loc_approx_buffer": self.dist_buffer,
            "distance_method": DEFAULT_DIST_METHOD,
            "distance_units": DEFAULT_DIST_UNITS
        }
        scoring_weights = {
           ScoreComponents.LS: self.ls_weight,
           ScoreComponents.DS: self.ds_weight, 
           ScoreComponents.PS: self.ps_weight, 
           ScoreComponents.RS: self.rs_weight, 
           ScoreComponents.VS: self.vs_weight, 
           ScoreComponents.AS: self.as_weight, 
           ScoreComponents.TSS: self.tss_weight, 
           ScoreComponents.TTS: self.tts_weight, 
           ScoreComponents.ESS: self.es_weight
        }
        for gw in germane_warnings.index:
            gw_data = germane_warnings.ix[gw].to_dict()
            gw_score = self.score_one(gw_data, gsr_data, scoring_weights, scoring_params)["Results"]
            result_key = gw_data["sequence"]
            result_dict[result_key] = gw_score
        result_df = pd.DataFrame(result_dict)
        new_index = self.score_components
        result_df = result_df.reindex(new_index)
        mean_results = result_df.mean(axis=1)

        return mean_results

    def make_date_diff_mat(self):

        if self.gsr_events is None or self.warnings is None:
            dd_mat = np.empty(shape=(0, 0))
        else:
            warn_dates = list(self.warnings[EVENT_DATE])
            gsr_dates = list(self.gsr_events[EVENT_DATE])
            warn_days = [parse(x).day for x in warn_dates]
            gsr_days = [parse(x).day for x in gsr_dates]

            days_array = make_combination_mats(warn_days, gsr_days)

            def get_date_diff(i,j):
                """
                Looks up the date score for the pair of days
                :param i: Date of the month for the warning
                :param j: Date of the month for the event
                :return: Date Score
                """
                return all_dd_mat[i,j]

            get_date_diff_vfunc = np.vectorize(get_date_diff)
            dd_mat = get_date_diff_vfunc(days_array[0], days_array[1])

        return dd_mat

        def lookup_dist(i, j):
            """
            Uses lookup table to find distance between warning coords i and gsr coords j
            """
            return dist_lookup_df[col_name_list[j]].iloc[row_loc_list[i]]

        lookup_dist_vfunc = np.vectorize(lookup_dist)

        return lookup_dist_vfunc(warn_i_array, gsr_i_array)

    def make_lead_time_mat(self):
        """Builds a matrix of date scores between dates in warnings and GSR."""
        if self.gsr_events is None or self.warnings is None:
            lt_mat = np.empty(shape=(0, 0))
        else:
            warn_dates = np.array(self.warnings[TIMESTAMP].apply(lambda x: str(parse(x).date())))
            gsr_dates = np.array(self.gsr_events[EARLIEST_REPORTED_DATE].apply(lambda x: str(parse(x).date())))
            warn_date_mat, gsr_date_mat = make_combination_mats(warn_dates, gsr_dates)
            lt_vfunc = np.vectorize(Scorer.lead_time)
            lt_mat = lt_vfunc(warn_date_mat, gsr_date_mat)
            #lt_mat = warn_dates[:, None] - gsr_dates -1
            #old_shape = lt_mat.shape
            #lt_mat = pd.to_timedelta(lt_mat.reshape(old_shape[0]*old_shape[1]), box=False)/np.timedelta64(1, "D")
            #lt_mat = lt_mat.reshape(old_shape)
            #lt_mat = -1*lt_mat
        return lt_mat

    def make_ds_mat(self):
        """Builds a matrix of date scores between places in gsr1 and gsr2.
        :returns:
        """
        if self.gsr_events is None or self.warnings is None:
            ds_mat = np.empty(shape=(0, 0))
        else:
            warn_days = self.warnings[EVENT_DATE].apply(
                lambda x: (parse(x).date()-self.first_history_date).days
            )
            gsr_days = self.gsr_events[EVENT_DATE].apply(
                lambda x: (parse(x).date()-self.first_history_date).days
            )

            days_array = make_combination_mats(warn_days, gsr_days)

            def get_date_score(i,j):
                """
                Looks up the date score for the pair of days
                :param i: Date of the month for the warning
                :param j: Date of the month for the event
                :return: Date Score
                """
                return self.all_ds_mat[i,j]

            get_date_score_vfunc = np.vectorize(get_date_score)
            ds_mat = get_date_score_vfunc(days_array[0], days_array[1])

        return ds_mat

    def make_facet_mat(self, col_name):
        """Builds a matrix of facet matches between gsr1 and gsr2."""

        def concat_list(x):
            if isinstance(x, (list, tuple)):
                return ";".join(x)
            else:
                return x

        n_warn = len(self.warnings)
        n_gsr = len(self.gsr_events)
        if min(n_gsr, n_warn) == 0:
            fs_mat = np.empty(shape=(0, 0))
        else:
            warn_facet = np.array(n_gsr*list(self.warnings[col_name].values))
            warn_facet = warn_facet.reshape(n_gsr, n_warn)
            warn_facet = warn_facet.T
            gsr_facet = self.gsr_events[col_name].values
            gsr_facet = [concat_list(x) for x in gsr_facet]
            gsr_facet = np.array(n_warn*gsr_facet)
            gsr_facet = gsr_facet.reshape(n_warn, n_gsr)

            def warn_in_gsr(warn_val, gsr_val):
                return int(warn_val in gsr_val)

            warn_in_gsr_vfunc = np.vectorize(warn_in_gsr)
            fs_mat = warn_in_gsr_vfunc(warn_facet, gsr_facet)
        return fs_mat

    def make_qs_mat(self, principal_weight_dict, facet_weight_dict, max_score=4):
        """
        Build a Quality Score Matrix from the components
        :param max_score: Maximum value of quality score
        :param principal_weight_dict: Dict {principal: weight}
        :param facet_weight_dict: Dict {facet: weight}
        :return: Matrix
        """
        if min(len(self.gsr_events), len(self.warnings)) == 0:
            out_mat = np.empty(shape=(0, 0))
        else:
            weight_dict = dict(ChainMap(facet_weight_dict, principal_weight_dict))
            component_mat_dict = dict(ChainMap(self.facet_score_mats,
                                               self.principal_score_mats))
            out_mat = sum([weight_dict[k]*component_mat_dict[k]
                           for k in weight_dict])
            for psm in self.principal_score_mats:
                out_mat[self.principal_score_mats[psm] == 0] = 0

            lead_time_mat = self.make_lead_time_mat()
            out_mat[lead_time_mat <= 0] = 0

        return out_mat

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
            """
            Scores a specific warning against a specific event.  This stub only computes lead time and probability
            :param warn_data: JSON formatted warning details
            :param gsr_data: JSON formatted GSR details
            :param weights: Scoring weights.  No effect on this method
            :param params: Scoring parameters.  No effect on this method
            :return: Dict with scoring results and/or error details.
            """
            score_results = dict()
            error_flag = False
            error_dict = dict()
            out_dict = dict()
            if JSONField.TIMESTAMP in warn_data and JSONField.EARLIEST_REPORTED_DATE in gsr_data:
                lead_time = cls.lead_time(warn_data[JSONField.TIMESTAMP],
                                             gsr_data[JSONField.EARLIEST_REPORTED_DATE])
                score_results[ScoreComponents.LT] = lead_time
            if JSONField.PROBABILITY in warn_data:
                prob = warn_data[JSONField.PROBABILITY]
                if prob < 0 or prob > 1:
                    error_flag = True
                    error_dict["Invalid Probability"] = prob
                else:
                    prob_m = cls.probability_score(1, warn_data[JSONField.PROBABILITY])
                    score_results["Prob-M"] = prob_m
            out_dict["Results"] = score_results
            if error_flag:
                out_dict["Error"] = error_dict

            return out_dict


class LocationScorer(Scorer):
    """
    Abstract class for all location and time based scoring
    """
    score_components = Scorer.score_components + [ScoreComponents.LS, ScoreComponents.DS]

    def __init__(self, country, start_date, end_date,
                 event_type, max_dist, max_date_diff, dist_buffer, 
                 date_buffer, ls_weight, ds_weight, ps_weight, rs_weight,
                 vs_weight, as_weight, tss_weight, tts_weight, es_weight, 
                 dist_method=DEFAULT_DIST_METHOD, dist_units=DEFAULT_DIST_UNITS,
                 performer_id=None, **kwargs):
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Type of event to be scored
        :param max_dist: What's the maximum distance to have a non-zero score?
        :param dist_method: How are distances to be computed?
        :param units: What are the distance units?
        :param approx_buffer: What is free distance for approximate locations?
        :param max_diff: The maximum date difference for a non-zero score
        :param date_buffer: Time buffer for searching for better matching events
        :param performer_id: The ID of the performer being scored.
        :return:
        """
        self.event_type = event_type
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.date_buffer = date_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.performer_id = performer_id
        self.country = country
        self.start_date = str(parse(start_date).date())
        self.end_date = str(parse(end_date).date())
        self.first_history_date = (parse(self.start_date) - datetime.timedelta(self.date_buffer)).date()
        self.last_history_date = (parse(self.end_date) + datetime.timedelta(self.date_buffer)).date()
        self.historian = Historian()
        search_params = {
          'country' : self.country,
          'event_type' : self.event_type,
          'start_date' : self.first_history_date,
          'end_date' : self.last_history_date,
          'performer_id': self.performer_id
          }
        search_params.update(kwargs)
        self.gsr_events = self.historian.get_history(db.GSR_TYPE, **search_params)
        self.submitted_warnings = self.historian.get_history(db.WARNING_TYPE, **search_params)
        if self.submitted_warnings is not None:
            self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
        else:
            self.warnings = None
        self.dist_method = dist_method
        self.dist_units = dist_units
        self.date_buffer = date_buffer
        self.principal_score_mats = self.set_principal_score_mats()
        self.facet_score_mats = self.set_facet_score_mats()
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)

    def set_principal_score_mats(self):
        """
        Adds LS and DS matrices
        """

        psm = dict()
        if self.gsr_events is not None and self.warnings is not None:
            ls_mat = self.make_ls_mat()
            psm["Location Score"] = ls_mat
            ds_mat = self.make_ds_mat()
            psm["Date Score"] = ds_mat

        return psm

    def do_scoring(self, ls_weight=2,
                   ds_weight=2,
                   max_score=4, allow_zero_scores=False):
        """
        Do scoring
        :param ls_weight: Weight to location score
        :param ds_weight: Weight to date score
        :param max_score: Maximum quality score
        :param allow_zero_scores: Should scores of zero be included in computations
        :return: Dict of score results
        """

        pwd = {"Location Score": self.ls_weight if self.ls_weight is not None else ls_weight,
               "Date Score": self.ds_weight if self.ds_weight is not None else ds_weight,}

        fwd = dict()
        scoring = Scorer.do_scoring(self, pwd, fwd, max_score, allow_zero_scores)
        if self.warnings is None:
            pass
        else:
            matches = scoring["Matches"]
            matched_warn = [x[0] for x in matches]
            match_ser = pd.Series(1, index=matched_warn)
            match_ser = match_ser.reindex(range(len(self.warnings))).fillna(0)
            prob_ser = self.warnings.Probability
            prob_df = pd.DataFrame({"Occurred": match_ser, "Probability": prob_ser})
            prob_df["Prob-M"] = prob_df.apply(lambda x:
                                              self.probability_score(x.Occurred, x.Probability),
                                              axis=1)
#            prob_df["Prob-M"] = prob_df["Prob-M"].apply(lambda x: float(x))
            prob_m_values = [float(pm) for pm in prob_df["Prob-M"].values]
            scoring["Prob-M"] = prob_m_values
            prob_m_value = prob_df["Prob-M"].mean()
            scoring["Results"]["Probability Score"] = prob_m_value

        return scoring

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        out_dict = dict()
        scorer_output = Scorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        score_params = {
            "max_date_diff": MAX_DATE_DIFF,
            "max_dist": DEFAULT_MAX_DIST,
            "loc_approx_buffer": DEFAULT_LOC_APPROX_BUFFER,
            "distance_method": DEFAULT_DIST_METHOD,
            "distance_units": DEFAULT_DIST_UNITS
        }
        if params:
            score_params = dict(ChainMap(params, score_params))
        dist_method = score_params["distance_method"]
        dist_units = score_params["distance_units"]
        max_dist = score_params["max_dist"]
        loc_approx_buffer = score_params["loc_approx_buffer"]
        max_date_diff = score_params["max_date_diff"]
        score_weights = {
            ScoreComponents.LS: 2.0,
            ScoreComponents.DS: 2.0
        }
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        ls_weight = score_weights[ScoreComponents.LS]
        ds_weight = score_weights[ScoreComponents.DS]
        bad_loc_flag = False
        try:
            warn_place = (warn_data[JSONField.LATITUDE], warn_data[JSONField.LONGITUDE])
        except KeyError as e:
            bad_loc_flag = True
            error_dict["Bad Warning Location"] = repr(e)
        try:
            gsr_place = (gsr_data[JSONField.LATITUDE], gsr_data[JSONField.LONGITUDE])
        except KeyError as e:
            bad_loc_flag = True
            error_dict["Bad GSR Location"] = repr(e)
        if JSONField.APPROXIMATE_LOCATION in gsr_data:
            loc_approximate = eval(gsr_data[JSONField.APPROXIMATE_LOCATION])
        else:
            loc_approximate = False
        if not bad_loc_flag:
            distance = LocationScorer.distance(warn_place, gsr_place, dist_method, dist_units)
            score_results["Distance"] = distance
            score_results[JSONField.APPROXIMATE_LOCATION] = loc_approximate
            ls = LocationScorer.distance_score(cls, dist=distance)
            score_results[ScoreComponents.LS] = ls
        bad_date_flag = False
        try:
            warn_date = parse(warn_data[JSONField.EVENT_DATE])
        except KeyError as e:
            bad_date_flag = True
            error_dict["Bad Warning Date"] = repr(e)
        try:
            gsr_date = parse(gsr_data[JSONField.EVENT_DATE])
        except KeyError as e:
            bad_date_flag = True
            error_dict["Bad GSR Date"] = repr(e)

        if not bad_date_flag:
            date_diff = (gsr_date - warn_date).days
            score_results["Date Difference"] = date_diff
            ds = date_score(date_diff=date_diff, max_diff=score_params["max_date_diff"])
            score_results[ScoreComponents.DS] = ds

        error_flag = error_flag or bad_loc_flag or bad_date_flag
        if error_flag:
            out_dict["Error"] = error_dict
        else:
            qs = ls_weight*ls + ds_weight*ds
            score_results[ScoreComponents.QS] = qs
        if score_results:
            out_dict["Results"] = score_results

        return out_dict

    @staticmethod
    def distance(place1, place2, method=DEFAULT_DIST_METHOD, units="km"):
        """
        Computes the distance between place1 and place2 using the method
        :param place1: (Lat,Long) pair
        :param place2: (Lat,Long) pair
        :param method: how to compute distance, default is vincenty
        :param units: distance units, default km
        :return: Distance in specified units
        """
        try:
            dist = method(place1, place2)
            dist = getattr(dist, units)
        except UnboundLocalError:
            dist = False

        return dist

    def distance_score(self, dist):
        """
        Computes distance score between place 1 and place 2
        :param is_approximate_location: Is either location approximate?
        :param dist: Distance value
        :param max_dist: What distance is considered to be 0
        :return: Value between 0 and 1
        """
        max_dist_score = DEFAULT_MAX_DIST if not hasattr(self, 'max_dist') else self.max_dist
        dist_score = max(0.0, (max_dist_score-dist)/max_dist_score)
        return dist_score

    def make_dist_mat(self, method=DEFAULT_DIST_METHOD, units="km"):

        warn_loc = list(zip(self.warnings[JSONField.LATITUDE], self.warnings[JSONField.LONGITUDE]))
        gsr_loc = list(zip(self.gsr_events[JSONField.LATITUDE], self.gsr_events[JSONField.LONGITUDE]))
        warn_loc_u = pd.Series(warn_loc).unique()
        gsr_loc_u = pd.Series(gsr_loc).unique()
        col_name_list = [gsr_loc_u[list(gsr_loc_u).index(x)] for x in gsr_loc]
        row_loc_list = [list(warn_loc_u).index(w) for w in warn_loc]

        warn_i_array, gsr_i_array = make_index_mats(warn_loc, gsr_loc)

        warn_u_i_array, gsr_u_i_array = make_index_mats(warn_loc_u, gsr_loc_u)

        def how_far(i, j):
            warn_coord = warn_loc_u[i]
            gsr_coord = gsr_loc_u[j]
            dist = method(warn_coord, gsr_coord)
            dist = getattr(dist, units)
            return dist

        how_far_vfunc = np.vectorize(how_far)
        dist_lookup = how_far_vfunc(warn_u_i_array, gsr_u_i_array)
        dist_lookup_df = pd.DataFrame(dist_lookup, index=warn_loc_u, columns=gsr_loc_u)

        def lookup_dist(i, j):
            """
            Uses lookup table to find distance between warning coords i and gsr coords j
            """
            return dist_lookup_df[col_name_list[j]].iloc[row_loc_list[i]]

        lookup_dist_vfunc = np.vectorize(lookup_dist)

        return lookup_dist_vfunc(warn_i_array, gsr_i_array)

    def make_ls_mat(self):
        """Builds a matrix of location scores between places in gsr1 and gsr2.
           This assumes that we already have Latitude and Longitude in each
           GSR.
           :param units: Distance units
           :param method: Method for computing distance, default vincenty
           :param gsr1: DataFrame with event or warning data
           :param gsr2: DataFrame with event or warning data
           """
        n_warn, n_gsr = (len(self.warnings), len(self.gsr_events))
        gsr_facet = self.gsr_events[JSONField.APPROXIMATE_LOCATION].values
        gsr_facet = [int(x=="True") for x in gsr_facet]
        gsr_facet = np.array(n_warn*gsr_facet)
        gsr_facet = gsr_facet.reshape(n_warn, n_gsr)

        dist_mat = self.make_dist_mat(method=self.dist_method, units=self.dist_units)
        dist_mat = dist_mat - self.dist_buffer*gsr_facet
        dist_mat = np.maximum(dist_mat, 0)
        ds_vfunc = np.vectorize(self.distance_score)
        ls_mat = ds_vfunc(dist_mat)
        return ls_mat


class CivilUnrestScorer(LocationScorer):

    score_components = LocationScorer.score_components + [ScoreComponents.PS, ScoreComponents.RS, ScoreComponents.VS]

    def __init__(self, country, start_date, end_date,
                 max_dist, max_date_diff, dist_buffer, date_buffer, ls_weight, 
                 ds_weight, ps_weight, rs_weight, vs_weight, as_weight, tss_weight, 
                 tts_weight, es_weight, dist_method=DEFAULT_DIST_METHOD, 
                 dist_units=DEFAULT_DIST_UNITS, performer_id=None, **kwargs):

        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param max_dist: What's the maximum distance to have a non-zero score?
        :param dist_method: How are distances to be computed?
        :param units: What are the distance units?
        :param approx_buffer: What is free distance for approximate locations?
        :param max_diff: The maximum date difference for a non-zero score
        :param date_buffer: Time buffer for searching for better matching events
        :param performer_id: The ID of the performer being scored.
        :return:
        """
        super().__init__(start_date=start_date, end_date=end_date, country=country,
                         event_type=EventType.CIVIL_UNREST, max_dist=max_dist, max_date_diff=max_date_diff, 
                         dist_buffer=dist_buffer, date_buffer=date_buffer, ls_weight=ls_weight, 
                         ds_weight=ds_weight, ps_weight=ps_weight, rs_weight=rs_weight, 
                         vs_weight=vs_weight, as_weight=as_weight, tss_weight=tss_weight, 
                         tts_weight=tts_weight, es_weight=es_weight, dist_method=dist_method, 
                         dist_units=dist_units, performer_id=performer_id, **kwargs)

    def set_facet_score_mats(self):
        """
        Assigns the Facet score matrices
        :return: Dict of labeled score matrices
        """
        facet_score_mats = dict()
        if self.gsr_events is not None and self.warnings is not None:
            facet_score_mats["Population Score"] = self.make_facet_mat(JSONField.POPULATION)
            facet_score_mats["Reason Score"] = self.make_facet_mat(JSONField.REASON)
            facet_score_mats["Violence Score"] = self.make_facet_mat(JSONField.VIOLENT)

        return facet_score_mats

    def do_scoring(self, ls_weight=1, ds_weight=1,
                   ps_weight=1, rs_weight=0.5, vs_weight=0.5,
                   max_score=4, allow_zero_scores=False):
        """
        Scores Civil Unrest
        :param vs_weight: Weight for Violent facet
        :param rs_weight: Weight for Reason facet
        :param ps_weight: Weight for Population facet
        :param ls_weight: How much weight is given to Location Score
        :param ds_weight: How much weight is given to Date Score
        :param max_score: What is the maximum value of QS
        :param allow_zero_scores:
        :return: Dict with scores
        """
        pwd = {"Location Score": self.ls_weight if self.ls_weight is not None else ls_weight,
               "Date Score": self.ds_weight if self.ds_weight is not None else ds_weight}

        fwd = {"Population Score": self.ps_weight if self.ps_weight is not None else ps_weight,
               "Reason Score": self.rs_weight if self.rs_weight is not None else rs_weight,
               "Violence Score": self.rs_weight if self.rs_weight is not None else rs_weight}
        scoring = Scorer.do_scoring(self, pwd, fwd, max_score, allow_zero_scores)
        if self.warnings is None:
            pass
        else:
            matches = scoring["Matches"]
            matched_warn = [x[0] for x in matches]
            match_ser = pd.Series(1, index=matched_warn)
            match_ser = match_ser.reindex(range(len(self.warnings))).fillna(0)
            prob_ser = self.warnings.Probability
            prob_df = pd.DataFrame({"Occurred": match_ser, "Probability": prob_ser})
            prob_df["Prob-M"] = prob_df.apply(lambda x:
                                              self.probability_score(x.Occurred, x.Probability),
                                              axis=1)
            prob_m_values = [float(pm) for pm in prob_df["Prob-M"].values]
            scoring["Prob-M"] = prob_m_values
            prob_m_value = prob_df["Prob-M"].mean()
            scoring["Results"]["Probability Score"] = prob_m_value

        return scoring

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        out_dict = dict()
        scorer_output = LocationScorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        score_params = {
            "max_date_diff": MAX_DATE_DIFF,
            "max_dist": DEFAULT_MAX_DIST,
            "loc_approx_buffer": DEFAULT_LOC_APPROX_BUFFER,
            "distance_method": DEFAULT_DIST_METHOD,
            "distance_units": DEFAULT_DIST_UNITS
        }
        if params:
            score_params = dict(ChainMap(params, score_params))
        dist_method = score_params["distance_method"]
        dist_units = score_params["distance_units"]
        max_dist = score_params["max_dist"]
        loc_approx_buffer = score_params["loc_approx_buffer"]
        max_date_diff = score_params["max_date_diff"]
        score_weights = {
            ScoreComponents.LS: 1.0,
            ScoreComponents.DS: 1.0,
            ScoreComponents.PS: 2/3,
            ScoreComponents.RS: 2/3,
            ScoreComponents.VS: 2/3
        }
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        #have to address this for score_one calls
        ls_weight = score_weights[ScoreComponents.LS]
        ds_weight = score_weights[ScoreComponents.DS]
        ps_weight = score_weights[ScoreComponents.PS]
        rs_weight = score_weights[ScoreComponents.RS]
        vs_weight = score_weights[ScoreComponents.VS]
        if score_results:
            del score_results[ScoreComponents.QS]
            ls = score_results[ScoreComponents.LS]
            ds = score_results[ScoreComponents.DS]
            for field_name in [JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT]:
                try:
                    warn_value = warn_data[field_name]
                except KeyError as e:
                    error_flag = True
                    error_dict["No {0} in Warning".format(field_name)] = repr(e)
                try:
                    gsr_value = gsr_data[field_name]
                except KeyError as e:
                    error_flag = True
                    error_dict["No {0} in GSR".format(field_name)] = repr(e)
            if error_flag:
                pass
            else:
                ps = int(warn_data[JSONField.POPULATION] == gsr_data[JSONField.POPULATION])
                score_results[ScoreComponents.PS] = ps
                rs = int(warn_data[JSONField.REASON] == gsr_data[JSONField.REASON])
                score_results[ScoreComponents.RS] = rs
                vs = int(warn_data[JSONField.VIOLENT] == gsr_data[JSONField.VIOLENT])
                score_results[ScoreComponents.VS] = vs
                qs = ls_weight*ls + ds_weight*ds + ps_weight*ps + rs_weight*rs + vs_weight*vs
                score_results[ScoreComponents.QS] = qs

        if error_flag:
            out_dict["Error"] = error_dict
        if score_results:
            out_dict["Results"] = score_results

        return out_dict


class WidespreadCivilUnrestScorer(CivilUnrestScorer):

    score_components = [ScoreComponents.QS, ScoreComponents.LT] + [ScoreComponents.DS, ScoreComponents.PS,
                                                                   ScoreComponents.RS, ScoreComponents.VS]

    def __init__(self, country, start_date, end_date, max_dist, max_date_diff,
                 dist_buffer, date_buffer, ls_weight, ds_weight, ps_weight, 
                 rs_weight, vs_weight, as_weight, tss_weight, tts_weight, 
                 es_weight, performer_id=None, **kwargs):
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param date_buffer: Buffer of days around scoring period to look for events.
        :param kwargs: Additional keyword arguments
        :return:
        """
        self.event_type = "Widespread Civil Unrest"
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.date_buffer = date_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.performer_id = performer_id
        self.country=country
        self.start_date = str(parse(start_date).date())
        self.end_date = str(parse(end_date).date())
        self.first_history_date = (parse(self.start_date) - datetime.timedelta(self.date_buffer)).date()
        self.last_history_date = (parse(self.end_date) + datetime.timedelta(self.date_buffer)).date()
        self.historian = Historian()
        params = {
          'event_type' : self.event_type,
          'country' : self.country,
          'start_date' : self.first_history_date,
          'end_date' : self.last_history_date,
          'performer_id' : self.performer_id
          }
        params.update(kwargs)
        self.gsr_events = self.historian.get_history(db.GSR_TYPE, **params)
        self.submitted_warnings = self.historian.get_history(db.WARNING_TYPE, **params)
        if self.submitted_warnings is not None:
            print("Submitted Warnings: {0}".format(len(self.submitted_warnings)))
            self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
            print("Latest Warnings: {0}".format(len(self.warnings)))
            print("Late Warnings: {0}".format(len(self.warnings[self.warnings.timestamp > self.end_date])))
        else:
            self.warnings = None

        self.lead_time_mat = self.make_lead_time_mat()

#        self._start_me(country=country, start_date=start_date, end_date=end_date, date_buffer=date_buffer,
#                       performer_id=performer_id)
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        self.principal_score_mats = self.set_principal_score_mats(max_date_diff)
        self.facet_score_mats = self.set_facet_score_mats()

    def set_principal_score_mats(self, max_diff):
        """
        Adds DS matrix
        """

        psm = dict()
        if self.gsr_events is not None and self.warnings is not None:
            ds_mat = self.make_ds_mat()
            psm["Date Score"] = ds_mat

        return psm

    def do_scoring(self, ds_weight=4./3,
                   ps_weight=4./3, rs_weight=2./3, vs_weight=2./3,
                   max_score=4, allow_zero_scores=False):
        """
        Scores Widespread Civil Unrest
        :param vs_weight: Weight for Violent facet
        :param rs_weight: Weight for Reason facet
        :param ps_weight: Weight for Population facet
        :param ds_weight: How much weight is given to Date Score
        :param max_score: What is the maximum value of QS
        :param allow_zero_scores:
        :return: Dict with scores
        """
        pwd = {"Date Score": self.ds_weight if self.ds_weight is not None else ds_weight}

        fwd = {"Population Score": self.ps_weight if self.ps_weight is not None else ps_weight,
               "Reason Score": self.rs_weight if self.rs_weight is not None else rs_weight,
               "Violence Score": self.vs_weight if self.vs_weight is not None else vs_weight}
        scoring = Scorer.do_scoring(self, pwd, fwd, max_score, allow_zero_scores)
        if self.warnings is None:
            pass
        else:
            matches = scoring["Matches"]
            matched_warn = [x[0] for x in matches]
            match_ser = pd.Series(1, index=matched_warn)
            match_ser = match_ser.reindex(range(len(self.warnings))).fillna(0)
            prob_ser = self.warnings.Probability
            prob_df = pd.DataFrame({"Occurred": match_ser, "Probability": prob_ser})
            prob_df["Prob-M"] = prob_df.apply(lambda x:
                                              self.probability_score(x.Occurred, x.Probability),
                                              axis=1)
            prob_m_values = [float(pm) for pm in prob_df["Prob-M"].values]
            scoring["Prob-M"] = prob_m_values
            prob_m_value = prob_df["Prob-M"].mean()
            scoring["Results"]["Probability Score"] = prob_m_value

        return scoring

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        out_dict = dict()
        scorer_output = Scorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        score_params = {
            "max_date_diff": MAX_DATE_DIFF
        }
        if params:
            score_params = dict(ChainMap(params, score_params))
        max_date_diff = score_params["max_date_diff"]
        score_weights = {
            ScoreComponents.DS: 4./3.,
            ScoreComponents.PS: 4./3,
            ScoreComponents.RS: 2./3,
            ScoreComponents.VS: 2./3
        }
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        ds_weight = score_weights[ScoreComponents.DS]
        ps_weight = score_weights[ScoreComponents.PS]
        rs_weight = score_weights[ScoreComponents.RS]
        vs_weight = score_weights[ScoreComponents.VS]
        if score_results:
            if ScoreComponents.QS in score_results:
                del score_results[ScoreComponents.QS]
            w_evt_date = parse(warn_data[JSONField.EVENT_DATE])
            w_gsr_date = parse(gsr_data[JSONField.EVENT_DATE])
            date_diff = (w_evt_date - w_gsr_date).days
            ds = date_score(date_diff, score_params["max_date_diff"])
            score_results[ScoreComponents.DS] = ds
            for field_name in [JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT]:
                try:
                    warn_value = warn_data[field_name]
                except KeyError as e:
                    error_flag = True
                    error_dict["No {0} in Warning".format(field_name)] = repr(e)
                try:
                    gsr_value = gsr_data[field_name]
                except KeyError as e:
                    error_flag = True
                    error_dict["No {0} in GSR".format(field_name)] = repr(e)
            if error_flag:
                pass
            else:
                ps = int(warn_data[JSONField.POPULATION] == gsr_data[JSONField.POPULATION])
                score_results[ScoreComponents.PS] = ps
                rs = int(warn_data[JSONField.REASON] == gsr_data[JSONField.REASON])
                score_results[ScoreComponents.RS] = rs
                vs = int(warn_data[JSONField.VIOLENT] == gsr_data[JSONField.VIOLENT])
                score_results[ScoreComponents.VS] = vs
                qs = ds_weight*ds + ps_weight*ps + rs_weight*rs + vs_weight*vs
                score_results[ScoreComponents.QS] = qs

        if error_flag:
            out_dict["Error"] = error_dict
        if score_results:
            out_dict["Results"] = score_results

        return out_dict


class MaNsaScorer(LocationScorer):
    """
    Parent class for scoring military actions or non-state actor events.
    Don't instantiate
    """

    score_components = LocationScorer.score_components + [ScoreComponents.AS, ScoreComponents.ESS,
                                                          ScoreComponents.TSS, ScoreComponents.TTS]

    def __init__(self, country, start_date, end_date, event_type,
                 max_dist, max_date_diff, dist_buffer, date_buffer, ls_weight, 
                 ds_weight, ps_weight, rs_weight, vs_weight, as_weight, tss_weight, 
                 tts_weight, es_weight, dist_method=DEFAULT_DIST_METHOD, dist_units=DEFAULT_DIST_UNITS,
                 performer_id=None, **kwargs):
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Military Action or Non-State Actor
        :param max_dist: What's the maximum distance to have a non-zero score?
        :param dist_method: How are distances to be computed?
        :param units: What are the distance units?
        :param approx_buffer: What is free distance for approximate locations?
        :param max_diff: The maximum date difference for a non-zero score
        :param date_buffer: Time buffer for searching for better matching events
        :param performer_id: The ID of the performer being scored.
        :return:
        """
        super().__init__(start_date=start_date, end_date=end_date, country=country,
                         event_type=event_type, max_dist=max_dist, max_date_diff=max_date_diff, 
                         dist_buffer=dist_buffer, date_buffer=date_buffer, ls_weight=ls_weight,
                         ds_weight=ds_weight, ps_weight=ps_weight, rs_weight=rs_weight, 
                         vs_weight=vs_weight, as_weight=as_weight, tss_weight=tss_weight, 
                         tts_weight=tts_weight, es_weight=es_weight, dist_method=dist_method, 
                         dist_units=dist_units, performer_id=performer_id, **kwargs)

    def set_facet_score_mats(self):
        """
        Assigns the Facet score matrices
        :return: Dict of labeled score matrices
        """

        facet_score_mats = dict()
        if self.gsr_events is not None and self.warnings is not None:
            facet_score_mats[ScoreComponents.AS] = self._make_actor_score_mat()
            ts_mats = self._make_target_score_mats()
            facet_score_mats[ScoreComponents.TSS] = ts_mats[0]
            facet_score_mats[ScoreComponents.TTS] = ts_mats[1]
            facet_score_mats[ScoreComponents.ESS] = self.make_facet_mat(EVENT_SUBTYPE)

        return facet_score_mats

    def _make_actor_score_mat(self):
        """Builds a matrix of target matches between gsr1 and gsr2."""
        len1 = len(self.warnings)
        len2 = len(self.gsr_events)
        fs_mat = np.zeros(len1*len2).reshape(len1, len2)
        col_name = JSONField.ACTOR
        unspecified_list = UNSPECIFIED_ACTORS
        gsr_actors = [np.zeros(len2)]
        warning_actors = np.zeros(len1)
        actor_id_map = dict()
        actor_id = 1
        UNSPECIFIED_ID = -1
        for j in range(len2):
            facet2 = self.gsr_events.iloc[j][col_name]
            if isinstance(facet2, list):
                for facet_idx in range(len(facet2)):
                    if facet_idx >= len(gsr_actors):
                        gsr_actors.append(np.zeros(len2))
                    if facet2[facet_idx] in unspecified_list:
                        gsr_actors[facet_idx][j] = UNSPECIFIED_ID
                        continue
                    if not facet2[facet_idx] in actor_id_map:
                        actor_id_map[facet2[facet_idx]] = actor_id
                        actor_id += 1
                    gsr_actors[facet_idx][j] = actor_id_map[facet2[facet_idx]]
            else:
                if facet2 in unspecified_list:
                    gsr_actors[0][j] = UNSPECIFIED_ID
                    continue

                if not facet2 in actor_id_map:
                    actor_id_map[facet2] = actor_id
                    actor_id += 1
                gsr_actors[0][j] = actor_id_map[facet2]

        for i in range(len1):
            facet1 = self.warnings.iloc[i][col_name]
            if facet1 in unspecified_list:
                warning_actors[i] = UNSPECIFIED_ID
                continue
            if not facet1 in actor_id_map:
                actor_id_map[facet1] = actor_id
                actor_id += 1
            warning_actors[i] = actor_id_map[facet1]

        # First handle exact equality for warnings and first GSR entry
        warning_tile = np.tile(warning_actors, (len2, 1) ).transpose()
        fs_mat = (warning_tile == np.tile(gsr_actors[0], (len1, 1)))
        fs_mat = fs_mat.astype(np.int32)

        # Now check for exact equality on other GSR entries (GSR might have multiple actors)
        for i in range(1, len(gsr_actors)):
            intermediate_array = (warning_tile == np.tile(gsr_actors[i], (len1, 1)))
            fs_mat = fs_mat.astype(np.int32)

        # Now check for exact equality on other GSR entries (GSR might list multiple actors)
        for i in range(1, len(gsr_actors)):
            intermediate_array = (warning_tile == np.tile(gsr_actors[i], (len1, 1)))
            fs_mat += intermediate_array.astype(np.int32)
        fs_mat = (fs_mat > 0)
        fs_mat = fs_mat.astype(np.int32)

        # Finally hardcode score to 1 if GSR contains any unspecified actor
        for i in range(1, len(gsr_actors)):
            fs_mat[:, np.argwhere(gsr_actors[i] == UNSPECIFIED_ID)] = 1

        return fs_mat

    def _make_target_score_mats(self):
        """
        Builds a matrix of target status matches between GSR and warnings.
        """
        len1 = len(self.warnings)
        len2 = len(self.gsr_events)
        col_name = JSONField.TARGETS
        status_name = JSONField.GSR_TARGET_STATUS
        tgt_name = JSONField.GSR_TARGET
        gsr_targets = [np.zeros(len2)]
        warning_targets = np.zeros(len1)
        gsr_status = [np.zeros(len2)]
        warning_status = np.zeros(len1)
        target_id_map = dict()
        target_id = 1
        status_id_map = dict()
        status_id = 1
        WILDCARD_ID = -1

        for j in range(len2):
            gsr_entry = self.gsr_events.iloc[j][col_name]

            if isinstance(gsr_entry, list):
                for entry_idx in range(len(gsr_entry)):
                    if entry_idx >= len(gsr_targets):
                        gsr_targets.append(np.zeros(len2))
                        gsr_status.append(np.zeros(len2))

                    if gsr_entry[entry_idx][tgt_name] == Wildcards.TARGET_TARGET:
                        gsr_targets[entry_idx][j] = WILDCARD_ID
                    else:
                        if not gsr_entry[entry_idx][tgt_name] in target_id_map:
                            target_id_map[gsr_entry[entry_idx][tgt_name]] = target_id
                            target_id += 1
                        gsr_targets[entry_idx][j] = target_id_map[gsr_entry[entry_idx][tgt_name]]

                    if gsr_entry[entry_idx][status_name] == Wildcards.TARGET_STATUS:
                        gsr_status[entry_idx][j] = WILDCARD_ID
                    else:
                        if not gsr_entry[entry_idx][status_name] in status_id_map:
                            status_id_map[gsr_entry[entry_idx][status_name]] = status_id
                            status_id += 1
                        gsr_status[entry_idx][j] = status_id_map[gsr_entry[entry_idx][status_name]]

            else:
                if gsr_entry[tgt_name] == Wildcards.TARGET_TARGET:
                    gsr_targets[0][j] = WILDCARD_ID
                else:
                    if not gsr_entry[tgt_name] in target_id_map:
                        target_id_map[gsr_entry[tgt_name]] = target_id
                        target_id += 1
                    gsr_targets[0][j] = target_id_map[gsr_entry[tgt_name]]

                if gsr_entry[status_name] == Wildcards.TARGET_STATUS:
                    gsr_status[0][j] = WILDCARD_ID
                else:
                    if not gsr_entry[status_name] in status_id_map:
                        status_id_map[gsr_entry[status_name]] = status_id
                        status_id += 1
                    gsr_status[0][j] = status_id_map[gsr_entry[status_name]]

        for i in range(len1):
            warning_entry = self.warnings.iloc[i][col_name]
            if len(warning_entry) > 1:
                print("ERROR!! len(warning_entry) > 1")
            warning_entry = warning_entry[0]
            if warning_entry[tgt_name] == Wildcards.TARGET_TARGET:
                warning_targets[i] = WILDCARD_ID
            else:
                if not warning_entry[tgt_name] in target_id_map:
                    target_id_map[warning_entry[tgt_name]] = target_id
                    target_id += 1
                warning_targets[i] = target_id_map[warning_entry[tgt_name]]

            if warning_entry[status_name] == Wildcards.TARGET_STATUS:
                warning_status[i] = WILDCARD_ID
            else:
                if not warning_entry[status_name] in status_id_map:
                    status_id_map[warning_entry[status_name]] = status_id
                    status_id += 1
                warning_status[i] = status_id_map[warning_entry[status_name]]

        # Do for targets
        # First handle exact equality for warnings and first GSR entry
        warning_target_tile = np.tile(warning_targets, (len2, 1)).transpose()
        tgt_mat = (warning_target_tile == np.tile(gsr_targets[0], (len1, 1)))
        tgt_mat = tgt_mat.astype(np.int32)

        # Now check for exact equality on other GSR entries (GSR might list multiple targets)
        for i in range(1, len(gsr_targets)):
            tgt_mat += (warning_target_tile == np.tile(gsr_targets[i], (len1, 1)))
        tgt_mat = (tgt_mat > 0)
        tgt_mat = tgt_mat.astype(np.int32)

        # Finally, hardcode score to 1 if GSR contains any wildcard target
        for i in range(len(gsr_targets)):
            tgt_mat[:, np.argwhere(gsr_targets[i] == WILDCARD_ID)] = 1

        # Now do for status
        # First, handle exact equality for warnings and first GSR entry
        warning_status_tile = np.tile(warning_status, (len2, 1)).transpose()
        status_mat = (warning_status_tile == np.tile(gsr_status[0], (len1, 1)))
        status_mat = status_mat.astype(np.int32)

        # Now check for exact equality on other GSR entries (GSR might list multiple targets)
        for i in range(1, len(gsr_status)):
            status_mat += (warning_status_tile == np.tile(gsr_status[i], (len1, 1)))
        status_mat = (status_mat > 0)
        status_mat = status_mat.astype(np.int32)

        # Finally, hardcode score to 1 if GSR contains any wildcard target
        for i in range(len(gsr_status)):
            status_mat[:, np.argwhere(gsr_status[i] == WILDCARD_ID)] = 1

        return status_mat, tgt_mat

    def do_scoring(self, ls_weight=1, ds_weight=1,
                   as_weight=2/3, tss_weight=1/3, tts_weight=1/3, es_weight=2/3,
                   max_score=4, allow_zero_scores=False):
        """
        Scores MA or NSA
        :param ls_weight: Weight given to location score, default 1
        :param ds_weight: Weight given to date score, default 1
        :param as_weight: Weight given to actor score, default 2/3
        :param tss_weight: Weight given to target status score, default 1/3
        :param tts_weight: Weight given to target target score, default 1/3
        :param es_weight: Weight given to event subtype score, default 2/3
        :param max_score: The maximum value of QS, default is 4
        :param allow_zero_scores: Should matching include items with QS=0?  Default False
        :return: Dict with scores
        """
        pwd = {ScoreComponents.LS: self.ls_weight if self.ls_weight is not None else ls_weight,
               ScoreComponents.DS: self.ds_weight if self.ds_weight is not None else ds_weight}

        fwd = {ScoreComponents.AS: self.as_weight if self.as_weight is not None else as_weight,
               ScoreComponents.TSS: self.tss_weight if self.tss_weight is not None else tss_weight,
               ScoreComponents.TTS: self.tts_weight if self.tts_weight is not None else tts_weight,
               ScoreComponents.ESS: self.es_weight if self.es_weight is not None else es_weight}
        scoring = Scorer.do_scoring(self, pwd, fwd, max_score, allow_zero_scores)
        if self.warnings is None:
            pass
        else:
            matches = scoring["Matches"]
            matched_warn = [x[0] for x in matches]
            match_ser = pd.Series(1, index=matched_warn)
            match_ser = match_ser.reindex(range(len(self.warnings))).fillna(0)
            prob_ser = self.warnings.Probability
            prob_df = pd.DataFrame({"Occurred": match_ser, "Probability": prob_ser})
            prob_df["Prob-M"] = prob_df.apply(lambda x:
                                              self.probability_score(x.Occurred, x.Probability),
                                              axis=1)
            prob_m_values = [float(pm) for pm in prob_df["Prob-M"].values]
            scoring["Prob-M"] = prob_m_values
            prob_m_value = prob_df["Prob-M"].mean()
            scoring["Results"]["Probability Score"] = prob_m_value

        return scoring


    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """

        #
        out_dict = dict()
        scorer_output = LocationScorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        score_weights = {
            ScoreComponents.LS: 1.0,
            ScoreComponents.DS: 1.0,
            ScoreComponents.AS: 2./3.,
            ScoreComponents.ESS: 2./3.,
            ScoreComponents.TSS: 1./3.,
            ScoreComponents.TTS: 1./3.
        }
        if score_results:
            try:
                del score_results[ScoreComponents.QS]
            except KeyError:
                pass
            try:
                ls = score_results[ScoreComponents.LS]
            except KeyError:
                ls = 0
            try:
                ds = score_results[ScoreComponents.DS]
            except KeyError:
                ds = 0
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        ls_weight = score_weights[ScoreComponents.LS]
        ds_weight = score_weights[ScoreComponents.DS]
        acs_weight = score_weights[ScoreComponents.AS]
        ess_weight = score_weights[ScoreComponents.ESS]
        tss_weight = score_weights[ScoreComponents.TSS]
        tts_weight = score_weights[ScoreComponents.TTS]
        for field_name in [JSONField.ACTOR, JSONField.SUBTYPE]:
            try:
                warn_value = warn_data[field_name]
            except KeyError as e:
                error_flag = True
                error_dict["No {0} in Warning".format(field_name)] = repr(e)
            try:
                gsr_value = gsr_data[field_name]
            except KeyError as e:
                error_flag = True
                error_dict["No {0} in GSR".format(field_name)] = repr(e)
        for field_name in [JSONField.GSR_TARGET_STATUS, JSONField.GSR_TARGET]:
            try:
                if JSONField.TARGETS in warn_data:
                    if isinstance(warn_data[JSONField.TARGETS], list):
                        warn_value_list = [x[field_name] for x in warn_data[JSONField.TARGETS]]
                    else:
                        warn_value_list = [warn_data[JSONField.TARGETS][field_name]]
                else:
                    warn_value = warn_data[field_name]
            except KeyError as e:
                error_flag = True
                error_dict["No {0} in Warning".format(field_name)] = repr(e)
            try:
                if JSONField.TARGETS in gsr_data:
                    if isinstance(gsr_data[JSONField.TARGETS], list):
                        gsr_value_list = [x[field_name] for x in gsr_data[JSONField.TARGETS]]
                    else:
                        gsr_value_list = [gsr_data[JSONField.TARGETS][field_name]]
                else:
                    gsr_value = gsr_data[field_name]
            except KeyError as e:
                error_flag = True
                error_dict["No {0} in GSR".format(field_name)] = repr(e)
        # Validate Target Status
        target_stati = set([x["Target_Status"] for x in Dictionary.getTargets()])
        if JSONField.TARGETS in warn_data:
            if isinstance(warn_data[JSONField.TARGETS], list):
                warn_ts = warn_data[JSONField.TARGETS][0][JSONField.GSR_TARGET_STATUS]
                warn_tt = warn_data[JSONField.TARGETS][0][JSONField.GSR_TARGET]
            else:
                warn_ts = warn_data[JSONField.TARGETS][JSONField.GSR_TARGET_STATUS]
                warn_tt = warn_data[JSONField.TARGETS][JSONField.TARGET]
        else:
            warn_ts = [warn_data[JSONField.GSR_TARGET_STATUS]]
            warn_tt = [warn_data[JSONField.GSR_TARGET]]
        if JSONField.TARGETS in gsr_data:
            if isinstance(gsr_data[JSONField.TARGETS], list):
                gsr_ts = [x[JSONField.GSR_TARGET_STATUS] for x in gsr_data[JSONField.TARGETS]]
                gsr_tt = [x[JSONField.GSR_TARGET] for x in gsr_data[JSONField.TARGETS]]
            else:
                gsr_ts = [gsr_data[JSONField.TARGETS][JSONField.GSR_TARGET_STATUS]]
                gsr_tt = [gsr_data[JSONField.TARGETS][JSONField.GSR_TARGET]]
        else:
            if isinstance(gsr_data[JSONField.GSR_TARGET_STATUS], list):
                gsr_ts = [x for x in gsr_data[JSONField.GSR_TARGET_STATUS]]
            else:
                gsr_ts = [gsr_data[JSONField.GSR_TARGET_STATUS]]
            if isinstance(gsr_data[JSONField.GSR_TARGET], list):
                gsr_tt = [x for x in gsr_data[JSONField.GSR_TARGET]]
            else:
                gsr_tt = [gsr_data[JSONField.GSR_TARGET]]
        bad_warn_ts_list = []
        if not isinstance(warn_ts, list):
            warn_ts = [warn_ts]
        if not isinstance(warn_tt, list):
            warn_tt = [warn_tt]
        for wts in warn_ts:
            if wts not in target_stati:
                bad_warn_ts_list.append(wts)
        if bad_warn_ts_list:
            error_flag = True
            error_dict["Invalid Warning Target Status"] = bad_warn_ts_list
        bad_gsr_ts_list = []
        for x in gsr_ts:
            if x not in target_stati:
                bad_gsr_ts_list.append(x)
        if bad_gsr_ts_list:
            error_flag = True
            error_dict["Invalid GSR Target Status"] = bad_gsr_ts_list
        target_targets = set([x["Target"] for x in Dictionary.getTargets()])
        bad_gsr_tt_list = []
        for x in gsr_tt:
            if x not in target_targets:
                bad_gsr_tt_list.append(x)
        if bad_gsr_tt_list:
            error_flag = True
            error_dict["Invalid GSR Target Target"] = bad_gsr_tt_list
        bad_warn_tt_list = []
        for wtt in warn_tt:
            if wtt not in target_targets:
                bad_warn_tt_list.append(wtt)
        if bad_warn_tt_list:
            error_flag = True
            error_dict["Invalid Warning Target Target"] = bad_warn_tt_list
        if error_flag:
            pass
        else:
            # Need to call this acs because as is a reserved word
            acs = int(warn_data[JSONField.ACTOR] == gsr_data[JSONField.ACTOR]
                     or gsr_data[JSONField.ACTOR] in Wildcards.ALL_ACTORS)
            score_results[ScoreComponents.AS] = acs
            ess = int(warn_data[JSONField.SUBTYPE] == gsr_data[JSONField.SUBTYPE])
            score_results[ScoreComponents.ESS] = ess
            tss = int(len(set(warn_ts).intersection(set(gsr_ts))) > 0
                      or Wildcards.TARGET_STATUS in gsr_ts)
            score_results[ScoreComponents.TSS] = tss
            tts = int(len(set(warn_tt).intersection(set(gsr_tt))) > 0
                      or Wildcards.TARGET_TARGET in gsr_tt)
            score_results[ScoreComponents.TTS] = tts
            qs = ls_weight * ls + ds_weight * ds + acs_weight * acs + ess_weight * ess\
                 + tss_weight * tss + tts_weight * tts
            score_results[ScoreComponents.QS] = qs

        if error_flag:
            out_dict["Error"] = error_dict
        if score_results:
            out_dict["Results"] = score_results

        return out_dict


class NsaScorer(MaNsaScorer):
    """
    Scoring class for Non-State Actor events.
    """

    def __init__(self, country, start_date, end_date,
                 max_dist, max_date_diff, dist_buffer, date_buffer, ls_weight, 
                 ds_weight, ps_weight, rs_weight, vs_weight, as_weight, 
                 tss_weight, tts_weight, es_weight, dist_method=DEFAULT_DIST_METHOD, 
                 dist_units=DEFAULT_DIST_UNITS, performer_id=None, **kwargs):
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param max_dist: What's the maximum distance to have a non-zero score?
        :param dist_method: How are distances to be computed?
        :param units: What are the distance units?
        :param approx_buffer: What is free distance for approximate locations?
        :param max_diff: The maximum date difference for a non-zero score
        :param date_buffer: Time buffer for searching for better matching events
        :param performer_id: The ID of the performer being scored.
        :return:
        """
        super().__init__(start_date=start_date, end_date=end_date, country=country,
                         event_type=EventType.NONSTATE_ACTOR, max_dist=max_dist, max_date_diff=max_date_diff,
                         dist_buffer=dist_buffer, date_buffer=date_buffer, ls_weight=ls_weight,
                         ds_weight=ds_weight, ps_weight=ps_weight, rs_weight=rs_weight, 
                         vs_weight=vs_weight, as_weight=as_weight, tss_weight=tss_weight, 
                         tts_weight=tts_weight, es_weight=es_weight, dist_method=dist_method,
                         dist_units=dist_units, performer_id=performer_id, **kwargs)


    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        out_dict = dict()
        scorer_output = MaNsaScorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        # Validate subtypes
        nsa_subtypes = [Subtype.ARMED_ASSAULT, Subtype.BOMBING, Subtype.HOSTAGE_TAKING]
        if warn_data[JSONField.SUBTYPE] not in nsa_subtypes:
            error_flag = True
            error_dict["Invalid Warning Subtype"] = warn_data[JSONField.SUBTYPE]
        if gsr_data[JSONField.SUBTYPE] not in nsa_subtypes:
            error_flag = True
            error_dict["Invalid GSR Subtype"] = gsr_data[JSONField.SUBTYPE]
        # Validate that actors are in the NSA Dictionary
        nsa_actors = Dictionary.getNonStateActors()
        if warn_data[JSONField.ACTOR] not in nsa_actors:
            error_flag = True
            error_dict["Invalid Warning Non-State Actor"] = warn_data[JSONField.ACTOR]
        if gsr_data[JSONField.ACTOR] not in nsa_actors:
            error_flag = True
            error_dict["Invalid GSR Non-State Actor"] = gsr_data[JSONField.ACTOR]

        if error_flag:
            out_dict["Error"] = error_dict
        if score_results:
            out_dict["Results"] = score_results

        return out_dict


class MaScorer(MaNsaScorer):
    """
    Scoring class for Non-State Actor events.
    """

    def __init__(self, country, start_date, end_date,
                 max_dist, max_date_diff, dist_buffer, date_buffer, ls_weight, 
                 ds_weight, ps_weight, rs_weight, vs_weight, as_weight, tss_weight, 
                 tts_weight, es_weight, dist_method=DEFAULT_DIST_METHOD, 
                 dist_units=DEFAULT_DIST_UNITS, performer_id=None, **kwargs):
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param max_dist: What's the maximum distance to have a non-zero score?
        :param dist_method: How are distances to be computed?
        :param units: What are the distance units?
        :param approx_buffer: What is free distance for approximate locations?
        :param max_diff: The maximum date difference for a non-zero score
        :param date_buffer: Time buffer for searching for better matching events
        :param performer_id: The ID of the performer being scored.
        :return:
        """
        super().__init__(start_date=start_date, end_date=end_date, country=country,
                         event_type=EventType.MILITARY_ACTION, max_dist=max_dist, max_date_diff=max_date_diff, 
                         dist_buffer=dist_buffer, date_buffer=date_buffer, ls_weight=ls_weight,
                         ds_weight=ds_weight, ps_weight=ps_weight, rs_weight=rs_weight, 
                         vs_weight=vs_weight, as_weight=as_weight, tss_weight=tss_weight, 
                         tts_weight=tts_weight, es_weight=es_weight, dist_method=dist_method,
                         dist_units=dist_units, performer_id=performer_id, **kwargs)

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        out_dict = dict()
        scorer_output = MaNsaScorer.score_one(warn_data, gsr_data, weights, params)
        if "Error" in scorer_output:
            error_flag = True
            error_dict = scorer_output["Error"]
        else:
            error_flag = False
            error_dict = dict()
        if "Results" in scorer_output:
            score_results = scorer_output["Results"]
        else:
            score_results = dict()
        # Validate subtypes
        ma_subtypes = [Subtype.ARMED_CONFLICT, Subtype.FORCE_POSTURE]
        if warn_data[JSONField.SUBTYPE] not in ma_subtypes:
            error_flag = True
            error_dict["Invalid Warning Subtype"] = warn_data[JSONField.SUBTYPE]
        if gsr_data[JSONField.SUBTYPE] not in ma_subtypes:
            error_flag = True
            error_dict["Invalid GSR Subtype"] = gsr_data[JSONField.SUBTYPE]
        # Validate that actors are in the NSA Dictionary
        ma_actors = Dictionary.getStateActors()
        if warn_data[JSONField.ACTOR] not in ma_actors:
            error_flag = True
            error_dict["Invalid Warning State Actor"] = warn_data[JSONField.ACTOR]
        if gsr_data[JSONField.ACTOR] not in ma_actors:
            error_flag = True
            error_dict["Invalid GSR State Actor"] = gsr_data[JSONField.ACTOR]

        if error_flag:
            out_dict["Error"] = error_dict
        if score_results:
            out_dict["Results"] = score_results

        return out_dict


class RareDiseaseScorer(LocationScorer):
    """
    Scoring class for Rare Disease events.
    """

    def __init__(self, country, start_date, end_date,
                 max_dist, max_date_diff, dist_buffer, date_buffer, ls_weight, 
                 ds_weight, ps_weight, rs_weight, vs_weight, as_weight, tss_weight, 
                 tts_weight, es_weight, dist_method=DEFAULT_DIST_METHOD, 
                 dist_units="km", performer_id=None, **kwargs):
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)
        """
        Sets scorer parameters
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param es: An Elasticsearch instance with GSR and Warning data
        :return:
        """
        self.event_type = EventType.DISEASE
        self.country = country
        if not isinstance(start_date, str):
            self.start_date = str(start_date.date())
        else:
            self.start_date = start_date
        if not isinstance(end_date, str):
            self.end_date = str(end_date.date())
        else:
            self.end_date = end_date
        self.date_buffer = date_buffer
        self.historian = RareDiseaseHistorian()

        self.first_history_date = (parse(self.start_date) - datetime.timedelta(self.date_buffer)).date()
        self.last_history_date = (parse(self.end_date) + datetime.timedelta(self.date_buffer)).date()
        self.gsr_events = self.historian.get_history(doc_type=db.GSR_TYPE, country=self.country,
                                                     start_date=self.first_history_date,
                                                     end_date=self.last_history_date, **kwargs)
        self.submitted_warnings = self.historian.get_history(doc_type=db.WARNING_TYPE, country=self.country,
                                                             start_date=self.first_history_date,
                                                             end_date = self.last_history_date, **kwargs)
        if self.submitted_warnings is not None:
            self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
        else:
            self.warnings = None
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.dist_method = dist_method
        self.dist_units = dist_units
        self.performer_id = performer_id
        self.principal_score_mats = dict()
        self.facet_score_mats = dict()
        if self.gsr_events is not None and self.warnings is not None:
            ls_mat = self.make_ls_mat()
            self.principal_score_mats[ScoreComponents.LS] = ls_mat
            ds_mat = self.make_ds_mat()
            self.principal_score_mats[ScoreComponents.DS] = ds_mat
            dis_mat = self.make_facet_mat(JSONField.DISEASE)
            self.principal_score_mats[ScoreComponents.DIS] = dis_mat

    def set_facet_score_mats(self):
        """
        Assigns the Facet score matrices
        :return: Dict of labeled score matrices
        """

        return dict()


    def get_history(self, doc_type=db.GSR_TYPE, verbose=False, **kwargs):
        """
        Convenience Method for retrieving Rare Disease History
        :param doc_type: Which document type to query?  Default is "gsr"
        :param verbose: Whether to print details, default is False
        :param kwargs: Additional arguments to be passed to the query
        :returns: DataFrame of results
        """
        history = None
        chik_args = {JSONField.DISEASE : "Chikungunya"}
        chik_history = self.historian.get_history(doc_type=doc_type, verbose=verbose,
                                                  country=self.country, **chik_args)
        if chik_history is not None:
            history = chik_history
        dengue_args = {JSONField.DISEASE : "Dengue"}
        dengue_history = self.historian.get_history(doc_type=doc_type, verbose=verbose,
                                                    country=self.country, **dengue_args)
        if dengue_history is not None:
            if history is not None:
                history = pd.concat([history, dengue_history],
                                    ignore_index=True)
            else:
                history = dengue_history
        # MERS Non-SA History
        mers_args = {JSONField.DISEASE : "MERS"}
        if self.country != CountryName.SAUDI_ARABIA:
            mers_history = self.historian.get_history(doc_type=doc_type, verbose=verbose,
                                                      country=self.country, **mers_args)
            if mers_history is not None:
                if history is not None:
                    history = pd.concat([history, mers_history],
                                        ignore_index=True)
                else:
                    history = mers_history
        # Avian Flu Non-Egypt History
        af_args = {JSONField.DISEASE : "Avian Influenza"}
        if self.country != CountryName.EGYPT:
            af_history = self.historian.get_history(doc_type=doc_type, verbose=verbose,
                                                    country=self.country, **af_args)
            if af_history is not None:
                if history is not None:
                    history = pd.concat([history, af_history],
                                        ignore_index=True)
                else:
                    history = af_history
        germane_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
                           JSONField.DISEASE,
                           JSONField.LATITUDE, JSONField.LONGITUDE]
        if doc_type == db.GSR_TYPE:
            germane_columns += [EVENT_ID, JSONField.EARLIEST_REPORTED_DATE]
        elif doc_type == db.WARNING_TYPE:
            germane_columns += [WARNING_ID]
        if history is not None:
            return history[germane_columns]
        else:
            return None

    def make_qs_mat_deprecated(self, ls_weight=1, ds_weight=1, dis_weight=2):
        """
        Make a matrix of quality scores
        :param dis_weight: Weight for Disease facet
        :param ls_weight: How much weight is given to Location Score
        :param ds_weight: How much weight is given to Date Score
        :return: Matrix of quality scores
        """

        if min(len(self.warnings), len(self.gsr_events)) == 0:
            qs_mat = np.empty(shape=(0, 0))
        else:
            ls_mat = self.principal_score_mats[ScoreComponents.LS]
            ds_mat = self.principal_score_mats[ScoreComponents.DS]
            dis_mat = self.facet_score_mats[ScoreComponents.DIS]
            principals = [(ls_mat, ls_weight, ScoreComponents.LS),
                          (ds_mat, ds_weight, ScoreComponents.DS),
                          (dis_mat, dis_weight, ScoreComponents.DIS)]
            facets = [(dis_mat, dis_weight, ScoreComponents.DIS)]
            qs_mat = Scorer.make_qs_mat(principals, facets, max_score=4)

        return qs_mat

    def do_scoring(self, ls_weight=1, ds_weight=1, dis_weight=2,
                   max_score=4, allow_zero_scores=False):
        """
        Scores MA or NSA
        :return: Dict with scores
        """
        #ls_mat = self.principal_score_mats["Location Score"]
        #ds_mat = self.principal_score_mats["Date Score"]
        #dis_mat = self.facet_score_mats["Disease Score"]
        pwd = {"Location Score": self.ls_weight if self.ls_weight is not None else ls_weight,
               "Date Score": self.ds_weight if self.ds_weight is not None else ds_weight,
               "Disease Score": dis_weight}

        fwd = dict()
        scoring = Scorer.do_scoring(self, pwd, fwd, max_score, allow_zero_scores)
        if self.warnings is None:
            pass
        else:
            matches = scoring["Matches"]
            matched_warn = [x[0] for x in matches]
            match_ser = pd.Series(1, index=matched_warn)
            match_ser = match_ser.reindex(range(len(self.warnings))).fillna(0)
            prob_ser = self.warnings.Probability
            prob_df = pd.DataFrame({"Occurred": match_ser, "Probability": prob_ser})
            prob_df["Prob-M"] = prob_df.apply(lambda x:
                                              self.probability_score(x.Occurred, x.Probability),
                                              axis=1)
            prob_m_values = [float(pm) for pm in prob_df["Prob-M"].values]
            scoring["Prob-M"] = prob_m_values
            prob_m_value = prob_df["Prob-M"].mean()
            scoring["Results"]["Probability Score"] = prob_m_value

        return scoring

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        score_params = {
            "max_date_diff": MAX_DATE_DIFF,
            "max_dist": DEFAULT_MAX_DIST,
            "loc_approx_buffer": DEFAULT_LOC_APPROX_BUFFER,
            "distance_method": DEFAULT_DIST_METHOD,
            "distance_units": DEFAULT_DIST_UNITS
        }
        if params:
            score_params = dict(ChainMap(params, score_params))
        dist_method = score_params["distance_method"]
        dist_units = score_params["distance_units"]
        max_dist = score_params["max_dist"]
        loc_approx_buffer = score_params["loc_approx_buffer"]
        max_date_diff = score_params["max_date_diff"]
        score_weights = {
            ScoreComponents.LS: 2.,
            ScoreComponents.DS: 2.
        }
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        ls_weight = score_weights[ScoreComponents.LS]
        ds_weight = score_weights[ScoreComponents.DS]
        location_scorer_output = LocationScorer.score_one(warn_data, gsr_data,
                                                          score_weights, score_params)
        error_dict = dict()
        score_results = dict()
        if "Error" in location_scorer_output:
            error_flag = True
            error_dict = location_scorer_output["Error"]
        if "Results" in location_scorer_output:
            score_results = location_scorer_output["Results"].copy()
            ls = score_results[ScoreComponents.LS]
            ds = score_results[ScoreComponents.DS]
            error_flag = False
            error_dict = dict()
            bad_disease_flag = False
            diseases = [DiseaseType.AVIAN_INFLUENZA, DiseaseType.MERS, DiseaseType.DENGUE]
            for field_name in [JSONField.DISEASE]:
                try:
                    warn_value = warn_data[field_name]
                    if warn_value not in diseases:
                        error_flag = True
                        bad_disease_flag = True
                        error_dict["Invalid Warning Disease"] = warn_value
                    not_rd = (warn_value == DiseaseType.AVIAN_INFLUENZA and
                              warn_data[JSONField.COUNTRY] == CountryName.EGYPT) or\
                             (warn_value == DiseaseType.MERS and
                              warn_data[JSONField.COUNTRY] == CountryName.SAUDI_ARABIA)
                    if not_rd:
                        bad_disease_flag = True
                        error_flag = True
                        error_dict["Warning Disease Isn't Rare"] = {JSONField.COUNTRY: warn_data[JSONField.COUNTRY],
                                                                    JSONField.DISEASE: warn_value}
                except KeyError as e:
                    error_flag = True
                    bad_disease_flag = True
                    error_dict["No {0} in Warning".format(field_name)] = repr(e)
                try:
                    gsr_value = gsr_data[field_name]
                    if gsr_value not in diseases:
                        error_flag = True
                        bad_disease_flag = True
                        error_dict["Invalid GSR Disease"] = gsr_value
                    not_rd = (gsr_value == DiseaseType.AVIAN_INFLUENZA and
                              gsr_data[JSONField.COUNTRY] == CountryName.EGYPT) or\
                             (gsr_value == DiseaseType.MERS and
                              gsr_data[JSONField.COUNTRY] == CountryName.SAUDI_ARABIA)
                    if not_rd:
                        bad_disease_flag = True
                        error_flag = True
                        error_dict["GSR Disease Isn't Rare"] = {JSONField.COUNTRY: gsr_data[JSONField.COUNTRY],
                                                                    JSONField.DISEASE: gsr_value}
                except KeyError as e:
                    error_flag = True
                    bad_disease_flag = True
                    error_dict["No {0} in GSR".format(field_name)] = repr(e)

            if not bad_disease_flag:
                dis_match = warn_data[JSONField.DISEASE] == gsr_data[JSONField.DISEASE]
                if not dis_match:
                    error_flag = True
                    error_dict["Disease Mismatch"] = {"Warning Disease": warn_data[JSONField.DISEASE],
                                                      "GSR Disease": gsr_data[JSONField.DISEASE]}

            out_dict = {"Results": score_results}
            if error_flag:
                out_dict["Error"] = error_dict

            return out_dict


    @staticmethod
    def score_one(warn_data, gsr_data, weights=None, params=None):
        """
        Scores a single warning against a single event
        :param warn_data: Dict with warning data
        :param gsr_data: Dict with event data
        :param weights: Dict with scoring weights
        :param params: Dict with parameters used in scoring
        :return:
        """
        score_params = {
            "max_date_diff": MAX_DATE_DIFF,
            "max_dist": DEFAULT_MAX_DIST,
            "loc_approx_buffer": DEFAULT_LOC_APPROX_BUFFER,
            "distance_method": DEFAULT_DIST_METHOD,
            "distance_units": DEFAULT_DIST_UNITS
        }
        if params:
            score_params = dict(ChainMap(params, score_params))
        dist_method = score_params["distance_method"]
        dist_units = score_params["distance_units"]
        max_dist = score_params["max_dist"]
        loc_approx_buffer = score_params["loc_approx_buffer"]
        max_date_diff = score_params["max_date_diff"]
        score_weights = {
            ScoreComponents.LS: 2.,
            ScoreComponents.DS: 2.
        }
        if weights:
            score_weights = dict(ChainMap(weights, score_weights))
        ls_weight = score_weights[ScoreComponents.LS]
        ds_weight = score_weights[ScoreComponents.DS]
        location_scorer_output = LocationScorer.score_one(warn_data, gsr_data,
                                                          score_weights, score_params)
        error_dict = dict()
        score_results = dict()
        if "Error" in location_scorer_output:
            error_flag = True
            error_dict = location_scorer_output["Error"]
        if "Results" in location_scorer_output:
            score_results = location_scorer_output["Results"].copy()
            ls = score_results[ScoreComponents.LS]
            ds = score_results[ScoreComponents.DS]
            error_flag = False
            error_dict = dict()
            bad_disease_flag = False
            diseases = [DiseaseType.AVIAN_INFLUENZA, DiseaseType.MERS, DiseaseType.DENGUE]
            for field_name in [JSONField.DISEASE]:
                try:
                    warn_value = warn_data[field_name]
                    if warn_value not in diseases:
                        error_flag = True
                        bad_disease_flag = True
                        error_dict["Invalid Warning Disease"] = warn_value
                    not_rd = (warn_value == DiseaseType.AVIAN_INFLUENZA and
                              warn_data[JSONField.COUNTRY] == CountryName.EGYPT) or\
                             (warn_value == DiseaseType.MERS and
                              warn_data[JSONField.COUNTRY] == CountryName.SAUDI_ARABIA)
                    if not_rd:
                        bad_disease_flag = True
                        error_flag = True
                        error_dict["Warning Disease Isn't Rare"] = {JSONField.COUNTRY: warn_data[JSONField.COUNTRY],
                                                                    JSONField.DISEASE: warn_value}
                except KeyError as e:
                    error_flag = True
                    bad_disease_flag = True
                    error_dict["No {0} in Warning".format(field_name)] = repr(e)
                try:
                    gsr_value = gsr_data[field_name]
                    if gsr_value not in diseases:
                        error_flag = True
                        bad_disease_flag = True
                        error_dict["Invalid GSR Disease"] = gsr_value
                    not_rd = (gsr_value == DiseaseType.AVIAN_INFLUENZA and
                              gsr_data[JSONField.COUNTRY] == CountryName.EGYPT) or\
                             (gsr_value == DiseaseType.MERS and
                              gsr_data[JSONField.COUNTRY] == CountryName.SAUDI_ARABIA)
                    if not_rd:
                        bad_disease_flag = True
                        error_flag = True
                        error_dict["GSR Disease Isn't Rare"] = {JSONField.COUNTRY: gsr_data[JSONField.COUNTRY],
                                                                    JSONField.DISEASE: gsr_value}
                except KeyError as e:
                    error_flag = True
                    bad_disease_flag = True
                    error_dict["No {0} in GSR".format(field_name)] = repr(e)

            if not bad_disease_flag:
                dis_match = warn_data[JSONField.DISEASE] == gsr_data[JSONField.DISEASE]
                if not dis_match:
                    error_flag = True
                    error_dict["Disease Mismatch"] = {"Warning Disease": warn_data[JSONField.DISEASE],
                                                      "GSR Disease": gsr_data[JSONField.DISEASE]}

            out_dict = {"Results": score_results}
            if error_flag:
                out_dict["Error"] = error_dict

            return out_dict


class BooleanScorer(Scorer):

    @staticmethod
    def quality_score(event, warning):
        prob_score = Scorer.probability_score(event, warning["Probability"])
        return 4*prob_score


class CaseCountScorer(Scorer):

    @staticmethod
    def quality_score(predicted, actual,
                      occurrence_weight=0.5,
                      accuracy_denominator=4.0):
        """Computes the occurrence-based QS for predicted and actual values.

        Args:
            predicted:  What count is predicted?
            actual: What count occurred?
            occurrence_weight:  What weight to give to the occurrence component?
                               Default changed to 0.5 on 10/24/14
            accuracy_denominator: What's the minimum denominator in the accuracy
                                  score component?  Default changed to 4 on 10/24

        Returns:
            Dict with QS and components
        """

        accuracy_weight = 4 - occurrence_weight
        occurrence_score = ((predicted == 0) == (actual == 0))
        accuracy_score = 1 - (abs(actual - predicted)*1.0/\
                              max(actual, predicted, accuracy_denominator))
        qs = occurrence_weight*occurrence_score + accuracy_weight*accuracy_score
        result_dict = dict()
        result_dict["Quality Score"] = qs
        result_dict["Occurrence Score"] = int(occurrence_score)
        result_dict["Accuracy Score"] = accuracy_score

        return result_dict

    @classmethod
    def score_one(cls, warn_data, gsr_data, weights, params):
        """
        Scores a single case count warning
        :param warn_data:
        :param gsr_data:
        :param weights:
        :param params:
        :return: Dict with results
        """
        #TODO: Add Lead Time to this
        out_dict = dict()
        error_flag = False
        error_details = dict()
        # Check the warning and GSR event inputs for legitimacy
        warn_event_type = warn_data[EVENT_TYPE]
        gsr_event_type = gsr_data[EVENT_TYPE]
        event_type_mismatch_flag = warn_event_type != gsr_event_type
        if event_type_mismatch_flag:
            error_flag = True
            error_details["Event Type Mismatch"] = {"Warning": warn_event_type,
                                                    "GSR": gsr_event_type}
        bad_country_flag = warn_data[JSONField.COUNTRY] != gsr_data[JSONField.COUNTRY]
        if bad_country_flag:
            error_flag = True
            error_details["Country Mismatch"] = {"Warning": warn_data[JSONField.COUNTRY],
                                                 "GSR": gsr_data[JSONField.COUNTRY]}
        bad_date_flag = warn_data[EVENT_DATE] != gsr_data[EVENT_DATE]
        if bad_date_flag:
            error_flag = True
            error_details["Event Date Mismatch"] = {"Warning": warn_data[EVENT_DATE],
                                                    "GSR": gsr_data[EVENT_DATE]}
        mers_flag = (gsr_data[JSONField.COUNTRY] == CountryName.SAUDI_ARABIA) and \
                    (gsr_event_type == JSONField.DISEASE) and \
                    (gsr_data[JSONField.DISEASE] == DiseaseType.MERS)
        eaf_flag = (gsr_data[JSONField.COUNTRY] == CountryName.EGYPT) and \
                    (gsr_event_type == JSONField.DISEASE) and \
                    (gsr_data[JSONField.DISEASE] == DiseaseType.AVIAN_INFLUENZA)
        cc_disease_flag = mers_flag or eaf_flag
        icews_flag = (gsr_data[EVENT_TYPE] == EventType.ICEWS_PROTEST)
        case_count_flag = cc_disease_flag or icews_flag
        if not case_count_flag:
            error_flag = True
            if JSONField.DISEASE in gsr_data:
                bad_disease_flag = JSONField.DISEASE not in (DiseaseType.AVIAN_INFLUENZA,
                                                             DiseaseType.MERS)
            else:
                bad_disease_flag = False
            if bad_disease_flag:
                error_details["Invalid Disease"] = gsr_data[JSONField.DISEASE]
            bad_mers_country_flag = (gsr_data[JSONField.COUNTRY] != CountryName.SAUDI_ARABIA) and \
                    (gsr_event_type == JSONField.DISEASE)
            if bad_mers_country_flag:
                error_details["Invalid MERS Country"] = gsr_data[JSONField.COUNTRY]
            bad_eaf_country_flag = (gsr_data[JSONField.COUNTRY] != CountryName.EGYPT and
                            gsr_event_type == JSONField.DISEASE and
                            gsr_data[JSONField.DISEASE] == DiseaseType.AVIAN_INFLUENZA)
            if bad_eaf_country_flag:
                error_details["Invalid EAF Country"] = gsr_data[JSONField.COUNTRY]
            error_details["Bad Event Type"] = warn_data[EVENT_TYPE]

        if error_flag:
            out_dict["Error"] = error_details
        # Check the weights for legitimacy
        # Check the parameters for legitimacy
        # Do the scoring
        else:
            if "Occurrence Weight" in weights:
                occurrence_weight = weights["Occurrence Weight"]
            else:
                occurrence_weight = 0.5
            if "Accuracy Denominator" in weights:
                accuracy_denominator = weights["Accuracy Denominator"]
            else:
                accuracy_denominator = 4.0
            warn_case_count = warn_data[JSONField.CASE_COUNT]
            gsr_case_count = gsr_data[JSONField.CASE_COUNT]
            out_dict["Results"] = CaseCountScorer.quality_score(
                predicted=warn_case_count,
                actual=gsr_case_count,
                occurrence_weight = occurrence_weight,
                accuracy_denominator = accuracy_denominator
            )


        return out_dict



    def do_scoring(self, max_score=4, allow_zero_scores=True):
        """
        Perform scoring on ICEWS Protest Events
        :param max_score: What's the highest allowable score for an event-warning pair?
        :param allow_zero_scores: Should we allow scores of 0?  Default True
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


class IcewsScorer(CaseCountScorer):

    def __init__(self, country, start_date, end_date, max_dist, max_date_diff, dist_buffer, 
                date_buffer, ls_weight, ds_weight, ps_weight, rs_weight, vs_weight, 
                as_weight, tss_weight, tts_weight, es_weight, performer_id=None):
        self.event_type = EventType.ICEWS_PROTEST
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.date_buffer = date_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.performer_id = performer_id
        self._start_me(country, start_date, end_date)
        self.principal_score_mats = dict()
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)

    def _start_me(self, country, start_date, end_date, **kwargs):
        """
        Method to group the setting of common items across all subclasses
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Type of event to be scored
        :param es: An Elasticsearch instance with GSR and Warning data
        :return:
        """

        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.historian = IcewsHistorian()
        params = {
          'country' : self.country,
          'start_date' : self.start_date,
          'end_date' : self.end_date,
          'performer_id' : self.performer_id,
          }
        params.update(kwargs)
        self.gsr_events = self.historian.get_history(db.GSR_TYPE, **params)
        self.submitted_warnings = self.historian.get_history(db.WARNING_TYPE, **params)
        if self.submitted_warnings is not None:
            if "latest" in self.submitted_warnings:
                self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
            else:
                self.warnings = self.submitted_warnings
        else:
            self.warnings = self.submitted_warnings


class CaseCountDiseaseScorer(CaseCountScorer):

    def __init__(self, country, start_date, end_date, max_dist, max_date_diff, dist_buffer, 
                 date_buffer, ls_weight, ds_weight, ps_weight, rs_weight, vs_weight, 
                 as_weight, tss_weight, tts_weight, es_weight, performer_id=None):
        self.event_type = EventType.DISEASE
        self.max_dist = max_dist
        self.max_date_diff = max_date_diff
        self.dist_buffer = dist_buffer
        self.date_buffer = date_buffer
        self.ls_weight = ls_weight
        self.ds_weight = ds_weight
        self.ps_weight = ps_weight
        self.rs_weight = rs_weight
        self.vs_weight = vs_weight
        self.as_weight = as_weight
        self.tss_weight = tss_weight
        self.tts_weight = tts_weight
        self.es_weight = es_weight
        self.performer_id = performer_id
        self._start_me(country, start_date, end_date)
        self.principal_score_mats = dict()
        self.all_ds_mat = ds_vfunc(all_dd_mat, max_date_diff)

    def _start_me(self, country, start_date, end_date, **kwargs):
        """
        Method to group the setting of common items across all subclasses
        :type kwargs: Dict of additional arguments
        :param country: Sets the country
        :param start_date: First date of scoring interval
        :param end_date: Last date of scoring interval
        :param event_type: Type of event to be scored
        :param es: An Elasticsearch instance with GSR and Warning data
        :return:
        """

        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.historian = CaseCountDiseaseHistorian()
        params = {
          'country' : self.country,
          'start_date' : self.start_date,
          'end_date' : self.end_date,
          'performer_id' : self.performer_id,
          }
        params.update(kwargs)
        self.gsr_events = self.historian.get_history(db.GSR_TYPE, **params)
        self.submitted_warnings = self.historian.get_history(db.WARNING_TYPE, **params)
        if self.submitted_warnings is not None:
            if "latest" in self.submitted_warnings:
                self.warnings = self.submitted_warnings[self.submitted_warnings.latest]
            else:
                self.warnings = self.submitted_warnings
        else:
            self.warnings = self.submitted_warnings

SCORER = {
  EventType.CIVIL_UNREST : CivilUnrestScorer,
  EventType.WIDESPREAD_CIVIL_UNREST: WidespreadCivilUnrestScorer,
  EventType.NONSTATE_ACTOR : NsaScorer,
  EventType.MILITARY_ACTION : MaScorer,
  EventType.DISEASE : {
    DiseaseType.MERS : CaseCountDiseaseScorer,
    DiseaseType.AVIAN_INFLUENZA : CaseCountDiseaseScorer,
    DiseaseType.RARE : RareDiseaseScorer,
    },
  EventType.ICEWS_PROTEST : IcewsScorer,
  }

