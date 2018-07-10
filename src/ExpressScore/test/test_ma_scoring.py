import unittest
import sys
sys.path.append("..")
from main.express_score import (
    Scorer,
    MaScorer,
    Defaults
)
from main.schema import (
    JSONField,
    ScoreComponents
)
import pandas as pd
import numpy as np
from dateutil.parser import parse
import json
import os

EXPRESS_SCORE_HOME = os.path.abspath("..")
RESOURCE_PATH = os.path.join(EXPRESS_SCORE_HOME, "resources")
TEST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "test")
LB_MA_TEST_PATH = os.path.join(TEST_RESOURCE_PATH, "lb_ma_may_2018")
SA_MA_TEST_PATH = os.path.join(TEST_RESOURCE_PATH, "sa_ma_may_2018")
EG_MA_TEST_PATH = os.path.join(TEST_RESOURCE_PATH, "eg_ma_may_2018")
IQ_MA_TEST_PATH = os.path.join(TEST_RESOURCE_PATH, "iq_ma_may_2018")


class ScorerTest(unittest.TestCase):

    def test_slope_score(self):
        min_value = 0
        max_value = 100
        too_low = -5
        too_high = 105
        just_right = 50
        result = Scorer.slope_score(too_low, min_value, max_value)
        self.assertAlmostEqual(result, 1)
        result = Scorer.slope_score(min_value, min_value, max_value)
        self.assertAlmostEqual(result, 1)
        result = Scorer.slope_score(just_right, min_value, max_value)
        self.assertAlmostEqual(result, 0.5)
        result = Scorer.slope_score(max_value, min_value, max_value)
        self.assertAlmostEqual(result, 0)
        result = Scorer.slope_score(too_high, min_value, max_value)
        self.assertAlmostEqual(result, 0)
        self.assertRaises(ValueError, Scorer.slope_score, just_right, min_value, min_value)
        self.assertRaises(ValueError, Scorer.slope_score, just_right, max_value, min_value)

    def test_f1(self):
        """
        Tests Scorer.f1
        :return:
        """
        p, r = (0,0)
        expected = 0
        result = Scorer.f1(p,r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (1,1)
        expected = 1
        result = Scorer.f1(p, r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (1,1)
        expected = 1
        result = Scorer.f1(p, r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (.5,.5)
        expected = .5
        result = Scorer.f1(p, r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (0,1)
        expected = 0
        result = Scorer.f1(p, r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (.25,.75)
        expected = 0.375
        result = Scorer.f1(p, r)
        self.assertAlmostEqual(result, expected, 3)

        p, r = (-.5, 1)
        self.assertRaises(ValueError, Scorer.f1, p, r)

        p, r = (2,1)
        self.assertRaises(ValueError, Scorer.f1, p, r)

        p, r = (.5, -.1)
        self.assertRaises(ValueError, Scorer.f1, p, r)

        p, r = (.5,2)
        self.assertRaises(ValueError, Scorer.f1, p, r)

    def test_date_diff(self):

        # Test when both are 0
        warn_date = "2018-06-22"
        gsr_date_range = pd.date_range("2018-06-17", "2018-06-27")
        gsr_dates = [d.strftime("%Y-%m-%d") for d in gsr_date_range]
        expected_values = range(-5, 6)
        for i, d in enumerate(gsr_dates):
            result = Scorer.date_diff(warn_date, d)
            expected = expected_values[i]
            self.assertAlmostEqual(result, expected)

    def test_date_score(self):

        date_diffs = range(-6, 7)
        results = [Scorer.date_score(dd) for dd in date_diffs]
        expected = [0, 0, 0, .25, .5, .75, 1, .75, .5, .25, 0, 0, 0]
        for i, e in enumerate(expected):
            self.assertAlmostEqual(results[i], e, 3)
        max_date_diff = 5
        results = [Scorer.date_score(dd, max_date_diff) for dd in date_diffs]
        expected = [0, 0, .2, .4, .6, .8, 1, .8, .6, .4, .2, 0, 0]
        for i, e in enumerate(expected):
            self.assertAlmostEqual(results[i], e, 3)

    def test_make_index_mats(self):
        """
        Tests Scorer.make_index_mats method
        :return:
        """
        row_names = ["a", "b", "c", "d"]
        row_indices = list(range(len(row_names)))
        col_names = ["x", "y", "z"]
        col_indices = list(range(len(col_names)))
        row_array = np.array(row_indices*3).reshape(3,4).T
        col_array = np.array(col_indices*4).reshape(4,3)
        results = Scorer.make_index_mats(row_names, col_names)
        try:
            np.testing.assert_equal(row_array, results[0])
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        try:
            np.testing.assert_equal(col_array, results[1])
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_combination_mats(self):
        """
        Tests Scorer.make_combination_mats method
        :return:
        """
        row_names = ["a", "b", "c", "d"]
        col_names = ["x", "y", "z"]
        row_array = np.array(row_names*3).reshape(3,4).T
        col_array = np.array(col_names*4).reshape(4,3)
        results = Scorer.make_combination_mats(row_names, col_names)
        try:
            np.testing.assert_equal(row_array, results[0])
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        try:
            np.testing.assert_equal(col_array, results[1])
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)


class MaScorerTest(unittest.TestCase):

    country = "Egypt"
    scorer = MaScorer(country=country)
    warn_dict = dict()
    warn_dict[JSONField.WARNING_ID] = "test_1"
    warn_dict[JSONField.EVENT_TYPE] = "Military Action"
    warn_dict[JSONField.COUNTRY] = country
    warn_dict[JSONField.EVENT_DATE] = "2018-05-27"
    warn_dict[JSONField.TIMESTAMP] = "20160324T00:01:01"
    gsr_dict = dict()
    gsr_dict[JSONField.EVENT_TYPE] = "Disease"
    gsr_dict[JSONField.EVENT_ID] = "Disease_Saudi_Arabia_MERS_2016-03-27"
    gsr_dict[JSONField.DISEASE] =  "MERS"
    gsr_dict[JSONField.COUNTRY] = "Saudi Arabia"
    gsr_dict[JSONField.EVENT_DATE] = "2016-03-27"
    gsr_dict[JSONField.EARLIEST_REPORTED_DATE] = "2016-04-01"
    result_dict = dict()
    result_dict[JSONField.WARNING_ID] = "test_1"
    result_dict[JSONField.EVENT_ID] = "Disease_Saudi_Arabia_MERS_2016-03-27"

    def test_ls(self):
        result = MaScorer.location_score(0, is_approximate=False)
        expected = 1.0
        self.assertAlmostEqual(result, expected)
        result = MaScorer.location_score(0, is_approximate="False")
        expected = 1.0
        self.assertAlmostEqual(result, expected)
        # 22 km distance
        result = MaScorer.location_score(22)
        expected = 0.78
        self.assertAlmostEqual(result, expected, 2)
        result = MaScorer.location_score(22.17, is_approximate=True)
        expected = 0.934
        self.assertAlmostEqual(result, expected, 3)
        result = MaScorer.location_score(22.1, max_dist=44.2)
        expected = 0.50
        self.assertAlmostEqual(result, expected, 2)
        result = MaScorer.location_score(150)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def test_make_dist_mat(self):
        """
        Tests MaScorer.make_dist_mat
        :return:
        """

        test_warn_filename = "ma_test_warnings.json"
        test_warn_path = os.path.join(TEST_RESOURCE_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "ma_test_gsr.json"
        test_gsr_path = os.path.join(TEST_RESOURCE_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        result = MaScorer.make_dist_mat(test_warnings, test_gsr)
        expected = np.array([156.672, 156.672, 22.173, 22.173]).reshape(4,1)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        test_gsr.append(test_warnings[-1])
        expected = np.array([156.672, 145.956, 156.672, 145.956, 22.173, 0, 22.173, 0]).reshape(4,2)
        result = MaScorer.make_dist_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)
        dist_mat_filename = "test_lb_dist_matrix.csv"
        dist_mat_path = os.path.join(LB_MA_TEST_PATH, dist_mat_filename)
        expected = np.genfromtxt(dist_mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_dist_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_ls_mat(self):
        """
        Tests MaScorer.make_ls_mat
        :return:
        """
        test_warn_filename = "ma_test_warnings.json"
        test_warn_path = os.path.join(TEST_RESOURCE_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "ma_test_gsr.json"
        test_gsr_path = os.path.join(TEST_RESOURCE_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        expected = np.array([0, 0, 0.778, 0.778]).reshape(4,1)
        result = MaScorer.make_ls_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)
        test_gsr.append(test_warnings[-1])
        test_gsr[-1]["Approximate_Location"] = "False"
        expected = np.array([0, 0, 0, 0, 0.778, 1, 0.778, 1]).reshape(4,2)
        result = MaScorer.make_ls_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        ls_mat_filename = "test_ls_matrix.csv"
        ls_mat_path = os.path.join(LB_MA_TEST_PATH, ls_mat_filename)
        expected = np.genfromtxt(ls_mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_ls_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_ds_mat(self):
        """
        Tests MaScorer.make_ds_mat
        :return:
        """
        test_warn_filename = "ma_test_warnings.json"
        test_warn_path = os.path.join(TEST_RESOURCE_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "ma_test_gsr.json"
        test_gsr_path = os.path.join(TEST_RESOURCE_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        expected = np.array([0, .75, 0, .75]).reshape(4,1)
        result = MaScorer.make_ds_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        ds_mat_filename = "test_ds_matrix.csv"
        ds_mat_path = os.path.join(LB_MA_TEST_PATH, ds_mat_filename)
        expected = np.genfromtxt(ds_mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_ds_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_ess_mat(self):
        """
        Tests MaScorer.make_ess_mat
        :return:
        """
        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        ess_mat_filename = "test_es_match_matrix.csv"
        ess_mat_path = os.path.join(LB_MA_TEST_PATH, ess_mat_filename)
        expected = np.genfromtxt(ess_mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_ess_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_as_mat(self):
        """
        Tests MaScorer.make_as_mat
        :return:
        """
        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        acs_mat_filename = "test_actor_match_matrix.csv"
        acs_mat_path = os.path.join(LB_MA_TEST_PATH, acs_mat_filename)
        expected = np.genfromtxt(acs_mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_as_mat(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_make_qs_mat(self):
        """
        Tests MaScorer.make_qs_df
        :return:
        """
        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        mat_filename = "test_qs_mat.csv"
        mat_path = os.path.join(LB_MA_TEST_PATH, mat_filename)
        expected = pd.read_csv(mat_path, index_col=0)
        result = MaScorer.make_qs_df(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_cc_warnings.json"
        test_warn_path = os.path.join(EG_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_cc_gsr.json"
        test_gsr_path = os.path.join(EG_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        mat_filename = "test_qs_mat.csv"
        mat_path = os.path.join(EG_MA_TEST_PATH, mat_filename)
        expected = np.genfromtxt(mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_qs_df(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_cc_warnings.json"
        test_warn_path = os.path.join(SA_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_cc_gsr.json"
        test_gsr_path = os.path.join(SA_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        mat_filename = "test_qs_mat.csv"
        mat_path = os.path.join(SA_MA_TEST_PATH, mat_filename)
        expected = np.genfromtxt(mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_qs_df(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

        test_warn_filename = "test_cc_warnings.json"
        test_warn_path = os.path.join(IQ_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_cc_gsr.json"
        test_gsr_path = os.path.join(IQ_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        mat_filename = "test_qs_mat.csv"
        mat_path = os.path.join(IQ_MA_TEST_PATH, mat_filename)
        expected = np.genfromtxt(mat_path, delimiter=",", skip_header=True)[:, 1:]
        result = MaScorer.make_qs_df(test_warnings, test_gsr)
        try:
            np.testing.assert_allclose(result, expected, 3)
            test_res = True
        except AssertionError as e:
            test_res = False
            print(repr(e))
        self.assertTrue(test_res)

    def test_facet_score(self):
        """
        Tests Scorer.facet_score
        :return:
        """
        wildcards = ["Unspecified", "Wildcard"]
        warn_value = "Fred"
        gsr_value = "Ethel"
        expected = 0
        result = Scorer.facet_score(warn_value, gsr_value, wildcards)
        self.assertEqual(result, expected)
        gsr_value = wildcards[0]
        expected = 1
        result = Scorer.facet_score(warn_value, gsr_value, wildcards)
        self.assertEqual(result, expected)
        gsr_value = wildcards[1]
        expected = 1
        result = Scorer.facet_score(warn_value, gsr_value, wildcards)
        self.assertEqual(result, expected)
        warn_value = wildcards[0]
        gsr_value = "Ethel"
        expected = 0
        result = Scorer.facet_score(warn_value, gsr_value, wildcards)
        self.assertEqual(result, expected)
        warn_value = "Fred"
        gsr_value = ["Ethel", "Fred"]
        expected = 1
        result = Scorer.facet_score(warn_value, gsr_value, wildcards)
        self.assertEqual(result, expected)

    def test_actor_score(self):
        """
        Test MaScorer.actor_score
        :return:
        """
        wildcards = ["Unspecified", "Wildcard"]
        legits = ["Fred", "Ethel"]
        warn_value = "Fred"
        gsr_value = "Ethel"
        expected = 0
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)
        gsr_value = wildcards[0]
        expected = 1
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)
        gsr_value = wildcards[1]
        expected = 1
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)
        warn_value = wildcards[0]
        gsr_value = "Ethel"
        expected = 0
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)
        warn_value = "Fred"
        gsr_value = ["Ethel", "Fred"]
        expected = 1
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)
        warn_value = "Lucy"
        gsr_value = ["Ethel", "Fred"]
        expected = 0
        result = MaScorer.actor_score(warn_value, gsr_value, legits, wildcards)
        self.assertEqual(result, expected)

    def test_subtype_score(self):
        """
        Test MaScorer.event_subtype_score
        :return:
        """
        warn_value = "Force Posture"
        gsr_value = "Conflict"
        expected = 0
        result = MaScorer.event_subtype_score(warn_value, gsr_value)
        self.assertEqual(result, expected)
        gsr_value = "Force Posture"
        expected = 1
        result = MaScorer.event_subtype_score(warn_value, gsr_value)
        self.assertEqual(result, expected)
        warn_value = "Sharpening Swords"
        expected = 0
        result = MaScorer.event_subtype_score(warn_value, gsr_value)
        self.assertEqual(result, expected)

    def test_score(self):
        """
        Tests MaScorer.score method
        :return:
        """

        test_warn_filename = "test_lb_warnings.json"
        test_warn_path = os.path.join(LB_MA_TEST_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings = json.load(f)
        test_gsr_filename = "test_lb_gsr.json"
        test_gsr_path = os.path.join(LB_MA_TEST_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        result = MaScorer.score(test_warnings, test_gsr)
        expected_filename = "match_results.json"
        path_ = os.path.join(LB_MA_TEST_PATH, expected_filename)
        with open(path_, "r", encoding="utf8") as f:
            expected = json.load(f)
        expected_matches = sorted(set([(m["Warning"], m["Event"]) for m in expected["Matches"]]))
        expected_qs_ser = expected["Details"]["Quality Scores"]
        expected_qs_mean = expected["Quality Score"]
        expected_precision = expected["Precision"]
        expected_recall = expected["Recall"]
        expected_f1 = expected["F1"]
        expected_merc_score = expected["Mercury Score"]
        self.assertEqual(sorted(set(result["Matches"])), expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Mercury Score"], expected_merc_score, 3)
        self.assertAlmostEqual(result["Precision"], expected_precision, 3)
        self.assertAlmostEqual(result["Recall"], expected_recall, 3)
        self.assertAlmostEqual(result["F1"], expected_f1, 3)
        for i, qs in enumerate(expected_qs_ser):
            res_qs = result["Details"]["Quality Scores"][i]
            self.assertAlmostEqual(res_qs, qs, 3)

    def test_match(self):
        """
        Tests MaScorer.match
        :return:
        """
        # Simple Matrix, 3 by 4
        test_matrix_filename = "test_qs_matrix_1.csv"
        path_ = os.path.join(TEST_RESOURCE_PATH, test_matrix_filename)
        test_mat = pd.read_csv(path_, index_col=0)
        expected_matches = [("warn_0", "evt_0"), ("warn_1", "evt_1"), ("warn_2", "evt_3")]
        expected_qs_ser = [4, 3.4, 3.2]
        expected_qs_mean = np.mean(expected_qs_ser)
        result = MaScorer.match(input_matrix=test_mat)
        self.assertEqual(result["Matches"], expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Precision"], 1.0)
        self.assertAlmostEqual(result["Recall"], 0.75)
        self.assertAlmostEqual(result["F1"], 1.5/1.75)
        self.assertAlmostEqual(result["Mercury Score"], expected_qs_mean/4.0 + 1.5/1.75)
        self.assertAlmostEqual(result["Details"]["Quality Scores"], expected_qs_ser, 3)
        # Simple matrix, 4 by 3
        test_matrix_filename = "test_qs_matrix_2.csv"
        path_ = os.path.join(TEST_RESOURCE_PATH, test_matrix_filename)
        test_mat = pd.read_csv(path_, index_col=0)
        expected_matches = [("warn_0", "evt_0"), ("warn_1", "evt_1"), ("warn_3", "evt_2")]
        expected_qs_ser = [4, 3.4, 3]
        expected_qs_mean = np.mean(expected_qs_ser)
        result = MaScorer.match(input_matrix=test_mat)
        self.assertEqual(result["Matches"], expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Precision"], 0.75)
        self.assertAlmostEqual(result["Recall"], 1.00)
        self.assertAlmostEqual(result["F1"], 1.5/1.75)
        self.assertAlmostEqual(result["Mercury Score"], expected_qs_mean/4.0 + 1.5/1.75)
        self.assertAlmostEqual(result["Details"]["Quality Scores"], expected_qs_ser, 3)
        # Null Matrix
        test_matrix_filename = "test_null_matrix.csv"
        path_ = os.path.join(TEST_RESOURCE_PATH, test_matrix_filename)
        test_mat = pd.read_csv(path_, index_col=0)
        expected_matches = []
        expected_qs_ser = []
        expected_qs_mean = 0
        result = MaScorer.match(input_matrix=test_mat)
        self.assertEqual(result["Matches"], expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Precision"], 0)
        self.assertAlmostEqual(result["Recall"], 0)
        self.assertAlmostEqual(result["F1"], 0)
        self.assertAlmostEqual(result["Mercury Score"], 0)
        self.assertAlmostEqual(result["Details"]["Quality Scores"], expected_qs_ser, 3)
        # Matrix with negative entries
        test_matrix_filename = "test_neg_matrix.csv"
        path_ = os.path.join(TEST_RESOURCE_PATH, test_matrix_filename)
        test_mat = pd.read_csv(path_, index_col=0)
        expected_matches = [("warn_0", "evt_0"), ("warn_2", "evt_2")]
        expected_qs_ser = [3, 4]
        expected_qs_mean = 3.5
        result = MaScorer.match(input_matrix=test_mat)
        self.assertEqual(result["Matches"], expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Precision"], 0.5)
        self.assertAlmostEqual(result["Recall"], 0.667, 3)
        self.assertAlmostEqual(result["F1"], 0.667/1.167, 3)
        self.assertAlmostEqual(result["Mercury Score"], expected_qs_mean/4.0 + 0.667/1.167, 3)
        self.assertAlmostEqual(result["Details"]["Quality Scores"], expected_qs_ser, 3)

        # Matrix with Lebanon data
        test_matrix_filename = "test_qs_mat.csv"
        path_ = os.path.join(LB_MA_TEST_PATH, test_matrix_filename)
        test_mat = pd.read_csv(path_, index_col=0)
        expected_filename = "match_results.json"
        path_ = os.path.join(LB_MA_TEST_PATH, expected_filename)
        with open(path_, "r", encoding="utf8") as f:
            expected = json.load(f)
        expected_matches = sorted(set([(m["Warning"], m["Event"]) for m in expected["Matches"]]))
        expected_qs_ser = expected["Details"]["Quality Scores"]
        expected_qs_mean = expected["Quality Score"]
        expected_precision = expected["Precision"]
        expected_recall = expected["Recall"]
        expected_f1 = expected["F1"]
        result = MaScorer.match(input_matrix=test_mat)
        self.assertEqual(sorted(set(result["Matches"])), expected_matches)
        self.assertAlmostEqual(result["Quality Score"], expected_qs_mean, 3)
        self.assertAlmostEqual(result["Precision"], expected_precision, 3)
        self.assertAlmostEqual(result["Recall"], expected_recall, 3)
        self.assertAlmostEqual(result["F1"], expected_f1, 3)
        self.assertAlmostEqual(result["Mercury Score"], expected["Quality Score"]/4.0 + expected["F1"], 3)
        for i, qs in enumerate(expected_qs_ser):
            res_qs = result["Details"]["Quality Scores"][i]
            self.assertAlmostEqual(res_qs, qs, 3)

    def test_score_one(self):
        """
        Test MaScorer.score_one weights input
        :return:
        """
        test_warn_filename = "ma_test_warnings.json"
        test_warn_path = os.path.join(TEST_RESOURCE_PATH, test_warn_filename)
        with open(test_warn_path, "r", encoding="utf8") as f:
            test_warnings= json.load(f)
        test_gsr_filename = "ma_test_gsr.json"
        test_gsr_path = os.path.join(TEST_RESOURCE_PATH, test_gsr_filename)
        with open(test_gsr_path, "r", encoding="utf8") as f:
            test_gsr = json.load(f)

        #print(test_gsr[0])

        LEGIT_ACTORS = ["Egyptian Police"]

        bad_weight = -1
        sub_weight = .5
        super_weight = 2

        # Test with default weights


        # Test with bad weights
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ls_weight=bad_weight)
        self.assertTrue("Errors" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ds_weight=bad_weight)
        self.assertTrue("Errors" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    as_weight=bad_weight)
        self.assertTrue("Errors" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ess_weight=bad_weight)
        self.assertTrue("Errors" in result)

        # Test with weights summing to less than 4
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ls_weight=sub_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ds_weight=sub_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    as_weight=sub_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ess_weight=sub_weight)
        self.assertTrue("Notices" in result)

        # Test with weights summing to more than 4
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ls_weight=super_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ds_weight=super_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    as_weight=super_weight)
        self.assertTrue("Notices" in result)
        result = MaScorer.score_one(test_warnings[0], test_gsr[0], legit_actors=LEGIT_ACTORS,
                                    ess_weight=super_weight)
        self.assertTrue("Notices" in result)

        print("Result using default weights")
        result = MaScorer.score_one(test_warnings[3], test_gsr[0])
        print(result)

        # Test a warning with LS = 0
        result = MaScorer.score_one(test_warnings[1], test_gsr[0])
        self.assertEqual(result[ScoreComponents.QS], 0)
        self.assertEqual(result[ScoreComponents.LS], 0)
        self.assertAlmostEqual(result[ScoreComponents.DS], 0.75)

        # Test a warning with DS = 0
        result = MaScorer.score_one(test_warnings[2], test_gsr[0])
        self.assertEqual(result[ScoreComponents.QS], 0)
        self.assertAlmostEqual(result[ScoreComponents.LS], 0.778, 3)
        self.assertEqual(result[ScoreComponents.DS], 0)

        # Test a legitimately matched warning
        result = MaScorer.score_one(test_warnings[3], test_gsr[0], legit_actors=LEGIT_ACTORS)
        self.assertEqual(result[ScoreComponents.AS], 1)
        self.assertEqual(result[ScoreComponents.ESS], 1)
        self.assertAlmostEqual(result[ScoreComponents.LS], 0.778, 3)
        self.assertAlmostEqual(result[ScoreComponents.DS], 0.75)
        self.assertAlmostEqual(result[ScoreComponents.QS], 3.528, 3)
        self.assertFalse("Notices" in result)
        self.assertFalse("Errors" in result)


if __name__ == "__main__":
    unittest.main()
