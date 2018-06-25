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

    def test_ds(self):

        # Test when both are 0
        warn_date = "2018-06-22"
        gsr_date_range = pd.date_range("2018-06-17", "2018-06-27")
        gsr_dates = [d.strftime("%Y-%m-%d") for d in gsr_date_range]
        expected_values = [0, 0] + [(1- np.abs(i)/Defaults.MAX_DATE_DIFF) for i in range(-3, 4)] + [0, 0]
        for i, d in enumerate(gsr_dates):
            result = MaScorer.date_score(warn_date, d)
            expected = expected_values[i]
            self.assertAlmostEqual(result, expected)

    def test_ls(self):
        lat1, long1 = (30.0, 30.0)
        lat2, long2 = (30.0, 30.0)
        result = MaScorer.location_score(lat1, long1, lat2, long2)
        expected = 1.0
        self.assertAlmostEqual(result, expected)
        # 22 km distance
        lat2 = 30.2
        result = MaScorer.location_score(lat1, long1, lat2, long2)
        expected = 0.78
        self.assertAlmostEqual(result, expected, 2)
        result = MaScorer.location_score(lat1, long1, lat2, long2, is_approximate=True)
        expected = 0.934
        self.assertAlmostEqual(result, expected, 3)
        result = MaScorer.location_score(lat1, long1, lat2, long2, max_dist=44.2)
        expected = 0.50
        self.assertAlmostEqual(result, expected, 2)
        lat2 = 31.0
        result = MaScorer.location_score(lat1, long1, lat2, long2)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

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



    def test_match(self):
        pass

    def test_score_one_weights(self):
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
