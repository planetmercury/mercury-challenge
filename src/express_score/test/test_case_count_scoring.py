import unittest
import sys
sys.path.append("..")
from main.express_score import CaseCountScorer
from main.schema import (
    JSONField,
    ScoreComponents
)


class CaseCountScorerTest(unittest.TestCase):

    scorer = CaseCountScorer(event_type=JSONField.DISEASE)
    warn_dict = dict()
    warn_dict[JSONField.WARNING_ID] = "test_1"
    warn_dict[JSONField.EVENT_TYPE] = "Disease"
    warn_dict[JSONField.DISEASE] = "MERS"
    warn_dict[JSONField.COUNTRY] = "Saudi Arabia"
    warn_dict[JSONField.EVENT_DATE] = "2016-03-27"
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


    def test_qs_both_zero(self):

        # Test when both are 0
        value = CaseCountScorer.quality_score(0, 0)
        self.assertEqual(1.0, value)

    def test_score_one_both_zero(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 0
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 0
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 1.0
        result_[JSONField.WARN_VALUE] = 0
        result_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_qs_same_nonzero(self):

        value = self.scorer.quality_score(10, 10)
        self.assertEqual(1.0, value)

    def test_score_one_same_nonzero(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 10
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 1.0
        result_[JSONField.WARN_VALUE] = 10
        result_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_qs_1_0(self):

        value = self.scorer.quality_score(1, 0)
        self.assertEqual(0.75, value)

    def test_score_one_1_0(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 1
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 0
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 0.75
        result_[JSONField.WARN_VALUE] = 1
        result_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_qs_3_0(self):

        value = self.scorer.quality_score(0,3)
        self.assertEqual(0.25, value)

    def test_score_one_3_0(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 3
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 0
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 0.25
        result_[JSONField.WARN_VALUE] = 3
        result_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_score_one_9_10(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 9
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 0.9
        result_[JSONField.WARN_VALUE] = 9
        result_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_qs_9_10(self):

        value = self.scorer.quality_score(9, 10)
        self.assertAlmostEqual(0.9, value)

    def test_qs_1_10(self):

        value = self.scorer.quality_score(1, 10)
        self.assertAlmostEqual(0.1, value)

    def test_score_one_1_10(self):
        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 1
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        value = self.scorer.score_one(warn_, event_)
        result_ = self.result_dict.copy()
        result_[ScoreComponents.QS] = 0.1
        result_[JSONField.WARN_VALUE] = 1
        result_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in result_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(value[k], result_[k])
                else:
                    self.assertEqual(value[k], result_[k])

    def test_match(self):
        pass


if __name__ == "__main__":
    unittest.main()
