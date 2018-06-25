import unittest
import sys
sys.path.append("..")
from main.express_score import CaseCountScorer
from main.schema import (
    JSONField,
    ScoreComponents,
    LocationName,
    EventType,
    DiseaseType
)
import json
import os

EXPRESS_SCORE_HOME = os.path.abspath("..")
RESOURCE_PATH = os.path.join(EXPRESS_SCORE_HOME, "resources")
TEST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "test")


class CaseCountScorerTest(unittest.TestCase):

    scorer = CaseCountScorer(event_type=JSONField.DISEASE, location=LocationName.SAUDI_ARABIA)
    warn_dict = dict()
    warn_dict[JSONField.WARNING_ID] = "test_1"
    warn_dict[JSONField.EVENT_TYPE] = EventType.DIS
    warn_dict[JSONField.DISEASE] = DiseaseType.MERS
    warn_dict[JSONField.COUNTRY] = LocationName.SAUDI_ARABIA
    warn_dict[JSONField.EVENT_DATE] = "2016-03-27"
    warn_dict[JSONField.TIMESTAMP] = "20160324T00:01:01"
    gsr_dict = dict()
    gsr_dict[JSONField.EVENT_TYPE] = EventType.DIS
    gsr_dict[JSONField.EVENT_ID] = "Disease_Saudi_Arabia_MERS_2016-03-27"
    gsr_dict[JSONField.DISEASE] =  DiseaseType.MERS
    gsr_dict[JSONField.COUNTRY] = LocationName.SAUDI_ARABIA
    gsr_dict[JSONField.EVENT_DATE] = "2016-03-27"
    gsr_dict[JSONField.EARLIEST_REPORTED_DATE] = "2016-04-01"
    result_dict = dict()
    result_dict[JSONField.WARNING_ID] = "test_1"
    result_dict[JSONField.EVENT_ID] = "Disease_Saudi_Arabia_MERS_2016-03-27"
    
    # Load CU Count Warnings
    # Egypt
    warn_filename = "test_egypt_daily_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        eg_daily_warn = json.load(f)
    # Tahrir
    warn_filename = "test_tahrir_weekly_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        tahrir_weekly_warn = json.load(f)
    # Jordan
    warn_filename = "test_jordan_weekly_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        jo_weekly_warn = json.load(f)
    # Amman
    warn_filename = "test_amman_monthly_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        amman_monthly_warn = json.load(f)
    # Irbid
    warn_filename = "test_irbid_monthly_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        irbid_monthly_warn = json.load(f)
    # Madaba
    warn_filename = "test_madaba_monthly_cu_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        madaba_monthly_warn = json.load(f)

    # Load CU Count GSR
    # Egypt
    gsr_filename = "test_egypt_daily_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        eg_daily_gsr = json.load(f)
    # Tahrir
    gsr_filename = "test_tahrir_weekly_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        tahrir_weekly_gsr = json.load(f)
    # Jordan
    gsr_filename = "test_jordan_weekly_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        jo_weekly_gsr = json.load(f)
    # Amman
    gsr_filename = "test_amman_monthly_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        amman_monthly_gsr = json.load(f)
    # Irbid
    gsr_filename = "test_irbid_monthly_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        irbid_monthly_gsr = json.load(f)
    # Madaba
    gsr_filename = "test_madaba_monthly_cu_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        madaba_monthly_gsr = json.load(f)

    def test_qs(self):

        # Test when both are 0
        value = CaseCountScorer.quality_score(0, 0)
        self.assertEqual(1.0, value)
        value = self.scorer.quality_score(10, 10)
        self.assertEqual(1.0, value)
        value = self.scorer.quality_score(1, 0)
        self.assertEqual(0.75, value)
        value = self.scorer.quality_score(0,3)
        self.assertEqual(0.25, value)
        value = self.scorer.quality_score(9, 10)
        self.assertAlmostEqual(0.9, value)
        value = self.scorer.quality_score(1, 10)
        self.assertAlmostEqual(0.1, value)

    def test_score_one(self):
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
        """
        Tests the CaseCountScorer match function
        :return:
        """
        scorer = CaseCountScorer(location=LocationName.EGYPT, event_type=EventType.CU)
        warn_ = self.eg_daily_warn[-4:]
        gsr_ = self.eg_daily_gsr[-5:]
        match_list = [("test_Egypt-05-29", "CU_Count_Egypt_2018-05-29")]
        match_list.append(("test_Egypt-05-30", "CU_Count_Egypt_2018-05-30"))
        match_list.append(("test_Egypt-05-31", "CU_Count_Egypt_2018-05-31"))
        unmatched_warn_list = ["test_Egypt-06-01"]
        unmatched_gsr_list = ["CU_Count_Egypt_2018-05-27","CU_Count_Egypt_2018-05-28"]
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))

    def test_score(self):
        """
        Tests CaseCountScorer.score method
        :return:
        """
        scorer = CaseCountScorer(location=LocationName.EGYPT, event_type=EventType.CU)
        warn_ = self.eg_daily_warn[-4:]
        gsr_ = self.eg_daily_gsr[-5:]
        expected_precision = 0.75
        expected_recall = 0.60
        expected_qs = 0.25
        expected_qs_ser = [0, 0.25, 0.50]
        result = scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs)
        self.assertEqual(details["QS Values"], expected_qs_ser)

    
    def test_fill_out_location(self):
        """
        Tests CaseCountScorer.fill_out_location method
        :return:
        """
        test_loc = "Freedonia"
        test_evt_type = EventType.DIS
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.DIS}
        self.assertEqual(result, expected)
        test_loc = LocationName.SAUDI_ARABIA
        test_evt_type = EventType.DIS
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.DIS,
                    JSONField.COUNTRY: LocationName.SAUDI_ARABIA}
        self.assertEqual(result, expected)
        test_loc = LocationName.EGYPT
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.EGYPT}
        self.assertEqual(result, expected)
        test_loc = LocationName.TAHRIR
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.EGYPT,
                    JSONField.CITY: LocationName.TAHRIR}
        self.assertEqual(result, expected)
        test_loc = LocationName.JORDAN
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.JORDAN}
        self.assertEqual(result, expected)
        test_loc = LocationName.AMMAN
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.JORDAN,
                    JSONField.STATE: LocationName.AMMAN}
        self.assertEqual(result, expected)
        test_loc = LocationName.IRBID
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.JORDAN,
                    JSONField.STATE: LocationName.IRBID}
        self.assertEqual(result, expected)
        test_loc = LocationName.MADABA
        test_evt_type = EventType.CU
        result = CaseCountScorer.fill_out_location(event_type=test_evt_type, location=test_loc)
        expected = {JSONField.EVENT_TYPE: EventType.CU,
                    JSONField.COUNTRY: LocationName.JORDAN,
                    JSONField.STATE: LocationName.MADABA}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
