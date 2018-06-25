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
import numpy as np

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
    expected_dict = dict()
    expected_dict[JSONField.WARNING_ID] = "test_1"
    expected_dict[JSONField.EVENT_ID] = "Disease_Saudi_Arabia_MERS_2016-03-27"
    
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

    # Load Disease Warnings
    warn_filename = "dis_test_warnings.json"
    warn_path = os.path.join(TEST_RESOURCE_PATH, warn_filename)
    with open(warn_path, "r") as f:
        mers_warn = json.load(f)
    # Disease GSR
    gsr_filename = "dis_test_gsr.json"
    gsr_path = os.path.join(TEST_RESOURCE_PATH, gsr_filename)
    with open(gsr_path, "r") as f:
        mers_gsr = json.load(f)


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
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 1.0
        expected_[JSONField.WARN_VALUE] = 0
        expected_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 10
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 1.0
        expected_[JSONField.WARN_VALUE] = 10
        expected_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 1
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 0
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 0.75
        expected_[JSONField.WARN_VALUE] = 1
        expected_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 3
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 0
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 0.25
        expected_[JSONField.WARN_VALUE] = 3
        expected_[JSONField.EVENT_VALUE] = 0
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 9
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 0.9
        expected_[JSONField.WARN_VALUE] = 9
        expected_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

        warn_ = self.warn_dict.copy()
        warn_["Case_Count"] = 1
        event_ = self.gsr_dict.copy()
        event_["Case_Count"] = 10
        result_ = self.scorer.score_one(warn_, event_)
        expected_ = self.expected_dict.copy()
        expected_[ScoreComponents.QS] = 0.1
        expected_[JSONField.WARN_VALUE] = 1
        expected_[JSONField.EVENT_VALUE] = 10
        for k in warn_.keys():
            if k in expected_.keys():
                if k == ScoreComponents.QS:
                    self.assertAlmostEqual(result_[k], expected_[k])
                else:
                    self.assertEqual(result_[k], expected_[k])

    def test_match(self):
        """
        Tests the CaseCountScorer match function
        :return:
        """
        # Egypt Daily CU
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
        # Tahrir Weekly CU
        scorer = CaseCountScorer(location=LocationName.TAHRIR, event_type=EventType.CU)
        warn_ = self.tahrir_weekly_warn
        gsr_ = self.tahrir_weekly_gsr
        match_list = [("test_Tahrir-05-02", "CU_Count_Tahrir_2018-05-02")]
        match_list.append(("test_Tahrir-05-09", "CU_Count_Tahrir_2018-05-09"))
        match_list.append(("test_Tahrir-05-16", "CU_Count_Tahrir_2018-05-16"))
        match_list.append(("test_Tahrir-05-23", "CU_Count_Tahrir_2018-05-23"))
        match_list.append(("test_Tahrir-05-30", "CU_Count_Tahrir_2018-05-30"))
        unmatched_warn_list = ["test_Tahrir-06-06", "test_Tahrir-06-13"]
        unmatched_gsr_list = ["CU_Count_Tahrir_2018-04-18", "CU_Count_Tahrir_2018-04-25"]
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))
        # Jordan Weekly CU
        scorer = CaseCountScorer(location=LocationName.JORDAN, event_type=EventType.CU)
        warn_ = self.jo_weekly_warn
        gsr_ = self.jo_weekly_gsr
        match_list = [("test_Jordan-05-02", "CU_Count_Jordan_2018-05-02")]
        match_list.append(("test_Jordan-05-09", "CU_Count_Jordan_2018-05-09"))
        match_list.append(("test_Jordan-05-16", "CU_Count_Jordan_2018-05-16"))
        match_list.append(("test_Jordan-05-23", "CU_Count_Jordan_2018-05-23"))
        match_list.append(("test_Jordan-05-30", "CU_Count_Jordan_2018-05-30"))
        unmatched_warn_list = ["test_Jordan-06-06"]
        unmatched_gsr_list = []
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))
        # Amman Monthly CU
        scorer = CaseCountScorer(location=LocationName.AMMAN, event_type=EventType.CU)
        warn_ = self.amman_monthly_warn
        gsr_ = self.amman_monthly_gsr
        match_list = [("test_Amman-01-01", "CU_Count_Amman_2018-01-01")]
        match_list.append(("test_Amman-02-01", "CU_Count_Amman_2018-02-01"))
        match_list.append(("test_Amman-03-01", "CU_Count_Amman_2018-03-01"))
        match_list.append(("test_Amman-04-01", "CU_Count_Amman_2018-04-01"))
        match_list.append(("test_Amman-05-01", "CU_Count_Amman_2018-05-01"))
        unmatched_warn_list = ["test_Amman-06-01"]
        unmatched_gsr_list = ["CU_Count_Amman_2017-12-01"]
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))
        # Irbid Monthly CU
        scorer = CaseCountScorer(location=LocationName.IRBID, event_type=EventType.CU)
        warn_ = self.irbid_monthly_warn
        gsr_ = self.irbid_monthly_gsr
        match_list = [("test_Irbid-01-01", "CU_Count_Irbid_2018-01-01")]
        match_list.append(("test_Irbid-02-01", "CU_Count_Irbid_2018-02-01"))
        match_list.append(("test_Irbid-03-01", "CU_Count_Irbid_2018-03-01"))
        match_list.append(("test_Irbid-04-01", "CU_Count_Irbid_2018-04-01"))
        match_list.append(("test_Irbid-05-01", "CU_Count_Irbid_2018-05-01"))
        unmatched_warn_list = ["test_Irbid-06-01"]
        unmatched_gsr_list = ["CU_Count_Irbid_2017-08-01", "CU_Count_Irbid_2017-09-01",
                              "CU_Count_Irbid_2017-10-01", "CU_Count_Irbid_2017-11-01",
                              "CU_Count_Irbid_2017-12-01"]
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))
        # Madaba Monthly CU
        scorer = CaseCountScorer(location=LocationName.MADABA, event_type=EventType.CU)
        warn_ = self.madaba_monthly_warn
        gsr_ = self.madaba_monthly_gsr
        match_list = [("test_Madaba_2017-09-01", "CU_Count_Madaba_2017-09-01")]
        match_list.append(("test_Madaba_2017-10-01", "CU_Count_Madaba_2017-10-01"))
        match_list.append(("test_Madaba_2017-11-01", "CU_Count_Madaba_2017-11-01"))
        match_list.append(("test_Madaba_2017-12-01", "CU_Count_Madaba_2017-12-01"))
        match_list.append(("test_Madaba-01-01", "CU_Count_Madaba_2018-01-01"))
        match_list.append(("test_Madaba-02-01", "CU_Count_Madaba_2018-02-01"))
        match_list.append(("test_Madaba-03-01", "CU_Count_Madaba_2018-03-01"))
        match_list.append(("test_Madaba-04-01", "CU_Count_Madaba_2018-04-01"))
        match_list.append(("test_Madaba-05-01", "CU_Count_Madaba_2018-05-01"))
        unmatched_warn_list = ["test_Madaba-06-01"]
        unmatched_gsr_list = []
        result = scorer.match(warn_data=warn_, gsr_data=gsr_)
        # Unmatched warnings
        self.assertEqual(unmatched_warn_list, result["Unmatched Warnings"])
        self.assertEqual(unmatched_gsr_list, result["Unmatched GSR"])
        self.assertEqual(set(match_list), set(result["Matches"]))

        # MERS
        scorer = CaseCountScorer(location=LocationName.SAUDI_ARABIA, event_type=EventType.DIS)
        warn_ = self.mers_warn
        gsr_ = self.mers_gsr
        match_list = [("test_2018-04-29", "Disease_Saudi_Arabia_MERS_2018-04-29")]
        match_list.append(("test_2018-05-06", "Disease_Saudi_Arabia_MERS_2018-05-06"))
        match_list.append(("test_2018-05-13", "Disease_Saudi_Arabia_MERS_2018-05-13"))
        match_list.append(("test_2018-05-20", "Disease_Saudi_Arabia_MERS_2018-05-20"))
        unmatched_warn_list = ["test_2018-05-27", "test_2018-06-03"]
        unmatched_gsr_list = ["Disease_Saudi_Arabia_MERS_2018-04-22"]
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
        # Egypt Daily CU
        eg_scorer = CaseCountScorer(location=LocationName.EGYPT, event_type=EventType.CU)
        warn_ = self.eg_daily_warn[-4:]
        gsr_ = self.eg_daily_gsr[-5:]
        expected_precision = 0.75
        expected_recall = 0.60
        expected_qs = 0.25
        expected_qs_ser = [0, 0.25, 0.50]
        result = eg_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Tahrir Weekly CU
        tahrir_scorer = CaseCountScorer(location=LocationName.TAHRIR, event_type=EventType.CU)
        warn_ = self.tahrir_weekly_warn
        gsr_ = self.tahrir_weekly_gsr
        expected_precision = 5./7
        expected_recall = 5./7
        expected_qs = 0.935
        expected_qs_ser = [0.923, 1, 1, 0.75, 1]
        result = tahrir_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall, 3)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Jordan Weekly CU
        jo_scorer = CaseCountScorer(location=LocationName.JORDAN, event_type=EventType.CU)
        warn_ = self.jo_weekly_warn
        gsr_ = self.jo_weekly_gsr
        expected_precision = 5./6
        expected_recall = 1
        expected_qs = 1
        expected_qs_ser = [1, 1, 1, 1, 1]
        result = jo_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall, 3)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Amman Monthly CU
        amman_scorer = CaseCountScorer(location=LocationName.AMMAN, event_type=EventType.CU)
        warn_ = self.amman_monthly_warn
        gsr_ = self.amman_monthly_gsr
        expected_precision = 5./6
        expected_recall = 5./6
        expected_qs_ser = [1, 16./20, 18./21, 10./13, 18./20]
        expected_qs = np.mean(expected_qs_ser)
        result = amman_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Irbid Monthly CU
        irbid_scorer = CaseCountScorer(location=LocationName.IRBID, event_type=EventType.CU)
        warn_ = self.irbid_monthly_warn
        gsr_ = self.irbid_monthly_gsr
        expected_precision = 5./6
        expected_recall = 5./10
        expected_qs_ser = [0.5, 1, 0.5, 1, 0.45]
        expected_qs = np.mean(expected_qs_ser)
        result = irbid_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Madaba Monthly CU
        madaba_scorer = CaseCountScorer(location=LocationName.MADABA, event_type=EventType.CU)
        warn_ = self.madaba_monthly_warn
        gsr_ = self.madaba_monthly_gsr
        expected_precision = 9./10
        expected_recall = 1
        expected_qs_ser = [1,1,1,1,1, 0.8, 1, 0, 1]
        expected_qs = np.mean(expected_qs_ser)
        result = madaba_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Saudi Arabia MERS
        mers_scorer = CaseCountScorer(location=LocationName.SAUDI_ARABIA, event_type=EventType.DIS)
        warn_ = self.mers_warn
        gsr_ = self.mers_gsr
        expected_precision = 0.667
        expected_recall = 0.80
        expected_qs = 0.8125
        expected_qs_ser = [1, 0.25, 1, 1]
        result = mers_scorer.score(warn_, gsr_)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs, 3)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)

        # Mixed event types
        mixed_warn = self.mers_warn + self.eg_daily_warn[-4:]
        mixed_gsr = self.mers_gsr + self.eg_daily_gsr[-5:]
        expected_precision = 0.75
        expected_recall = 0.60
        expected_qs = 0.25
        expected_qs_ser = [0, 0.25, 0.50]
        result = eg_scorer.score(mixed_warn, mixed_gsr)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs)
        self.assertEqual(details["QS Values"], expected_qs_ser)
        expected_precision = 0.667
        expected_recall = 0.80
        expected_qs = 0.8125
        expected_qs_ser = [1, 0.25, 1, 1]
        result = mers_scorer.score(mixed_warn, mixed_gsr)
        metrics = result["Results"]
        details = result["Details"]
        self.assertAlmostEqual(metrics["Precision"], expected_precision, 3)
        self.assertAlmostEqual(metrics["Recall"], expected_recall)
        self.assertAlmostEqual(metrics[ScoreComponents.QS], expected_qs)
        for i, qs in enumerate(details["QS Values"]):
            self.assertAlmostEqual(qs, expected_qs_ser[i], 3)


    
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
