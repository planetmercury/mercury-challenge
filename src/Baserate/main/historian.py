import os
import json
import pandas as pd
import sys
sys.path.append("../..")
from ExpressScore.main.schema import JSONField as JF
from ExpressScore.main.schema import EventType

MC_HOME = os.path.join("..", "..", "..")
DATA_PATH = os.path.join(MC_HOME, "data", "gsr", "ma_gsr")


class Historian:

    def __init__(self, country):
        self.country = country

    def get_history(self, start_date, end_date, gsr):
        """
        Retrieves the history for the given event type
        :param start_date: First date for the history
        :param end_date: Last date for the history
        :param gsr: List of JSON formatted GSR events
        :return: DataFrame with country GSR from start_date to end_date
        """

        gsr_df = pd.DataFrame(gsr)
        gsr_df = gsr_df[gsr_df.Country == self.country]
        gsr_df = gsr_df[gsr_df.Event_Date <= end_date]
        gsr_df = gsr_df[gsr_df.Event_Date >= start_date]
        gsr_df.index = gsr_df.Event_ID
        gsr_df.drop("Event_ID", axis=1, inplace=True)

        return gsr_df

    def get_history_from_path(self, start_date, end_date, gsr_path=DATA_PATH):
        """
        Convenience method to retrieve GSR from a path
        :param start_date:
        :param end_date:
        :param gsr_path:
        :return:
        """
        gsr = self.load_gsr_from_path(gsr_path)
        return self.get_history(start_date, end_date, gsr)

    @staticmethod
    def load_gsr_from_path(gsr_path=DATA_PATH):
        """
        Loads GSR JSON from a path
        :param gsr_path: path to GSR files
        :return: List of JSON-formatted events
        """
        out_gsr = []
        gsr_filenames = [x for x in os.listdir(gsr_path) if x.endswith(".json")]
        for filename_ in gsr_filenames:
            try:
                path_ = os.path.join(gsr_path, filename_)
                with open(path_, "r", encoding="utf8") as f:
                    out_gsr += json.load(f)
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError with {}".format(filename_))

        return out_gsr

    def event_rate(self, gsr_df, start_date=None, end_date=None):
        """
        Determines the rate of events over the interval in the GSR
        :param gsr_df: DataFrame of GSR data
        :param start_date:  First date from which to compute history
        :param end_date: Last date from which to compute history
        :return:
        """
        df_ = gsr_df.copy()
        df_ = df_[df_.Country == self.country]
        if start_date is None:
            start_date = df_.Event_Date.min()
        if end_date is None:
            end_date = df_.Event_Date.max()
        df_ = df_[df_.Event_Date <= end_date]
        df_ = df_[df_.Event_Date >= start_date]
        event_count = len(df_)
        dr_ = pd.date_range(start_date, end_date)
        n_days = len(dr_)
        rate_ = 1.*event_count/n_days

        return rate_


class MaHistorian(Historian):

    def __init__(self, country):
        self.event_type = EventType.MA
        super().__init__(country)
