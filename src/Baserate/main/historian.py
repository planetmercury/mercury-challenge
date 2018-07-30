import os
import json
import pandas as pd

MC_HOME = os.path.join("..", "..", "..")
DATA_PATH = os.path.join(MC_HOME, "data", "gsr", "ma_gsr")


class Historian():

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
        for filename_ in os.listdir(gsr_path):
            path_ = os.path.join(gsr_path, filename_)
            with open(path_, "r", encoding="utf8") as f:
                out_gsr += json.load(f)

        return out_gsr
