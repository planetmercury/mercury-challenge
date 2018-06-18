'''
Script to run and score multiple baserates for a combination of country and event type
'''

import subprocess
import os
import sys
import json
import re

import pandas as pd
import numpy as np

from dateutil.parser import parse
import datetime



class Defaults():
    """
    Placeholder for default values
    """
    NUM_ITER = 100
    SCORE_START_DAY = 22
    SCORE_END_DAY = 21
    BUFFER_START_DAY = 15
    BUFFER_END_DAY = 28
    MERC_HOME = os.path.join("..", "..", "..", "..", "..", "..")
    DATA_HOME = os.path.join(MERC_HOME, "data")
    RESULTS_HOME = os.path.join(DATA_HOME, "scoring_results")
    WARN_HOME = os.path.join(DATA_HOME, "baserate_warnings", "Multiple_Baserates")
    PERF_PREFIX = "BR"

    SCORE_CMD = ["curl", "-k","-H", "Content-Type: application/json", "-XPOST", "https://localhost:8053/score", "-d"]

    CLEAR_WARN_CMD = ["curl", "-k","https://localhost:8029/warning/remove_all/BR_ITER"]

    LOAD_WARN_CMD = ["curl", "-k","-H", "Content-Type: application/json",
                     "-XPOST", "https://localhost:8029/warning/dev-intake", "--data-binary"]

    FILENAME_TEMPLATE = "MONTHSTR/Baserate_EVTTYPE_ITER_MONTHSTR.json"


    NSA_COUNTRIES = ["Bahrain", "Egypt", "Jordan", "Lebanon", "Qatar", "Saudi Arabia"]
    MA_EXTRAS = ["Palestine", "Turkey", "Yemen"]
    ma_countries = NSA_COUNTRIES + MA_EXTRAS
    IQ_SY = ["Iraq", "Syria"]

    score_dict_template = {"Include Matching": "false"}
    TOP_METRICS = ["Quality Score", "Lead Time", "Precision", "Recall"]

    TM_XLIM_DICT = {"Quality Score": (2.0, 4.0), "Precision": (0.0, 1.0), "Recall": (0.0, 1.0)}

    ALPHA = 0.05

    EVT_ABBR_DICT = {"Civil Unrest": "CU", "Military Action": "MA",
                     "Non-State Actor": "NSA"}


def clear_warnings(n_iter):
    """
    Clears warnings with performer ids <PERF>_<ITER> from Elasticsearch
    :param n_iter: How many to clear out
    :return: None
    """
    for i in range(n_iter):
        clear_cmd = Defaults.CLEAR_WARN_CMD.copy()
        clear_cmd[-1] = re.sub("ITER", str(i), clear_cmd[-1])
        # print(clear_cmd)
        # Issue the clear command
        proc = subprocess.Popen(clear_cmd, stdout=subprocess.PIPE)
        clearout, clearerrs = proc.communicate()
        if clearerrs is not None:
            print("Errors while clearing warnings:")
            print(clearerrs)


def load_warnings(iter_i, evt_prefix, last_month_str, this_month_str):
    """
    Loads the warnings for iteration iter_i
    :param iter_i: which iteration to load
    :param evt_prefix: Short name for event type
    :param last_month_str: string name for last month, e.g. "April_2017"
    :param this_month_str: string name for this month, e.g. "May_2017"
    :return:
    """

    for month_str in [last_month_str, this_month_str]:
        filepath = re.sub("EVTTYPE", evt_prefix, Defaults.FILENAME_TEMPLATE)
        filepath = re.sub("ITER", str(iter_i), filepath)
        filepath = re.sub("MONTHSTR", month_str, filepath)
        filepath = os.path.join(Defaults.WARN_HOME, filepath)
        filepath = os.path.abspath(filepath)
        jsonfilepath = "@" + filepath
        #print(filepath)
        load_warn_cmd = Defaults.LOAD_WARN_CMD.copy()
        load_warn_cmd.append(jsonfilepath)
        #print(load_warn_cmd)
        proc = subprocess.Popen(load_warn_cmd, stdout=subprocess.PIPE)
        out, errs = proc.communicate()
        #print(out)
        if errs is not None:
            print(iter_i, errs)



def score_iteration(score_dict_shell, perf_name, iter_i):
    """
    Scores an iteration of the simulation
    :return:
    """
    score_dict = score_dict_shell.copy()
    score_dict["Performer ID"] = "{0}_{1}".format(perf_name, iter_i)
    score_params = json.dumps(score_dict)
    score_cmd = Defaults.SCORE_CMD.copy()
    score_cmd.append(score_params)
    #score_cmd.append(">")
    #score_cmd.append("fred.json")
    #print(score_cmd)
    #output_filename = "../../scoring_results/{2}/Multiple_Baserates/results_MA_{0}_BR_{1}.json".format(cc,
                                                                                                       #i,
                                                                                                       #this_month_str)
#        score_cmd = re.sub("OUTPUT_FILENAME", output_filename, score_cmd)
    #print(score_cmd)

    proc = subprocess.Popen(score_cmd, stdout=subprocess.PIPE)
    scoreout, scoreerrs = proc.communicate()
    if scoreerrs is not None:
        print(iter_i, country, scoreerrs)
    scoring = json.loads(scoreout.decode("utf-8"))["Scoring"]["Results"]

    return scoring


def scoring_date_range(month_str,
                       score_start_day=Defaults.SCORE_START_DAY,
                       score_end_day=Defaults.SCORE_END_DAY):
    """
    Returns date range for scoring
    :param month_str:
    :param score_start_day:
    :param score_end_day:
    :return:
    """
    month_date = parse(month_str).replace(day=1)
    last_month_date = month_date - datetime.timedelta(1)
    scoring_start_date = str(last_month_date.replace(day=score_start_day).date())
    scoring_end_date = str(month_date.replace(day=score_end_day).date())
    return (scoring_start_date, scoring_end_date)



def main(country, event_type, last_month_str, score_month_str, num_iter=Defaults.NUM_ITER,
         perf_name=Defaults.PERF_PREFIX):
    """
    Runs num_iter simulations of country/event_type baserate and scores them
    :param country: Which country to be scored
    :param event_type: What event type
    :param last_month_str: The month before the scoring month
    :param score_month_str: Underscore joined name of the month being scored, e.g. "May_2017"
    :param num_iter: How many simulations to run?  Default is DEFAULT_NUM_ITER
    :param perf_name: Name of the performer, usually "BR"
    :return: None; writes output to a file
    """
    evt_prefix = Defaults.EVT_ABBR_DICT[event_type]
    score_month = " ".join(score_month_str.split("_"))
    score_start, score_end = scoring_date_range(score_month)
    score_dict_shell = Defaults.score_dict_template.copy()
    score_dict_shell["Country"] = country
    score_dict_shell["Event Type"] = event_type
    score_dict_shell["Start Date"] = score_start
    score_dict_shell["End Date"] = score_end

    clear_warnings(num_iter)
    result_dict = dict()
    for i in range(num_iter):
        if num_iter >= 20:
            if i%(num_iter//20) == 0:
                print("Iteration {}".format(i))
        load_warnings(i, evt_prefix, last_month_str, score_month_str)
        _scoring = score_iteration(score_dict_shell, perf_name, i)
        _key = "{0}_{1}".format(country, i)
        result_dict[_key] = _scoring
    out_filename = "{0} {1} {2} Multiple Baserate Results {3} Iterations.json".format(country,
                                                                                      evt_prefix,
                                                                                      score_month_str,
                                                                                      num_iter)
    out_path = os.path.join(Defaults.RESULTS_HOME, score_month_str, "Multiple_Baserates", out_filename)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    country, event_type, last_month_str, score_month_str = sys.argv[1:5]
    event_type = " ".join(event_type.split("_"))
    if len(sys.argv) > 5:
        num_iter = int(sys.argv[5])
    else:
        num_iter = Defaults.NUM_ITER
    main(country, event_type, last_month_str, score_month_str, num_iter)
