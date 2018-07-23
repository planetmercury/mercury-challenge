import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (plot_acf, plot_pacf)
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import datetime
from dateutil.parser import parse
import sys
sys.path.append(("../.."))
from Baserate.main.timeseries import (evaluate_arima_model, evaluate_models, mean_qs_error, TsDefaults)
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("fivethirtyeight")

MC_HOME = os.path.abspath("../../..")
DATA_PATH = os.path.join(MC_HOME, "data")
GSR_PATH = os.path.join(DATA_PATH, "gsr")
CU_COUNT_GSR_PATH = os.path.join(GSR_PATH, "cu_count_gsr")
DISEASE_GSR_PATH = os.path.join(GSR_PATH, "disease_gsr")

P_MAX = TsDefaults.P_MAX
D_MAX = TsDefaults.D_MAX
Q_MAX = TsDefaults.Q_MAX

filename_ = "Egypt_Daily_Counts.json"
filepath_ = os.path.join(CU_COUNT_GSR_PATH, filename_)
with open(filepath_, "r", encoding="utf8") as f:
    gsr_ = json.load(f)
count_dict = {parse(e["Event_Date"]): e["Case_Count"] for e in gsr_}
count_ser = pd.Series(count_dict)

evaluate_models(count_ser, range(1,P_MAX+1), range(1,D_MAX+1), range(2,Q_MAX+1), verbose=True)