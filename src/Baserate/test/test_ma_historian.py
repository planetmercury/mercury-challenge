import sys, os, json
sys.path.append("..")
from main.mahistorian import MaHistorian

MC_HOME = os.path.join("..", "..", "..")
DATA_PATH = os.path.join(MC_HOME, "data", "gsr", "ma_gsr")

h = MaHistorian("Egypt")

gsr = []
for filename_ in os.listdir(DATA_PATH):
    filepath_ = os.path.join(DATA_PATH, filename_)
    with open(filepath_, "r", encoding="utf8") as f:
        gsr_ = json.load(f)
        gsr += gsr_

start_date = "2018-06-01"
end_date = "2018-06-30"

eg_history = h.get_history(start_date, end_date, gsr)
len1 = len(eg_history)
print(len1)
#print(eg_history.tail())

eg_history = h.get_history_from_path(start_date, end_date, DATA_PATH)
len2 = len(eg_history)
print(len2)

print(len1 == len2)