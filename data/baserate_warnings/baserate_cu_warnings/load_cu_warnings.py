#!/usr/bin/env python
import os
import subprocess

LOAD_WARN_CMD = ["curl", "-H", "Content-Type: application/json",
                 "-XPOST", "http://localhost:8029/warning/dev-intake", "--data-binary"]

places = [c for c in os.listdir(".") if c != ".DS_Store" and not c.endswith("py")]
print(places)
for place in places:
    place_path = os.path.join(".", place)
    warn_files = [wf for wf in os.listdir(place_path) if wf.endswith("json")]
    for wf in warn_files:
        load_warn_cmd = LOAD_WARN_CMD.copy()
        path_ = os.path.join(place_path, wf)
        load_warn_cmd.append("@{}".format(path_))
        #print(load_warn_cmd)
        proc = subprocess.Popen(load_warn_cmd, stdout=subprocess.PIPE)
        out, errs = proc.communicate()
        if "rror" in out.__str__():
            print (wf, out)
        if errs is not None:
            print(errs)

