#!/usr/bin/env python
import os
import subprocess

LOAD_WARN_CMD = ["curl", "-H", "Content-Type: application/json",
                 "-XPOST", "http://localhost:8029/warning/dev-intake", "--data-binary"]

ma_countries = [c for c in os.listdir(".") if c != ".DS_Store" and not c.endswith("py")]
print(ma_countries)
for cc in ma_countries:
    cc_path = os.path.join(".", cc)
    warn_files = [wf for wf in os.listdir(cc_path) if wf.endswith("json")]
    for wf in warn_files:
        load_warn_cmd = LOAD_WARN_CMD.copy()
        path_ = os.path.join(cc_path, wf)
        load_warn_cmd.append("@{}".format(path_))
        print(load_warn_cmd)
        proc = subprocess.Popen(load_warn_cmd, stdout=subprocess.PIPE)
        out, errs = proc.communicate()
        if errs is not None:
            print(errs)

