#!/usr/bin/env python
import os
import subprocess

LOAD_WARN_CMD = ["curl", "-H", "Content-Type: application/json",
                 "-XPOST", "http://localhost:8029/warning/dev-intake", "--data-binary"]

warn_files = [wf for wf in os.listdir(".") if wf.endswith("json")]
for wf in warn_files:
    load_warn_cmd = LOAD_WARN_CMD.copy()
    load_warn_cmd.append("@{}".format(wf))
    print(load_warn_cmd)
    proc = subprocess.Popen(load_warn_cmd, stdout=subprocess.PIPE)
    out, errs = proc.communicate()
    if errs is not None:
        print(errs)

