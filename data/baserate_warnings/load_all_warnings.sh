#!/usr/bin/env bash
cd baserate_cu_warnings
python load_cu_warnings.py
cd ../baserate_disease_warnings
python load_dis_warnings.py
cd ../baserate_ma_warnings
python load_ma_warnings.py
cd ..