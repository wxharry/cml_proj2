#!/bin/sh

current_dir=$(pwd) 

pip install $current_dir/requirements.txt
python $current_dir/main.py
