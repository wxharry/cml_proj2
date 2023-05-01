#!/bin/sh

current_dir=$(cd $(dirname $0); pwd)
echo $current_dir

pip install $current_dir/requirements.txt
python $current_dir/main.py
