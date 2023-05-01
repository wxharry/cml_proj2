#!/bin/sh

current_dir=$(cd $(dirname $0); pwd)
echo $current_dir

pip install -r $current_dir/requirements.txt
python $current_dir/main.py --num-workers 10