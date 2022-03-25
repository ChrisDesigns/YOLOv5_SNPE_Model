#!/usr/bin/env bash

# in the current directory, download the data and prepare it for quantization
python3 scripts/setup_yolov5.py -a $PWD -d -r dsp