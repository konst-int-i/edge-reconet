#!/bin/bash

echo '##################### RUNNING MODEL #########################'
python3 edge_tpu_video_style/train.py --debug 1

echo '##################### QUANTISING MODEL #########################'
python3 edge_tpu_video_style/tpu_compile/compile.py

echo '##################### RUNNING TPU COMPILER #########################'
edgetpu_compiler saved_models/style_transfer.tflite

