#!/bin/bash

cd ..
python attention_processor.py \
    --image_resolution 512 \
    --mode 'select' \
    --visualize_now \
    --prompt 'a round cake with orange frosting on a wooden plate' \
    --num_inference_steps 10 \
    --seed 319 \
    --index 3 \
    --filter_ids -1 