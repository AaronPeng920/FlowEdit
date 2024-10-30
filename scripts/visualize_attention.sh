#!/bin/bash

cd ..
python attention_processor.py \
    --image_resolution 512 \
    --use_t5 \
    --mode 'all' \
    --visualize_now \
    --prompt 'a photo of a cat and a dog' \
    --num_inference_steps 25 \
    --index 5 \
    --filter_ids 23 