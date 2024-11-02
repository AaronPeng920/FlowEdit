#!/bin/bash

cd ..
python run_pie_benchmark.py \
    --source_root "/data/pengzhengwei/datasets/PIE-Bench/PIE-Bench_v1" \
    --result_root "/data/pengzhengwei/datasets/PIE-Bench/flowedit_result" \
    --device "cuda:5" \
    --image_resolution 512 \
    --num_inference_steps 12 \
    --source_guidance_scale 2.0 \
    --target_guidance_scale 7.0 \
    --cross_start_step 0.0 \
    --cross_end_step 0.3 \
    --self_start_step 0.3 \
    --self_end_step 0.7 \
    --seed 319
