from metrics_calculator import MetricsCalculator
import numpy as np
import torch
from PIL import Image
import os
import csv
import argparse


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_prompt, tgt_prompt):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)


all_tgt_image_folders = {
    "ip2p": "afhq_result/ip2p",
    "flowedit": "afhq_result/flowedit",
    "di": "afhq_result/di",
    "infedit": "infedit",
    "pix2pix_zero": "pix2pix_zero/out",
    "masactrl": "afhq_result/masactrl",
    "cycled": "afhq_result/cycled"
}
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", nargs="+", type=str,
        default=["clip_similarity_target_image", "lpips", "psnr", "ssim"]
    )
    parser.add_argument(
        "--src_image_folder", type=str, default="datasets/afhq/afhq/val/cat"
    )
    parser.add_argument(
        "--tgt_methods", nargs="+", type=str,
        default=list(all_tgt_image_folders.keys())
    )
    parser.add_argument("--result_path", type=str, default="evaluation_pie_result.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    src_image_folder = args.src_image_folder
    tgt_methods = args.tgt_methods
    result_path = args.result_path
    device = args.device
    metrics = args.metrics
    src_prompt = "cat"
    tgt_prompt = "dog"

    metrics_calculator = MetricsCalculator(device)
    src_image_filenames = os.listdir(src_image_folder)

    for method in tgt_methods:
        result_path = f"{method}_{result_path}"
        tgt_image_folder = all_tgt_image_folders[method]
        
        with open(result_path, 'w', newline="") as f:
            csv_write = csv.writer(f)
            csv_head = ["filename"] + metrics
            csv_write.writerow(csv_head)


        for i, image_filename in enumerate(src_image_filenames):
            evaluation_results = [image_filename]
            
            src_image_filename = os.path.join(src_image_folder, image_filename)
            tgt_image_filename = os.path.join(tgt_image_folder, image_filename)
            
            src_image = Image.open(src_image_filename).convert("RGB").resize([512, 512])
            tgt_image = Image.open(tgt_image_filename).convert("RGB").resize([512, 512])
            
            for metric in metrics:
                m = calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_prompt, tgt_prompt)
                evaluation_results.append(m)
                print(m)
            print(src_image_filename, "DONE!")
            with open(result_path,'a+', newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_results)
        
        