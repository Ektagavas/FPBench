import json
import numpy as np

import cv2
import matplotlib.pyplot as plt
import os

def parse_min_file(min_file):
    minutiae = []
    with open(min_file, "r") as f:
        for line in f:
            if ":" in line and ("RIG" in line or "BIF" in line):
                parts = line.split(":")
                # x,y is in the second field
                coords = parts[1].strip().split(",")
                x, y = int(coords[0]), int(coords[1])
                m_type = parts[4].strip()
                minutiae.append((x, y, m_type))
    return minutiae

def plot_minutiae(image_path, min_file, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    minutiae = parse_min_file(min_file)
    
    for (x, y, m_type) in minutiae:
        if m_type == "RIG":   # ridge ending → red
            cv2.circle(color_img, (x, y), 4, (0, 0, 255), 2) # Red circle for ending
        elif m_type == "BIF": # bifurcation → green
            cv2.circle(color_img, (x, y), 4, (57, 255, 20), 2) # Red circle for ending

    cv2.imwrite(save_path, color_img)

with open('./FPBench/benchmark/minutiae_multiple_all_data.json','r') as f:
    bank = json.load(f)
    data = bank["questions"]

flag = False
for qid in data:
    if 'overlay' in data[qid]["question_type"]:
        for i, img in enumerate(data[qid]["image_paths"]):
            if 'FVC' in img:
                min_path = img.replace('.png','.min')
                save_path = img.replace('.png','_min.png')
            else:
                t = img.replace('_imgs_conv', '_op')
                min_path = t.replace('.png','.min')
                save_path = t.replace('.png','_min.png')
                if not os.path.exists(save_path):
                    plot_minutiae(img, min_path, save_path)
                print(img)
            data[qid]["image_paths"][i] = save_path
                



output_json = './FPBench/benchmark/minutiae_multiple_min_all_data.json'
with open(output_json, "w") as f:
    bank["questions"] = data
    json.dump(bank, f, indent=4)