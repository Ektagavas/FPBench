import json
import numpy as np

import cv2
import matplotlib.pyplot as plt
import os

import math

f = lambda x,y: 2*x*y
g = lambda x,y: x**2 - y**2

def get_line_ends(x, y, tang, block_size, offset=0):
	x, y = x*block_size, y*block_size
	half_block = (block_size/float(2))

	if offset < 0:
		offset = 0
	elif offset > block_size/2:
		offset = block_size/2

	if -1 <= tang <= 1:
		x1 = x + offset
		y1 = y + half_block - (tang * half_block)
		x2 = x + block_size - offset
		y2 = y + half_block + (tang * half_block)
	else:
		x1 = x + half_block + (half_block/(2*tang))
		y1 = y + block_size - offset
		x2 = x + half_block - (half_block/(2*tang))
		y2 = y + offset

	return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))

def draw_lines(im, h, w, c, angles, block_size):
    
    for i in range(w//block_size):
        for j in range(h//block_size):
            angle = angles.item(j, i)
            
            if angle != 0:
                angle = -1/math.tan(math.radians(angle))
                p1, p2 = get_line_ends(i, j, angle, block_size, 2)
                cv2.line(im, p1, p2, (0,0,255), 1)
    return im

def orientation(img):
    smooth = False

    block_size = 11

    color_img = cv2.imread(img)
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    # make a reflect border frame to simplify kernel operation on borders
    borderedImg = cv2.copyMakeBorder(img, block_size,block_size,block_size,block_size, cv2.BORDER_DEFAULT)

    # apply a gradient in both axis
    sobelx = cv2.Sobel(borderedImg, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(borderedImg, cv2.CV_64F, 0, 1, ksize=3)

    angles = np.zeros((h//block_size, w//block_size), np.float32)

    for i in range(w//block_size):
        for j in range(h//block_size):
            nominator = 0.
            denominator = 0.

            # calculate the summation of nominator (2*Gx*Gy)
            # and denominator (Gx^2 - Gy^2), where Gx and Gy
            # are the gradient values in the position (j, i)
            for k in range(block_size):
                for l in range(block_size):
                    posX = block_size-1 + (i*block_size) + k
                    posY = block_size-1 + (j*block_size) + l
                    valX = sobelx.item(posY, posX)
                    valY = sobely.item(posY, posX)

                    nominator += f(valX, valY)
                    denominator += g(valX, valY)
            
            # if the strength (norm) of the vector 
            # is not greater than a threshold
            if math.sqrt(nominator**2 + denominator**2) < 1000000:
                angle = 0.
            else:
                if denominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator)
                elif denominator < 0 and nominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator) + math.pi
                else:
                    angle = cv2.fastAtan2(nominator, denominator) - math.pi
                angle /= float(2)

            angles[j][i] = angle

    if smooth:
        angles = cv2.GaussianBlur(angles, (3,3), 0, 0)


    orientationImg = draw_lines(color_img, h, w, 3, angles, block_size)

    return orientationImg

def draw_orientation(h, w, angles, block_size):
	im = np.zeros((h, w), np.uint8)

	for i in range(w//block_size):
		for j in range(h//block_size):	
			dangle = 2*angles.item(j, i)
			v = int(round(dangle * (255/float(360))))
			for k in range(block_size):
				for l in range(block_size):
					im.itemset((j*block_size+l,i*block_size+k), v)
	return im


with open('./FPBench/benchmark/orientation_single_all_data.json','r') as f1:
    bank = json.load(f1)
    data = bank["questions"]

root = './FPBench/orientation_op'

flag = False
for qid in data:
    if True:
        for i, img in enumerate(data[qid]["image_paths"]):
            print(img)
            
            if 'FVC' in img:
                save_path = os.path.join(root,os.path.basename(img))
                if not os.path.exists(save_path):
                    orImg = orientation(img)
                    cv2.imwrite(save_path, orImg)
            else:
                
                save_path = os.path.join(root,os.path.basename(img))
                if not os.path.exists(save_path):
                    orImg = orientation(img)
                    cv2.imwrite(save_path, orImg)
            data[qid]["image_paths"][i] = save_path
            

output_json = './FPBench/benchmark/orientation_single_or_all_data.json'
with open(output_json, "w") as f1:
    bank["questions"] = data
    json.dump(bank, f1, indent=4)