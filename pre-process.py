import cv2
import numpy as np
import rglob
from PIL import Image
import os
import shutil

file_list = rglob.rglob("processed_train\\organized\\banana", "*")

#%%

def rgb_of_pixel(img_path, x, y):
    im = Image.open(img_path).convert('RGB')
    r, g, b = im.getpixel((x, y))
    a = (r, g, b)
    return a

counter = 0

bananaEliminated = []

for i in range(3842):
    counter = 0
    for x in range(32):
        for y in range(32):
            color = rgb_of_pixel(file_list[i], x, y)
            if(174 < color[0] < 256):
                if(174 < color[1] < 256):
                    if(0 < color[2] < 141):
                        counter = counter + 1
    if counter > 9:
        bananaEliminated.append(os.path.basename(file_list[i]))
        print(file_list[i] + " saved")
    else:
        print(file_list[i] + " removed")
        
dir = os.path.join("processed_train\\eliminated")
if not os.path.exists(dir):
    os.mkdir(dir)
dir = os.path.join("processed_train\\eliminated\\all")
if not os.path.exists(dir):
    os.mkdir(dir)

    
for i in range(len(bananaEliminated)):
    original = "processed_train\\organized\\banana\\" + bananaEliminated[i]
    target = "processed_train\\eliminated\\all\\" + bananaEliminated[i]
    shutil.copyfile(original, target)
    
#%%

import csv
from random import seed
from random import randint

notBanana = []
data = []
with open('processed_train\\organized\\all.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

for i in range(len(data)):
    if(data[i][1] == '-1'):
        notBanana.append(data[i][0])
        
notBananaEliminated = []
for i in range(len(bananaEliminated)):
    randomIndex = randint(0, len(notBanana) - i)
    notBananaEliminated.append(notBanana[randomIndex])
    notBanana.remove(notBanana[randomIndex])
    
for i in range(len(bananaEliminated)):
    original = "processed_train\\organized\\notbanana\\" + notBananaEliminated[i]
    target = "processed_train\\eliminated\\all\\" + notBananaEliminated[i]
    shutil.copyfile(original, target)
    
#%%

mergedList = bananaEliminated + notBananaEliminated
finalList = sorted(mergedList)
finalLabel = []

for i in range(len(finalList)):  
    for j in range(len(data)):
        if(data[j][0] == finalList[i]):
            finalLabel.append(data[j][1])

#%%

with open('processed_train\\eliminated\\datasetFinal.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["name", "label"])
    for i in range(len(finalList)):        
        writer.writerow([finalList[i], finalLabel[i]])
    