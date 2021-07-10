from keras.models import load_model
import random
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from skimage import img_as_ubyte
#%%

# parameters can be changed
image_path = "test_images\\" + str(3) + ".png"
distance = 10
generated_image = 1000
strict_check = 200

#%%

model = load_model('model.h5')

results={
    0:'BANANA',
    1:'notBANANA'
}

cropped_size = 32
channels = 3

image_original = Image.open(image_path)
image_original = np.array(image_original)

#%%

# crop imagesand find the best banana

start_x_list = []
start_y_list = []
plots = []
results_plot = []
yellow_list = []
pred_list = []

for i in range(generated_image):
    image_cropped = np.zeros((cropped_size,cropped_size,channels))
    start_x = random.randint(0,(len(image_original) - 32))
    start_y = random.randint(0,(len(image_original) - 32))
    start_x_list.append(start_x)
    start_y_list.append(start_y)
    
    for y in range(32):
        for x in range(32):
            for z in range(3):
                image_cropped[y][x][z] = image_original[start_y+y][start_x+x][z]
                
    img = img_as_ubyte(image_cropped/255)
    plots.append(img)
    image_cropped = np.expand_dims(image_cropped,axis=0)
    image_cropped = np.array(image_cropped)
    image_cropped = image_cropped/255
    pred = model.predict_classes([image_cropped])[0]
    results_plot.append(results[pred])
    pred_list.append(model.predict([image_cropped]))
    
    counter = 0
    if results[pred] == 'BANANA':
        for y in range(32):
            for x in range(32):
                for z in range(3):
                    r = image_original[start_y+y][start_x+x][0]
                    g = image_original[start_y+y][start_x+x][1]
                    b = image_original[start_y+y][start_x+x][2]
                    
                    if(174 < r < 256):
                        if(174 < g < 256):
                            if(0 < b < 141):
                                counter = counter + 1
                                
    yellow_list.append(counter)
    
    print(str(i+1)," . cropped image is ",results[pred]," and has ",yellow_list[i]," yellow pixels ")
    
#%%

#create figure from cropped images
fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 4
for i in range(16):
    fig.add_subplot(rows, columns, (i+1))
    # showing image
    plt.imshow(plots[i])
    plt.axis('off')
    title = str(results_plot[i]) + " " + str(yellow_list[i])
    plt.title(title)
    
    
    
#%%

# check around of the chosen banana to find all possible banana images

yellowest = yellow_list.index(max(yellow_list))
plots = []
results_plot = []
yellow_list = []
new_x_start = []
new_y_start = []

for i in range(generated_image):
     
    image_cropped = np.zeros((cropped_size,cropped_size,channels))
    
    x_value_start = start_x_list[yellowest] - distance
    if x_value_start < 0:
        x_value_start = 0
    x_value_end = start_x_list[yellowest] + distance
    if x_value_end >= len(image_original):
        x_value_end = len(image_original)   
    start_x = random.randint(x_value_start,x_value_end)
    new_x_start.append(start_x)
     
    y_value_start = start_y_list[yellowest] - distance
    if y_value_start < 0:
        y_value_start = 0
    y_value_end = start_y_list[yellowest] + distance
    if y_value_end >= len(image_original):
        y_value_end = len(image_original)       
    start_y = random.randint(y_value_start,y_value_end)
    new_y_start.append(start_y)
    
    for y in range(32):
        for x in range(32):
            for z in range(3):
                y_value = start_y+y
                if y_value > 255:
                    y_value = 255
                x_value = start_x+x
                if x_value > 255:
                    x_value = 255
                    
                image_cropped[y][x][z] = image_original[y_value][x_value][z]
                
    
    img = img_as_ubyte(image_cropped/255)
    plots.append(img)
    image_cropped = np.expand_dims(image_cropped,axis=0)
    image_cropped = np.array(image_cropped)
    image_cropped = image_cropped/255
    pred = model.predict_classes([image_cropped])[0]
    results_plot.append(results[pred])
    
    
    counter = 0
    if results[pred] == 'BANANA':
        for y in range(32):
            for x in range(32):
                for z in range(3):
                    
                    y_value = start_y+y
                    if y_value > 255:
                        y_value = 255
                    x_value = start_x+x
                    if x_value > 255:
                        x_value = 255
                    
                    r = image_original[y_value][x_value][0]
                    g = image_original[y_value][x_value][1]
                    b = image_original[y_value][x_value][2]
                    
                    if(174 < r < 256):
                        if(174 < g < 256):
                            if(0 < b < 141):
                                counter = counter + 1
                                
    yellow_list.append(counter)
    
    print(str(i+1)," . image from around of selected banana image is ",results[pred]," and has ",yellow_list[i]," yellow pixels")

#%%

#create figure from bananas
fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 4
for i in range(16):
    fig.add_subplot(rows, columns, (i+1))
    # showing image
    plt.imshow(plots[i])
    plt.axis('off')
    title = str(results_plot[i]) + " " + str(yellow_list[i])
    plt.title(title)
    
#%%

# merge detected bananas and draw rectangle

detect_x = []
detect_y = []

bbox_x1 = 256
bbox_y1 = 256
bbox_x2 = 0
bbox_y2 = 0

for i in range(len(yellow_list)):
    if yellow_list[i] > strict_check :
        if new_x_start[i] < bbox_x1:
            bbox_x1 = new_x_start[i]
        if new_x_start[i]+32 > bbox_x2:
            bbox_x2 = new_x_start[i]+32
        if new_y_start[i] < bbox_y1:
            bbox_y1 = new_y_start[i]
        if new_y_start[i]+32 > bbox_y2:
            bbox_y2 = new_y_start[i]+32


import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches

img = matplotlib.image.imread(image_path)

figure, ax = plt.subplots(1)
rect = patches.Rectangle((bbox_x1,bbox_y1),bbox_x2-bbox_x1,bbox_y2-bbox_y1, edgecolor='r', facecolor="none", linewidth = 3)
ax.imshow(img)
ax.add_patch(rect)
figure.savefig("detected_banana.png")


#%%

# find best prediction

# biggest = 0
# take_index = 0
# for i in range(len(pred_list)):
#     a = pred_list[i]
#     a = a[0]
#     if a[0] > a[1]:
#         biggest = a[0]
#         take_index = i
    