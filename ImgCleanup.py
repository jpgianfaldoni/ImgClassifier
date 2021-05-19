import cv2
import os
import os.path
import numpy as np
from sklearn.model_selection import train_test_split


origin_dir =  os.getcwd() + "\\Assets\\Data_Filtered_Resized" #r"C:\Users\jpgia\Documents\Projetos\ImgClassifier\Assets\Data_Filtered_Resized"
destination_dir = os.getcwd() + "\\Assets\\Data_Filtered_Resized2"
output_size = (120,120)

current_dir = destination_dir[:]
j = 0

for character in os.listdir(origin_dir):
    
    if character not in os.listdir(destination_dir):
        os.mkdir(os.path.join(destination_dir, character))
        
    current_dir = os.path.join(origin_dir, character)

    for k, img in enumerate(os.listdir(current_dir)):
        if not img.endswith(('.gif','.svg','.asp')):
            j += 1
            original_img = cv2.imread(os.path.join(current_dir, img), cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(original_img, output_size)
            success = cv2.imwrite(os.path.join(destination_dir, character, str(j) + '.jpg'), resized)
            
        else:
            print(character,img)
            
print('Done.')