import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from cannyedge import *
import os

DIR = '.\\images\\'

def save_image(img,name,ID):
    cv2.imwrite(DIR+name+ID+'.png',img)
def edge_classify(edge_image,gradient_image,edge_angle):
    """
    Classifies edge pixels based on gradient intensity and orientation
    """
    
    rows,cols = edge_image.shape
    edge_color_map = np.zeros((rows,cols,3),dtype=np.uint8)
    
    max_gradient = np.max(gradient_image)
    min_gradient = np.min(gradient_image)
    
    #Hue values for each orientation
    VERT_COLOR = 0
    DIAG_COLOR = 45
    HOR_COLOR = 90
    MDIAG_COLOR = 135
    
    for i in range(rows):
        for j in range(cols):
            try:
                #normalising gradient value to use in Value in HSV
                norm_gradient = (255.0 - 0.0)*((1.0*gradient_image[i][j] - min_gradient)/(max_gradient - min_gradient))
            except ZeroDivisionError as error:
                pass
            if edge_image[i][j]:
                direction = get_angle_direction(edge_angle[i][j])
                if direction == 0:
                    edge_color_map[i][j] = [VERT_COLOR,255,norm_gradient]
                elif direction == 45:
                    edge_color_map[i][j] = [DIAG_COLOR,255,norm_gradient]
                elif direction == 90:
                    edge_color_map[i][j] = [HOR_COLOR,255,norm_gradient]
                else:
                    edge_color_map[i][j] = [MDIAG_COLOR,255,norm_gradient]
                
    return cv2.cvtColor(edge_color_map,cv2.COLOR_HSV2RGB)

try:
    image_file_path = sys.argv[1]
except IndexError as error:
    print error
    sys.exit()

#Sobel Kernel size
KSIZE =[5]

#Gaussian Kernel size
GSIZE = [3]

#Hysteresis Thresholding
threshold = [(100,200),(200,250),(250,300)]

for ksize in KSIZE:
    for gsize in GSIZE:
        for thresh in threshold:
            
            outname = "G="+str(gsize)+"_"+"S="+str(ksize)+"_"+"T=("+str(thresh[0])+","+str(thresh[1])+")"

            #Reading the image
            img = cv2.imread(image_file_path,0)

            #Applying Canny edge detector
            edge_image = Canny(img,outname,thresh[0],thresh[1],ksize,gsize)

            #Classifying edge pixels according to gradient intensity and orientation
            edge_gradient,edge_angle = get_gradient(img,ksize)
            edge_color_map = edge_classify(edge_image,edge_gradient,edge_angle)

            save_image(edge_image,'cannyOut',outname)
            save_image(edge_color_map,'edgeColorMap',outname)
            
            print outname + " : done"
